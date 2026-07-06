from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import requests
import xarray as xr

from .config import LEVEL_HPA

LOG = logging.getLogger(__name__)


class ModelDownloadError(RuntimeError):
    """Raised when a model field cannot be downloaded or decoded."""


@dataclass(slots=True)
class ModelField:
    key: str
    label: str
    source: str
    cycle: datetime
    latitude: np.ndarray
    longitude: np.ndarray
    u_ms: np.ndarray
    v_ms: np.ndarray


def _download(url: str, target: Path, *, params: dict[str, str] | None = None) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    LOG.info("Downloading %s", url)
    with requests.get(url, params=params, stream=True, timeout=(30, 180)) as response:
        response.raise_for_status()
        with target.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    _validate_grib(target)
    return target


def _validate_grib(path: Path) -> None:
    if path.stat().st_size < 100:
        raise ModelDownloadError(f"Downloaded file is too small to be GRIB2: {path}")
    with path.open("rb") as handle:
        magic = handle.read(4)
    if magic != b"GRIB":
        raise ModelDownloadError(f"Downloaded content is not GRIB2: {path}")


def _coordinate(dataset: xr.Dataset, candidates: Iterable[str]) -> np.ndarray:
    for name in candidates:
        if name in dataset.coords:
            return np.asarray(dataset[name].values, dtype=float)
    raise ModelDownloadError(f"Could not find coordinate among {tuple(candidates)}")


def _variable(dataset: xr.Dataset, candidates: Iterable[str]) -> np.ndarray:
    for name in candidates:
        if name in dataset.data_vars:
            values = dataset[name]
            if "isobaricInhPa" in values.dims:
                values = values.sel(isobaricInhPa=LEVEL_HPA)
            return np.asarray(values.squeeze().values, dtype=float)
    raise ModelDownloadError(
        f"Could not find a wind component among {tuple(candidates)}; found {list(dataset.data_vars)}"
    )


def _open_uv_file(path: Path, key: str, label: str, source: str, cycle: datetime) -> ModelField:
    try:
        dataset = xr.open_dataset(
            path,
            engine="cfgrib",
            backend_kwargs={"indexpath": "", "filter_by_keys": {"typeOfLevel": "isobaricInhPa"}},
        )
    except Exception:
        dataset = xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})

    latitude = _coordinate(dataset, ("latitude", "lat"))
    longitude = _coordinate(dataset, ("longitude", "lon"))
    u_ms = _variable(dataset, ("u", "u_component_of_wind", "UGRD"))
    v_ms = _variable(dataset, ("v", "v_component_of_wind", "VGRD"))
    dataset.close()
    return ModelField(key, label, source, cycle, latitude, longitude, u_ms, v_ms)


def _open_component(path: Path, component: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        dataset = xr.open_dataset(
            path,
            engine="cfgrib",
            backend_kwargs={"indexpath": "", "filter_by_keys": {"typeOfLevel": "isobaricInhPa"}},
        )
    except Exception:
        dataset = xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})
    latitude = _coordinate(dataset, ("latitude", "lat"))
    longitude = _coordinate(dataset, ("longitude", "lon"))
    values = _variable(dataset, (component, component.upper(), f"{component}_component_of_wind"))
    dataset.close()
    return latitude, longitude, values


def fetch_gfs(cycle: datetime, workdir: Path) -> ModelField:
    date = cycle.strftime("%Y%m%d")
    hour = cycle.strftime("%H")
    target = workdir / f"gfs_{date}{hour}_300.grib2"
    params = {
        "file": f"gfs.t{hour}z.pgrb2.0p25.f000",
        "lev_300_mb": "on",
        "var_UGRD": "on",
        "var_VGRD": "on",
        "leftlon": "0",
        "rightlon": "360",
        "toplat": "90",
        "bottomlat": "-90",
        "dir": f"/gfs.{date}/{hour}/atmos",
    }
    _download("https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl", target, params=params)
    return _open_uv_file(target, "gfs", "GFS", "NOAA/NCEP NOMADS", cycle)


def fetch_ecmwf(cycle: datetime, workdir: Path) -> ModelField:
    try:
        from ecmwf.opendata import Client
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise ModelDownloadError("ecmwf-opendata is not installed") from exc

    target = workdir / f"ecmwf_{cycle:%Y%m%d%H}_300.grib2"
    client = Client(source="aws", model="ifs", resol="0p25")
    try:
        client.retrieve(
            date=cycle.strftime("%Y%m%d"),
            time=cycle.hour,
            step=0,
            stream="oper",
            type="fc",
            levtype="pl",
            param=["u", "v"],
            levelist=[LEVEL_HPA],
            target=str(target),
        )
    except Exception as exc:  # pragma: no cover - network/client behavior
        raise ModelDownloadError(f"ECMWF Open Data retrieval failed: {exc}") from exc
    _validate_grib(target)
    return _open_uv_file(target, "ecmwf", "ECMWF IFS", "ECMWF Open Data (AWS mirror)", cycle)


def _gdps_filename(cycle: datetime, component: str) -> str:
    return (
        f"{cycle:%Y%m%d}T{cycle:%H}Z_MSC_GDPS_{component}_"
        f"IsbL-0{LEVEL_HPA:03d}_LatLon0.15_PT000H.grib2"
    )


def _gdps_legacy_filename(cycle: datetime, component: str) -> str:
    return (
        f"CMC_glb_{component}_ISBL_{LEVEL_HPA:04d}_latlon.15x.15_"
        f"{cycle:%Y%m%d%H}_P000.grib2"
    )


def _download_first(candidates: list[str], target: Path) -> Path:
    errors: list[str] = []
    for url in candidates:
        try:
            return _download(url, target)
        except Exception as exc:  # pragma: no cover - depends on remote archive layout
            errors.append(f"{url}: {exc}")
            target.unlink(missing_ok=True)
    raise ModelDownloadError("No GDPS URL candidate succeeded:\n" + "\n".join(errors))


def fetch_gdps(cycle: datetime, workdir: Path) -> ModelField:
    date = cycle.strftime("%Y%m%d")
    hour = cycle.strftime("%H")
    component_files: dict[str, Path] = {}

    for component in ("UGRD", "VGRD"):
        filename = _gdps_filename(cycle, component)
        legacy = _gdps_legacy_filename(cycle, component)
        candidates = [
            f"https://dd.weather.gc.ca/{date}/WXO-DD/model_gdps/15km/{hour}/000/{filename}",
            f"https://dd.weather.gc.ca/today/model_gdps/15km/{hour}/000/{filename}",
            (
                f"https://dd.weather.gc.ca/{date}/WXO-DD/model_gem_global/15km/"
                f"grib2/lat_lon/{hour}/000/{legacy}"
            ),
        ]
        target = workdir / f"gdps_{date}{hour}_{component}_300.grib2"
        component_files[component] = _download_first(candidates, target)

    lat_u, lon_u, u_ms = _open_component(component_files["UGRD"], "u")
    lat_v, lon_v, v_ms = _open_component(component_files["VGRD"], "v")
    if lat_u.shape != lat_v.shape or lon_u.shape != lon_v.shape:
        raise ModelDownloadError("GDPS U and V grids do not match")
    return ModelField(
        "gdps",
        "Canadian GDPS",
        "Environment and Climate Change Canada MSC Datamart",
        cycle,
        lat_u,
        lon_u,
        u_ms,
        v_ms,
    )


FETCHERS = {"gfs": fetch_gfs, "ecmwf": fetch_ecmwf, "gdps": fetch_gdps}

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
GRAVITY = 9.80665


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
    height_m: np.ndarray


def _download(url: str, target: Path, *, params: dict[str, str] | None = None) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    LOG.info("Downloading %s", url)
    with requests.get(url, params=params, stream=True, timeout=(30, 240)) as response:
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


def _select_level(values: xr.DataArray) -> xr.DataArray:
    for coord in ("isobaricInhPa", "level", "pressure_level"):
        if coord in values.dims or coord in values.coords:
            try:
                return values.sel({coord: LEVEL_HPA})
            except Exception:
                pass
    return values


def _data_array(dataset: xr.Dataset, candidates: Iterable[str]) -> xr.DataArray:
    for name in candidates:
        if name in dataset.data_vars:
            return _select_level(dataset[name]).squeeze()
    raise ModelDownloadError(
        f"Could not find any of {tuple(candidates)}; found {list(dataset.data_vars)}"
    )


def _height_values(values: xr.DataArray) -> np.ndarray:
    output = np.asarray(values.values, dtype=float)
    units = str(values.attrs.get("units", "")).lower().replace(" ", "")
    if "m**2" in units or "m2s-2" in units or "m^2" in units or np.nanmedian(output) > 20000:
        output = output / GRAVITY
    return output


def _open_dataset(path: Path) -> xr.Dataset:
    try:
        return xr.open_dataset(
            path,
            engine="cfgrib",
            backend_kwargs={"indexpath": "", "filter_by_keys": {"typeOfLevel": "isobaricInhPa"}},
        )
    except Exception:
        return xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})


def _open_field(path: Path, key: str, label: str, source: str, cycle: datetime) -> ModelField:
    dataset = _open_dataset(path)
    try:
        latitude = _coordinate(dataset, ("latitude", "lat"))
        longitude = _coordinate(dataset, ("longitude", "lon"))
        u_ms = np.asarray(_data_array(dataset, ("u", "u_component_of_wind", "UGRD")).values, dtype=float)
        v_ms = np.asarray(_data_array(dataset, ("v", "v_component_of_wind", "VGRD")).values, dtype=float)
        height_m = _height_values(
            _data_array(dataset, ("gh", "z", "hgt", "HGT", "geopotential_height"))
        )
    finally:
        dataset.close()
    return ModelField(key, label, source, cycle, latitude, longitude, u_ms, v_ms, height_m)


def _open_component(path: Path, candidates: tuple[str, ...], *, height: bool = False):
    dataset = _open_dataset(path)
    try:
        latitude = _coordinate(dataset, ("latitude", "lat"))
        longitude = _coordinate(dataset, ("longitude", "lon"))
        data_array = _data_array(dataset, candidates)
        values = _height_values(data_array) if height else np.asarray(data_array.values, dtype=float)
    finally:
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
        "var_HGT": "on",
        "leftlon": "0",
        "rightlon": "360",
        "toplat": "90",
        "bottomlat": "-90",
        "dir": f"/gfs.{date}/{hour}/atmos",
    }
    _download("https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl", target, params=params)
    return _open_field(target, "gfs", "GFS", "NOAA/NCEP NOMADS", cycle)


def fetch_ecmwf(cycle: datetime, workdir: Path) -> ModelField:
    try:
        from ecmwf.opendata import Client
    except ImportError as exc:
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
            param=["u", "v", "gh"],
            levelist=[LEVEL_HPA],
            target=str(target),
        )
    except Exception as exc:
        raise ModelDownloadError(f"ECMWF Open Data retrieval failed: {exc}") from exc
    _validate_grib(target)
    return _open_field(target, "ecmwf", "ECMWF IFS", "ECMWF Open Data (AWS mirror)", cycle)


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
        except Exception as exc:
            errors.append(f"{url}: {exc}")
            target.unlink(missing_ok=True)
    raise ModelDownloadError("No GDPS URL candidate succeeded:\n" + "\n".join(errors))


def fetch_gdps(cycle: datetime, workdir: Path) -> ModelField:
    date = cycle.strftime("%Y%m%d")
    hour = cycle.strftime("%H")
    component_files: dict[str, Path] = {}

    for component in ("UGRD", "VGRD", "HGT"):
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

    lat_u, lon_u, u_ms = _open_component(component_files["UGRD"], ("u", "UGRD"))
    lat_v, lon_v, v_ms = _open_component(component_files["VGRD"], ("v", "VGRD"))
    lat_h, lon_h, height_m = _open_component(
        component_files["HGT"], ("gh", "z", "hgt", "HGT"), height=True
    )
    if not (
        lat_u.shape == lat_v.shape == lat_h.shape
        and lon_u.shape == lon_v.shape == lon_h.shape
    ):
        raise ModelDownloadError("GDPS U, V, and height grids do not match")
    return ModelField(
        "gdps",
        "Canadian GDPS",
        "Environment and Climate Change Canada MSC Datamart",
        cycle,
        lat_u,
        lon_u,
        u_ms,
        v_ms,
        height_m,
    )


FETCHERS = {"gfs": fetch_gfs, "ecmwf": fetch_ecmwf, "gdps": fetch_gdps}

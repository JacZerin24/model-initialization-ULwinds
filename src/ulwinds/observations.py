from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
import pandas as pd
import requests

from .config import LEVEL_HPA

LOG = logging.getLogger(__name__)
IEM_RAOB_URL = "https://mesonet.agron.iastate.edu/json/raob.py"
IEM_RAOB_NETWORK_URL = "https://mesonet.agron.iastate.edu/geojson/network.py"
IGRA_STATION_LIST_URL = "https://www.ncei.noaa.gov/pub/data/igra/igra2-station-list.txt"
REQUEST_HEADERS = {
    "User-Agent": "model-initialization-ULwinds/0.2 (GitHub Pages verification project)"
}


class ObservationError(RuntimeError):
    """Raised when radiosonde observations cannot be retrieved or parsed."""


def wind_direction_from_uv(u: np.ndarray | float, v: np.ndarray | float) -> np.ndarray:
    return (270.0 - np.degrees(np.arctan2(v, u))) % 360.0


def uv_from_speed_direction(speed, direction) -> tuple[np.ndarray, np.ndarray]:
    radians = np.radians(np.asarray(direction, dtype=float))
    speed_values = np.asarray(speed, dtype=float)
    return -speed_values * np.sin(radians), -speed_values * np.cos(radians)


def _request(url: str, params: dict[str, object] | None = None) -> requests.Response:
    try:
        response = requests.get(
            url,
            params=params,
            headers=REQUEST_HEADERS,
            timeout=(20, 180),
        )
        response.raise_for_status()
        return response
    except requests.RequestException as exc:
        raise ObservationError(f"Observation metadata request failed for {url}: {exc}") from exc


def _get_json(url: str, params: dict[str, object]) -> dict[str, object]:
    response = _request(url, params)
    try:
        payload = response.json()
    except ValueError as exc:
        raise ObservationError(f"IEM returned invalid JSON for {url}") from exc
    if not isinstance(payload, dict):
        raise ObservationError(f"IEM returned an unexpected payload for {url}")
    return payload


def _metadata_record(name: str, latitude: float, longitude: float, source: str) -> dict[str, object]:
    return {
        "name": name,
        "latitude": latitude,
        "longitude": longitude,
        "metadata_source": source,
    }


def _parse_iem_station_metadata(payload: dict[str, object]) -> dict[str, dict[str, object]]:
    stations: dict[str, dict[str, object]] = {}
    for feature in payload.get("features", []) or []:
        if not isinstance(feature, dict):
            continue
        properties = feature.get("properties") or {}
        geometry = feature.get("geometry") or {}
        coordinates = geometry.get("coordinates") or []
        station = str(feature.get("id") or properties.get("sid") or "").strip().upper()
        if not station or len(coordinates) < 2:
            continue
        try:
            longitude = float(coordinates[0])
            latitude = float(coordinates[1])
        except (TypeError, ValueError):
            continue
        stations[station] = _metadata_record(
            str(properties.get("sname") or station), latitude, longitude, "IEM RAOB network"
        )
    return stations


_parse_station_metadata = _parse_iem_station_metadata


def _parse_igra_station_list(text: str) -> dict[str, dict[str, object]]:
    """Parse the fixed-width IGRA inventory and index ICAO/WMO aliases."""
    stations: dict[str, dict[str, object]] = {}
    for line in text.splitlines():
        if len(line) < 71:
            continue
        station_id = line[0:11].strip().upper()
        try:
            latitude = float(line[12:20])
            longitude = float(line[21:30])
        except ValueError:
            continue
        if latitude <= -98 or longitude <= -998 or not (-90 <= latitude <= 90):
            continue
        name = line[41:71].strip().title() or station_id
        record = _metadata_record(name, latitude, longitude, "NOAA/NCEI IGRA station inventory")
        aliases = {station_id}
        if len(station_id) == 11:
            network = station_id[2]
            if network == "M":
                aliases.add(station_id[-5:])
                aliases.add(station_id[-5:].lstrip("0") or "0")
            elif network == "I":
                aliases.add(station_id[-4:])
            elif network == "W":
                aliases.add(station_id[-5:])
        for alias in aliases:
            stations.setdefault(alias, record)
    return stations


def _station_key_candidates(station: str) -> list[str]:
    value = station.strip().upper()
    candidates = [value]
    if value.isdigit():
        candidates.extend([value.zfill(5), value.lstrip("0") or "0"])
    if len(value) == 4 and value.startswith("K"):
        candidates.append(value[1:])
    return list(dict.fromkeys(candidates))


def _parse_raob_profiles(
    payload: dict[str, object],
    metadata: dict[str, dict[str, object]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    missing_metadata = 0

    for profile in payload.get("profiles", []) or []:
        if not isinstance(profile, dict):
            continue
        station = str(profile.get("station") or "").strip().upper()
        station_meta = next(
            (metadata[key] for key in _station_key_candidates(station) if key in metadata),
            None,
        )
        if station_meta is None:
            missing_metadata += 1
            continue

        best: dict[str, object] | None = None
        best_distance = np.inf
        for point in profile.get("profile", []) or []:
            if not isinstance(point, dict):
                continue
            pressure = point.get("pres")
            speed = point.get("sknt")
            direction = point.get("drct")
            if pressure is None or speed is None or direction is None:
                continue
            try:
                distance = abs(float(pressure) - LEVEL_HPA)
            except (TypeError, ValueError):
                continue
            if distance < best_distance:
                best = point
                best_distance = distance

        if best is None or best_distance > 1.5:
            continue
        try:
            speed_kt = float(best["sknt"])
            direction_deg = float(best["drct"]) % 360.0
        except (TypeError, ValueError, KeyError):
            continue
        if not np.isfinite(speed_kt) or not 0 <= speed_kt <= 250:
            continue
        try:
            obs_height_m = float(best.get("hght")) if best.get("hght") is not None else np.nan
        except (TypeError, ValueError):
            obs_height_m = np.nan

        u_kt, v_kt = uv_from_speed_direction(speed_kt, direction_deg)
        rows.append(
            {
                "station": station,
                "name": station_meta["name"],
                "latitude": float(station_meta["latitude"]),
                "longitude": float(station_meta["longitude"]),
                "metadata_source": station_meta.get("metadata_source", "unknown"),
                "obs_u_kt": float(u_kt),
                "obs_v_kt": float(v_kt),
                "obs_speed_kt": speed_kt,
                "obs_height_m": obs_height_m,
                "obs_direction_deg": direction_deg,
                "vertical_method": "observed 300 hPa level",
                "observation_time": str(profile.get("valid") or ""),
            }
        )

    if missing_metadata:
        LOG.warning("Skipped %d RAOB profiles without station metadata", missing_metadata)

    data = pd.DataFrame(rows)
    if data.empty:
        return data
    data = data[
        data["latitude"].between(-90, 90) & data["longitude"].between(-180, 360)
    ].copy()
    data["longitude"] = ((data["longitude"] + 180) % 360) - 180
    return data.drop_duplicates(subset=["station"], keep="first").reset_index(drop=True)


def fetch_raob_300(cycle: datetime) -> pd.DataFrame:
    """Retrieve global 300-hPa RAOB winds with IEM and IGRA station metadata."""
    igra_text = _request(IGRA_STATION_LIST_URL).text
    metadata = _parse_igra_station_list(igra_text)

    iem_payload = _get_json(IEM_RAOB_NETWORK_URL, {"network": "RAOB"})
    metadata.update(_parse_iem_station_metadata(iem_payload))
    if not metadata:
        raise ObservationError("Combined IEM/IGRA RAOB station metadata was empty")

    profile_payload = _get_json(
        IEM_RAOB_URL,
        {"ts": cycle.strftime("%Y%m%d%H00"), "pressure": LEVEL_HPA},
    )
    observations = _parse_raob_profiles(profile_payload, metadata)
    if observations.empty:
        raise ObservationError(f"No quality-controlled 300-hPa RAOB winds for {cycle:%Y-%m-%d %HZ}")

    non_us_like = (~observations["station"].str.startswith("K")) & (
        ~observations["station"].str.startswith("P")
    )
    LOG.info(
        "Prepared %d global RAOB stations at 300 hPa (%d non-K/P identifiers)",
        len(observations),
        int(non_us_like.sum()),
    )
    return observations

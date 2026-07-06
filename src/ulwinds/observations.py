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
REQUEST_HEADERS = {
    "User-Agent": "model-initialization-ULwinds/0.1 (GitHub Pages verification project)"
}


class ObservationError(RuntimeError):
    """Raised when radiosonde observations cannot be retrieved or parsed."""


def wind_direction_from_uv(u: np.ndarray | float, v: np.ndarray | float) -> np.ndarray:
    """Meteorological wind direction in degrees from which the wind blows."""
    return (270.0 - np.degrees(np.arctan2(v, u))) % 360.0


def uv_from_speed_direction(
    speed: np.ndarray | pd.Series | float,
    direction: np.ndarray | pd.Series | float,
) -> tuple[np.ndarray, np.ndarray]:
    radians = np.radians(np.asarray(direction, dtype=float))
    speed_values = np.asarray(speed, dtype=float)
    return -speed_values * np.sin(radians), -speed_values * np.cos(radians)


def _get_json(url: str, params: dict[str, object]) -> dict[str, object]:
    try:
        response = requests.get(
            url,
            params=params,
            headers=REQUEST_HEADERS,
            timeout=(20, 120),
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise ObservationError(f"IEM request failed for {url}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ObservationError(f"IEM returned an unexpected payload for {url}")
    return payload


def _parse_station_metadata(payload: dict[str, object]) -> dict[str, dict[str, object]]:
    stations: dict[str, dict[str, object]] = {}
    for feature in payload.get("features", []) or []:
        if not isinstance(feature, dict):
            continue
        properties = feature.get("properties") or {}
        geometry = feature.get("geometry") or {}
        coordinates = geometry.get("coordinates") or []
        station = str(feature.get("id") or properties.get("sid") or "").strip()
        if not station or len(coordinates) < 2:
            continue
        try:
            longitude = float(coordinates[0])
            latitude = float(coordinates[1])
        except (TypeError, ValueError):
            continue
        stations[station] = {
            "name": str(properties.get("sname") or station),
            "latitude": latitude,
            "longitude": longitude,
        }
    return stations


def _parse_raob_profiles(
    payload: dict[str, object],
    metadata: dict[str, dict[str, object]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    missing_metadata = 0

    for profile in payload.get("profiles", []) or []:
        if not isinstance(profile, dict):
            continue
        station = str(profile.get("station") or "").strip()
        station_meta = metadata.get(station)
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

        u_kt, v_kt = uv_from_speed_direction(speed_kt, direction_deg)
        rows.append(
            {
                "station": station,
                "name": station_meta["name"],
                "latitude": float(station_meta["latitude"]),
                "longitude": float(station_meta["longitude"]),
                "obs_u_kt": float(u_kt),
                "obs_v_kt": float(v_kt),
                "obs_speed_kt": speed_kt,
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
    """Retrieve global IEM RAOB winds observed at the 300-hPa mandatory level."""
    metadata_payload = _get_json(IEM_RAOB_NETWORK_URL, {"network": "RAOB"})
    metadata = _parse_station_metadata(metadata_payload)
    if not metadata:
        raise ObservationError("IEM RAOB station metadata response was empty")

    profile_payload = _get_json(
        IEM_RAOB_URL,
        {"ts": cycle.strftime("%Y%m%d%H00"), "pressure": LEVEL_HPA},
    )
    observations = _parse_raob_profiles(profile_payload, metadata)
    if observations.empty:
        raise ObservationError(f"No quality-controlled 300-hPa RAOB winds for {cycle:%Y-%m-%d %HZ}")

    LOG.info("Prepared %d global RAOB stations at 300 hPa", len(observations))
    return observations

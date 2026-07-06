from __future__ import annotations

import html as html_lib
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd
import requests

from .config import LEVEL_HPA, MPS_TO_KT

LOG = logging.getLogger(__name__)
IEM_RAOB_URL = "https://mesonet.agron.iastate.edu/json/raob.py"
IEM_RAOB_NETWORK_URL = "https://mesonet.agron.iastate.edu/geojson/network.py"
IGRA_STATION_LIST_URL = "https://www.ncei.noaa.gov/pub/data/igra/igra2-station-list.txt"
UWYO_SOUNDING_URL = "https://weather.uwyo.edu/wsgi/sounding"
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; model-initialization-ULwinds/0.3; +https://github.com/JacZerin24/model-initialization-ULwinds)",
    "Accept": "text/html,application/xhtml+xml,application/json,text/plain",
}
UWYO_WORKERS = 8
MIN_INTERNATIONAL_PROFILES = 20


class ObservationError(RuntimeError):
    """Raised when radiosonde observations cannot be retrieved or parsed."""


def wind_direction_from_uv(u: np.ndarray | float, v: np.ndarray | float) -> np.ndarray:
    return (270.0 - np.degrees(np.arctan2(v, u))) % 360.0


def uv_from_speed_direction(speed, direction) -> tuple[np.ndarray, np.ndarray]:
    radians = np.radians(np.asarray(direction, dtype=float))
    speed_values = np.asarray(speed, dtype=float)
    return -speed_values * np.sin(radians), -speed_values * np.cos(radians)


def _request(
    url: str,
    params: dict[str, object] | None = None,
    *,
    timeout: tuple[int, int] = (20, 180),
) -> requests.Response:
    try:
        response = requests.get(
            url,
            params=params,
            headers=REQUEST_HEADERS,
            timeout=timeout,
        )
        response.raise_for_status()
        return response
    except requests.RequestException as exc:
        raise ObservationError(f"Observation request failed for {url}: {exc}") from exc


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


def _active_international_wmo_stations(text: str, cycle_year: int) -> list[dict[str, object]]:
    """Return recently active non-U.S. IGRA stations with traditional WMO IDs."""
    stations: dict[str, dict[str, object]] = {}
    for line in text.splitlines():
        if len(line) < 81:
            continue
        station_id = line[0:11].strip().upper()
        if len(station_id) != 11 or station_id[2] != "M" or station_id[:2] == "US":
            continue
        try:
            latitude = float(line[12:20])
            longitude = float(line[21:30])
            last_year = int(line[77:81])
        except ValueError:
            continue
        if latitude <= -98 or longitude <= -998 or last_year < cycle_year - 1:
            continue
        wmo_id = station_id[-5:]
        name = line[41:71].strip().title() or wmo_id
        stations[wmo_id] = {
            "station": wmo_id,
            "name": name,
            "latitude": latitude,
            "longitude": longitude,
            "metadata_source": "NOAA/NCEI IGRA station inventory",
            "country_code": station_id[:2],
        }
    return list(stations.values())


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
                "profile_source": "Iowa Environmental Mesonet RAOB archive",
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
        LOG.warning("Skipped %d IEM RAOB profiles without station metadata", missing_metadata)
    return pd.DataFrame(rows)


def _pre_blocks(page: str) -> list[str]:
    return [
        re.sub(r"<[^>]+>", "", html_lib.unescape(block))
        for block in re.findall(r"<pre[^>]*>(.*?)</pre>", page, flags=re.IGNORECASE | re.DOTALL)
    ]


def _parse_uwyo_300_hpa(page: str) -> dict[str, float] | None:
    """Extract the observed 300-hPa row from Wyoming's current text-list page."""
    block = next((item for item in _pre_blocks(page) if "PRES" in item and "HGHT" in item), None)
    if block is None:
        return None
    header = "\n".join(block.splitlines()[:8]).lower()
    speed_is_ms = "m/s" in header or "m s-1" in header or "m s**-1" in header
    best: dict[str, float] | None = None
    best_distance = np.inf
    for line in block.splitlines():
        if len(line) < 56:
            continue
        try:
            pressure = float(line[0:7])
            height_m = float(line[7:14])
            direction_deg = float(line[42:49]) % 360.0
            raw_speed = float(line[49:56])
        except ValueError:
            continue
        distance = abs(pressure - LEVEL_HPA)
        if distance < best_distance:
            speed_kt = raw_speed * MPS_TO_KT if speed_is_ms else raw_speed
            best = {
                "pressure_hpa": pressure,
                "height_m": height_m,
                "direction_deg": direction_deg,
                "speed_kt": speed_kt,
            }
            best_distance = distance
    if best is None or best_distance > 1.5 or not 0 <= best["speed_kt"] <= 250:
        return None
    return best


def _fetch_uwyo_station(cycle: datetime, station: dict[str, object]) -> dict[str, object] | None:
    params = {
        "datetime": cycle.strftime("%Y-%m-%d %H:%M:%S"),
        "id": station["station"],
        "src": "FM35",
        "type": "TEXT:LIST",
    }
    for attempt in range(2):
        try:
            response = requests.get(
                UWYO_SOUNDING_URL,
                params=params,
                headers=REQUEST_HEADERS,
                timeout=(10, 35),
            )
            if response.status_code in (400, 404):
                return None
            response.raise_for_status()
            point = _parse_uwyo_300_hpa(response.text)
            if point is None:
                return None
            u_kt, v_kt = uv_from_speed_direction(point["speed_kt"], point["direction_deg"])
            return {
                "station": str(station["station"]),
                "name": str(station["name"]),
                "latitude": float(station["latitude"]),
                "longitude": float(station["longitude"]),
                "metadata_source": str(station["metadata_source"]),
                "profile_source": "University of Wyoming global radiosonde archive",
                "obs_u_kt": float(u_kt),
                "obs_v_kt": float(v_kt),
                "obs_speed_kt": float(point["speed_kt"]),
                "obs_height_m": float(point["height_m"]),
                "obs_direction_deg": float(point["direction_deg"]),
                "vertical_method": "observed 300 hPa level",
                "observation_time": cycle.isoformat().replace("+00:00", "Z"),
            }
        except requests.RequestException:
            if attempt == 0:
                time.sleep(0.5)
    return None


def _fetch_international_uwyo(cycle: datetime, stations: list[dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    completed = 0
    with ThreadPoolExecutor(max_workers=UWYO_WORKERS) as executor:
        futures = {executor.submit(_fetch_uwyo_station, cycle, station): station for station in stations}
        for future in as_completed(futures):
            completed += 1
            try:
                row = future.result()
            except Exception as exc:
                LOG.debug("Wyoming station fetch failed: %s", exc)
                row = None
            if row is not None:
                rows.append(row)
            if completed % 100 == 0:
                LOG.info("Checked %d/%d international stations; %d profiles found", completed, len(stations), len(rows))
    return pd.DataFrame(rows)


def _normalize_observations(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return data
    data = data[
        data["latitude"].between(-90, 90) & data["longitude"].between(-180, 360)
    ].copy()
    data["longitude"] = ((data["longitude"] + 180) % 360) - 180
    data["location_key"] = (
        data["latitude"].round(2).astype(str) + ":" + data["longitude"].round(2).astype(str)
    )
    data = data.drop_duplicates(subset=["location_key"], keep="first").drop(columns="location_key")
    return data.reset_index(drop=True)


def fetch_raob_300(cycle: datetime) -> pd.DataFrame:
    """Retrieve U.S. IEM profiles and true international profiles from Wyoming."""
    igra_text = _request(IGRA_STATION_LIST_URL).text
    metadata = _parse_igra_station_list(igra_text)
    international_stations = _active_international_wmo_stations(igra_text, cycle.year)

    iem_payload = _get_json(IEM_RAOB_NETWORK_URL, {"network": "RAOB"})
    metadata.update(_parse_iem_station_metadata(iem_payload))
    profile_payload = _get_json(
        IEM_RAOB_URL,
        {"ts": cycle.strftime("%Y%m%d%H00"), "pressure": LEVEL_HPA},
    )
    iem_observations = _parse_raob_profiles(profile_payload, metadata)
    international_observations = _fetch_international_uwyo(cycle, international_stations)

    if len(international_observations) < MIN_INTERNATIONAL_PROFILES:
        raise ObservationError(
            "Global RAOB acquisition returned only "
            f"{len(international_observations)} international profiles; refusing to deploy a misleading U.S.-only dataset"
        )

    observations = _normalize_observations(
        pd.concat([iem_observations, international_observations], ignore_index=True, sort=False)
    )
    if observations.empty:
        raise ObservationError(f"No quality-controlled 300-hPa RAOB winds for {cycle:%Y-%m-%d %HZ}")

    LOG.info(
        "Prepared %d total 300-hPa RAOBs: %d IEM and %d international Wyoming profiles",
        len(observations),
        len(iem_observations),
        len(international_observations),
    )
    return observations

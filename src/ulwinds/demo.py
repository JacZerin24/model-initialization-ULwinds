from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from .config import MODEL_LABELS, MODEL_ORDER
from .models import ModelField
from .verification import analysis_payload, station_records, summarize, verify_stations


def _base_fields(lat: np.ndarray, lon: np.ndarray):
    lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")
    jet_n = 25 + 70 * np.exp(-((lat2d - 42) / 14) ** 2)
    jet_s = 18 + 55 * np.exp(-((lat2d + 48) / 16) ** 2)
    wave = 12 * np.sin(np.radians(lon2d * 2 + lat2d))
    u_kt = jet_n + jet_s + wave
    v_kt = 14 * np.sin(np.radians(lon2d + lat2d * 1.7))
    height_m = (
        9300
        - 380 * np.sin(np.radians(lat2d))
        + 130 * np.cos(np.radians(lon2d * 2 - lat2d))
        + 70 * np.sin(np.radians(lon2d + lat2d * 3))
    )
    return u_kt / 1.9438444924406, v_kt / 1.9438444924406, height_m


def build_demo_payload() -> dict[str, object]:
    rng = np.random.default_rng(300)
    cycle = datetime(2026, 7, 4, 12, tzinfo=UTC)
    lat = np.arange(-90, 90.25, 2.5)
    lon = np.arange(0, 360, 2.5)
    base_u, base_v, base_height = _base_fields(lat, lon)

    station_count = 70
    station_lats = rng.uniform(-72, 72, station_count)
    station_lons = rng.uniform(-180, 180, station_count)
    nearest_lat = np.abs(lat[:, None] - station_lats).argmin(axis=0)
    nearest_lon = np.abs((lon[:, None] - (station_lons % 360) + 180) % 360 - 180).argmin(axis=0)
    obs_u = base_u[nearest_lat, nearest_lon] * 1.9438444924406 + rng.normal(0, 4, station_count)
    obs_v = base_v[nearest_lat, nearest_lon] * 1.9438444924406 + rng.normal(0, 4, station_count)
    obs_height = base_height[nearest_lat, nearest_lon] + rng.normal(0, 18, station_count)
    obs = pd.DataFrame(
        {
            "station": [f"W{i:04d}" for i in range(station_count)],
            "name": [f"Global Demo RAOB {i:03d}" for i in range(station_count)],
            "latitude": station_lats,
            "longitude": station_lons,
            "metadata_source": "NOAA/NCEI IGRA station inventory",
            "obs_u_kt": obs_u,
            "obs_v_kt": obs_v,
            "obs_speed_kt": np.hypot(obs_u, obs_v),
            "obs_direction_deg": (270 - np.degrees(np.arctan2(obs_v, obs_u))) % 360,
            "obs_height_m": obs_height,
            "vertical_method": "demonstration",
            "observation_time": cycle.isoformat().replace("+00:00", "Z"),
        }
    )

    perturbations = {
        "gfs": (1.5, 1.0, 1.0, 18),
        "ecmwf": (-0.5, 0.5, 0.7, -8),
        "gdps": (3.0, -1.5, 1.3, 30),
    }
    models: dict[str, object] = {}
    lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")
    for key in MODEL_ORDER:
        du, dv, wave_scale, height_bias = perturbations[key]
        u = base_u + du / 1.9438444924406 + wave_scale * np.sin(np.radians(lon2d))
        v = base_v + dv / 1.9438444924406 + 0.5 * wave_scale * np.cos(np.radians(lat2d * 2))
        height = base_height + height_bias + 15 * wave_scale * np.sin(np.radians(lon2d - lat2d))
        field = ModelField(key, MODEL_LABELS[key], "Demonstration field", cycle, lat, lon, u, v, height)
        verified = verify_stations(field, obs)
        models[key] = {
            "label": MODEL_LABELS[key],
            "source": "Demonstration field",
            "status": "ok",
            "metrics": summarize(verified),
            "analysis": analysis_payload(field, spacing_degrees=5.0),
            "stations": station_records(verified),
        }

    return {
        "schema_version": 2,
        "demo": True,
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "cycle": cycle.isoformat().replace("+00:00", "Z"),
        "level_hpa": 300,
        "models": models,
        "observation_summary": {
            "station_count": station_count,
            "metadata_sources": {"NOAA/NCEI IGRA station inventory": station_count},
        },
        "notes": [
            "Demonstration data are shown until the live-data workflow completes successfully.",
            "Filled colors show wind speed; black lines show 300-hPa geopotential height every 12 dam.",
        ],
    }

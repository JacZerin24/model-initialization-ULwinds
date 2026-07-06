from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from .config import MODEL_LABELS, MODEL_ORDER
from .models import ModelField
from .verification import sample_wind_vectors, station_records, summarize, verify_stations


def _base_uv(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")
    jet_n = 25 + 70 * np.exp(-((lat2d - 42) / 14) ** 2)
    jet_s = 18 + 55 * np.exp(-((lat2d + 48) / 16) ** 2)
    wave = 12 * np.sin(np.radians(lon2d * 2 + lat2d))
    u = jet_n + jet_s + wave
    v = 14 * np.sin(np.radians(lon2d + lat2d * 1.7))
    return u / 1.9438444924406, v / 1.9438444924406


def build_demo_payload() -> dict[str, object]:
    rng = np.random.default_rng(300)
    cycle = datetime(2026, 7, 4, 12, tzinfo=UTC)
    lat = np.arange(-90, 90.25, 2.5)
    lon = np.arange(0, 360, 2.5)
    base_u, base_v = _base_uv(lat, lon)

    station_count = 48
    station_lats = rng.uniform(-72, 72, station_count)
    station_lons = rng.uniform(-180, 180, station_count)
    nearest_lat = np.abs(lat[:, None] - station_lats).argmin(axis=0)
    nearest_lon = np.abs((lon[:, None] - (station_lons % 360) + 180) % 360 - 180).argmin(axis=0)
    obs_u = base_u[nearest_lat, nearest_lon] * 1.9438444924406 + rng.normal(0, 4, station_count)
    obs_v = base_v[nearest_lat, nearest_lon] * 1.9438444924406 + rng.normal(0, 4, station_count)
    obs = pd.DataFrame(
        {
            "station": [f"D{i:03d}" for i in range(station_count)],
            "name": [f"Demo RAOB {i:03d}" for i in range(station_count)],
            "latitude": station_lats,
            "longitude": station_lons,
            "obs_u_kt": obs_u,
            "obs_v_kt": obs_v,
            "obs_speed_kt": np.hypot(obs_u, obs_v),
            "obs_direction_deg": (270 - np.degrees(np.arctan2(obs_v, obs_u))) % 360,
            "vertical_method": "demonstration",
            "observation_time": cycle.isoformat().replace("+00:00", "Z"),
        }
    )

    perturbations = {
        "gfs": (1.5, 1.0, 1.0),
        "ecmwf": (-0.5, 0.5, 0.7),
        "gdps": (3.0, -1.5, 1.3),
    }
    models: dict[str, object] = {}
    for key in MODEL_ORDER:
        du, dv, wave_scale = perturbations[key]
        lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")
        u = base_u + du / 1.9438444924406 + wave_scale * np.sin(np.radians(lon2d))
        v = base_v + dv / 1.9438444924406 + 0.5 * wave_scale * np.cos(np.radians(lat2d * 2))
        field = ModelField(key, MODEL_LABELS[key], "Demonstration field", cycle, lat, lon, u, v)
        verified = verify_stations(field, obs)
        models[key] = {
            "label": MODEL_LABELS[key],
            "source": "Demonstration field",
            "status": "ok",
            "metrics": summarize(verified),
            "vectors": sample_wind_vectors(field, spacing_degrees=30.0),
            "stations": station_records(verified),
        }

    return {
        "schema_version": 1,
        "demo": True,
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "cycle": cycle.isoformat().replace("+00:00", "Z"),
        "level_hpa": 300,
        "models": models,
        "notes": [
            "Demonstration data are shown until the live-data workflow completes successfully.",
            "RAOB dots are colored by absolute model wind-speed error in knots.",
        ],
    }

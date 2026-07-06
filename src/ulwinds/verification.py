from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from .config import MPS_TO_KT
from .models import ModelField
from .observations import wind_direction_from_uv


@dataclass(slots=True)
class Interpolators:
    u: RegularGridInterpolator
    v: RegularGridInterpolator


def _prepare_grid(field: ModelField) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lat = np.asarray(field.latitude, dtype=float).squeeze()
    lon = np.asarray(field.longitude, dtype=float).squeeze()
    u = np.asarray(field.u_ms, dtype=float).squeeze()
    v = np.asarray(field.v_ms, dtype=float).squeeze()

    if lat.ndim != 1 or lon.ndim != 1 or u.ndim != 2 or v.ndim != 2:
        raise ValueError("Only regular latitude/longitude model grids are supported")

    if u.shape == (lon.size, lat.size):
        u = u.T
        v = v.T
    if u.shape != (lat.size, lon.size):
        raise ValueError(
            f"Wind array shape {u.shape} does not match latitude/longitude sizes {(lat.size, lon.size)}"
        )

    lat_order = np.argsort(lat)
    lat = lat[lat_order]
    u = u[lat_order, :]
    v = v[lat_order, :]

    lon = lon % 360.0
    lon_order = np.argsort(lon)
    lon = lon[lon_order]
    u = u[:, lon_order]
    v = v[:, lon_order]

    unique_lon, unique_index = np.unique(lon, return_index=True)
    lon = unique_lon
    u = u[:, unique_index]
    v = v[:, unique_index]

    if lon[0] > 0.001:
        lon = np.insert(lon, 0, 0.0)
        u = np.column_stack([u[:, -1], u])
        v = np.column_stack([v[:, -1], v])
    if lon[-1] < 359.999:
        lon = np.append(lon, 360.0)
        u = np.column_stack([u, u[:, 0]])
        v = np.column_stack([v, v[:, 0]])

    return lat, lon, u, v


def make_interpolators(field: ModelField) -> Interpolators:
    lat, lon, u, v = _prepare_grid(field)
    kwargs = {"bounds_error": False, "fill_value": np.nan}
    return Interpolators(
        RegularGridInterpolator((lat, lon), u, **kwargs),
        RegularGridInterpolator((lat, lon), v, **kwargs),
    )


def _direction_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs((a - b + 180.0) % 360.0 - 180.0)


def verify_stations(field: ModelField, observations: pd.DataFrame) -> pd.DataFrame:
    interpolators = make_interpolators(field)
    points = np.column_stack(
        [observations["latitude"].to_numpy(), observations["longitude"].to_numpy() % 360]
    )
    model_u_kt = interpolators.u(points) * MPS_TO_KT
    model_v_kt = interpolators.v(points) * MPS_TO_KT

    verified = observations.copy()
    verified["model_u_kt"] = model_u_kt
    verified["model_v_kt"] = model_v_kt
    verified["model_speed_kt"] = np.hypot(model_u_kt, model_v_kt)
    verified["model_direction_deg"] = wind_direction_from_uv(model_u_kt, model_v_kt)
    verified["speed_error_kt"] = verified["model_speed_kt"] - verified["obs_speed_kt"]
    verified["abs_speed_error_kt"] = verified["speed_error_kt"].abs()
    verified["vector_error_kt"] = np.hypot(
        verified["model_u_kt"] - verified["obs_u_kt"],
        verified["model_v_kt"] - verified["obs_v_kt"],
    )
    verified["direction_error_deg"] = _direction_difference(
        verified["model_direction_deg"].to_numpy(),
        verified["obs_direction_deg"].to_numpy(),
    )
    verified = verified.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["model_u_kt", "model_v_kt", "obs_u_kt", "obs_v_kt"]
    )
    return verified.reset_index(drop=True)


def summarize(verified: pd.DataFrame) -> dict[str, float | int | None]:
    if verified.empty:
        return {"n": 0, "mae_kt": None, "bias_kt": None, "rmse_kt": None, "vector_rmse_kt": None}
    speed_error = verified["speed_error_kt"].to_numpy(dtype=float)
    vector_error = verified["vector_error_kt"].to_numpy(dtype=float)
    return {
        "n": int(len(verified)),
        "mae_kt": round(float(np.mean(np.abs(speed_error))), 1),
        "bias_kt": round(float(np.mean(speed_error)), 1),
        "rmse_kt": round(float(np.sqrt(np.mean(speed_error**2))), 1),
        "vector_rmse_kt": round(float(np.sqrt(np.mean(vector_error**2))), 1),
        "median_abs_error_kt": round(float(np.median(np.abs(speed_error))), 1),
    }


def sample_wind_vectors(field: ModelField, spacing_degrees: float = 10.0) -> list[dict[str, float]]:
    interpolators = make_interpolators(field)
    latitudes = np.arange(-80.0, 80.1, spacing_degrees)
    longitudes = np.arange(-180.0, 180.0, spacing_degrees)
    lat_grid, lon_grid = np.meshgrid(latitudes, longitudes, indexing="ij")
    points = np.column_stack([lat_grid.ravel(), lon_grid.ravel() % 360])
    u_kt = interpolators.u(points) * MPS_TO_KT
    v_kt = interpolators.v(points) * MPS_TO_KT
    speed = np.hypot(u_kt, v_kt)
    direction_from = wind_direction_from_uv(u_kt, v_kt)
    direction_to = (direction_from + 180.0) % 360.0

    vectors: list[dict[str, float]] = []
    for lat, lon, u, v, spd, from_dir, to_dir in zip(
        lat_grid.ravel(), lon_grid.ravel(), u_kt, v_kt, speed, direction_from, direction_to
    ):
        if np.isfinite(spd):
            vectors.append(
                {
                    "lat": round(float(lat), 2),
                    "lon": round(float(lon), 2),
                    "u_kt": round(float(u), 1),
                    "v_kt": round(float(v), 1),
                    "speed_kt": round(float(spd), 1),
                    "direction_from_deg": round(float(from_dir), 0),
                    "direction_to_deg": round(float(to_dir), 0),
                }
            )
    return vectors


def station_records(verified: pd.DataFrame) -> list[dict[str, object]]:
    numeric_columns = [
        "latitude",
        "longitude",
        "obs_u_kt",
        "obs_v_kt",
        "obs_speed_kt",
        "obs_direction_deg",
        "model_u_kt",
        "model_v_kt",
        "model_speed_kt",
        "model_direction_deg",
        "speed_error_kt",
        "abs_speed_error_kt",
        "vector_error_kt",
        "direction_error_deg",
    ]
    records: list[dict[str, object]] = []
    for row in verified.to_dict(orient="records"):
        record: dict[str, object] = {
            "station": str(row.get("station", "UNKNOWN")),
            "name": str(row.get("name", row.get("station", "UNKNOWN"))),
            "vertical_method": str(row.get("vertical_method", "unknown")),
            "observation_time": str(row.get("observation_time", "")),
        }
        for column in numeric_columns:
            value = row.get(column)
            record[column] = None if value is None or not np.isfinite(value) else round(float(value), 1)
        records.append(record)
    return records

from __future__ import annotations

from dataclasses import dataclass

import contourpy
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
    height: RegularGridInterpolator


def _orient(values: np.ndarray, lat_size: int, lon_size: int) -> np.ndarray:
    values = np.asarray(values, dtype=float).squeeze()
    if values.shape == (lon_size, lat_size):
        return values.T
    if values.shape != (lat_size, lon_size):
        raise ValueError(f"Field shape {values.shape} does not match {(lat_size, lon_size)}")
    return values


def _prepare_grid(field: ModelField):
    lat = np.asarray(field.latitude, dtype=float).squeeze()
    lon = np.asarray(field.longitude, dtype=float).squeeze()
    if lat.ndim != 1 or lon.ndim != 1:
        raise ValueError("Only regular latitude/longitude model grids are supported")

    u = _orient(field.u_ms, lat.size, lon.size)
    v = _orient(field.v_ms, lat.size, lon.size)
    height = _orient(field.height_m, lat.size, lon.size)

    lat_order = np.argsort(lat)
    lat = lat[lat_order]
    u, v, height = u[lat_order], v[lat_order], height[lat_order]

    lon = lon % 360.0
    lon_order = np.argsort(lon)
    lon = lon[lon_order]
    u, v, height = u[:, lon_order], v[:, lon_order], height[:, lon_order]

    lon, unique_index = np.unique(lon, return_index=True)
    u, v, height = u[:, unique_index], v[:, unique_index], height[:, unique_index]

    if lon[0] > 0.001:
        lon = np.insert(lon, 0, 0.0)
        u = np.column_stack([u[:, -1], u])
        v = np.column_stack([v[:, -1], v])
        height = np.column_stack([height[:, -1], height])
    if lon[-1] < 359.999:
        lon = np.append(lon, 360.0)
        u = np.column_stack([u, u[:, 0]])
        v = np.column_stack([v, v[:, 0]])
        height = np.column_stack([height, height[:, 0]])

    return lat, lon, u, v, height


def make_interpolators(field: ModelField) -> Interpolators:
    lat, lon, u, v, height = _prepare_grid(field)
    kwargs = {"bounds_error": False, "fill_value": np.nan}
    return Interpolators(
        RegularGridInterpolator((lat, lon), u, **kwargs),
        RegularGridInterpolator((lat, lon), v, **kwargs),
        RegularGridInterpolator((lat, lon), height, **kwargs),
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
    model_height_m = interpolators.height(points)

    verified = observations.copy()
    verified["model_u_kt"] = model_u_kt
    verified["model_v_kt"] = model_v_kt
    verified["model_speed_kt"] = np.hypot(model_u_kt, model_v_kt)
    verified["model_direction_deg"] = wind_direction_from_uv(model_u_kt, model_v_kt)
    verified["model_height_m"] = model_height_m
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
    if "obs_height_m" in verified:
        verified["height_error_m"] = verified["model_height_m"] - verified["obs_height_m"]
    else:
        verified["obs_height_m"] = np.nan
        verified["height_error_m"] = np.nan
    verified = verified.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["model_u_kt", "model_v_kt", "obs_u_kt", "obs_v_kt"]
    )
    return verified.reset_index(drop=True)


def summarize(verified: pd.DataFrame) -> dict[str, float | int | None]:
    if verified.empty:
        return {
            "n": 0,
            "mae_kt": None,
            "bias_kt": None,
            "rmse_kt": None,
            "vector_rmse_kt": None,
            "height_mae_m": None,
        }
    speed_error = verified["speed_error_kt"].to_numpy(dtype=float)
    vector_error = verified["vector_error_kt"].to_numpy(dtype=float)
    height_error = verified["height_error_m"].dropna().to_numpy(dtype=float)
    return {
        "n": int(len(verified)),
        "mae_kt": round(float(np.mean(np.abs(speed_error))), 1),
        "bias_kt": round(float(np.mean(speed_error)), 1),
        "rmse_kt": round(float(np.sqrt(np.mean(speed_error**2))), 1),
        "vector_rmse_kt": round(float(np.sqrt(np.mean(vector_error**2))), 1),
        "median_abs_error_kt": round(float(np.median(np.abs(speed_error))), 1),
        "height_mae_m": round(float(np.mean(np.abs(height_error))), 0) if height_error.size else None,
        "height_n": int(height_error.size),
    }


def _height_contours(
    longitudes: np.ndarray,
    latitudes: np.ndarray,
    height_dam: np.ndarray,
    interval_dam: int = 12,
) -> tuple[list[dict[str, object]], list[dict[str, float]]]:
    cyclic_lons = np.append(longitudes, 180.0)
    cyclic_heights = np.column_stack([height_dam, height_dam[:, 0]])
    finite = cyclic_heights[np.isfinite(cyclic_heights)]
    if not finite.size:
        return [], []
    start = int(np.ceil(finite.min() / interval_dam) * interval_dam)
    stop = int(np.floor(finite.max() / interval_dam) * interval_dam)
    generator = contourpy.contour_generator(
        x=cyclic_lons,
        y=latitudes,
        z=cyclic_heights,
        line_type="Separate",
        corner_mask=True,
    )
    contours: list[dict[str, object]] = []
    labels: list[dict[str, float]] = []
    for level in range(start, stop + 1, interval_dam):
        lines = generator.lines(float(level))
        usable: list[np.ndarray] = []
        for line in lines:
            if len(line) < 3:
                continue
            simplified = line[::2] if len(line) > 8 else line
            if not np.array_equal(simplified[-1], line[-1]):
                simplified = np.vstack([simplified, line[-1]])
            coordinates = [
                [round(float(lon), 2), round(float(lat), 2)] for lon, lat in simplified
            ]
            contours.append({"level_dam": level, "coordinates": coordinates})
            usable.append(line)
        if usable:
            longest = max(usable, key=len)
            point = longest[len(longest) // 2]
            labels.append(
                {
                    "level_dam": level,
                    "lon": round(float(point[0]), 2),
                    "lat": round(float(point[1]), 2),
                }
            )
    return contours, labels


def analysis_payload(field: ModelField, spacing_degrees: float = 2.5) -> dict[str, object]:
    """Create a compact scalar grid and labeled height contours for the webpage."""
    interpolators = make_interpolators(field)
    latitudes = np.arange(-90.0, 90.01, spacing_degrees)
    longitudes = np.arange(-180.0, 180.0, spacing_degrees)
    lat_grid, lon_grid = np.meshgrid(latitudes, longitudes, indexing="ij")
    points = np.column_stack([lat_grid.ravel(), lon_grid.ravel() % 360])
    u_kt = interpolators.u(points).reshape(lat_grid.shape) * MPS_TO_KT
    v_kt = interpolators.v(points).reshape(lat_grid.shape) * MPS_TO_KT
    wind_speed = np.hypot(u_kt, v_kt)
    height_dam = interpolators.height(points).reshape(lat_grid.shape) / 10.0
    contours, labels = _height_contours(longitudes, latitudes, height_dam)
    return {
        "spacing_degrees": spacing_degrees,
        "latitudes": [round(float(value), 2) for value in latitudes],
        "longitudes": [round(float(value), 2) for value in longitudes],
        "wind_speed_kt": [
            [None if not np.isfinite(value) else round(float(value), 1) for value in row]
            for row in wind_speed
        ],
        "height_contour_interval_dam": 12,
        "height_contours": contours,
        "height_labels": labels,
    }


def station_records(verified: pd.DataFrame) -> list[dict[str, object]]:
    numeric_columns = [
        "latitude", "longitude", "obs_u_kt", "obs_v_kt", "obs_speed_kt",
        "obs_direction_deg", "obs_height_m", "model_u_kt", "model_v_kt",
        "model_speed_kt", "model_direction_deg", "model_height_m", "height_error_m",
        "speed_error_kt", "abs_speed_error_kt", "vector_error_kt", "direction_error_deg",
    ]
    records: list[dict[str, object]] = []
    for row in verified.to_dict(orient="records"):
        record: dict[str, object] = {
            "station": str(row.get("station", "UNKNOWN")),
            "name": str(row.get("name", row.get("station", "UNKNOWN"))),
            "vertical_method": str(row.get("vertical_method", "unknown")),
            "observation_time": str(row.get("observation_time", "")),
            "metadata_source": str(row.get("metadata_source", "unknown")),
            "profile_source": str(row.get("profile_source", "unknown")),
        }
        for column in numeric_columns:
            value = row.get(column)
            record[column] = None if value is None or not np.isfinite(value) else round(float(value), 1)
        records.append(record)
    return records

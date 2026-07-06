from datetime import UTC, datetime

import numpy as np
import pandas as pd

from ulwinds.models import ModelField
from ulwinds.observations import uv_from_speed_direction, wind_direction_from_uv
from ulwinds.verification import analysis_payload, summarize, verify_stations


def test_wind_direction_round_trip():
    speed = np.array([20.0, 35.0, 50.0, 10.0])
    direction = np.array([0.0, 90.0, 180.0, 270.0])
    u, v = uv_from_speed_direction(speed, direction)
    assert np.allclose(wind_direction_from_uv(u, v), direction)


def test_periodic_interpolation_metrics_and_analysis_grid():
    lat = np.array([-10.0, 0.0, 10.0])
    lon = np.array([0.0, 120.0, 240.0])
    u_ms = np.full((3, 3), 10.0)
    v_ms = np.zeros((3, 3))
    height_m = np.array(
        [[9000, 9060, 9120], [9060, 9120, 9180], [9120, 9180, 9240]],
        dtype=float,
    )
    field = ModelField(
        "test",
        "Test",
        "Unit test",
        datetime(2026, 1, 1, tzinfo=UTC),
        lat,
        lon,
        u_ms,
        v_ms,
        height_m,
    )
    obs = pd.DataFrame(
        {
            "station": ["A", "B"],
            "name": ["A", "B"],
            "latitude": [0.0, 0.0],
            "longitude": [179.0, -179.0],
            "obs_u_kt": [19.438444924406, 19.438444924406],
            "obs_v_kt": [0.0, 0.0],
            "obs_speed_kt": [19.438444924406, 19.438444924406],
            "obs_direction_deg": [270.0, 270.0],
            "obs_height_m": [9150.0, 9150.0],
            "vertical_method": ["test", "test"],
            "metadata_source": ["test", "test"],
        }
    )
    verified = verify_stations(field, obs)
    metrics = summarize(verified)
    analysis = analysis_payload(field, spacing_degrees=10.0)
    assert len(verified) == 2
    assert metrics["mae_kt"] == 0.0
    assert metrics["vector_rmse_kt"] == 0.0
    assert len(analysis["wind_speed_kt"]) == 19
    assert analysis["height_contour_interval_dam"] == 12
    assert analysis["height_contours"]

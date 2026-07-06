import numpy as np

from ulwinds.observations import _parse_raob_profiles, _parse_station_metadata


def test_parse_iem_metadata_and_300_hpa_profile():
    metadata_payload = {
        "features": [
            {
                "id": "TEST",
                "properties": {"sid": "TEST", "sname": "Test Sounding"},
                "geometry": {"type": "Point", "coordinates": [-90.5, 30.25]},
            }
        ]
    }
    profile_payload = {
        "profiles": [
            {
                "station": "TEST",
                "valid": "2026-07-04T12:00:00Z",
                "profile": [{"pres": 300.0, "sknt": 50.0, "drct": 270.0}],
            }
        ]
    }

    metadata = _parse_station_metadata(metadata_payload)
    result = _parse_raob_profiles(profile_payload, metadata)

    assert len(result) == 1
    assert result.loc[0, "name"] == "Test Sounding"
    assert result.loc[0, "longitude"] == -90.5
    assert np.isclose(result.loc[0, "obs_u_kt"], 50.0)
    assert np.isclose(result.loc[0, "obs_v_kt"], 0.0, atol=1e-10)

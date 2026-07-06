import numpy as np

from ulwinds.observations import (
    _parse_igra_station_list,
    _parse_iem_station_metadata,
    _parse_raob_profiles,
)


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
                "profile": [{"pres": 300.0, "hght": 9300, "sknt": 50.0, "drct": 270.0}],
            }
        ]
    }
    result = _parse_raob_profiles(profile_payload, _parse_iem_station_metadata(metadata_payload))
    assert len(result) == 1
    assert result.loc[0, "name"] == "Test Sounding"
    assert result.loc[0, "obs_height_m"] == 9300
    assert np.isclose(result.loc[0, "obs_u_kt"], 50.0)


def test_igra_wmo_alias_matches_international_profile():
    line = f"{'FRM00007145':<11} {48.7700:8.4f} {2.0100:9.4f} {160.0:6.1f} {'':2} {'TRAPPES':<30} {1945:4d} {2026:4d} {10000:6d}"
    metadata = _parse_igra_station_list(line)
    assert "07145" in metadata
    payload = {
        "profiles": [
            {
                "station": "07145",
                "valid": "2026-07-04T12:00:00Z",
                "profile": [{"pres": 300, "hght": 9250, "sknt": 80, "drct": 250}],
            }
        ]
    }
    result = _parse_raob_profiles(payload, metadata)
    assert len(result) == 1
    assert result.loc[0, "name"] == "Trappes"
    assert result.loc[0, "metadata_source"].startswith("NOAA/NCEI")

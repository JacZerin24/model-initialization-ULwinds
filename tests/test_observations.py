import numpy as np

from ulwinds.observations import (
    _active_international_wmo_stations,
    _parse_igra_station_list,
    _parse_iem_station_metadata,
    _parse_raob_profiles,
    _parse_uwyo_300_hpa,
)


def _igra_line(station_id="FRM00007145", last_year=2026):
    return (
        f"{station_id:<11} {48.7700:8.4f} {2.0100:9.4f} {160.0:6.1f} "
        f"{'':2} {'TRAPPES':<30} {1945:4d} {last_year:4d} {10000:6d}"
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
    assert result.loc[0, "profile_source"].startswith("Iowa Environmental")
    assert np.isclose(result.loc[0, "obs_u_kt"], 50.0)


def test_igra_wmo_alias_and_active_international_catalog():
    line = _igra_line()
    metadata = _parse_igra_station_list(line)
    active = _active_international_wmo_stations(line, 2026)
    assert "07145" in metadata
    assert active[0]["station"] == "07145"
    assert active[0]["country_code"] == "FR"


def test_parse_wyoming_current_text_list_300_hpa_row():
    header = """<html><pre>
-----------------------------------------------------------------------------
   PRES   HGHT   TEMP   DWPT   RELH   MIXR   DRCT   SKNT   THTA   THTE   THTV
    hPa      m      C      C      %   g/kg    deg   knot      K      K      K
-----------------------------------------------------------------------------
"""
    row = (
        f"{300.0:7.1f}{9250.0:7.0f}{-40.0:7.1f}{-50.0:7.1f}{30.0:7.0f}"
        f"{0.2:7.2f}{250.0:7.0f}{80.0:7.0f}{350.0:7.1f}{351.0:7.1f}{350.0:7.1f}"
    )
    point = _parse_uwyo_300_hpa(header + row + "\n</pre></html>")
    assert point is not None
    assert point["height_m"] == 9250
    assert point["speed_kt"] == 80
    assert point["direction_deg"] == 250


def test_old_international_profile_metadata_match_still_supported():
    metadata = _parse_igra_station_list(_igra_line())
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

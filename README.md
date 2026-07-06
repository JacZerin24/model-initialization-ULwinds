# Global 300-hPa Model Initialization Verification

A GitHub Pages dashboard and automated Python pipeline comparing initialized 300-hPa winds and geopotential heights from the GFS, ECMWF IFS, and Canadian GDPS with worldwide radiosonde observations.

## Dashboard

- Filled, zoom-aware 300-hPa wind-speed shading.
- Labeled 300-hPa geopotential-height contours every 12 dam, with 24-dam contours emphasized.
- Worldwide RAOB error dots using IEM sounding profiles plus NOAA/NCEI IGRA station metadata.
- Station popups with observed and modeled wind, height, signed bias, vector error, direction error, and height error.
- Summary metrics for wind MAE, wind-speed bias, vector RMSE, height MAE, and station count.
- Graceful partial updates when one model source is unavailable.

## Scientific interpretation

This evaluates **initialization fit**, not independent forecast skill. Radiosonde observations are commonly assimilated into global analyses. Differences can identify spatially coherent upper-level analysis errors, but the statistics should not be presented as clean out-of-sample forecast verification.

## Data sources

- **GFS:** NOAA/NCEP NOMADS, 0.25-degree f000 pressure-level U/V wind and geopotential height.
- **ECMWF IFS:** ECMWF Open Data, 0.25-degree step-0 pressure-level U/V wind and geopotential height.
- **Canadian GDPS:** Environment and Climate Change Canada MSC Datamart, 0.15-degree step-0 pressure-level U/V wind and height.
- **RAOB profiles:** Iowa Environmental Mesonet 300-hPa sounding profiles.
- **International station locations:** NOAA/NCEI Integrated Global Radiosonde Archive station inventory. IEM metadata remains preferred for U.S. ICAO identifiers.

The pipeline accepts only 00 and 12 UTC cycles. In automatic mode, it selects the newest common cycle that is at least 18 hours old, giving the model and observation sources time to arrive.

## Run locally

Python 3.11 or newer and an ecCodes runtime are required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python -m cfgrib selfcheck
python -m ulwinds.cli --cycle 2026070400 --output site/data/latest.json
python -m http.server 8000 --directory site
```

Generate deterministic demonstration data and run tests:

```bash
python -m ulwinds.cli --demo --output site/data/latest.json
pip install -e ".[test]"
pytest -q
```

## GitHub Pages operation

- A push to `main` builds a compatible demonstration dataset and deploys the updated webpage.
- Scheduled and manually triggered runs build live data.
- The live workflow runs at 06:20 and 18:20 UTC daily.
- Generated JSON is deployed inside the Pages artifact rather than committed twice per day.

## Verification definitions

- **Speed bias:** model wind speed minus observed wind speed.
- **Absolute speed error:** absolute value of speed bias; controls RAOB dot color and size.
- **Vector error:** magnitude of the model-minus-observed `(u, v)` difference.
- **Vector RMSE:** root-mean-square of station vector-error magnitudes.
- **Height error:** model 300-hPa geopotential height minus observed 300-hPa height.

Station verification uses bilinear interpolation on each model's native latitude/longitude grid. The webpage scalar field is sampled to a regular 2.5-degree display grid for browser performance; this does not change the native-grid station verification.

## Operational limitations

- ECMWF Open Data retains only a short rolling archive, so older manual cycles may not be available.
- A scheduled build that cannot retrieve RAOB data fails and leaves the previously deployed page in place.
- Individual model failures are recorded in the JSON and shown as disabled model tabs.
- The RAOB pipeline requests the observed 300-hPa mandatory level directly and does not vertically blend nearby levels.

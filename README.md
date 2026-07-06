# Global 300-hPa Model Initialization Verification

A GitHub Pages dashboard and automated Python pipeline that compares initialized **300-hPa winds** from the **GFS**, **ECMWF IFS**, and **Canadian GDPS** with global radiosonde observations.

The dashboard provides:

- A switchable global map for each model's step-0/f000 300-hPa wind field.
- Radiosonde locations colored and sized by absolute wind-speed error.
- Station popups with observed and modeled wind, signed speed bias, vector error, direction error, and observation time.
- Model summary cards for sample count, mean absolute error, speed bias, and vector RMSE.
- Graceful partial updates: one unavailable model is disabled without hiding the other models.

## Scientific interpretation

This evaluates **initialization fit**, not independent forecast skill. Radiosonde observations are commonly assimilated into global analyses. Differences can still identify spatially coherent upper-level wind analysis errors, but the statistics should not be presented as a clean out-of-sample forecast verification.

## Data sources

- **GFS:** [NOAA/NCEP NOMADS GRIB Filter](https://nomads.ncep.noaa.gov/), 0.25-degree f000 pressure-level U/V winds.
- **ECMWF IFS:** [ECMWF Open Data](https://www.ecmwf.int/en/forecasts/datasets/open-data), 0.25-degree step-0 pressure-level U/V winds through `ecmwf-opendata`.
- **Canadian GDPS:** [Environment and Climate Change Canada MSC Datamart](https://eccc-msc.github.io/open-data/msc-data/nwp_gdps/readme_gdps-datamart_en/), 0.15-degree step-0 pressure-level U/V winds.
- **RAOBs and station metadata:** [Iowa Environmental Mesonet RAOB JSON and network GeoJSON services](https://mesonet.agron.iastate.edu/api/). The IEM upper-air archive is derived from NCEI/IGRA holdings and can lag recent launches.

The pipeline accepts only 00 and 12 UTC cycles. In automatic mode, it selects the newest common cycle that is at least 18 hours old. That conservative lag gives all model and observation sources time to arrive.

## Repository layout

```text
.github/workflows/
  pages.yml             Builds live data twice daily and deploys GitHub Pages
  tests.yml             Runs unit tests and validates demo-data generation
site/
  index.html            Dashboard layout
  styles.css            Responsive styling
  app.js                Leaflet map, model controls, legends, and popups
  data/latest.json      Committed demonstration dataset
src/ulwinds/
  config.py             Cycle selection and shared constants
  models.py             Model downloads and GRIB decoding
  observations.py       Global RAOB and station-metadata retrieval
  verification.py       Native-grid interpolation and error calculations
  demo.py               Deterministic demonstration data
  cli.py                Command-line pipeline
```

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

Then open `http://localhost:8000`.

Generate deterministic demonstration data without downloading weather data:

```bash
python -m ulwinds.cli --demo --output site/data/latest.json
```

Run tests:

```bash
pip install -e ".[test]"
pytest -q
```

## GitHub Pages setup

1. Merge the initial pull request.
2. Open **Settings → Pages** and set **Source** to **GitHub Actions**.
3. Run **Build and deploy dashboard** manually once. An optional `YYYYMMDDHH` UTC cycle can be supplied.
4. The workflow will then rebuild and deploy the site at 06:20 and 18:20 UTC each day.

Scheduled builds deploy `site/data/latest.json` inside the Pages artifact rather than committing a large generated JSON file twice per day. This keeps repository history small while the live page remains current.

## Verification definitions

- **Speed bias:** model wind speed minus observed wind speed.
- **Absolute speed error:** absolute value of speed bias; controls dot color and size.
- **Scalar RMSE:** root-mean-square of wind-speed errors.
- **Vector error:** magnitude of the model-minus-observed `(u, v)` difference.
- **Vector RMSE:** root-mean-square of station vector-error magnitudes.
- **Direction error:** smallest angular difference between modeled and observed wind direction.

Station verification uses bilinear interpolation on each model's native latitude/longitude grid. The display arrows are sampled every 10 degrees for browser performance; this display sampling does not affect station verification.

## Operational limitations

- ECMWF Open Data retains only a short rolling archive, so older manual cycles may not be available.
- A scheduled build that cannot retrieve RAOB data fails and leaves the previously deployed page in place.
- Individual model failures are recorded in the generated JSON and shown as disabled model tabs.
- The IEM RAOB service provides mandatory-level winds in knots. The pipeline requests the observed 300-hPa level directly and does not vertically blend nearby levels.

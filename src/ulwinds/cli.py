from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from .config import MODEL_LABELS, MODEL_ORDER, ensure_parent, parse_cycle, validate_cycle
from .demo import build_demo_payload
from .models import FETCHERS
from .observations import fetch_raob_300
from .verification import analysis_payload, station_records, summarize, verify_stations

LOG = logging.getLogger("ulwinds")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify global 300-hPa model initialization winds against radiosondes."
    )
    parser.add_argument("--cycle", default="auto", help="auto, YYYYMMDDHH, or YYYY-MM-DDTHH")
    parser.add_argument("--output", type=Path, default=Path("site/data/latest.json"))
    parser.add_argument("--work-dir", type=Path, default=Path(".cache/ulwinds"))
    parser.add_argument("--models", nargs="+", choices=MODEL_ORDER, default=list(MODEL_ORDER))
    parser.add_argument("--demo", action="store_true", help="Generate deterministic demonstration data")
    parser.add_argument("--strict", action="store_true", help="Fail when any requested model fails")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser


def _write(payload: dict[str, object], output: Path) -> None:
    ensure_parent(output)
    output.write_text(
        json.dumps(payload, allow_nan=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    LOG.info("Wrote %s", output)


def run_live(cycle, models: list[str], work_dir: Path, strict: bool) -> dict[str, object]:
    observations = fetch_raob_300(cycle)
    model_payload: dict[str, object] = {}
    failures = 0

    for key in models:
        try:
            field = FETCHERS[key](cycle, work_dir)
            verified = verify_stations(field, observations)
            model_payload[key] = {
                "label": field.label,
                "source": field.source,
                "status": "ok",
                "metrics": summarize(verified),
                "analysis": analysis_payload(field),
                "stations": station_records(verified),
            }
            LOG.info("%s verified at %d stations", field.label, len(verified))
        except Exception as exc:
            failures += 1
            LOG.exception("%s failed", key)
            model_payload[key] = {
                "label": MODEL_LABELS[key],
                "source": "Unavailable",
                "status": "error",
                "error": str(exc),
                "metrics": {"n": 0},
                "analysis": None,
                "stations": [],
            }

    if failures == len(models) or (strict and failures):
        raise RuntimeError(f"{failures} of {len(models)} model retrievals failed")

    return {
        "schema_version": 2,
        "demo": False,
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "cycle": cycle.isoformat().replace("+00:00", "Z"),
        "level_hpa": 300,
        "models": model_payload,
        "observation_summary": {
            "station_count": int(len(observations)),
            "metadata_sources": observations["metadata_source"].value_counts().to_dict(),
        },
        "notes": [
            "Wind speed is displayed as a filled scalar field with 300-hPa geopotential-height contours every 12 dam.",
            "International station locations are matched through NOAA/NCEI's IGRA inventory; IEM metadata remains preferred for U.S. ICAO stations.",
            "This measures initialization fit, not independent forecast skill; many RAOBs may have been assimilated.",
        ],
    }


def main() -> None:
    args = _parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.demo:
        _write(build_demo_payload(), args.output)
        return
    cycle = validate_cycle(parse_cycle(args.cycle))
    _write(run_live(cycle, args.models, args.work_dir, args.strict), args.output)


if __name__ == "__main__":
    main()

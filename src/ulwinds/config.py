from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

LEVEL_HPA = 300
MPS_TO_KT = 1.9438444924406
MODEL_ORDER = ("gfs", "ecmwf", "gdps")
MODEL_LABELS = {
    "gfs": "GFS",
    "ecmwf": "ECMWF IFS",
    "gdps": "Canadian GDPS",
}


def default_cycle(now: datetime | None = None) -> datetime:
    """Return the newest 00/12 UTC cycle at least 18 hours old.

    The lag leaves time for all three model analyses and the global RAOB archive
    to become available while supporting twice-daily automated updates.
    """
    now = (now or datetime.now(UTC)).astimezone(UTC) - timedelta(hours=18)
    cycle_hour = 12 if now.hour >= 12 else 0
    return now.replace(hour=cycle_hour, minute=0, second=0, microsecond=0)


def parse_cycle(value: str | None) -> datetime:
    if not value or value.lower() == "auto":
        return default_cycle()

    cleaned = value.strip().upper().replace("Z", "")
    formats = ("%Y%m%d%H", "%Y-%m-%dT%H", "%Y-%m-%d %H")
    for fmt in formats:
        try:
            return datetime.strptime(cleaned, fmt).replace(tzinfo=UTC)
        except ValueError:
            continue
    raise ValueError("Cycle must be 'auto', YYYYMMDDHH, or YYYY-MM-DDTHH (00 or 12 UTC).")


def validate_cycle(cycle: datetime) -> datetime:
    cycle = cycle.astimezone(UTC)
    if cycle.hour not in (0, 12) or cycle.minute or cycle.second:
        raise ValueError("The common GFS/IFS/GDPS and RAOB comparison cycle must be 00 or 12 UTC.")
    return cycle


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

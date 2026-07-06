from datetime import UTC, datetime

from ulwinds.config import default_cycle


def test_default_cycle_uses_safe_18_hour_lag():
    assert default_cycle(datetime(2026, 7, 5, 18, 20, tzinfo=UTC)) == datetime(
        2026, 7, 5, 0, tzinfo=UTC
    )
    assert default_cycle(datetime(2026, 7, 5, 6, 20, tzinfo=UTC)) == datetime(
        2026, 7, 4, 12, tzinfo=UTC
    )

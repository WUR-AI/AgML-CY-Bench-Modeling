"""Filesystem-safe tags for crop-calendar forecast horizons (lead times)."""

from __future__ import annotations

import re

# Values accepted by dataset.temporal.season.end_of_sequence (see alignment.py).
_HORIZON_ALIASES: dict[str, str] = {
    "eos": "eos",
    "middle-of-season": "mid_season",
    "mid-season": "mid_season",
    "quarter-of-season": "quarter_season",
    "quarter-season": "quarter_season",
}


def prediction_horizon_tag(end_of_sequence: str) -> str:
    """Map ``end_of_sequence`` to a short, path-safe label for run / file names."""
    key = end_of_sequence.strip().lower()
    if key in _HORIZON_ALIASES:
        return _HORIZON_ALIASES[key]
    if key.startswith("eos-"):
        days = key.split("-", 1)[1]
        if days.isdigit():
            return f"eos_{days}"
    return re.sub(r"[^a-z0-9]+", "_", key).strip("_") or "unknown"


def parse_run_name_suffix(suffix: str) -> tuple[str | None, str]:
    """Parse ``<horizon>_<timestamp>`` or legacy ``<timestamp>`` after validation phase."""
    if re.fullmatch(r"\d{8}_\d{6}", suffix):
        return None, suffix
    match = re.fullmatch(r"(.+)_(\d{8}_\d{6})", suffix)
    if match:
        return match.group(1), match.group(2)
    return None, suffix

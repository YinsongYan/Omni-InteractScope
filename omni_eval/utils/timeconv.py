"""Time helpers.

All annotation temporal fields (query_time, notify_trigger_time, valid_response_window,
long_memory_start) are **millisecond timestamps**. Internally the harness works in
**seconds** (float). These helpers centralise the conversion so the unit convention lives
in exactly one place.
"""
from __future__ import annotations

from typing import List, Optional


def ms_to_sec(ms: Optional[float]) -> Optional[float]:
    """Convert a millisecond timestamp to seconds. ``None`` passes through."""
    if ms is None:
        return None
    return float(ms) / 1000.0


def sec_to_ms(sec: Optional[float]) -> Optional[int]:
    """Convert seconds back to an integer millisecond timestamp. ``None`` passes through."""
    if sec is None:
        return None
    return int(round(float(sec) * 1000.0))


def window_ms_to_sec(window: Optional[List[float]]) -> Optional[List[float]]:
    """Convert a ``[start_ms, end_ms]`` window to ``[start_sec, end_sec]``."""
    if window is None:
        return None
    if len(window) != 2:
        raise ValueError(f"valid_response_window must have 2 elements, got {window!r}")
    return [ms_to_sec(window[0]), ms_to_sec(window[1])]


def fmt_ts(sec: Optional[float]) -> str:
    """Human-readable ``MM:SS.mmm`` for logs."""
    if sec is None:
        return "--:--"
    if sec < 0:
        return "-" + fmt_ts(-sec)
    m, s = divmod(sec, 60.0)
    return f"{int(m):02d}:{s:06.3f}"

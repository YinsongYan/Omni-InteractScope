"""Which on-disk annotation layout is this?

Two layouts are supported, and both loaders start from the same record list:

* **video_grouped** — the annotation release (``data/examples/annotated_verified-*.json``): a list
  of *video* records with video-level metadata plus a nested ``interactions`` list. Flattened by
  :mod:`omni_eval.data.release`.
* **flat** — a JSON list of ``Interaction`` dicts (``data/fixtures/``). The original layout, kept
  frozen in :mod:`omni_eval.data.legacy_flat`.

Detection is structural rather than name- or version-based: a record that nests ``interactions``
is a video record, full stop. That survives new top-level metadata keys in future releases.
"""
from __future__ import annotations

from typing import Any, List


def detect_format(data: Any) -> str:
    """``"video_grouped"`` when records nest an ``interactions`` list, else ``"flat"``."""
    if isinstance(data, dict):
        data = data.get("videos") or data.get("data") or []
    if isinstance(data, list):
        for rec in data:
            if isinstance(rec, dict) and isinstance(rec.get("interactions"), list):
                return "video_grouped"
            if isinstance(rec, dict):
                break
    return "flat"


def records_of(data: Any) -> List[dict]:
    """The record list, unwrapping a ``{"videos": [...]}`` / ``{"data": [...]}`` envelope."""
    if isinstance(data, dict):
        data = data.get("videos") or data.get("data") or []
    if not isinstance(data, list):
        raise ValueError("expected a JSON list (of interactions, or of video records)")
    return [r for r in data if isinstance(r, dict)]

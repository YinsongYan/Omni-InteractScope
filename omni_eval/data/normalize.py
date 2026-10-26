"""Normalization for the ``annotated_verified-*.json`` release format.

Human-annotated releases arrive dirty: the same ``task_type`` shows up in ~29 spellings
(``proactive _notifying``, ``simultaneous_tools_calls_ search_ui``, ``state_dependen``), enum
fields occasionally hold a whole Chinese sentence, and every temporal field exists twice — once
as a **frame index** (``notify_trigger_time``) and once as **float milliseconds**
(``notify_trigger_time_ms``).

This module is the single place that cleans that up:

* ``normalize_*`` map a raw string to a canonical enum value, or ``None`` when the value is
  garbage (a sentence, ``"500"``, ``"test"``). Matching is: clean -> exact -> alias table ->
  fuzzy (``difflib``, so truncations/letter swaps like ``txet`` or ``verbal_visual_interjectio``
  resolve without enumerating every typo).
* ``parse_time_ms`` reads a timestamp preferring the ``*_ms`` field and falling back to
  ``frame_index / fps * 1000``.
* ``NormalizeReport`` accumulates what was rewritten or dropped so the loader can surface a
  data-quality summary instead of silently guessing.

Unit convention downstream is unchanged (float seconds everywhere; see
``omni_eval.utils.timeconv``); this module only produces milliseconds.
"""
from __future__ import annotations

import difflib
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

# ---- canonical vocabularies ----------------------------------------------------------
CANONICAL_TASK_TYPES: List[str] = [
    "proactive_notifying",
    "state_dependent",
    "verbal_visual_interjection",
    "simultaneous_tool_calls_search_ui",
    "simultaneous_tool_calls_search_generative_ui",
    "simultaneous_speech",
    "time_awareness",
    "visual_hints",
    "proactive_agent_assistance",
    "seamless_dialog_management",
    "impairment_assistance",
]
CANONICAL_QUERY_TYPES: List[str] = ["text", "text_image", "audio", "text_audio", "none"]
CANONICAL_TRIGGER_TYPES: List[str] = ["vision", "audio", "vision_audio", "text", "other"]
CANONICAL_RESPONSE_TYPES: List[str] = ["text", "toolcall", "ui", "multimodal"]
CANONICAL_LONG_MEMORY: List[str] = ["yes", "no"]

# Structural variants fuzzy matching should not have to guess at (or would get wrong).
TASK_TYPE_ALIASES: Dict[str, str] = {
    "proactive_notification": "proactive_notifying",
    "proactive_notify": "proactive_notifying",
    "forward_active_responding": "proactive_notifying",
}
QUERY_TYPE_ALIASES: Dict[str, str] = {
    # a lone "vision" query_type in the wild carries only query_text -> plain text query
    "vision": "text",
    "image": "text_image",
    "text_vision": "text_image",
    "audio_text": "text_audio",
    "image_text": "text_image",
}
TRIGGER_TYPE_ALIASES: Dict[str, str] = {
    "multimodal": "vision_audio",
    "audio_vision": "vision_audio",
    "visual": "vision",
    "none": "other",
}
RESPONSE_TYPE_ALIASES: Dict[str, str] = {
    "tool_call": "toolcall",
    "tool": "toolcall",
    "vision_audio": "multimodal",
    "generative_ui": "ui",
}
LONG_MEMORY_ALIASES: Dict[str, str] = {"true": "yes", "false": "no", "y": "yes", "n": "no"}

SILENCE_TOKEN = "<silence>"

_STRIP_CHARS = "\"'“”‘’：:，,。.、·`~ \t\r\n"


def clean_token(raw: Any) -> str:
    """Lowercase / de-quote / collapse whitespace-and-underscores into a comparable token."""
    s = "" if raw is None else str(raw)
    s = s.strip().strip(_STRIP_CHARS).strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s*_\s*", "_", s)
    s = s.replace(" ", "_")
    return re.sub(r"_+", "_", s).strip("_")


def _resolve(
    raw: Any, canonical: Sequence[str], aliases: Dict[str, str], cutoff: float
) -> Optional[str]:
    tok = clean_token(raw)
    if not tok:
        return None
    if tok in canonical:
        return tok
    if tok in aliases:
        return aliases[tok]
    match = difflib.get_close_matches(tok, list(canonical), n=1, cutoff=cutoff)
    return match[0] if match else None


def normalize_task_type(raw: Any) -> Optional[str]:
    return _resolve(raw, CANONICAL_TASK_TYPES, TASK_TYPE_ALIASES, 0.85)


def normalize_query_type(raw: Any) -> Optional[str]:
    return _resolve(raw, CANONICAL_QUERY_TYPES, QUERY_TYPE_ALIASES, 0.7)


def normalize_trigger_type(raw: Any) -> Optional[str]:
    return _resolve(raw, CANONICAL_TRIGGER_TYPES, TRIGGER_TYPE_ALIASES, 0.75)


def normalize_response_type(raw: Any) -> Optional[str]:
    return _resolve(raw, CANONICAL_RESPONSE_TYPES, RESPONSE_TYPE_ALIASES, 0.75)


def normalize_long_memory(raw: Any) -> Optional[str]:
    return _resolve(raw, CANONICAL_LONG_MEMORY, LONG_MEMORY_ALIASES, 0.8)


SUB_TASK_UNSPECIFIED = "unspecified"


def normalize_sub_task_type(raw: Any) -> str:
    """Sub-tasks are single letters (A-H). Anything else buckets as ``unspecified``."""
    tok = clean_token(raw).upper()
    return tok if len(tok) == 1 and "A" <= tok <= "Z" else SUB_TASK_UNSPECIFIED


def is_silence_reply(reply: Any) -> bool:
    """True when an ``assistant_reply_text`` marks a should-stay-silent target."""
    return clean_token(reply) == clean_token(SILENCE_TOKEN)


# ---- timestamps ----------------------------------------------------------------------
def _as_float(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def parse_time_ms(
    d: Dict[str, Any],
    ms_key: str,
    frame_key: Optional[str] = None,
    fps: Optional[float] = None,
) -> Optional[float]:
    """Timestamp in float milliseconds: ``*_ms`` wins, frame index / fps is the fallback."""
    ms = _as_float(d.get(ms_key))
    if ms is not None:
        return ms
    if frame_key and fps and fps > 0:
        frames = _as_float(d.get(frame_key))
        if frames is not None:
            return frames / float(fps) * 1000.0
    return None


def parse_window_ms(
    d: Dict[str, Any],
    ms_key: str,
    frame_key: Optional[str] = None,
    fps: Optional[float] = None,
) -> Optional[List[float]]:
    """``[start_ms, end_ms]``: ``*_ms`` wins, frame-index pair / fps is the fallback."""
    for key, scale in ((ms_key, None), (frame_key, fps)):
        if not key:
            continue
        win = d.get(key)
        if not win or not isinstance(win, (list, tuple)) or len(win) != 2:
            continue
        lo, hi = _as_float(win[0]), _as_float(win[1])
        if lo is None or hi is None:
            continue
        if scale:
            if scale <= 0:
                continue
            lo, hi = lo / float(scale) * 1000.0, hi / float(scale) * 1000.0
        return [lo, hi] if lo <= hi else [hi, lo]
    return None


def parse_fps(raw: Any, default: float = 30.0) -> float:
    fps = _as_float(raw)
    return fps if fps and fps > 0 else default


# ---- reporting -----------------------------------------------------------------------
@dataclass
class NormalizeReport:
    """Counters describing what normalization had to rewrite or give up on."""

    rewritten: Dict[str, Counter] = field(default_factory=lambda: {})
    unresolved: Dict[str, Counter] = field(default_factory=lambda: {})
    missing: Counter = field(default_factory=Counter)

    def note(self, field_name: str, raw: Any, resolved: Optional[str]) -> None:
        raw_s = "" if raw is None else str(raw)
        if not clean_token(raw_s):
            # An absent field is not dirt — it falls back to a documented default.
            self.missing[field_name] += 1
        elif resolved is None:
            self.unresolved.setdefault(field_name, Counter())[raw_s] += 1
        elif clean_token(raw_s) != resolved:
            self.rewritten.setdefault(field_name, Counter())[f"{raw_s!r} -> {resolved}"] += 1

    def as_dict(self) -> Dict[str, Any]:
        return {
            "rewritten": {k: dict(v) for k, v in self.rewritten.items()},
            "unresolved": {k: dict(v) for k, v in self.unresolved.items()},
            "missing": dict(self.missing),
        }

    @property
    def n_rewritten(self) -> int:
        return sum(sum(c.values()) for c in self.rewritten.values())

    @property
    def n_unresolved(self) -> int:
        return sum(sum(c.values()) for c in self.unresolved.values())

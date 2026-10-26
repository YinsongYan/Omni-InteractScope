"""Annotation data model for the Omni Proactive Interaction Benchmark.

Two on-disk layouts feed this model (both handled by ``omni_eval.data.loader``):

1. **flat** — a JSON list of ``Interaction`` objects (the original 4-file category layout,
   still used by ``data/fixtures/``). Temporal fields are millisecond timestamps.
2. **video_grouped** — the annotation-release layout
   (``data/examples/annotated_verified-*.json``): a JSON list of *video* records, each holding
   video-level metadata plus a nested ``interactions`` list. There every temporal field appears
   twice — as a frame index (``query_time``) and as float milliseconds (``query_time_ms``) — and
   the timestamps are **relative to the clip**, whose length is
   ``source_end_ms - source_start_ms``. The loader resolves both into the same ms fields here.

Unit convention: raw ms values are kept for round-tripping, and every consumer downstream uses
the derived ``*_sec`` (float seconds). See ``omni_eval.utils.timeconv``.

The model is intentionally lenient about ``task_type`` / ``sub_task_type`` values (the benchmark
authors are encouraged to add new categories); unknown values fall back to default metric
routing in ``omni_eval.taxonomy``.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .data.normalize import (
    CANONICAL_LONG_MEMORY,
    CANONICAL_QUERY_TYPES,
    CANONICAL_RESPONSE_TYPES,
    CANONICAL_TRIGGER_TYPES,
    is_silence_reply,
)
from .utils.timeconv import ms_to_sec, window_ms_to_sec

QUERY_TYPES = set(CANONICAL_QUERY_TYPES)
TRIGGER_TYPES = set(CANONICAL_TRIGGER_TYPES)
RESPONSE_TYPES = set(CANONICAL_RESPONSE_TYPES)
LONG_MEMORY = set(CANONICAL_LONG_MEMORY)
# Optional per-response expected behavior. "silence" marks a should-stay-silent negative
# used by the Proactive Turn-taking metric. Defaults to "respond".
EXPECTED_BEHAVIORS = {"respond", "silence"}


def _as_ms(v: Any) -> Optional[float]:
    """Raw millisecond timestamp as float (release data carries values like ``16733.33``)."""
    if v is None or v == "":
        return None
    return float(v)


def _check_enum(value: str, allowed: set, fieldname: str, ctx: str, strict: bool) -> str:
    if value in allowed:
        return value
    msg = f"[{ctx}] {fieldname}={value!r} not in {sorted(allowed)}"
    if strict:
        raise ValueError(msg)
    warnings.warn(msg)
    return value


@dataclass
class Response:
    response_id: int
    trigger_type: str = ""
    trigger_reason: str = ""
    response_type: str = ""
    assistant_reply_text: str = ""
    long_memory: str = "no"
    source_language: str = ""
    target_language: str = ""
    expected_behavior: str = "respond"

    # raw millisecond fields (kept for round-tripping / debugging)
    notify_trigger_time: Optional[float] = None
    valid_response_window: Optional[List[float]] = None
    long_memory_start: Optional[float] = None

    # derived seconds (populated in __post_init__)
    notify_trigger_sec: Optional[float] = None
    valid_response_window_sec: Optional[List[float]] = None
    long_memory_start_sec: Optional[float] = None

    # provenance for the video_grouped layout (raw pre-normalization enum values)
    raw: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.notify_trigger_sec = ms_to_sec(self.notify_trigger_time)
        self.valid_response_window_sec = window_ms_to_sec(self.valid_response_window)
        self.long_memory_start_sec = ms_to_sec(self.long_memory_start)

    @property
    def is_windowed(self) -> bool:
        return self.valid_response_window_sec is not None

    @property
    def is_silence(self) -> bool:
        """Should-stay-silent target.

        The flat layout marks these with ``expected_behavior: "silence"``; the release layout
        marks them by setting ``assistant_reply_text`` to ``<silence>``.
        """
        return self.expected_behavior == "silence" or is_silence_reply(self.assistant_reply_text)

    @property
    def has_gt_time(self) -> bool:
        return self.notify_trigger_sec is not None or self.valid_response_window_sec is not None

    @property
    def needs_audio(self) -> bool:
        return self.trigger_type in ("audio", "vision_audio")

    @property
    def is_crosslingual(self) -> bool:
        return bool(self.source_language) and bool(self.target_language)

    @classmethod
    def from_dict(cls, d: Dict[str, Any], ctx: str = "", strict: bool = False) -> "Response":
        rt = d.get("response_type", "") or ""
        tt = d.get("trigger_type", "") or ""
        lm = d.get("long_memory", "no") or "no"
        eb = d.get("expected_behavior", "respond") or "respond"
        if rt:
            _check_enum(rt, RESPONSE_TYPES, "response_type", ctx, strict)
        if tt:
            _check_enum(tt, TRIGGER_TYPES, "trigger_type", ctx, strict)
        _check_enum(lm, LONG_MEMORY, "long_memory", ctx, strict)
        _check_enum(eb, EXPECTED_BEHAVIORS, "expected_behavior", ctx, strict)
        vrw = d.get("valid_response_window")
        return cls(
            response_id=int(d.get("response_id", 1)),
            trigger_type=tt,
            trigger_reason=d.get("trigger_reason", "") or "",
            response_type=rt,
            assistant_reply_text=d.get("assistant_reply_text", "") or "",
            long_memory=lm,
            source_language=d.get("source_language", "") or "",
            target_language=d.get("target_language", "") or "",
            expected_behavior=eb,
            notify_trigger_time=_as_ms(d.get("notify_trigger_time")),
            valid_response_window=[float(x) for x in vrw] if vrw else None,
            long_memory_start=_as_ms(d.get("long_memory_start")),
        )


@dataclass
class Interaction:
    interaction_id: str
    task_type: str = ""
    sub_task_type: str = ""
    query_type: str = "text"
    query_text: str = ""
    query_image: str = ""
    query_audio: str = ""
    # Video reference. The example schema omitted it, but eval needs to know which clip an
    # interaction belongs to; accepts "video" / "video_id" / "video_path".
    video: str = ""
    responses: List[Response] = field(default_factory=list)

    query_time: Optional[float] = None      # raw ms
    query_time_sec: Optional[float] = None  # derived

    # ---- video-level metadata (video_grouped layout only; empty/None for flat fixtures) ----
    video_name: str = ""            # annotator-facing name, often without an extension
    video_url: str = ""             # remote (signed) URL the clip can be fetched from
    video_key: str = ""             # stable per-clip key = basename(unquote(url)) without suffix
    video_fps: Optional[float] = None
    video_duration_sec: Optional[float] = None  # source_end_ms - source_start_ms, in seconds
    source_start_ms: Optional[float] = None
    source_end_ms: Optional[float] = None
    scene: str = ""
    language: str = ""
    perspective: str = ""
    raw_interaction_id: str = ""    # pre-namespacing id ("inter_01"), not unique on its own
    is_available: Optional[bool] = None
    publication_status: str = ""
    target_group: str = ""
    qa_language: str = ""          # output/query language, distinct from scene audio language
    # Resolved attachment URLs (query_image / query_audio hold bare names in the release data).
    query_image_url: str = ""
    query_audio_url: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.query_time_sec = ms_to_sec(self.query_time)

    @property
    def needs_audio(self) -> bool:
        return self.query_type in ("audio", "text_audio") or any(
            r.needs_audio for r in self.responses
        )

    @classmethod
    def from_dict(cls, d: Dict[str, Any], strict: bool = False) -> "Interaction":
        iid = str(d.get("interaction_id", ""))
        qt = d.get("query_type", "text") or "text"
        if qt:
            _check_enum(qt, QUERY_TYPES, "query_type", iid, strict)
        video = d.get("video") or d.get("video_id") or d.get("video_path") or ""
        responses = [
            Response.from_dict(r, ctx=f"{iid}#resp{i}", strict=strict)
            for i, r in enumerate(d.get("responses", []) or [])
        ]
        return cls(
            interaction_id=iid,
            task_type=d.get("task_type", "") or "",
            sub_task_type=d.get("sub_task_type", "") or "",
            target_group=d.get("target_group", "") or "",
            qa_language=d.get("qa_language", "") or "",
            query_type=qt,
            query_text=d.get("query_text", "") or "",
            query_image=d.get("query_image", "") or "",
            query_audio=d.get("query_audio", "") or "",
            video=str(video),
            responses=responses,
            query_time=_as_ms(d.get("query_time")),
        )


def validate_interactions(items: List[Interaction], strict: bool = False) -> List[str]:
    """Return a list of human-readable validation issues (empty = clean)."""
    issues: List[str] = []
    seen = set()
    for it in items:
        if not it.interaction_id:
            issues.append("interaction with empty interaction_id")
        if it.interaction_id in seen:
            issues.append(f"duplicate interaction_id: {it.interaction_id}")
        seen.add(it.interaction_id)
        if not it.responses:
            issues.append(f"{it.interaction_id}: no responses")
        for r in it.responses:
            # A should-stay-silent target legitimately carries no timestamp (the release data
            # leaves notify_trigger_time null on nearly every <silence> response).
            if not r.has_gt_time and not r.is_silence:
                issues.append(
                    f"{it.interaction_id}#resp{r.response_id}: needs notify_trigger_time "
                    "or valid_response_window (or a <silence> reply)"
                )
    if strict and issues:
        raise ValueError("; ".join(issues))
    return issues

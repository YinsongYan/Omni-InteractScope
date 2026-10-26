"""Flatten the annotation release (``video_grouped``) into ``Interaction`` objects.

The release (``data/examples/annotated_verified-*.json``) is a list of *video* records carrying
video-level metadata (``video_url``, ``video_fps``, ``source_start_ms``/``source_end_ms``,
``scene``, ``language``, ``query_attachments``, …) plus a nested ``interactions`` list. Flattening
it is not a mechanical unwrap — four things have to be fixed up:

1. ``interaction_id`` is only ``inter_01..inter_04`` and repeats across every video, so ids are
   re-namespaced as ``<video_key>__<raw_id>`` (``video_key`` = the clip's file stem).
2. Every temporal field appears twice: a **frame index** (``query_time``) and float
   **milliseconds** (``query_time_ms``). ``normalize.parse_time_ms`` prefers ms and falls back
   to ``frame / video_fps``.
3. Enum fields are hand-typed and dirty (``proactive _notifying``, ``txet``, whole sentences);
   ``normalize.*`` canonicalises them, and un-resolvable ``task_type`` values are dropped.
4. ``query_image`` / ``query_audio`` hold bare names with no extension; the real filename+URL
   live in the video-level ``query_attachments`` list, keyed by (interaction_id, query_field).

Every rewrite and drop is recorded in the ``LoadReport``.
"""
from __future__ import annotations

import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import unquote, urlsplit

from ..schema import Interaction, Response
from .filters import Filters, LoadReport
from .formats import records_of
from .normalize import (
    SUB_TASK_UNSPECIFIED,
    is_silence_reply,
    normalize_long_memory,
    normalize_query_type,
    normalize_response_type,
    normalize_sub_task_type,
    normalize_task_type,
    normalize_trigger_type,
    parse_fps,
    parse_time_ms,
    parse_window_ms,
)
from .response_parser import infer_response_type

# How far past the clip end a ground-truth timestamp may sit before it is reported as suspect.
_OUT_OF_CLIP_TOL_SEC = 2.0


def url_basename(url: str) -> str:
    """Real filename behind a (possibly signed, possibly percent-encoded) URL."""
    if not url:
        return ""
    return os.path.basename(unquote(urlsplit(str(url)).path))


def video_key_of(rec: dict) -> str:
    """Stable per-clip key: the clip filename without its extension.

    Derived from ``video_url`` (unique across all 611 records in the current release) and
    falling back to ``video_name`` / the itag subtask id when the URL is absent.
    """
    base = url_basename(rec.get("video_url", ""))
    if not base:
        base = str(rec.get("video_name") or "").strip()
    if not base:
        base = str(rec.get("itag_subtask_id") or rec.get("platform_video_id") or "").strip()
    stem, _ = os.path.splitext(base)
    return stem or base


def video_filename_of(rec: dict) -> str:
    """Local filename to look for under ``paths.videos``.

    ``video_name`` is missing its extension for 505/611 records, so the URL basename wins.
    """
    base = url_basename(rec.get("video_url", ""))
    if base and os.path.splitext(base)[1]:
        return base
    name = str(rec.get("video_name") or "").strip()
    if name and os.path.splitext(name)[1]:
        return name
    return base or name


def t8_attachment_filename(video_key: str, filename: str) -> str:
    """Audio labels repeat across T8 videos; namespace local files to prevent overwrite."""
    return f"{video_key}__{os.path.basename(filename)}" if filename else ""


def _as_bool(v: Any) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    s = str(v or "").strip().lower()
    if s in ("true", "1", "yes", "y"):
        return True
    if s in ("false", "0", "no", "n"):
        return False
    return None


def _as_float(v: Any) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _attachment_index(rec: dict) -> Dict[Tuple[str, str], dict]:
    """``(interaction_id, query_field) -> attachment`` for the video-level attachment list."""
    idx: Dict[Tuple[str, str], dict] = {}
    for att in rec.get("query_attachments") or []:
        if not isinstance(att, dict):
            continue
        key = (str(att.get("interaction_id") or "").strip(),
               str(att.get("query_field") or "").strip())
        idx.setdefault(key, att)
    return idx


def _clip_duration_sec(rec: dict) -> Optional[float]:
    """Clip length in seconds. Annotation timestamps are relative to this clip's start."""
    lo, hi = _as_float(rec.get("source_start_ms")), _as_float(rec.get("source_end_ms"))
    if lo is None or hi is None:
        return None
    return (hi - lo) / 1000.0


def _build_response(
    rd: dict, ctx: str, fps: float, report: LoadReport
) -> Response:
    reply = str(rd.get("assistant_reply_text") or "")
    silence = is_silence_reply(reply)

    raw_trigger = rd.get("trigger_type")
    trigger = normalize_trigger_type(raw_trigger)
    report.normalize.note("trigger_type", raw_trigger, trigger)

    raw_rtype = rd.get("response_type")
    rtype = normalize_response_type(raw_rtype)
    report.normalize.note("response_type", raw_rtype, rtype)
    if rtype is None:
        # A garbage response_type is recoverable: the markup in the reply text tells us what
        # kind of response it is (<call:..> -> toolcall, <ui:..> -> ui, mixed -> multimodal).
        rtype = infer_response_type(reply)

    raw_lm = rd.get("long_memory")
    lm = normalize_long_memory(raw_lm)
    report.normalize.note("long_memory", raw_lm, lm)

    return Response(
        response_id=int(_as_float(rd.get("response_id")) or 1),
        trigger_type=trigger or "other",
        trigger_reason=str(rd.get("trigger_reason") or ""),
        response_type=rtype,
        assistant_reply_text=reply,
        long_memory=lm or "no",
        source_language=str(rd.get("source_language") or ""),
        target_language=str(rd.get("target_language") or ""),
        expected_behavior="silence" if silence else "respond",
        notify_trigger_time=parse_time_ms(rd, "notify_trigger_time_ms", "notify_trigger_time", fps),
        valid_response_window=parse_window_ms(
            rd, "valid_response_window_ms", "valid_response_window", fps
        ),
        long_memory_start=parse_time_ms(rd, "long_memory_start_ms", "long_memory_start", fps),
        raw={
            "ctx": ctx,
            "trigger_type": raw_trigger,
            "response_type": raw_rtype,
            "long_memory": raw_lm,
        },
    )


def _note_suspect_time(report: LoadReport, iid: str, label: str, t: float, dur: float) -> None:
    if len(report.suspect_times) < 15:
        report.suspect_times.append(f"{iid}: {label}={t:.2f}s > clip {dur:.2f}s")


def load_video_grouped(
    data: Any,
    filters: Optional[Filters] = None,
    report: Optional[LoadReport] = None,
) -> Tuple[List[Interaction], LoadReport]:
    """Flatten the release layout into ``Interaction`` objects, applying ``filters``."""
    rep = report or LoadReport()
    rep.fmt = "video_grouped"
    f = filters
    seen_ids: Counter = Counter()
    out: List[Interaction] = []

    for rec in records_of(data):
        rep.n_videos += 1
        interactions = [i for i in (rec.get("interactions") or []) if isinstance(i, dict)]
        n_inter = len(interactions)
        n_resp = sum(len(i.get("responses") or []) for i in interactions)
        rep.n_interactions_raw += n_inter
        rep.n_responses_raw += n_resp

        duration = _clip_duration_sec(rec)
        available = _as_bool(rec.get("is_available"))
        pub_status = str(rec.get("publication_status") or "")
        publishable = pub_status.strip().lower() != "unpublishable"

        def drop_video(reason: str) -> None:
            rep.dropped_videos[reason] += 1
            rep.dropped_interactions[reason] += n_inter
            rep.dropped_responses[reason] += n_resp

        if f and f.drop_invalid_duration and duration is not None and duration <= 0:
            drop_video("invalid_clip_duration")
            continue
        if f and f.require_available and available is False:
            drop_video("is_available_false")
            continue
        if f and f.require_publishable and not publishable:
            drop_video("unpublishable")
            continue
        if available is False:
            rep.tagged_unavailable += n_inter
        if not publishable:
            rep.tagged_unpublishable += n_inter

        rep.n_videos_kept += 1
        fps = parse_fps(rec.get("video_fps"))
        vkey = video_key_of(rec)
        vfile = video_filename_of(rec)
        attachments = _attachment_index(rec)

        for idict in interactions:
            # Keep the raw id for the attachment lookup (annotators typed e.g. "inter _02" in
            # both places) but build the namespaced id from a whitespace-free form.
            raw_id = str(idict.get("interaction_id") or "").strip() or "inter"
            iid = f"{vkey}__{re.sub(r'\s+', '', raw_id)}"
            seen_ids[iid] += 1
            if seen_ids[iid] > 1:  # same raw id twice inside one video record
                rep.id_collisions[iid] += 1
                iid = f"{iid}#{seen_ids[iid]}"

            raw_task = idict.get("task_type")
            task = normalize_task_type(raw_task)
            rep.normalize.note("task_type", raw_task, task)
            n_r = len(idict.get("responses") or [])
            if task is None:
                if f is None or f.drop_unknown_task_type:
                    rep.dropped_interactions["unknown_task_type"] += 1
                    rep.dropped_responses["unknown_task_type"] += n_r
                    continue
                task = ""
            if f and not f.keeps_task(task):
                rep.dropped_interactions[f"excluded_task:{task}"] += 1
                rep.dropped_responses[f"excluded_task:{task}"] += n_r
                continue

            raw_qtype = idict.get("query_type")
            qtype = normalize_query_type(raw_qtype)
            rep.normalize.note("query_type", raw_qtype, qtype)

            raw_sub = idict.get("sub_task_type")
            sub = normalize_sub_task_type(raw_sub)
            if sub == SUB_TASK_UNSPECIFIED and str(raw_sub or "").strip():
                rep.normalize.note("sub_task_type", raw_sub, None)

            responses: List[Response] = []
            for k, rd in enumerate(idict.get("responses") or []):
                if not isinstance(rd, dict):
                    rep.dropped_responses["malformed"] += 1
                    continue
                r = _build_response(rd, f"{iid}#resp{k}", fps, rep)
                if r.is_silence and f and not f.include_silence_negatives:
                    rep.dropped_responses["silence_negative"] += 1
                    continue
                if not r.has_gt_time and not r.is_silence:
                    rep.dropped_responses["no_gt_time"] += 1
                    continue
                if duration and duration > 0:
                    if (r.notify_trigger_sec or 0) > duration + _OUT_OF_CLIP_TOL_SEC:
                        _note_suspect_time(rep, iid, "notify_trigger",
                                           r.notify_trigger_sec, duration)
                    if r.valid_response_window_sec and (
                        r.valid_response_window_sec[1] > duration + _OUT_OF_CLIP_TOL_SEC
                    ):
                        _note_suspect_time(rep, iid, "window_end",
                                           r.valid_response_window_sec[1], duration)
                responses.append(r)

            if not responses:
                rep.dropped_interactions["no_evaluable_response"] += 1
                continue

            q_ms = parse_time_ms(idict, "query_time_ms", "query_time", fps)
            if duration and duration > 0 and q_ms is not None:
                q_sec = q_ms / 1000.0
                if q_sec > duration + _OUT_OF_CLIP_TOL_SEC:
                    _note_suspect_time(rep, iid, "query_time", q_sec, duration)

            img_att = attachments.get((raw_id, "query_image")) or {}
            aud_att = attachments.get((raw_id, "query_audio")) or {}

            it = Interaction(
                interaction_id=iid,
                task_type=task,
                sub_task_type=sub,
                target_group=str(idict.get("target_group") or "").strip(),
                qa_language=str(idict.get("qa_language") or "").strip().lower(),
                query_type=qtype or "text",
                query_text=str(idict.get("query_text") or ""),
                # bare annotation names carry no extension -> prefer the attachment filename
                query_image=str(img_att.get("file_name") or idict.get("query_image") or ""),
                query_audio=(t8_attachment_filename(vkey, str(aud_att.get("file_name") or idict.get("query_audio") or ""))
                             if task == "impairment_assistance" else
                             str(aud_att.get("file_name") or idict.get("query_audio") or "")),
                video=vfile,
                responses=responses,
                query_time=q_ms,
                video_name=str(rec.get("video_name") or ""),
                video_url=str(rec.get("video_url") or ""),
                video_key=vkey,
                video_fps=fps,
                video_duration_sec=duration,
                source_start_ms=_as_float(rec.get("source_start_ms")),
                source_end_ms=_as_float(rec.get("source_end_ms")),
                # stripped + lowercased: stray whitespace ("third-person ") would otherwise
                # split a subset in two when aggregating by these facets
                scene=str(rec.get("scene") or "").strip(),
                language=str(rec.get("language") or "").strip().lower(),
                perspective=str(rec.get("perspective") or "").strip().lower(),
                raw_interaction_id=raw_id,
                is_available=available,
                publication_status=pub_status,
                query_image_url=str(img_att.get("url") or ""),
                query_audio_url=str(aud_att.get("url") or ""),
                raw={"task_type": raw_task, "query_type": raw_qtype, "sub_task_type": raw_sub},
            )
            out.append(it)
            rep.n_interactions_kept += 1
            rep.n_responses_kept += len(responses)
            rep.task_type_dist[task] += 1

    return out, rep

"""Accepted Table 2 LiveCC onset grouping; preserve the scorer's two-second anchors."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Tuple



from omni_eval.config import get, load_config  # noqa: E402
from omni_eval.metrics.temporal import EmitItem, merge_emits  # noqa: E402

# LiveCC's trained continuation marker, at the end of a chunk only. Stripped when joining, because
# it is a streaming control token rather than something the model "said" -- leaving it in would put
# an ellipsis every few words through the joined text and change what the judge reads. A chunk that
# is ONLY this marker never reaches a prediction file: `LiveCCModel.should_speak` calls it standby.
_CONTINUATION = re.compile(r"(?:\.\.\.|…)\s*$")
_WS = re.compile(r"\s+")
# Guard against re-running the tool on its own output: the second pass would join across the 2 s
# boundaries the first pass established, which is not idempotent.
_SEGMENTATION_KEY = "emit_segmentation"


def _strip_chunk(text: str) -> str:
    return _WS.sub(" ", _CONTINUATION.sub("", text or "")).strip()


def _scorer_onsets(emits: List[dict], merge_gap: float) -> List[float]:
    """The onsets the FROZEN scorer derives from these emits, computed by its own code.

    Not a reimplementation: this calls ``temporal.merge_emits`` so the two can never drift.
    """
    items = [EmitItem(float(e["time_sec"]), e.get("reply_text", "") or "",
                      e.get("response_type", "text") or "text") for e in emits]
    return [e.time_sec for e in merge_emits(items, merge_gap)]


def merge_record(rec: Dict[str, Any], merge_gap: float) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """Return ``(new_record, counters)``. ``rec`` is not modified."""
    emits = sorted(rec.get("emits") or [], key=lambda e: float(e["time_sec"]))
    groups: List[List[dict]] = []
    for e in emits:
        t = float(e["time_sec"])
        # The comparison is against the group's FIRST tick, not its last -- character for character
        # what `temporal.merge_emits` does (`e.time_sec - out[-1].time_sec <= gap`, where `out[-1]`
        # is the kept onset and never advances). Chaining off the last tick instead would swallow a
        # whole monologue into one emit: 40 consecutive 1 s chunks would become 1 utterance where
        # the scorer sees 14 onsets, which moves the TIMING half of the score for this column only.
        # See `_scorer_onsets` below -- the invariant is asserted, not just documented.
        if groups and t - float(groups[-1][0]["time_sec"]) <= merge_gap:
            groups[-1].append(e)
        else:
            groups.append([e])

    out: List[dict] = []
    n_dups = 0
    n_empty = 0
    dropped: List[float] = []
    for g in groups:
        kept: List[dict] = []
        for e in g:
            txt = _strip_chunk(e.get("reply_text", ""))
            if kept and txt == _strip_chunk(kept[-1].get("reply_text", "")):
                n_dups += 1
                continue
            kept.append(e)
        reply = " ".join(_strip_chunk(e.get("reply_text", "")) for e in kept).strip()
        if not reply:
            # Every chunk in the group was pure markup/marker. Cannot happen through
            # `should_speak`, but a dropped fire must be counted rather than silently vanish.
            n_empty += 1
            dropped.append(float(g[0]["time_sec"]))
            continue
        raw = " ".join(_strip_chunk(e.get("raw", "") or e.get("reply_text", "")) for e in kept)
        out.append({
            "time_sec": float(g[0]["time_sec"]),          # onset, matching merge_emits' earliest
            "response_type": g[0].get("response_type", "text"),
            "reply_text": reply,
            "raw": raw.strip(),
            "n_chunks": len(g),
        })

    # THE invariant, asserted rather than trusted: re-segmenting the output must not move a single
    # onset. What we keep has to be exactly what the frozen scorer would have derived from the raw
    # file, so the timing half of this column comes from `metrics/` unmodified and only the judged
    # TEXT changes. An all-markup group (unreachable through `should_speak`) is the sole allowed
    # absence, and it is counted in `n_empty_groups_dropped`.
    want = _scorer_onsets(emits, merge_gap)
    for t in dropped:
        want.remove(t)
    got = [e["time_sec"] for e in out]
    if got != want:
        raise AssertionError(
            f"{rec.get('interaction_id')}: re-segmentation moved the onsets, so this file would "
            f"score a different TIMING than the raw one -- refusing to write it. "
            f"{len(got)} merged vs {len(want)} from temporal.merge_emits; first divergence at "
            f"{next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))}. "
            f"The grouping rule here and `merge_emits` have drifted apart.")

    new = dict(rec)
    new["emits"] = out
    meta = dict(rec.get("meta") or {})
    meta[_SEGMENTATION_KEY] = "merged"
    meta["n_raw_emits"] = len(emits)
    meta["n_utterances"] = len(out)
    meta["n_collapsed_dups"] = n_dups
    if n_empty:
        meta["n_empty_groups_dropped"] = n_empty
    meta["merge_gap_sec"] = merge_gap
    new["meta"] = meta
    return new, {"raw": len(emits), "utt": len(out), "dups": n_dups, "empty": n_empty}

"""Shared temporal scorer (used by BOTH protocols).

Emits (fires) are first merged (fires within ``merge_gap`` collapse to one), then matched to
GT triggers with tolerance ``+-tol``:

- **point** GT (notify_trigger_time): strict greedy 1-to-1; extra in-window fires are FP.
- **window** GT (valid_response_window, e.g. narration/interpretation): many-to-one — every
  fire within ``[w_s-tol, w_e+tol]`` belongs to the window (not FP); the window is a TP if it
  got >=1 fire. Coverage / redundancy / temporal-IoU are reported as diagnostics.

Timing P/R/F1: ``TP=|matched|``, ``FP=|unmatched emits|``, ``FN=|unmatched GT|``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class GTItem:
    gid: int
    center: float
    win_lo: float          # matching window low (raw_lo - tol)
    win_hi: float          # matching window high (raw_hi + tol)
    window_start: float    # for latency (raw_lo)
    is_window: bool
    raw_lo: float = 0.0    # GT window/point start (no tolerance)
    raw_hi: float = 0.0    # GT window/point end (no tolerance)
    payload: object = None  # (Response, MetricProfile) etc.


@dataclass
class EmitItem:
    time_sec: float
    reply_text: str = ""
    response_type: str = "text"
    redundant: int = 0


@dataclass
class MatchedUnit:
    gt: GTItem
    emits: List[EmitItem]
    primary: EmitItem

    @property
    def latency(self) -> float:
        return max(0.0, self.primary.time_sec - self.gt.window_start)

    @property
    def redundancy(self) -> int:
        return max(0, len(self.emits) - 1)


@dataclass
class MatchResult:
    matched: List[MatchedUnit] = field(default_factory=list)
    fp: List[EmitItem] = field(default_factory=list)
    fn: List[GTItem] = field(default_factory=list)

    @property
    def tp(self) -> int:
        return len(self.matched)

    def counts(self) -> Tuple[int, int, int]:
        return self.tp, len(self.fp), len(self.fn)


def make_gt(gid: int, *, center=None, window=None, tol: float = 2.0) -> GTItem:
    if window is not None:
        w_s, w_e = window
        return GTItem(gid, 0.5 * (w_s + w_e), w_s - tol, w_e + tol, w_s, True, raw_lo=w_s, raw_hi=w_e)
    return GTItem(gid, center, center - tol, center + tol, center, False, raw_lo=center, raw_hi=center)


def merge_emits(emits: List[EmitItem], gap: float) -> List[EmitItem]:
    """Collapse fires within ``gap`` seconds into one (earliest time, first reply)."""
    if not emits:
        return []
    ordered = sorted(emits, key=lambda e: e.time_sec)
    out = [EmitItem(ordered[0].time_sec, ordered[0].reply_text, ordered[0].response_type)]
    for e in ordered[1:]:
        if e.time_sec - out[-1].time_sec <= gap:
            out[-1].redundant += 1
        else:
            out.append(EmitItem(e.time_sec, e.reply_text, e.response_type))
    return out


def _dist(emit: EmitItem, gt: GTItem) -> float:
    if gt.is_window:
        return max(0.0, gt.win_lo + 0.0 - emit.time_sec, emit.time_sec - gt.win_hi)
    return abs(emit.time_sec - gt.center)


def match(emits: List[EmitItem], gts: List[GTItem], tol: float, merge_gap: float = 2.0,
          pre_merged: bool = False) -> MatchResult:
    if not pre_merged:
        emits = merge_emits(emits, merge_gap)
    res = MatchResult()
    remaining = list(emits)

    # 1) window GTs absorb all their in-window fires (many-to-one)
    for gt in [g for g in gts if g.is_window]:
        absorbed = [e for e in remaining if gt.win_lo <= e.time_sec <= gt.win_hi]
        remaining = [e for e in remaining if not (gt.win_lo <= e.time_sec <= gt.win_hi)]
        if absorbed:
            primary = min(absorbed, key=lambda e: e.time_sec)
            res.matched.append(MatchedUnit(gt, absorbed, primary))
        else:
            res.fn.append(gt)

    # 2) point GTs: greedy nearest 1-to-1
    point_gts = [g for g in gts if not g.is_window]
    cand = [
        (_dist(e, g), i, j)
        for i, e in enumerate(remaining)
        for j, g in enumerate(point_gts)
        if g.win_lo <= e.time_sec <= g.win_hi
    ]
    cand.sort(key=lambda x: x[0])
    used_e, used_g = set(), set()
    for _, i, j in cand:
        if i in used_e or j in used_g:
            continue
        used_e.add(i)
        used_g.add(j)
        res.matched.append(MatchedUnit(point_gts[j], [remaining[i]], remaining[i]))
    res.fp.extend([e for i, e in enumerate(remaining) if i not in used_e])
    res.fn.extend([g for j, g in enumerate(point_gts) if j not in used_g])
    return res


@dataclass
class FirstFireOutcome:
    """First-fire verdict for one point GT (used by the per-item accuracy scorer).

    ``status`` is one of:
      - ``"ontime"``    first relevant fire lands in ``[T-tol, T+tol]``  (content still checked)
      - ``"premature"`` first relevant fire lands BEFORE ``T-tol`` (抢跑 -> item fails)
      - ``"miss"``      no fire up to ``T+tol`` in this event's neighborhood
    """
    gt: GTItem
    status: str
    emit: Optional[EmitItem] = None


def assign_first_fire(
    emits: List[EmitItem], point_gts: List[GTItem], tol: float,
    merge_gap: float = 2.0, pre_merged: bool = False,
) -> List[FirstFireOutcome]:
    """Assign the model's FIRST proactive utterance to each point GT (locked scoring rule).

    Events are processed in time order; each GT claims the EARLIEST not-yet-consumed fire that
    belongs to its neighborhood ``(prev_midpoint, T+tol]`` (the lower bound is the midpoint to the
    previous GT, so one event's on-time fire can't be mis-charged as the next event's 抢跑). If
    that first fire is before ``T-tol`` it is premature (the item fails even if a later fire would
    have been correct); if it is in-window the caller still content-checks it; if no fire lands by
    ``T+tol`` the event is a miss.
    """
    if not pre_merged:
        emits = merge_emits(emits, merge_gap)
    gts = sorted(point_gts, key=lambda g: g.center)
    order = sorted(range(len(emits)), key=lambda i: emits[i].time_sec)
    consumed: set = set()
    out: List[FirstFireOutcome] = []
    for k, gt in enumerate(gts):
        lo_bound = float("-inf") if k == 0 else 0.5 * (gts[k - 1].center + gt.center)
        chosen: Optional[int] = None
        for i in order:
            if i in consumed:
                continue
            t = emits[i].time_sec
            if lo_bound < t <= gt.win_hi:
                chosen = i
                break
        if chosen is None:
            out.append(FirstFireOutcome(gt, "miss", None))
            continue
        consumed.add(chosen)
        e = emits[chosen]
        status = "premature" if e.time_sec < gt.win_lo else "ontime"
        out.append(FirstFireOutcome(gt, status, e))
    return out


def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def window_iou(unit: MatchedUnit) -> Optional[float]:
    """Temporal IoU of the predicted firing span vs the GT window (window units only)."""
    gt = unit.gt
    if not gt.is_window:
        return None
    w_s, w_e = gt.raw_lo, gt.raw_hi
    ts = min(e.time_sec for e in unit.emits)
    te = max(e.time_sec for e in unit.emits)
    inter = max(0.0, min(te, w_e) - max(ts, w_s))
    union = max(te, w_e) - min(ts, w_s)
    return inter / union if union > 0 else 0.0

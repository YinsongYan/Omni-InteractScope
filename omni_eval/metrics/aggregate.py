"""Per-interaction scoring + roll-up aggregation.

``score_interaction`` turns one (interaction, prediction) pair into raw counts and diagnostics.
``aggregate`` rolls a list of those up: **micro** within a sub_task, **macro** to category,
**macro** to overall, plus micro breakdowns for modality / audio-dependency / long-memory /
query-type subsets and — when the annotations carry video-level metadata — language /
perspective / trigger-modality subsets. Headline numbers per bucket: content-gated **joint_f1**,
plus **time_f1**, mean judge score (1-10), **turn_taking_f1**, and **silence_accuracy**.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

from ..config import get
from ..schema import Interaction
from ..taxonomy import MetricProfile, TaxonomyRouter
from .content import score_content
from .llm_judge import LLMJudge
from .temporal import (
    EmitItem,
    assign_first_fire,
    make_gt,
    match,
    merge_emits,
    prf,
    window_iou,
)
from .turn_taking import build_noise_regions, score_noise


def _emit_items(rec: Dict[str, Any]) -> List[EmitItem]:
    return [
        EmitItem(
            time_sec=float(e["time_sec"]),
            reply_text=e.get("reply_text", ""),
            response_type=e.get("response_type", "text"),
        )
        for e in rec.get("emits", [])
    ]


def _content_pass(
    profile: MetricProfile, question: str, gt_reply: str, pred_reply: str,
    judge: Optional[LLMJudge], tool_f1_pass: float, judge_pass: int,
) -> Tuple[bool, float, Optional[int]]:
    """Binary content verdict for one matched fire (uses the routed judge rubric)."""
    if profile.content_scorer == "none":
        return True, 1.0, None
    cs = score_content(
        scorer=profile.content_scorer, question=question,
        gt_reply=gt_reply, pred_reply=pred_reply, judge=judge,
        tool_f1_pass=tool_f1_pass, judge_pass=judge_pass, rubric=profile.judge_rubric,
    )
    return cs.ok, cs.value, cs.judge10


def _response_successes(
    it: Interaction, merged: List[EmitItem], router: TaxonomyRouter,
    judge: Optional[LLMJudge], tol: float, noise_window: float,
    tool_f1_pass: int, judge_pass: int,
) -> List[Dict[str, Any]]:
    """Per-Response binary success (the headline accuracy unit).

    - point  : FIRST-fire rule -> success iff first relevant fire is on-time AND content correct
               (premature/miss/wrong-content all fail).
    - window : success iff >=1 fire in [w_s-tol, w_e+tol] AND content correct.
    - noise  : success iff the model stayed silent in [c-w, c+w].
    status labels: ok | content_fail | premature | miss | spoke.
    """
    out: List[Dict[str, Any]] = []

    # noise (should-stay-silent) -> one item per silence response, in schema order
    sil = [r for r in it.responses if r.is_silence]
    for r, (lo, hi) in zip(sil, build_noise_regions(sil, noise_window)):
        spoke = any(lo <= e.time_sec <= hi for e in merged)
        out.append({"response_id": r.response_id, "matching_mode": "noise",
                    "status": "spoke" if spoke else "ok", "success": not spoke})

    # point + window speak GTs
    point_gts = []
    point_meta: Dict[int, tuple] = {}
    gid = 0
    for r in it.responses:
        if r.is_silence:
            continue
        profile = router.route(it, r)
        if r.valid_response_window_sec is not None:
            g = make_gt(gid, window=r.valid_response_window_sec, tol=tol)
            absorbed = [e for e in merged if g.win_lo <= e.time_sec <= g.win_hi]
            if not absorbed:
                out.append({"response_id": r.response_id, "matching_mode": "window",
                            "status": "miss", "success": False})
            else:
                pred = " ".join(e.reply_text for e in absorbed).strip()
                ok, _, _ = _content_pass(profile, it.query_text, r.assistant_reply_text,
                                         pred, judge, tool_f1_pass, judge_pass)
                out.append({"response_id": r.response_id, "matching_mode": "window",
                            "status": "ok" if ok else "content_fail", "success": ok})
        elif r.notify_trigger_sec is not None:
            point_gts.append(make_gt(gid, center=r.notify_trigger_sec, tol=tol))
            point_meta[gid] = (r, profile)
        else:
            gid += 1
            continue
        gid += 1

    if point_gts:
        for o in assign_first_fire(merged, point_gts, tol, pre_merged=True):
            r, profile = point_meta[o.gt.gid]
            if o.status != "ontime":
                out.append({"response_id": r.response_id, "matching_mode": "point",
                            "status": o.status, "success": False})  # premature | miss
            else:
                ok, _, _ = _content_pass(profile, it.query_text, r.assistant_reply_text,
                                         o.emit.reply_text, judge, tool_f1_pass, judge_pass)
                out.append({"response_id": r.response_id, "matching_mode": "point",
                            "status": "ok" if ok else "content_fail", "success": ok})
    return out


def _trigger_modality(it: Interaction) -> str:
    """Coarsest trigger modality over an interaction's speak targets (for subset reporting)."""
    kinds = {r.trigger_type for r in it.responses if not r.is_silence and r.trigger_type}
    if not kinds:
        return "other"
    if len(kinds) == 1:
        return next(iter(kinds))
    return "vision_audio" if kinds <= {"vision", "audio", "vision_audio"} else "mixed"


def score_interaction(
    it: Interaction,
    rec: Dict[str, Any],
    router: TaxonomyRouter,
    judge: Optional[LLMJudge],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    tol = get(cfg, "temporal.tolerance_sec", 2.0)
    merge_gap = get(cfg, "temporal.merge_gap_sec", 2.0)
    noise_window = get(cfg, "turn_taking.noise_window_sec", 4.0)
    tool_f1_pass = get(cfg, "content.tool_f1_pass", 1.0)
    judge_pass = get(cfg, "judge.pass_threshold", 6)

    merged = merge_emits(_emit_items(rec), merge_gap)

    # ---- noise / turn-taking negatives ----
    regions = build_noise_regions(it.responses, noise_window)
    noise_res = score_noise([e.time_sec for e in merged], regions)
    remaining = [e for i, e in enumerate(merged) if i not in noise_res.consumed_emit_idx]

    # ---- speak GTs (point/window) ----
    gts = []
    payloads = {}
    gid = 0
    for r in it.responses:
        if r.is_silence:
            continue
        profile = router.route(it, r)
        if r.valid_response_window_sec is not None:
            g = make_gt(gid, window=r.valid_response_window_sec, tol=tol)
        elif r.notify_trigger_sec is not None:
            g = make_gt(gid, center=r.notify_trigger_sec, tol=tol)
        else:
            continue
        gts.append(g)
        payloads[gid] = (r, profile)
        gid += 1

    mr = match(remaining, gts, tol, pre_merged=True)

    # ---- content on matched units ----
    content_vals: List[float] = []
    judge10: List[int] = []
    joint_tp = 0
    latencies: List[float] = []
    redundancies: List[int] = []
    ious: List[float] = []
    for unit in mr.matched:
        r, profile = payloads[unit.gt.gid]
        if profile.content_scorer == "none":
            ok, val, j10 = True, 1.0, None
        else:
            pred_reply = " ".join(e.reply_text for e in unit.emits).strip()
            cs = score_content(
                scorer=profile.content_scorer,
                question=it.query_text,
                gt_reply=r.assistant_reply_text,
                pred_reply=pred_reply,
                judge=judge,
                tool_f1_pass=tool_f1_pass,
                judge_pass=judge_pass,
                rubric=profile.judge_rubric,
            )
            ok, val, j10 = cs.ok, cs.value, cs.judge10
        content_vals.append(val)
        if j10 is not None:
            judge10.append(j10)
        if ok:
            joint_tp += 1
        latencies.append(unit.latency)
        redundancies.append(unit.redundancy)
        if unit.gt.is_window:
            iou = window_iou(unit)
            if iou is not None:
                ious.append(iou)

    tp, fp_time, fn = mr.counts()
    n_probes = int(rec.get("meta", {}).get("n_probes", 0))

    # ---- per-Response success (headline accuracy; point uses the first-fire rule) ----
    successes = _response_successes(
        it, merged, router, judge, tol, noise_window, tool_f1_pass, judge_pass
    )
    success_n = len(successes)
    success_hit = sum(1 for s in successes if s["success"])
    status_counts = dict(Counter(s["status"] for s in successes))

    return {
        "interaction_id": it.interaction_id,
        "task_type": it.task_type,
        "sub_task_type": it.sub_task_type,
        "protocol": rec.get("protocol", ""),
        "modality_tag": rec.get("modality_tag", "n/a"),
        "input_config": rec.get("input_config", ""),
        "audio_starved": bool(rec.get("audio_starved", False)),
        "supported": rec.get("supported", True),
        "long_memory": any(r.long_memory == "yes" for r in it.responses),
        "needs_audio": it.needs_audio,
        "query_type": it.query_type,
        # video-level facets (video_grouped layout; empty strings for the flat layout)
        "language": it.language,
        "perspective": it.perspective,
        "scene": it.scene,
        "trigger_modality": _trigger_modality(it),
        "is_available": it.is_available,
        "tp": tp,
        "fp_time": fp_time,
        "fn": fn,
        "joint_tp": joint_tp,
        "noise_regions": noise_res.n_regions,
        "noise_fp": noise_res.fp,
        "noise_tn": noise_res.tn,
        "content_vals": content_vals,
        "judge10": judge10,
        "latencies": latencies,
        "redundancies": redundancies,
        "ious": ious,
        "n_probes": n_probes,
        # headline accuracy (per-Response success)
        "success_n": success_n,
        "success_hit": success_hit,
        "status_counts": status_counts,
        "successes": successes,
    }


# --------------------------------------------------------------------------------------
def _mean(xs: List[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


def _micro(bucket: List[Dict[str, Any]]) -> Dict[str, Any]:
    tp = sum(r["tp"] for r in bucket)
    fp_time = sum(r["fp_time"] for r in bucket)
    fn = sum(r["fn"] for r in bucket)
    joint_tp = sum(r["joint_tp"] for r in bucket)
    noise_fp = sum(r["noise_fp"] for r in bucket)
    noise_tn = sum(r["noise_tn"] for r in bucket)
    n_probes = sum(r["n_probes"] for r in bucket)

    time_p, time_r, time_f1 = prf(tp, fp_time, fn)
    joint_p, joint_r, joint_f1 = prf(joint_tp, fp_time, fn)
    # turn-taking unifies over-firing (fp_time) + spoke-to-noise (noise_fp)
    tt_p, tt_r, tt_f1 = prf(tp, fp_time + noise_fp, fn)

    # Detection F1s are only defined when the bucket has at least one positive GT (tp+fn>0).
    # A pure should-stay-silent bucket has none -> report None (excluded from macro averages);
    # its behavior is captured by silence_accuracy instead.
    has_positives = (tp + fn) > 0
    time_f1_r = round(time_f1, 4) if has_positives else None
    joint_f1_r = round(joint_f1, 4) if has_positives else None
    tt_f1_r = round(tt_f1, 4) if has_positives else None

    content_vals = [v for r in bucket for v in r["content_vals"]]
    judge10 = [v for r in bucket for v in r["judge10"]]
    latencies = [v for r in bucket for v in r["latencies"]]
    redundancies = [v for r in bucket for v in r["redundancies"]]
    ious = [v for r in bucket for v in r["ious"]]

    # headline: per-Response success accuracy + status breakdown (first-fire scored)
    success_n = sum(r.get("success_n", 0) for r in bucket)
    success_hit = sum(r.get("success_hit", 0) for r in bucket)
    status_break: Counter = Counter()
    for r in bucket:
        status_break.update(r.get("status_counts", {}))

    return {
        "n": len(bucket),
        "accuracy": round(success_hit / success_n, 4) if success_n else None,
        "n_items": success_n,
        "status_breakdown": dict(status_break),
        "time_precision": round(time_p, 4) if has_positives else None,
        "time_recall": round(time_r, 4) if has_positives else None,
        "time_f1": time_f1_r,
        "joint_precision": round(joint_p, 4) if has_positives else None,
        "joint_recall": round(joint_r, 4) if has_positives else None,
        "joint_f1": joint_f1_r,
        "turn_taking_f1": tt_f1_r,
        "silence_accuracy": round(noise_tn / (noise_tn + noise_fp), 4) if (noise_tn + noise_fp) else None,
        "content_score_norm": round(_mean(content_vals), 4) if content_vals else None,
        "judge_1to10": round(_mean(judge10), 3) if judge10 else None,
        "mean_latency_sec": round(_mean(latencies), 3) if latencies else None,
        "mean_redundancy": round(_mean(redundancies), 3) if redundancies else None,
        "mean_window_iou": round(_mean(ious), 4) if ious else None,
        "false_alarm_rate": round(fp_time / n_probes, 4) if n_probes else None,
        "counts": {"tp": tp, "fp_time": fp_time, "fn": fn, "joint_tp": joint_tp,
                   "noise_fp": noise_fp, "noise_tn": noise_tn, "n_probes": n_probes},
    }


_MACRO_KEYS = [
    "accuracy",  # headline: per-Response success rate
    "time_f1", "joint_f1", "turn_taking_f1", "silence_accuracy",
    "content_score_norm", "judge_1to10", "mean_latency_sec", "false_alarm_rate",
]


def _macro(child_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    out = {"n_groups": len(child_metrics)}
    for k in _MACRO_KEYS:
        vals = [m[k] for m in child_metrics if m.get(k) is not None]
        out[k] = round(sum(vals) / len(vals), 4) if vals else None
    return out


def _by_value(results: List[Dict[str, Any]], subset, key: str) -> Dict[str, Any]:
    """Micro metrics per distinct value of ``key`` (skips blanks, so flat-layout runs stay clean)."""
    values = sorted({r.get(key) or "" for r in results} - {""})
    return {v: subset(lambda r, val=v, k=key: r.get(k) == val) for v in values}


def aggregate(results: List[Dict[str, Any]], cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    # group by category -> subtask
    by_cat: Dict[str, Dict[str, List[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        by_cat[r["task_type"]][r["sub_task_type"]].append(r)

    by_category: Dict[str, Any] = {}
    category_metrics = []
    for cat, subs in by_cat.items():
        sub_metrics = {}
        for sub, items in subs.items():
            sub_metrics[sub] = _micro(items)
        cat_macro = _macro(list(sub_metrics.values()))
        by_category[cat] = {"summary": cat_macro, "by_subtask": sub_metrics}
        category_metrics.append(cat_macro)

    overall_macro = _macro(category_metrics)
    overall_micro = _micro(results)

    # subsets (micro over filtered sets)
    def subset(pred):
        sel = [r for r in results if pred(r)]
        return _micro(sel) if sel else None

    subsets = {
        "modality": {
            tag: subset(lambda r, t=tag: r["modality_tag"] == t)
            for tag in ("native", "asr-substituted", "audio-dropped", "unsupported", "n/a")
        },
        "audio_dependency": {
            "needs_audio": subset(lambda r: r["needs_audio"]),
            "no_audio": subset(lambda r: not r["needs_audio"]),
            # audio-dependent items run in a cell that withholds the audio (cell ``v``): the
            # ablation's floor, kept in the denominator rather than skipped.
            "audio_starved": subset(lambda r: r.get("audio_starved", False)),
        },
        "long_memory": {
            "yes": subset(lambda r: r["long_memory"]),
            "no": subset(lambda r: not r["long_memory"]),
        },
        "query_type": {
            qt: subset(lambda r, q=qt: r["query_type"] == q)
            for qt in ("text", "text_image", "audio", "text_audio")
        },
        # data-driven facets: whatever values the loaded annotations / predictions actually carry
        "input_config": _by_value(results, subset, "input_config"),
        "language": _by_value(results, subset, "language"),
        "perspective": _by_value(results, subset, "perspective"),
        "trigger_modality": _by_value(results, subset, "trigger_modality"),
    }
    subsets = {k: {kk: vv for kk, vv in v.items() if vv is not None} for k, v in subsets.items()}

    return {
        "overall": {"macro": overall_macro, "micro": overall_micro},
        "by_category": by_category,
        "subsets": subsets,
    }

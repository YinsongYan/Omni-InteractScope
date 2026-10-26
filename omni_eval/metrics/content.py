"""Content routing: exact tool-name sets and the configured text or translation judge."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from ..data.response_parser import extract_tool_calls, parse_reply
from .llm_judge import LLMJudge

_REGIONS = {
    "top-left", "top-center", "top-right",
    "center-left", "center", "center-right",
    "bottom-left", "bottom-center", "bottom-right",
}


@dataclass
class ContentScore:
    ok: bool
    value: float                 # normalized [0,1]
    kind: str
    detail: str = ""
    judge10: Optional[int] = None  # raw 1-10 when judge used


def set_f1(gold: List[str], pred: List[str]) -> float:
    g = {x.strip().lower() for x in gold if x.strip()}
    p = {x.strip().lower() for x in pred if x.strip()}
    if not g and not p:
        return 1.0
    if not g or not p:
        return 0.0
    tp = len(g & p)
    prec = tp / len(p)
    rec = tp / len(g)
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def _last_int(s: str) -> Optional[int]:
    nums = re.findall(r"-?\d+", s or "")
    return int(nums[-1]) if nums else None


def _find_region(s: str) -> Optional[str]:
    s = (s or "").lower()
    for r in _REGIONS:
        if r in s:
            return r
    return None


def score_content(
    scorer: str,
    question: str,
    gt_reply: str,
    pred_reply: str,
    judge: Optional[LLMJudge] = None,
    tool_f1_pass: float = 1.0,
    judge_pass: int = 6,
    rubric: Optional[str] = None,
) -> ContentScore:
    if scorer == "none":
        return ContentScore(ok=True, value=1.0, kind="none")

    if scorer == "tool_set":
        g = parse_reply(gt_reply).tool_calls
        p = parse_reply(pred_reply).tool_calls
        f1 = set_f1(g, p)
        return ContentScore(ok=f1 >= tool_f1_pass, value=f1, kind="tool_set",
                            detail=f"gold={g} pred={p} f1={f1:.2f}")

    if scorer == "integer":
        gi, pi = _last_int(gt_reply), _last_int(pred_reply)
        ok = gi is not None and gi == pi
        return ContentScore(ok=ok, value=1.0 if ok else 0.0, kind="integer",
                            detail=f"gold={gi} pred={pi}")

    if scorer == "state":
        g = (parse_reply(gt_reply).text or gt_reply).strip().lower()
        p = (parse_reply(pred_reply).text or pred_reply).strip().lower()
        ok = bool(g) and (g == p or g in p)
        return ContentScore(ok=ok, value=1.0 if ok else 0.0, kind="state")

    if scorer == "region":
        g, p = _find_region(gt_reply), _find_region(pred_reply)
        ok = g is not None and g == p
        return ContentScore(ok=ok, value=1.0 if ok else 0.0, kind="region",
                            detail=f"gold={g} pred={p}")

    if scorer == "tool_name":
        # T1's headline scorer: timing gate x "did it call the right tool", where the tool name is
        # read out of the prediction format-tolerantly but compared STRICTLY (at the shipped
        # tool_f1_pass: 1.0, the predicted set must equal the GT primary set -- no extra tool, no
        # missing one). Two decisions are folded in here, both deliberate:
        #
        #  - the body judge is gone (see the tool_judge branch below, kept for reference). GT marker
        #    bodies hold the search RESULT, model bodies hold the search QUERY, and this benchmark
        #    never executes a tool -- so that gate was unpassable by construction (internvl3.5:
        #    tool name right + body judged wrong 67 times, both gates passed 0 times).
        #  - the parse is tolerant, the vocabulary is not. See response_parser.extract_tool_calls.
        g = parse_reply(gt_reply).tool_calls
        ex = extract_tool_calls(pred_reply)
        if not g:
            # Cannot happen on bench1200 (all 209 T1 GT responses carry exactly one clean marker,
            # verified) and would be a routing mistake, not a model failure. Scored 0 rather than
            # 1: set_f1([], []) == 1.0 would silently hand out free accuracy, and `detail` makes
            # the real cause visible in the per-item dump.
            return ContentScore(ok=False, value=0.0, kind="tool_name",
                                detail=f"no-gt-tool (not scorable) pred={ex.names}")
        f1 = set_f1(g, ex.names)
        return ContentScore(ok=f1 >= tool_f1_pass, value=f1, kind="tool_name",
                            detail=f"gold={g} pred={ex.names} f1={f1:.2f} shape={ex.shape}")

    if scorer == "tool_judge":
        gp, pp = parse_reply(gt_reply), parse_reply(pred_reply)
        f1 = set_f1(gp.tool_calls, pp.tool_calls)
        if not gp.tool_calls:
            # Some GT replies in the tool-call tasks carry no <call:> marker at all; there is no
            # tool set to compare, so fall back to judging the reply as plain text.
            js = _judge_text(question, gt_reply, pred_reply, judge, judge_pass, rubric or "general")
            return ContentScore(ok=js.ok, value=js.value, kind="tool_judge",
                                detail=f"no-gt-tool; {js.detail}", judge10=js.judge10)
        tool_ok = f1 >= tool_f1_pass
        if not gp.text:
            # GT is a bare tool call -> the tool set is the whole content requirement.
            return ContentScore(ok=tool_ok, value=f1, kind="tool_judge",
                                detail=f"gold={gp.tool_calls} pred={pp.tool_calls} f1={f1:.2f}")
        js = _judge_text(question, gp.text, pp.text, judge, judge_pass, rubric or "general")
        return ContentScore(
            ok=tool_ok and js.ok,
            value=(f1 + js.value) / 2.0,
            kind="tool_judge",
            detail=f"gold={gp.tool_calls} pred={pp.tool_calls} f1={f1:.2f}; {js.detail}",
            judge10=js.judge10,
        )

    # judge / translation
    # rubric selects the binary judge prompt (general | event | translation). Callers pass the
    # routed profile.judge_rubric; fall back to translation for the translation scorer, else general.
    if rubric is None:
        rubric = "translation" if scorer == "translation" else "general"
    js = _judge_text(question, gt_reply, pred_reply, judge, judge_pass, rubric)
    return ContentScore(ok=js.ok, value=js.value, kind=scorer,
                        detail=js.detail, judge10=js.judge10)


def _judge_text(
    question: str,
    gt: str,
    pred: str,
    judge: Optional[LLMJudge],
    judge_pass: int,
    rubric: str,
) -> ContentScore:
    """One LLM-judge verdict on a pair of reply texts (binary mode, or legacy 1-10)."""
    if judge is None:
        raise ValueError("judge required for rubric=%r" % rubric)
    js = judge.score(question=question, gt=gt, pred=pred, rubric=rubric)
    if js.correct is not None:  # binary judge: Correct/Incorrect, no 1-10 to report
        return ContentScore(ok=js.correct, value=1.0 if js.correct else 0.0, kind="judge",
                            detail=js.explanation, judge10=None)
    return ContentScore(ok=js.score >= judge_pass, value=js.score / 10.0, kind="judge",
                        detail=js.explanation, judge10=js.score)

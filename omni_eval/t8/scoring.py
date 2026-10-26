"""Exact A intervals, one-to-one ordered target matching, no invented silence labels.

The metric is timely annotated-target recall, NOT precision, safety, or closed-loop success.
Unmatched reminders need media adjudication; they cannot automatically be called false alarms.
"""
import math
from .utterances import scoring_emits
from ..metrics.llm_judge import LLMJudge, _BINARY_PROMPT, _BINARY_RUBRIC

_BINARY_RUBRIC['impairment_assistance'] = 'impairment_assistance'
_BINARY_PROMPT['impairment_assistance'] = '''Judge the content of one accessibility assistance response.
The reference is an annotated target, not the only valid thing to say in the scene.
Correct means the prediction communicates this same target information and any essential action
without contradictory directions, unsafe advice, incorrect facts or invented critical details.
Equivalent wording is fine. A generic warning alone does not cover a specific target. A C step
must provide the reference's currently actionable guidance, not just repeat the user's intention.
Ignore timing (scored separately). Do not penalize a different language here: report language
compliance separately. If the prediction concerns another event, mark Incorrect FOR THIS TARGET;
that does not mean it is an invalid reminder. Do not infer validity of unannotated events.
User profile/request: {question}
Reference target: {gt}
Prediction: {pred}
Return exactly Correct or Incorrect.'''


class StrictT8Judge(LLMJudge):
    def _call_api_binary(self, prompt):
        ok, raw = super()._call_api_binary(prompt)
        if raw.strip().lower().rstrip('.') not in ('correct', 'incorrect'):
            raise RuntimeError('judge failure/invalid verdict; cannot become a benchmark score')
        return ok, raw

    def score(self, question, gt, pred, rubric='impairment_assistance'):
        result = super().score(question, gt, pred, rubric)
        if result.explanation.strip().lower().rstrip('.') not in ('correct', 'incorrect'):
            raise RuntimeError('invalid cached judge verdict')
        return result


def ordered_matching(ok):
    """Maximum cardinality monotone one-to-one response/emit matching (C order preserved)."""
    n, m = len(ok), len(ok[0]) if ok else 0
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            dp[i][j] = max(dp[i + 1][j], dp[i][j + 1],
                           1 + dp[i + 1][j + 1] if ok[i][j] else 0)
    pairs, i, j = [], 0, 0
    while i < n and j < m:
        if ok[i][j] and dp[i][j] == 1 + dp[i + 1][j + 1]:
            pairs.append((i, j))
            i, j = i + 1, j + 1
        elif dp[i + 1][j] > dp[i][j + 1]:
            i += 1
        else:
            j += 1
    return pairs


def bounds(it, response, tolerance_sec=2.0):
    if it.sub_task_type == 'A':
        if response.valid_response_window_sec is None:
            raise ValueError('A needs its annotated valid interval')
        return response.valid_response_window_sec
    t = response.notify_trigger_sec
    return max(it.query_time_sec or 0, t - tolerance_sec), t + tolerance_sec


def score(it, rec, judge, tolerance_sec=2.0):
    emits = scoring_emits(rec)
    if any(not math.isfinite(e['time_sec']) for e in emits):
        raise ValueError('non-finite emission time')
    question = f'Target user: {it.target_group}. Subtask: {it.sub_task_type}. Request: {it.query_text}'
    ok = [[False] * len(emits) for _ in it.responses]
    verdicts = []
    for i, response in enumerate(it.responses):
        lo, hi = bounds(it, response, tolerance_sec)
        for j, emit in enumerate(emits):
            if lo - 1e-9 <= emit['time_sec'] <= hi + 1e-9:
                result = judge.score(question, response.assistant_reply_text, emit['reply_text'],
                                     rubric='impairment_assistance')
                ok[i][j] = bool(result.correct)
                verdicts.append(dict(response_id=response.response_id, emit_index=j, content_ok=bool(result.correct)))
    pairs = ordered_matching(ok)
    return dict(interaction_id=it.interaction_id, video_key=it.video_key,
                sub_task_type=it.sub_task_type, target_group=it.target_group, qa_language=it.qa_language,
                n_targets=len(it.responses), hit_targets=len(pairs),
                timely_target_recall=len(pairs) / len(it.responses),
                complete_interaction=len(pairs) == len(it.responses),
                n_emits=len(emits), n_raw_emits=len(rec['emits']),
                emit_segmentation='continuation_eos_completion_time' if rec.get('model') == 'livecc-7b' else 'adapter',
                n_unmatched_emits=len(emits) - len(pairs),
                matches=[dict(response_id=it.responses[i].response_id, emit_index=j,
                              time_sec=emits[j]['time_sec']) for i, j in pairs],
                content_verdicts=verdicts)

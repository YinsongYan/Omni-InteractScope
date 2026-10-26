"""LLM-as-judge (GPT-4o) for open-ended content. Timing is scored by clock math elsewhere;
the judge only rates *what* was said.

**Binary only** (Correct/Incorrect), aligned with StreamArena's judge: one lenient-wording /
strict-facts prompt, a tiny token budget, and ``"incorrect" in out.lower() -> wrong`` else right.
That is the locked scoring path — a window hard-gate in code plus a binary content verdict. The
former ``graded`` (1-10) mode is gone: it made every headline number depend on a
``pass_threshold`` knob, and two runs scored under different thresholds are not comparable.

- Real mode: OpenAI chat.completions, temperature 0.
- Mock mode (no API key / ``OMNI_JUDGE_MOCK=1`` / ``mock=True``): deterministic lexical-overlap
  stub so the pipeline + tests run and still discriminate exact-match from unrelated replies.
- Disk cache keyed by (model, mode, rubric, question, gt, pred) so re-scoring never re-calls.
  ``mode`` stays in the key so caches written before this change still hit.
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from ..utils.io import append_jsonl, read_jsonl

# ---- binary (Correct/Incorrect) prompts (mirrors StreamArena judge) -------------------
# ``general``: lenient on wording/language/format, strict on facts (open-ended short answers).
_BINARY_PROMPT = {
    "general": (
        "You are an impartial judge.\n"
        "Decide whether the model's prediction correctly answers the question, based on the "
        "ground-truth answer.\n\n"
        "Be lenient on:\n"
        "- Wording / phrasing / language (Chinese vs English)\n"
        "- Extra explanation around the right answer\n"
        "- Equivalent expressions\n"
        "- Numeric formats that mean the same thing\n\n"
        "Be strict on:\n"
        "- Wrong facts, wrong names, wrong numbers, wrong yes/no.\n"
        "- If the model says it cannot answer / refuses / hallucinates, mark Incorrect.\n\n"
        "Question: {question}\n"
        "Ground truth answer: {gt}\n"
        "Model prediction: {pred}\n\n"
        'Answer only with a single word: "Correct" or "Incorrect".'
    ),
    # ``event``: proactive event-detection — treat the GT as a trigger phrase / event signal,
    # not a literal wording requirement. Correct if the prediction refers to the SAME event.
    "event": (
        "You are an impartial judge for a proactive event-detection task.\n"
        "- The user asked the model to notify them when a SPECIFIC event happens.\n"
        "- 'Question' describes when the model should speak up.\n"
        "- 'Ground truth answer' is the SHORT trigger phrase for the target event; treat it as a "
        "signal that the event was correctly detected, NOT as a literal wording requirement.\n"
        "- 'Model prediction' is what the model said when it decided the event happened; "
        "descriptive/caption-style wording is fine as long as it refers to the SAME event.\n\n"
        "Mark Correct when the prediction describes the same event/scene/action the ground truth "
        "points to and captures the correct key subject/object/action.\n"
        "Mark Incorrect when it describes a different event, gets a factual detail wrong, is "
        "generic/hallucinated, or refuses.\n\n"
        "Question: {question}\n"
        "Ground truth (trigger phrase representing the target event): {gt}\n"
        "Model prediction (what was said when the event was detected): {pred}\n\n"
        'Answer only with a single word: "Correct" or "Incorrect".'
    ),
    "translation": (
        "You are an impartial judge of a real-time interpretation/translation.\n"
        "Decide whether the model's translation adequately preserves the meaning of the "
        "reference translation. Be lenient on fluency/synonyms/format; be strict on wrong "
        "meaning, omissions, or added content. Refusal/gibberish is Incorrect.\n\n"
        "Source request / context: {question}\n"
        "Reference translation: {gt}\n"
        "Model translation: {pred}\n\n"
        'Answer only with a single word: "Correct" or "Incorrect".'
    ),
}
# Routed rubric -> prompt key (an unknown rubric falls back to the strict-facts general prompt).
_BINARY_RUBRIC = {"general": "general", "translation": "translation", "event": "event"}

_PASS_MOCK = 6  # mock lexical-overlap threshold (out of 10) that counts as "Correct"


@dataclass
class JudgeScore:
    score: int                        # 10 if correct else 1 (kept for mean-score reporting)
    explanation: str = ""
    cached: bool = False
    correct: Optional[bool] = None    # the verdict; always set
    mode: str = "binary"


def _key(model: str, mode: str, rubric: str, question: str, gt: str, pred: str) -> str:
    h = hashlib.sha256()
    h.update("\x1f".join([model, mode, rubric, question or "", gt or "", pred or ""]).encode("utf-8"))
    return h.hexdigest()


def _tokens(s: str):
    s = (s or "").lower()
    words = re.findall(r"[a-z0-9]+", s)
    cjk = re.findall(r"[一-鿿]", s)
    return set(words) | set(cjk)


def _mock_score(gt: str, pred: str) -> int:
    if not (pred or "").strip():
        return 1
    a, b = _tokens(gt), _tokens(pred)
    if not a and not b:
        return 10
    if not a or not b:
        return 5
    jacc = len(a & b) / len(a | b)
    return max(1, min(10, int(round(1 + 9 * jacc))))


class LLMJudge:
    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.0,
        mock: Optional[bool] = None,
        cache_path: Optional[str] = None,
        max_retries: int = 3,
        mode: str = "binary",
    ):
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.cache_path = cache_path
        if mode != "binary":
            # Loud rather than silently re-interpreting an old config: a graded run's numbers are
            # not comparable with a binary run's, so falling back quietly would corrupt a table.
            raise ValueError(
                f"judge mode {mode!r} is not supported; the locked scoring path is 'binary' "
                "(the graded 1-10 mode was removed). Set judge.mode: binary in configs/eval.yaml."
            )
        self.mode = mode
        self._cache: Dict[str, dict] = {}
        self._client = None
        self.mock = mock
        if self.mock is None:
            self.mock = os.environ.get("OMNI_JUDGE_MOCK", "") == "1"
        self.n_mock_rows_ignored = 0
        if cache_path:
            for rec in read_jsonl(cache_path):
                if "key" not in rec:
                    continue
                # A real run must never reuse a mock verdict. The cache key contains the model
                # name but NOT the mock flag, so a mock run and a real run of the same judge
                # model share keys -- and the mock verdict is pure lexical overlap. This is not
                # hypothetical: 44 rows written under ``gpt-4o-2024-08-06`` during an early mock
                # pass (21 of them "Correct") sat in outputs/judge_cache.jsonl and were being
                # served to the 2026-09-03 scoring campaign. Mock rows are recognised by the
                # ``mock`` prefix ``_score_binary`` / ``_mock_score`` write into ``explanation``.
                if not self.mock and str(rec.get("explanation", "")).startswith("mock"):
                    self.n_mock_rows_ignored += 1
                    continue
                self._cache[rec["key"]] = rec
            if self.n_mock_rows_ignored:
                print(f"[judge] ignored {self.n_mock_rows_ignored} mock rows in {cache_path} "
                      f"(real judge run); they will be re-judged for real")

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI  # type: ignore

            self._client = OpenAI(base_url=os.environ.get("OPENAI_BASE_URL") or None)
        return self._client

    # ---- binary (Correct/Incorrect) -------------------------------------------------
    def _call_api_binary(self, prompt: str) -> Tuple[bool, str]:
        client = self._ensure_client()
        last = ""
        for _ in range(self.max_retries):
            try:
                resp = client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=10,
                    messages=[{"role": "user", "content": prompt}],
                )
                last = resp.choices[0].message.content or ""
                # StreamArena convention: any "incorrect" substring -> wrong, else correct.
                return ("incorrect" not in last.lower()), last
            except Exception as e:  # noqa
                last = f"error: {e}"
        # On persistent failure, be conservative: mark Incorrect (matches StreamArena fallback).
        return False, last

    def score(
        self, question: str, gt: str, pred: str, rubric: str = "general"
    ) -> JudgeScore:
        return self._score_binary(question, gt, pred, rubric)

    def cache_key_for(self, question: str, gt: str, pred: str, rubric: str = "general") -> str:
        """The key ``score()`` would use — same rubric normalization."""
        return _key(self.model, "binary", _BINARY_RUBRIC.get(rubric, "general"),
                    question, gt, pred)

    def prewarm(self, batch, workers: int = 16) -> int:
        """Populate the cache for many judgements concurrently, then return how many were called.

        The serial path is ~1300 API calls per run at ~1.5 s each. Callers score as usual after
        this; every ``score()`` then hits the in-memory cache, so no scoring logic changes.
        ``batch`` is an iterable of ``(question, gt, pred, rubric)``.
        """
        pending = {}
        for item in batch:
            question, gt, pred, rubric = item
            k = self.cache_key_for(question, gt, pred, rubric)
            if k not in self._cache and k not in pending:
                pending[k] = (question, gt, pred, rubric)
        if not pending:
            return 0
        # Mock mode is pure local CPU — threads would only add overhead and log noise.
        if self.mock or workers <= 1:
            for a in pending.values():
                self.score(*a)
            return len(pending)
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(lambda a: self.score(*a), pending.values()))
        return len(pending)

    def _score_binary(self, question: str, gt: str, pred: str, rubric: str) -> JudgeScore:
        rk = _BINARY_RUBRIC.get(rubric, "general")
        k = _key(self.model, "binary", rk, question, gt, pred)
        if k in self._cache:
            c = self._cache[k]
            correct = bool(c["correct"])
            return JudgeScore(10 if correct else 1, c.get("explanation", ""),
                              cached=True, correct=correct, mode="binary")

        if self.mock:
            correct = _mock_score(gt, pred) >= _PASS_MOCK
            raw = f"mock:{'Correct' if correct else 'Incorrect'}"
        else:
            prompt = _BINARY_PROMPT[rk].format(question=question or "", gt=gt or "", pred=pred or "")
            correct, raw = self._call_api_binary(prompt)

        rec = {"key": k, "model": self.model, "mode": "binary", "rubric": rk,
               "correct": bool(correct), "explanation": raw}
        self._cache[k] = rec
        if self.cache_path:
            append_jsonl(self.cache_path, rec)
        return JudgeScore(10 if correct else 1, raw, correct=bool(correct), mode="binary")


class JudgeRecorder:
    """A judge-shaped stand-in that records requests instead of answering them.

    Scoring never *branches* on a judge verdict (a content result only labels a status), so a
    scoring pass driven by this recorder reveals exactly which judgements are needed — markers
    already stripped, rubrics already routed — without calling the API. The caller then feeds
    ``batch`` to :meth:`LLMJudge.prewarm` and re-scores against the warm cache. Deriving the
    batch this way keeps it in sync with ``metrics/content.py`` by construction instead of by
    duplicating its argument-shaping rules.
    """

    def __init__(self, judge: LLMJudge):
        self.mode = judge.mode
        self.batch: list = []

    def score(self, question: str, gt: str, pred: str, rubric: str = "general") -> JudgeScore:
        self.batch.append((question, gt, pred, rubric))
        return JudgeScore(1, "", correct=False, mode="binary")

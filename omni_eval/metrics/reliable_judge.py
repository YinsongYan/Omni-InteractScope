"""Opt-in real binary judge: API failures are missing evidence, never verdicts.

Uses LLMJudge's unchanged prompts, cache keys and successful scoring semantics.
Only valid binary cache records are reused. Successful calls remain resumable.
"""
from __future__ import annotations

import os
import re
import time
from typing import Callable

from .llm_judge import LLMJudge


class JudgeUnavailable(RuntimeError):
    def __init__(self, reason: str, *, retryable: bool):
        super().__init__(reason)
        self.retryable = retryable


def binary_verdict(raw):
    if not isinstance(raw, str):
        return None
    match = re.fullmatch(r'''[\s"'`*]*(correct|incorrect)[\s.!"'`*]*''', raw, re.I)
    return match.group(1).lower() == "correct" if match else None


def valid_cache_record(rec, model):
    verdict = binary_verdict(rec.get("explanation"))
    return (rec.get("model") == model and rec.get("mode") == "binary"
            and rec.get("rubric") in {"general", "event", "translation", "impairment_assistance"}
            and type(rec.get("correct")) is bool and verdict is not None
            and rec["correct"] == verdict)


class ReliableLLMJudge(LLMJudge):
    def __init__(self, *args, sleep: Callable = time.sleep,
                 event: Callable | None = None, **kwargs):
        if kwargs.get("mock") is not False:
            raise ValueError("ReliableLLMJudge requires explicit mock=False")
        super().__init__(*args, **kwargs)
        if self.max_retries < 1:
            raise ValueError("max_retries must be positive")
        self.sleep = sleep
        self.event = event or (lambda **event: None)
        self.n_invalid_rows_ignored = 0
        for key, rec in list(self._cache.items()):
            if not valid_cache_record(rec, self.model):
                del self._cache[key]
                self.n_invalid_rows_ignored += 1

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(base_url=os.environ.get("OPENAI_BASE_URL") or None,
                                  timeout=60.0, max_retries=0)
        return self._client

    def _call_api_binary(self, prompt):
        client = self._ensure_client()
        reason = "no_valid_verdict"
        for attempt in range(self.max_retries):
            retry_after = 0.0
            try:
                response = client.chat.completions.create(
                    model=self.model, temperature=self.temperature, max_tokens=10,
                    messages=[{"role": "user", "content": prompt}])
                raw = response.choices[0].message.content
                verdict = binary_verdict(raw)
                if verdict is not None:
                    self.event(kind="valid_verdict", attempt=attempt + 1)
                    return verdict, raw
                reason = "invalid_binary_response"
            except Exception as exc:
                code = getattr(exc, "status_code", None)
                # Never log exception bodies, which can contain prompts or credentials.
                reason = f"{type(exc).__name__}:http_{code}"
                retryable = (code in {408, 409, 425, 429} or
                             (isinstance(code, int) and code >= 500) or
                             (code is None and type(exc).__name__ in {
                                 "APIConnectionError", "APITimeoutError", "TimeoutError",
                                 "ConnectionError", "IndexError", "AttributeError"}))
                if not retryable:
                    self.event(kind="permanent_error", reason=reason)
                    raise JudgeUnavailable(reason, retryable=False) from None
                headers = getattr(getattr(exc, "response", None), "headers", {})
                try:
                    retry_after = max(0.0, float(headers.get("retry-after", 0)))
                except (ValueError, TypeError):
                    pass
            if attempt + 1 < self.max_retries:
                delay = min(300.0, max(retry_after, min(120.0, 2.0 ** (attempt + 1))))
                self.event(kind="retry", reason=reason, attempt=attempt + 1, delay_sec=delay)
                self.sleep(delay)
        self.event(kind="retry_budget_exhausted", reason=reason)
        raise JudgeUnavailable(reason, retryable=True)

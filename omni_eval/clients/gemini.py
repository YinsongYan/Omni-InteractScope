#!/usr/bin/env python3
"""Native Gemini generateContent client with cached responses and explicit credentials."""
from __future__ import annotations

import hashlib
import json
import os
import random
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from omni_eval.utils.io import append_jsonl, read_jsonl

DEFAULT_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


class GeminiError(RuntimeError):
    pass


class GeminiClient:
    """One ``generateContent`` call per :meth:`generate`, cached on (model, prompt, config).

    The cache is a JSONL file appended under an advisory lock (``utils.io.append_jsonl``), the
    same pattern as the judge cache, so a re-run of the builder re-uses every earlier answer and
    an interrupted build resumes for free. For an eval run that means the resume granularity is
    the *probe*, not the item: a run killed at 80% re-reads every probe of the interrupted
    interaction from disk.
    """

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        cache_path: Optional[str] = None,
        temperature: float = 0.0,
        max_output_tokens: int = 8192,
        json_mode: bool = True,
        max_retries: int = 4,
        timeout: float = 240.0,
        thinking: Optional[dict] = None,
        media_resolution: Optional[str] = None,
    ):
        self.model = model
        self.base_url = (base_url or os.environ.get("GEMINI_BASE_URL")
                         or DEFAULT_BASE).rstrip("/")
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.json_mode = json_mode
        self.max_retries = max_retries
        self.timeout = timeout
        # ``generationConfig.thinkingConfig``, passed through verbatim. ``None`` (the default) sends
        # no such field at all, i.e. the provider's own dynamic budget — which is what the benchmark
        # reports, and what keeps this client's cache keys byte-identical to the builders'. The
        # field name is version-dependent (``thinkingLevel`` on Gemini 3, ``thinkingBudget`` on 2.5),
        # so it is the caller's literal dict rather than a knob we translate.
        self.thinking = dict(thinking) if thinking else None
        # ``generationConfig.media_resolution``. Gemini 3 charges per *frame* by this setting, not by
        # our pixel budget: measured on this gateway at 20 frames/request, HIGH (the provider default,
        # sent as no field at all) = 1098 tok/frame, MEDIUM = 525, LOW = 262. Two spellings exist and
        # only these work: top-level ``media_resolution`` (snake_case) and top-level
        # ``mediaResolution``; a *per-part* ``mediaResolution`` is rejected 400. ``None`` sends
        # nothing, keeps the builders' cache keys byte-identical, and is the provider default.
        self.media_resolution = media_resolution or None
        self.cache_path = cache_path
        self._cache: Dict[str, dict] = {}
        self._lock = threading.Lock()
        self.calls = 0
        self.cache_hits = 0
        self.prompt_tokens = 0
        self.output_tokens = 0
        self.thoughts_tokens = 0
        # Every distinct ``modelVersion`` the gateway answered with. This is the API analogue of an
        # HF ``revision`` pin: a ``-preview`` id is a moving target, and two halves of a run served
        # by two different builds are as incomparable as two different checkpoints would be. We
        # cannot pin it, so we record it — see run_stats() in the adapter.
        self.model_versions: set = set()
        if cache_path:
            for rec in read_jsonl(cache_path):
                if "key" in rec:
                    self._cache[rec["key"]] = rec

    # ---- plumbing -------------------------------------------------------------------
    def _key(self, prompt: str, parts: Optional[List[dict]] = None) -> str:
        h = hashlib.sha256()
        h.update("\x1f".join([
            self.model, prompt, f"{self.temperature}", f"{self.max_output_tokens}",
            "json" if self.json_mode else "text",
        ]).encode("utf-8"))
        # Only mixed in when set, so the default (provider-default thinking) keys exactly as this
        # client always has and the builders' existing caches stay valid.
        if self.thinking:
            h.update(b"\x1ethinking\x1f")
            h.update(json.dumps(self.thinking, sort_keys=True).encode("utf-8"))
        # Same "only when set" rule: it changes how many image tokens the model actually sees, so a
        # LOW answer must never be served from a default-resolution cache entry.
        if self.media_resolution:
            h.update(b"\x1emediares\x1f")
            h.update(self.media_resolution.encode("utf-8"))
        # Media parts go into the key by content hash, not by index: two calls that differ only
        # in a frame must not share a cache entry, and the base64 itself is too big to key on.
        for p in parts or []:
            if "text" in p:
                h.update(b"\x1etext\x1f")
                h.update(p["text"].encode("utf-8"))
            else:
                blob = p.get("inlineData") or p.get("inline_data") or {}
                h.update(b"\x1emedia\x1f")
                h.update((blob.get("mimeType") or blob.get("mime_type") or "").encode("utf-8"))
                h.update(hashlib.sha256((blob.get("data") or "").encode("utf-8")).digest())
        return h.hexdigest()

    def _post(self, prompt: str, parts: Optional[List[dict]] = None) -> Tuple[str, dict]:
        cfg: Dict[str, Any] = {
            "temperature": self.temperature,
            "maxOutputTokens": self.max_output_tokens,
        }
        if self.json_mode:
            cfg["responseMimeType"] = "application/json"
        if self.thinking:
            cfg["thinkingConfig"] = dict(self.thinking)
        if self.media_resolution:
            cfg["media_resolution"] = self.media_resolution
        # Media first, instruction last — Gemini follows the trailing instruction more reliably
        # when the evidence precedes it, and the caller controls interleaving (e.g. a "t=12.0s"
        # text part before each frame) because a bare image carries no timestamp.
        req_parts = list(parts or []) + [{"text": prompt}]
        body = json.dumps({
            "contents": [{"role": "user", "parts": req_parts}],
            "generationConfig": cfg,
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/{self.model}:generateContent",
            data=body, method="POST",
            headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            payload = json.load(resp)
        # Error responses may contain request metadata; do not include their body in logs.
        if not payload.get("candidates"):
            raise GeminiError('provider response has no candidates')
        cand = payload["candidates"][0]
        finish = cand.get("finishReason")
        parts = (cand.get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        usage = payload.get("usageMetadata") or {}
        version = payload.get("modelVersion") or ""
        if not text.strip():
            raise GeminiError(f"empty answer (finishReason={finish}, usage={usage})")
        return text, {"finish": finish, "usage": usage, "model_version": version}

    def generate(self, prompt: str, parts: Optional[List[dict]] = None) -> str:
        """Return the model's text. Raises :class:`GeminiError` after ``max_retries``.

        ``parts`` is an optional list of raw Gemini parts (``{"text": ...}`` or
        ``{"inlineData": {"mimeType": ..., "data": <base64>}}``) prepended to ``prompt``.
        Only the media *hash* enters the cache key, so the JSONL cache stays small.
        """
        k = self._key(prompt, parts)
        with self._lock:
            hit = self._cache.get(k)
            if hit is not None:
                self.cache_hits += 1
                if hit.get("model_version"):
                    self.model_versions.add(hit["model_version"])
                return hit["text"]
        last = ""
        for attempt in range(self.max_retries):
            try:
                text, meta = self._post(prompt, parts)
                rec = {"key": k, "model": self.model, "text": text, "finish": meta["finish"]}
                if meta.get("model_version"):
                    rec["model_version"] = meta["model_version"]
                u = meta["usage"] or {}
                with self._lock:
                    self.calls += 1
                    self.prompt_tokens += int(u.get("promptTokenCount") or 0)
                    self.output_tokens += int(u.get("totalTokenCount") or 0)
                    self.thoughts_tokens += int(u.get("thoughtsTokenCount") or 0)
                    if meta.get("model_version"):
                        self.model_versions.add(meta["model_version"])
                    self._cache[k] = rec
                if self.cache_path:
                    append_jsonl(self.cache_path, rec)
                return text
            except (GeminiError, urllib.error.URLError, urllib.error.HTTPError,
                    TimeoutError, json.JSONDecodeError, OSError) as e:
                last = type(e).__name__
                if isinstance(e, urllib.error.HTTPError):
                    last += f' HTTP {e.code}'
                if attempt < self.max_retries - 1:
                    # Jittered backoff: the gateway is shared and rate-limited (rpm 5000), so a
                    # thundering herd of 16 workers retrying in lockstep just re-fails together.
                    time.sleep(min(30.0, 1.5 * (2 ** attempt)) * (0.6 + 0.8 * random.random()))
        raise GeminiError(f"{self.model}: giving up after {self.max_retries} tries; last={last}")

    def generate_json(self, prompt: str, parts: Optional[List[dict]] = None) -> Any:
        """:meth:`generate` + ``json.loads``, tolerating a ```json fence if one slips through."""
        text = self.generate(prompt, parts).strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
            if text.rstrip().endswith("```"):
                text = text.rstrip()[:-3]
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise GeminiError(f"{self.model}: answer is not JSON ({e}): {text[:300]!r}") from e

    def stats(self) -> dict:
        return {
            "model": self.model, "api_calls": self.calls, "cache_hits": self.cache_hits,
            "prompt_tokens": self.prompt_tokens, "total_tokens": self.output_tokens,
            "thoughts_tokens": self.thoughts_tokens,
            "model_versions": sorted(self.model_versions),
            "thinking": self.thinking or "provider-default",
        }

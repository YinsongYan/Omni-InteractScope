#!/usr/bin/env python3
"""OpenAI-compatible chat client with strict response validation and cached responses."""
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

DEFAULT_BASE = "https://api.openai.com/v1"


class OpenAIError(RuntimeError):
    pass


class OpenAIClient:
    """One ``chat/completions`` call per :meth:`generate`, cached on (model, prompt, config).

    The cache is a JSONL file appended under an advisory lock (``utils.io.append_jsonl``), the same
    pattern as the judge cache. For an eval run that means the resume granularity is the *probe*,
    not the item: a run killed at 80% re-reads every probe of the interrupted interaction from disk.
    """

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        cache_path: Optional[str] = None,
        temperature: float = 0.0,
        max_output_tokens: int = 2048,
        json_mode: bool = True,
        max_retries: int = 4,
        timeout: float = 240.0,
    ):
        self.model = model
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or DEFAULT_BASE).rstrip("/")
        # Credentials are provided explicitly through the environment.
        # first so the relabelling path resolves exactly as it always has.
        self.api_key = (api_key or os.environ.get("OPENAI_API_KEY")
                        or os.environ.get("BENCH_API_KEY") or "")
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.json_mode = json_mode
        self.max_retries = max_retries
        self.timeout = timeout
        self.cache_path = cache_path
        self._cache: Dict[str, dict] = {}
        self._lock = threading.Lock()
        self.calls = 0
        self.cache_hits = 0
        self.prompt_tokens = 0
        self.output_tokens = 0
        self.completion_tokens = 0
        self.reasoning_tokens = 0
        # Every distinct ``model`` string the gateway answered with. This is the API analogue of an
        # HF ``revision`` pin: a served build is a moving target, and two halves of a run served by
        # two different builds are as incomparable as two different checkpoints would be. We cannot
        # pin it, so we record it — see run_stats() in the adapter.
        self.model_versions: set = set()
        if cache_path:
            for rec in read_jsonl(cache_path):
                if "key" in rec:
                    self._cache[rec["key"]] = rec

    # ---- plumbing -------------------------------------------------------------------
    @staticmethod
    def _part_material(p: dict) -> Tuple[str, str]:
        """(kind marker, payload string) for a non-text content block, for hashing only.

        Covers every shape this repo sends: ``image_url``/``audio_url``/``video_url`` carry the
        (possibly ``data:``) URL under ``.url``, ``input_audio`` carries raw base64 under ``.data``
        plus a container name under ``.format``.
        """
        kind = str(p.get("type") or "")
        body = p.get(kind)
        if isinstance(body, dict):
            return f"{kind}\x1f{body.get('format') or ''}", str(body.get("url")
                                                                 or body.get("data") or "")
        return kind, str(body or "")

    def _key(self, prompt: str, parts: Optional[List[dict]] = None) -> str:
        h = hashlib.sha256()
        h.update("\x1f".join([
            self.model, prompt, f"{self.temperature}", f"{self.max_output_tokens}",
            "json" if self.json_mode else "text",
        ]).encode("utf-8"))
        # Media parts go into the key by content hash, not by index: two calls that differ only in
        # one frame must not share a cache entry, and the base64 itself is too big to key on. Mixed
        # in only when there are parts at all, so the text-only caller keys exactly as this client
        # always has and ``scripts/relabel_tools.py``'s existing cache stays valid.
        for p in parts or []:
            if p.get("type") == "text":
                h.update(b"\x1etext\x1f")
                h.update((p.get("text") or "").encode("utf-8"))
            else:
                head, payload = self._part_material(p)
                h.update(b"\x1emedia\x1f")
                h.update(head.encode("utf-8"))
                h.update(hashlib.sha256(payload.encode("utf-8")).digest())
        return h.hexdigest()

    def _post(self, prompt: str, parts: Optional[List[dict]] = None) -> Tuple[str, dict]:
        # Media first, instruction last — the model follows the trailing instruction more reliably
        # when the evidence precedes it. With no parts the content stays a bare string, byte for
        # byte the request this client has always sent.
        content: Any = (list(parts) + [{"type": "text", "text": prompt}]) if parts else prompt
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
        }
        if self.json_mode:
            body["response_format"] = {"type": "json_object"}
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"), method="POST",
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            payload = json.load(resp)
        # Error responses may contain request metadata; do not include their body in logs.
        if not payload.get("choices"):
            raise OpenAIError('provider response has no choices')
        ch = payload["choices"][0]
        msg = ch.get("message") or {}
        text = msg.get("content") or ""
        usage = payload.get("usage") or {}
        version = str(payload.get("model") or "")
        if not text.strip():
            # A reasoning model that spent the whole budget thinking answers with an empty
            # ``content`` and a full ``reasoning_content``. Raising is the point: the caller turns
            # that into a `# generate-error:` marker, which the campaign's MAX_GEN_ERR_RATE gate
            # can see. Returning "" would be scored as a deliberate silence instead.
            reasoning = str(msg.get("reasoning_content") or "")
            why = ("budget spent on reasoning" if reasoning.strip() else "no content at all")
            raise OpenAIError(
                f"empty answer ({why}, finish={ch.get('finish_reason')}, "
                f"reasoning={len(reasoning)} chars, usage={usage})"
            )
        return text, {"finish": ch.get("finish_reason"), "usage": usage, "model_version": version}

    def generate(self, prompt: str, parts: Optional[List[dict]] = None) -> str:
        """Return the model's text. Raises :class:`OpenAIError` after ``max_retries``.

        ``parts`` is an optional list of OpenAI content blocks (``{"type": "image_url", ...}``,
        ``{"type": "input_audio", ...}``, ``{"type": "text", ...}``) placed before ``prompt``. Only
        the media *hash* enters the cache key, so the JSONL cache stays small.
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
                det = u.get("completion_tokens_details") or {}
                with self._lock:
                    self.calls += 1
                    self.prompt_tokens += int(u.get("prompt_tokens") or 0)
                    self.output_tokens += int(u.get("total_tokens") or 0)
                    self.completion_tokens += int(u.get("completion_tokens") or 0)
                    self.reasoning_tokens += int(det.get("reasoning_tokens") or 0)
                    if meta.get("model_version"):
                        self.model_versions.add(meta["model_version"])
                    self._cache[k] = rec
                if self.cache_path:
                    append_jsonl(self.cache_path, rec)
                return text
            except (OpenAIError, urllib.error.URLError, urllib.error.HTTPError,
                    TimeoutError, json.JSONDecodeError, OSError) as e:
                last = type(e).__name__
                if isinstance(e, urllib.error.HTTPError):
                    last += f' HTTP {e.code}'
                if attempt < self.max_retries - 1:
                    # Jittered backoff: the gateway is shared and rate-limited, so a thundering
                    # herd of 16 workers retrying in lockstep just re-fails together.
                    time.sleep(min(30.0, 1.5 * (2 ** attempt)) * (0.6 + 0.8 * random.random()))
        raise OpenAIError(f"{self.model}: giving up after {self.max_retries} tries; last={last}")

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
            raise OpenAIError(f"{self.model}: answer is not JSON ({e}): {text[:300]!r}") from e

    def stats(self) -> dict:
        return {
            "model": self.model, "api_calls": self.calls, "cache_hits": self.cache_hits,
            "prompt_tokens": self.prompt_tokens, "total_tokens": self.output_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "model_versions": sorted(self.model_versions),
        }

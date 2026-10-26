"""Gemini causal-prefix adapter: 32 frames, bounded pixels, native audio and provider-default thinking."""
from __future__ import annotations

import base64
import hashlib
import io
import os
import subprocess
import threading
from typing import List, Optional

from ..clients.gemini import GeminiClient
from ..data.media import Frame
from .base import Capabilities
from .prompted_base import (
    PromptedProactiveModel,
    build_prompt,
    frames_to_pil_timed,
    pixel_budget,
    true_frame_fps,
)

_KEY_HINT = "Set the API credential environment variable documented in README.md."

# The ONLY temporal information Gemini is given, and deliberately the same information Qwen3-VL
# gets in its prompt: how fast the frames were sampled. Not when they were taken. See the module
# docstring on why an absolute clock would break the T5 footnote.
_FRAME_PREAMBLE = (
    "The {n} frames below are sampled from the stream up to the current moment, oldest first, at "
    "about {fps:.2f} frames per second."
)
_IMAGE_LABEL = "[An image attached by the user with the request below]"
_QUERY_AUDIO_LABEL = "[The user's spoken request, as audio]"
_TRACK_AUDIO_LABEL = "[The stream's own audio track, up to the current moment]"

# Below this the "prefix" is a rounding artefact of the probe grid (t_p == origin), not audio that
# was dropped: there is nothing to send, and ffmpeg would emit a zero-frame mp3.
_MIN_AUDIO_SEC = 0.05


class GeminiAPIModel(PromptedProactiveModel):
    """Shared recipe; the two concrete classes only pick an ``api_model`` id."""

    capabilities = Capabilities(
        # No "video": we upload individually budgeted frames, never a clip. Gemini's native video
        # input would resample on its own terms and bypass all three budgets above.
        modalities_in={"text", "image", "audio"},
        supports_streaming=True,
        native_streaming=False,
        native_fps=1.0,
        thread_safe_probe=True,
    )
    _consumes_audio = True
    _api_model = ""

    def __init__(
        self,
        api_model: str = "",
        max_pixels: int = 360 * 420,
        jpeg_quality: int = 85,
        max_output_tokens: int = 2048,
        audio_bitrate: str = "32k",
        cache_path: str = "",
        base_url: str = "",
        max_retries: int = 4,
        timeout: float = 240.0,
        thinking: Optional[dict] = None,
        media_resolution: str = "",
        **kw,
    ):
        super().__init__(**kw)
        self.api_model = api_model or self._api_model
        self.max_pixels = int(max_pixels) if max_pixels else None
        self.jpeg_quality = int(jpeg_quality)
        self.max_output_tokens = int(max_output_tokens)
        self.audio_bitrate = audio_bitrate
        self.cache_path = cache_path
        self.base_url = base_url
        self.max_retries = int(max_retries)
        self.timeout = float(timeout)
        self.thinking = thinking or None
        # ``max_pixels`` bounds what we *upload*; this bounds what Gemini 3 *charges and sees* per
        # frame (measured on this gateway: default 1098 tok/frame, MEDIUM 525, LOW 262). Empty =
        # provider default, i.e. the same "ship it as it comes out of the box" rule as ``thinking``.
        self.media_resolution = (media_resolution or "").strip()
        self._client: Optional[GeminiClient] = None
        self._client_lock = threading.Lock()
        # An optional on-disk mp3 cache for the cumulative track, keyed by waveform bytes. Both
        # ``va`` runs (Pro and Flash) walk the same (video, origin, t_p) grid, so the second one
        # re-encodes ~12 k identical prefixes otherwise. Deliberately an env var and not a
        # model_arg: a scratch path must not enter the run fingerprint.
        self._mp3_cache_dir = os.environ.get("OMNI_GEMINI_AUDIO_CACHE", "") or ""
        self._counts = {"frames": 0, "mp3_encodes": 0, "mp3_cache_hits": 0, "empty_audio": 0}

    # ---- plumbing --------------------------------------------------------------------
    def _ensure_model(self) -> None:
        if self.mock or self._client is not None:
            return
        with self._client_lock:
            if self._client is not None:  # another worker won the race
                return
            if not (os.environ.get("GEMINI_API_KEY") or "").strip():
                raise RuntimeError(_KEY_HINT)
            if self.cache_path:
                os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
            self._client = GeminiClient(
                self.api_model,
                base_url=self.base_url or None,
                cache_path=self.cache_path or None,
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
                # Free text, not JSON: the answer format is this benchmark's markup convention
                # (<call:NAME> / ABSTAIN), which a JSON mime type would wrap and break.
                json_mode=False,
                max_retries=self.max_retries,
                timeout=self.timeout,
                thinking=self.thinking,
                media_resolution=self.media_resolution or None,
            )

    @staticmethod
    def _inline(mime: str, blob: bytes) -> dict:
        return {"inlineData": {"mimeType": mime, "data": base64.b64encode(blob).decode("ascii")}}

    def _jpeg(self, img) -> bytes:
        buf = io.BytesIO()
        pixel_budget(img, self.max_pixels).save(buf, format="JPEG", quality=self.jpeg_quality)
        return buf.getvalue()

    def _ffmpeg_mp3(self, in_args: List[str], stdin: Optional[bytes] = None) -> bytes:
        """Run one ffmpeg ->mp3 conversion. Raises on failure; never returns empty.

        ``-nostdin`` only for the file-fed form; the pipe-fed form obviously needs stdin.
        """
        cmd = (
            ["ffmpeg", "-v", "error"]
            + ([] if stdin is not None else ["-nostdin"])
            + in_args
            + ["-ac", "1", "-b:a", self.audio_bitrate, "-f", "mp3", "pipe:1"]
        )
        out = subprocess.run(cmd, input=stdin, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if out.returncode != 0 or not out.stdout:
            raise RuntimeError(
                f"ffmpeg mp3 encode failed (rc={out.returncode}, {len(out.stdout)} bytes out): "
                f"{(out.stderr or b'').decode('utf-8', 'replace')[:300]}"
            )
        self._counts["mp3_encodes"] += 1
        return out.stdout

    def _waveform_mp3(self, wav) -> Optional[bytes]:
        """Cumulative float32 @16 kHz track -> mp3 bytes, or ``None`` when there is no audio yet.

        ``media.audio_chunk`` hands back raw f32le mono at 16 kHz (it pipes ffmpeg's ``-f f32le``
        straight into ``np.frombuffer``), so that is what goes back in.
        """
        try:
            import numpy as np
        except Exception:  # pragma: no cover - numpy is a hard dep of the media layer
            return None
        arr = np.asarray(wav, dtype="float32").reshape(-1)
        if arr.size < int(16000 * _MIN_AUDIO_SEC):
            self._counts["empty_audio"] += 1
            return None
        raw = arr.tobytes()
        cache_file = ""
        if self._mp3_cache_dir:
            digest = hashlib.sha256(raw).hexdigest()
            cache_file = os.path.join(self._mp3_cache_dir, digest[:2], digest + ".mp3")
            if os.path.exists(cache_file):
                self._counts["mp3_cache_hits"] += 1
                with open(cache_file, "rb") as fh:
                    return fh.read()
        mp3 = self._ffmpeg_mp3(["-f", "f32le", "-ar", "16000", "-ac", "1", "-i", "pipe:0"], raw)
        if cache_file:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            tmp = cache_file + f".{os.getpid()}.{threading.get_ident()}.tmp"
            with open(tmp, "wb") as fh:
                fh.write(mp3)
            os.replace(tmp, cache_file)  # atomic: concurrent workers may hash the same prefix
        return mp3

    def _file_mp3(self, path: str) -> bytes:
        """Spoken-query file -> mp3 bytes. Transcoded even when already mp3.

        The release's query audio is 31 mp3 + 16 m4a; normalising both means one mime type in the
        request and no per-container support question at the gateway.
        """
        return self._ffmpeg_mp3(["-i", path])

    # ---- one probe -------------------------------------------------------------------
    def _generate(
        self,
        query_text: str,
        audio_ctx: str,
        frames: List[Frame],
        query_image: Optional[str] = None,
        query_audio: Optional[str] = None,
        audio_waveform=None,
    ) -> str:
        pil, times = frames_to_pil_timed(frames)
        parts: List[dict] = []

        if query_image and os.path.exists(query_image):
            with open(query_image, "rb") as fh:
                blob = fh.read()
            ext = os.path.splitext(query_image)[1].lower().lstrip(".") or "jpeg"
            mime = {"jpg": "jpeg", "jpe": "jpeg"}.get(ext, ext)
            parts.append({"text": _IMAGE_LABEL})
            parts.append(self._inline(f"image/{mime}", blob))

        if pil:
            # The rate the frames were REALLY taken at, from their own timestamps: max_frames
            # clamping stretches the gap well past the nominal 1 fps on long prefixes (median
            # 1.73 s, p90 11.21 s), so the nominal number would tell a model watching a 168 s
            # prefix that it spans 32 s. See prompted_base.true_frame_fps.
            fps = true_frame_fps(len(pil), times, fallback=self.capabilities.native_fps or 1.0)
            parts.append({"text": _FRAME_PREAMBLE.format(n=len(pil), fps=fps)})
            for img in pil:
                parts.append(self._inline("image/jpeg", self._jpeg(img)))
            self._counts["frames"] += len(pil)

        if self.use_audio and query_audio and os.path.exists(query_audio):
            parts.append({"text": _QUERY_AUDIO_LABEL})
            parts.append(self._inline("audio/mpeg", self._file_mp3(query_audio)))

        if self.use_audio and audio_waveform is not None:
            mp3 = self._waveform_mp3(audio_waveform)
            if mp3:
                parts.append({"text": _TRACK_AUDIO_LABEL})
                parts.append(self._inline("audio/mpeg", mp3))

        # The client appends the prompt as the trailing part: evidence first, instruction last.
        return self._client.generate(build_prompt(query_text, audio_ctx), parts).strip()

    # ---- run accounting --------------------------------------------------------------
    def run_stats(self) -> dict:
        """Token/version accounting, written next to the predictions by ``run_inference``.

        ``model_versions`` is why this exists: a ``-preview`` id is a moving target and we cannot
        pin it the way ``revision:`` pins a hub checkpoint, so the served build is recorded instead.
        More than one entry means the run straddled a provider-side model swap.
        """
        stats = dict(self._counts)
        stats["api_model"] = self.api_model
        stats["max_pixels"] = self.max_pixels
        stats["max_output_tokens"] = self.max_output_tokens
        stats["media_resolution"] = self.media_resolution or "provider-default"
        if self._client is not None:
            stats["client"] = self._client.stats()
        return stats


class Gemini31ProModel(GeminiAPIModel):
    _display_name = "gemini-3.1-pro"
    _api_model = "gemini-3.1-pro-preview"


class Gemini35FlashModel(GeminiAPIModel):
    _display_name = "gemini-3.5-flash"
    _api_model = "gemini-3.5-flash"

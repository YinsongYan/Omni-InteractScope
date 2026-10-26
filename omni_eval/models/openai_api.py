"""Causal image/audio input adapter for OpenAI-compatible chat endpoints."""
from __future__ import annotations

import base64
import hashlib
import io
import os
import subprocess
import threading
from typing import List, Optional

from ..clients.openai_compat import OpenAIClient
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

# The ONLY temporal information the model is given, and deliberately the same information Qwen3-VL
# and the Gemini pair get: how fast the frames were sampled. Not when they were taken. See the
# module docstring on why an absolute clock would break the T5 footnote. Kept byte-identical to
# gemini_api._FRAME_PREAMBLE so the two hosted columns differ by the model and not by the wording.
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

# The two audio content-block spellings this gateway might accept. Not a guess we make at runtime:
# trying both and keeping whichever answered would make the ``va`` cell's wire format depend on the
# order requests happened to fail in, and ``model_args`` is inside the run fingerprint precisely so
# that a choice like this is fixed before the run rather than discovered during it.
_AUDIO_STYLES = ("audio_url", "input_audio")


class OpenAICompatModel(PromptedProactiveModel):
    """Shared recipe for any chat-completions endpoint; concrete classes pick an ``api_model``."""

    capabilities = Capabilities(
        # No "video": we upload individually budgeted frames, never a clip. A native video part
        # would be resampled on the provider's terms and bypass all three budgets above.
        modalities_in={"text", "image", "audio"},
        supports_streaming=True,
        native_streaming=False,
        native_fps=1.0,
        thread_safe_probe=True,
    )
    _consumes_audio = True
    _api_model = ""
    _display_name = "openai-compat"

    def __init__(
        self,
        api_model: str = "",
        api_base: str = "",
        api_key_env: str = "BENCH_API_KEY",
        max_pixels: int = 360 * 420,
        jpeg_quality: int = 85,
        max_output_tokens: int = 4096,
        audio_bitrate: str = "32k",
        audio_part_style: str = "audio_url",
        cache_path: str = "",
        max_retries: int = 4,
        timeout: float = 240.0,
        **kw,
    ):
        super().__init__(**kw)
        self.api_model = api_model or self._api_model
        # Empty uses OPENAI_BASE_URL or the official OpenAI endpoint. The model YAML
        # specifies the inference endpoint separately from the content judge.
        self.api_base = (api_base or "").strip()
        self.api_key_env = api_key_env or "BENCH_API_KEY"
        self.max_pixels = int(max_pixels) if max_pixels else None
        self.jpeg_quality = int(jpeg_quality)
        self.max_output_tokens = int(max_output_tokens)
        self.audio_bitrate = audio_bitrate
        if audio_part_style not in _AUDIO_STYLES:
            raise ValueError(
                f"audio_part_style={audio_part_style!r} is not one of {_AUDIO_STYLES}"
            )
        self.audio_part_style = audio_part_style
        self.cache_path = cache_path
        self.max_retries = int(max_retries)
        self.timeout = float(timeout)
        self._client: Optional[OpenAIClient] = None
        self._client_lock = threading.Lock()
        # An optional on-disk mp3 cache for the cumulative track, keyed by waveform bytes. Every
        # ``va`` run walks the same (video, origin, t_p) grid, so the Gemini round already encoded
        # 11,920 of these prefixes (2.8 GB); pointing at that directory turns 195.8 h of audio
        # re-encoding into file reads. Deliberately an env var and not a model_arg: a scratch path
        # must not enter the run fingerprint.
        self._mp3_cache_dir = (os.environ.get("OMNI_API_AUDIO_CACHE")
                               or os.environ.get("OMNI_GEMINI_AUDIO_CACHE") or "")
        self._counts = {
            "frames": 0, "mp3_encodes": 0, "mp3_cache_hits": 0, "empty_audio": 0,
            "query_images": 0, "audio_parts": 0,
        }

    # ---- plumbing --------------------------------------------------------------------
    def _ensure_model(self) -> None:
        if self.mock or self._client is not None:
            return
        with self._client_lock:
            if self._client is not None:  # another worker won the race
                return
            key = ((os.environ.get(self.api_key_env) or os.environ.get("OPENAI_API_KEY")
                    or "").strip())
            if not key:
                raise RuntimeError(_KEY_HINT.format(var=self.api_key_env))
            if self.cache_path:
                os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
            self._client = OpenAIClient(
                self.api_model,
                base_url=self.api_base or None,
                api_key=key,
                cache_path=self.cache_path or None,
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
                # Free text, not JSON: the answer format is this benchmark's markup convention
                # (<call:NAME> / ABSTAIN), which a JSON response_format would wrap and break.
                json_mode=False,
                max_retries=self.max_retries,
                timeout=self.timeout,
            )

    @staticmethod
    def _data_uri(mime: str, blob: bytes) -> str:
        return f"data:{mime};base64," + base64.b64encode(blob).decode("ascii")

    def _image_part(self, mime: str, blob: bytes) -> dict:
        return {"type": "image_url", "image_url": {"url": self._data_uri(mime, blob)}}

    def _audio_part(self, mp3: bytes) -> dict:
        """One mp3 as a content block, in whichever of the two spellings is configured."""
        self._counts["audio_parts"] += 1
        if self.audio_part_style == "input_audio":
            return {
                "type": "input_audio",
                "input_audio": {"data": base64.b64encode(mp3).decode("ascii"), "format": "mp3"},
            }
        return {"type": "audio_url", "audio_url": {"url": self._data_uri("audio/mpeg", mp3)}}

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

    def _cache_file(self, digest: str) -> str:
        """Where a cached prefix mp3 lives. Plain ``<digest>.mp3`` means the 32 kbps default.

        The Gemini round wrote 11,920 files under that name and every run so far has used 32k, so
        keeping the default unqualified is what makes those files reusable; any other bitrate gets
        its own name rather than silently colliding with them.
        """
        stem = digest if self.audio_bitrate == "32k" else f"{digest}.{self.audio_bitrate}"
        return os.path.join(self._mp3_cache_dir, digest[:2], stem + ".mp3")

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
            cache_file = self._cache_file(hashlib.sha256(raw).hexdigest())
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
            parts.append({"type": "text", "text": _IMAGE_LABEL})
            parts.append(self._image_part(f"image/{mime}", blob))
            self._counts["query_images"] += 1

        if pil:
            # The rate the frames were REALLY taken at, from their own timestamps: max_frames
            # clamping stretches the gap well past the nominal 1 fps on long prefixes (median
            # 1.73 s, p90 11.21 s), so the nominal number would tell a model watching a 168 s
            # prefix that it spans 32 s. See prompted_base.true_frame_fps.
            fps = true_frame_fps(len(pil), times, fallback=self.capabilities.native_fps or 1.0)
            parts.append({"type": "text", "text": _FRAME_PREAMBLE.format(n=len(pil), fps=fps)})
            # Bare frames in time order, nothing interleaved: a per-frame "t=12.0s" text part is
            # exactly the clock the T5 footnote says probing does not have.
            for img in pil:
                parts.append(self._image_part("image/jpeg", self._jpeg(img)))
            self._counts["frames"] += len(pil)

        if self.use_audio and query_audio and os.path.exists(query_audio):
            parts.append({"type": "text", "text": _QUERY_AUDIO_LABEL})
            parts.append(self._audio_part(self._file_mp3(query_audio)))

        if self.use_audio and audio_waveform is not None:
            mp3 = self._waveform_mp3(audio_waveform)
            if mp3:
                parts.append({"type": "text", "text": _TRACK_AUDIO_LABEL})
                parts.append(self._audio_part(mp3))

        # The client appends the prompt as the trailing block: evidence first, instruction last.
        return self._client.generate(build_prompt(query_text, audio_ctx), parts).strip()

    # ---- run accounting --------------------------------------------------------------
    def run_stats(self) -> dict:
        """Token/version accounting, written next to the predictions by ``run_inference``.

        ``model_versions`` is why this exists: a gateway's served build is a moving target and we
        cannot pin it the way ``revision:`` pins a hub checkpoint, so it is recorded instead. More
        than one entry means the run straddled a provider-side model swap.
        """
        stats = dict(self._counts)
        stats["api_model"] = self.api_model
        stats["max_pixels"] = self.max_pixels
        stats["max_output_tokens"] = self.max_output_tokens
        stats["audio_part_style"] = self.audio_part_style
        if self._client is not None:
            stats["client"] = self._client.stats()
        return stats

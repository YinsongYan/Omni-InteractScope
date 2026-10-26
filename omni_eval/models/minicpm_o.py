"""MiniCPM-o 4.5 (9B) adapter (omni: audio+vision, text-out).

Two paths, one checkpoint:

- **Online Rolling (default, native full-duplex)** — the real streaming mechanism, aligned with
  StreamArena's ``MiniCPMOUnifiedRunner``: ``model.as_duplex(generate_audio=False)`` then per-tick
  ``streaming_prefill`` (1 frame + 1 s audio) → ``streaming_generate`` (``is_listen`` / ``end_of_turn``).
  A 30-unit ("context" mode) sliding window bounds KV growth. The harness routes MiniCPM-o here
  because ``capabilities.native_streaming`` is True.
- **Offline Probing (comparison only, prompted approximation)** — inherited from
  ``PromptedProactiveModel`` (``model.chat(..., omni_mode=...)`` + ABSTAIN sentinel). Kept so the
  same checkpoint can be compared under the probing protocol; NOT the default routing.

Recipe (AutoModel + trust_remote_code) adapted from OmniPro. ``_patch_whisper_attention`` is
required for the pinned transformers 4.51 (MiniCPM-o's encoder calls ``WhisperAttention.forward``
with ``past_key_values=`` and expects a 2-tuple). Runs in its own isolated env
(scripts/install_minicpm_o.sh, pins transformers==4.51.0); ``OMNI_HF_MOCK=1`` skips torch/weights
(both paths then abstain, so the whole pipeline + tests run with no GPU).
"""
from __future__ import annotations

from typing import List, Optional

from ..data.media import Frame
from ..data.response_parser import infer_response_type, strip_reasoning
from .base import Capabilities, Emit, SessionSpec
from .prompted_base import (
    PromptedProactiveModel,
    build_prompt,
    frames_to_pil,
    tool_disclosure,
)

_INSTALL_HINT = (
    "MiniCPM-o requires transformers==4.51.0, minicpmo-utils, decord, librosa "
    "(see scripts/install_minicpm_o.sh) and the MiniCPM-o-4_5 weights. "
    "Set OMNI_HF_MOCK=1 to run without them."
)

_SR = 16000  # MiniCPM-o omni audio sample rate

# Full-duplex system prompt: unified online-interaction framing (proactive-safe). The standing
# request is appended at begin(); premature speaking is undesirable so we tell the model to hold
# silence until the moment is right (mirrors StreamArena's E-task instruction). Timing is left to
# the duplex listen/speak head — no ABSTAIN sentinel here, that is the offline path's mechanism.
#
# The markup + tool vocabulary IS disclosed (2026-09-03; it was not before). Without it the
# ``simultaneous_tool_calls_*`` items are unscoreable for this model by construction, not by
# ability: ``metrics/content.py`` routes them to ``tool_judge``, which needs set-F1 == 1.0 over
# ``<call:NAME>`` names, and a model never told the vocabulary cannot produce them. The duplex
# runtime has no ``tools=`` channel (``prepare()`` takes only prefix_system_prompt / ref_audio /
# prompt_wav_path), so the prompt is the only place this can go.
_STREAM_SYS = (
    "You are watching a live video stream (frames + audio) in real time. "
    "Respond only when a response is warranted by the user's standing request or by what you "
    "see/hear; if the moment has not arrived yet, stay completely silent and keep watching -- "
    "do not speak early or guess. When you respond, be concise and do not restate the request.\n"
    + tool_disclosure()
)


def _patch_whisper_attention() -> None:  # pragma: no cover - needs weights
    """Make transformers 4.51 WhisperAttention.forward accept ``past_key_values=`` and return a
    2-tuple, as MiniCPM-o's custom encoder layer expects (OmniPro compatibility shim)."""
    try:
        from transformers.models.whisper import modeling_whisper as mw  # type: ignore
    except Exception:
        return
    orig = mw.WhisperAttention.forward
    if getattr(orig, "_minicpmo_patched", False):
        return

    def forward(self, *args, past_key_values=None, **kwargs):
        if past_key_values is not None and "past_key_value" not in kwargs:
            kwargs["past_key_value"] = past_key_values
        out = orig(self, *args, **kwargs)
        if isinstance(out, tuple) and len(out) == 3:
            out = (out[0], out[1])
        return out

    forward._minicpmo_patched = True
    mw.WhisperAttention.forward = forward


class MiniCPMoModel(PromptedProactiveModel):
    capabilities = Capabilities(
        modalities_in={"text", "image", "video", "audio"},
        supports_streaming=True,
        native_streaming=True,   # -> routed to Online Rolling (native duplex below)
        native_fps=1.0,
    )
    _consumes_audio = True
    _display_name = "minicpm-o"

    def __init__(
        self,
        model_path: str = "openbmb/MiniCPM-o-4_5",
        max_new_tokens: int = 512,
        device: str = "cuda:0",
        # --- native full-duplex (Online Rolling) knobs ---
        stream_window_units: int = 30,     # 30 frames @1fps sliding window ("context" mode)
        stream_max_speak_tokens: int = 20,  # duplex per-chunk speak cap (== SA default)
        decode_mode: str = "greedy",       # greedy -> deterministic (aligns temp=0)
        stream_temperature: float = 0.1,   # only used when decode_mode="sampling"
        **kw,
    ):
        super().__init__(model_path=model_path, max_new_tokens=max_new_tokens, device=device, **kw)
        self.stream_window_units = stream_window_units
        self.stream_max_speak_tokens = stream_max_speak_tokens
        self.decode_mode = decode_mode
        self.stream_temperature = stream_temperature
        # duplex streaming state
        self._dup = None
        self._buf = ""
        self._buf_t: Optional[float] = None   # tick of the FIRST spoken chunk of the current turn
        self._audio_chunks = 0
        self._audio_padded_samples = 0

    def _ensure_model(self) -> None:  # pragma: no cover - needs weights
        if self._model is not None:
            return
        try:
            import torch  # type: ignore
            from transformers import AutoModel, AutoTokenizer  # type: ignore
        except Exception as e:
            raise NotImplementedError(_INSTALL_HINT) from e
        _patch_whisper_attention()
        # The duplex API below is the checkpoint's own trust_remote_code code, so pin the snapshot:
        # `code_revision` is what fixes the .py files, `revision` the weights/config. See the
        # moss-vl-realtime docstring for the incident this guards against.
        pin = {"revision": self.revision, "code_revision": self.revision} if self.revision else {}
        self._model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
            init_vision=True,
            init_audio=True,
            init_tts=True,
            **pin,
        )
        self._model = self._model.eval().to(self.device)
        self._tok = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True, **pin
        )

    # ============================================================================== #
    # Offline Probing (prompted approximation, comparison only) — via _generate below
    # ============================================================================== #
    @staticmethod
    def _split_audio(waveform, n: int):
        """Split a 1-D waveform into ``n`` roughly-equal segments (one per frame)."""
        try:
            import numpy as np
        except Exception:
            return []
        wav = np.asarray(waveform, dtype="float32").reshape(-1)
        if n <= 0 or wav.size == 0:
            return []
        bounds = np.linspace(0, wav.size, n + 1, dtype=int)
        return [wav[bounds[i]:bounds[i + 1]] for i in range(n)]

    def _generate(
        self,
        query_text: str,
        audio_ctx: str,
        frames: List[Frame],
        query_image: Optional[str] = None,
        query_audio: Optional[str] = None,
        audio_waveform=None,
    ) -> str:  # pragma: no cover - needs weights
        pil = frames_to_pil(frames)
        if query_image:
            try:
                from PIL import Image  # type: ignore

                pil = [Image.open(query_image).convert("RGB")] + pil
            except Exception:
                pass

        segs = (
            self._split_audio(audio_waveform, len(pil))
            if (self.use_audio and audio_waveform is not None and pil)
            else []
        )
        use_omni = bool(segs)

        content: list = []
        if use_omni:
            # Official omni interleave: frame_i then its matching audio segment.
            for i, img in enumerate(pil):
                content.append(img)
                if i < len(segs):
                    content.append(segs[i])
        else:
            content.extend(pil)
        content.append(build_prompt(query_text, audio_ctx))
        msgs = [{"role": "user", "content": content}]

        response = self._model.chat(
            msgs=msgs,
            **self._gen_kwargs(),
            use_image_id=False,
            max_slice_nums=1,
            use_tts_template=False,
            enable_thinking=False,
            generate_audio=False,
            omni_mode=use_omni,
            tokenizer=self._tok,
        )
        if isinstance(response, tuple):
            response = response[0]
        return (response or "").strip()

    # ============================================================================== #
    # Online Rolling (NATIVE full-duplex) — overrides PromptedProactiveModel's
    # sliding-window re-prompt with the real duplex streaming mechanism.
    # ============================================================================== #
    def _ensure_duplex(self):  # pragma: no cover - needs weights
        if self._dup is not None:
            return self._dup
        self._ensure_model()
        try:
            self._model.init_tts()
        except Exception:
            pass
        self._dup = self._model.as_duplex(generate_audio=False)
        # 30-unit ("context" mode) sliding window: evict oldest chunks, keep KV bounded and the
        # system prompt protected. Without it, "off" mode accumulates the whole span -> KV blowup.
        try:
            wc = self._dup.decoder._window_config
            wc.sliding_window_mode = "context"
            wc.context_max_units = self.stream_window_units
            self._dup.decoder._window_enabled = True
        except AttributeError:
            pass
        return self._dup

    @staticmethod
    def _frame_to_pil(frame: Frame):
        pil = frames_to_pil([frame])
        return pil[0] if pil else None

    def _step(self, pil, audio):  # pragma: no cover - needs weights
        # The pinned duplex code increments audio_chunk_idx for VISION units too.
        # Its audio encoder, however, needs an index over actual audio chunks:
        # otherwise a visual-only initial tick skips first-audio padding and the
        # first real audio prefill fails (16000 < 16480 samples). Keep the native
        # unit clock for generation, but use the audio clock during audio prefill.
        has_audio = audio is not None and len(audio) > 0
        unit_index = self._dup.audio_chunk_idx
        if has_audio:
            # The native interface consumes one second per tick. ffmpeg can
            # return a slightly short chunk at codec/track boundaries. Right-pad
            # that missing tail with silence; never fetch later scene audio.
            if len(audio) < _SR:
                import numpy as np
                missing = _SR - len(audio)
                audio = np.pad(audio, (0, missing))
                self._audio_padded_samples += missing
            self._dup.audio_chunk_idx = self._audio_chunks
        try:
            prefill = self._dup.streaming_prefill(
                audio_waveform=audio, frame_list=[pil], max_slice_nums=1, batch_vision_feed=False
            )
            if not prefill.get("success"):
                raise RuntimeError("MiniCPM duplex prefill failed: " + str(prefill.get("reason", "")))
            if has_audio:
                self._audio_chunks += 1
        finally:
            if has_audio:
                self._dup.audio_chunk_idx = unit_index + 1
        return self._dup.streaming_generate(
            max_new_speak_tokens_per_chunk=self.stream_max_speak_tokens,
            decode_mode=self.decode_mode,
            temperature=self.stream_temperature,
        )

    def begin(self, spec: SessionSpec) -> None:
        self._spec = spec
        self._buf = ""
        self._buf_t = None
        self._audio_chunks = 0
        self._audio_padded_samples = 0
        if self.mock:
            return
        dup = self._ensure_duplex()  # pragma: no cover - needs weights
        req = (spec.query_text or "").strip()  # pragma: no cover
        prefix = _STREAM_SYS + (f"\n\n[Standing request] {req}" if req else "")
        dup.prepare(prefix_system_prompt=prefix, ref_audio=None, prompt_wav_path=None)

    def observe(self, frame: Frame, t_sec: float, audio_chunk=None) -> Optional[Emit]:
        if self.mock:
            return None
        pil = self._frame_to_pil(frame)  # pragma: no cover - needs weights
        if pil is None:
            return None
        try:  # pragma: no cover
            res = self._step(pil, audio_chunk)
        except RuntimeError as e:  # early ticks with empty audio: torch.cat on empty tensor
            if "torch.cat" in str(e) or "non-empty" in str(e):
                return None
            raise
        if not res.get("is_listen", True):  # pragma: no cover
            chunk = res.get("text") or ""
            if chunk and self._buf_t is None:
                self._buf_t = t_sec   # the tick the model decided to speak — see below
            self._buf += chunk
        if res.get("end_of_turn") and self._buf.strip():  # pragma: no cover
            raw = self._buf.strip()
            # An utterance spans as many ticks as it takes to decode: ``_buf`` accumulates chunks
            # across ``observe`` calls and only ``end_of_turn`` closes it. Stamping the emit at
            # ``t_sec`` therefore records when the model FINISHED speaking, while the GT records the
            # onset -- a systematic late bias of one tick per extra chunk, against a +-2 s window.
            # So the emit is anchored at the first spoken chunk instead (same convention as
            # moss_vl_realtime's ``_buf_t``; MMDuet2 decides and speaks within one tick, so its
            # ``t_sec`` is already an onset).
            at = self._buf_t if self._buf_t is not None else t_sec
            self._buf = ""
            self._buf_t = None
            # MiniCPM-o 4.5 has a thinking mode; keep only the spoken answer in the scored text.
            turn = strip_reasoning(raw)
            if not turn:
                return None
            # Verbatim repeats are NOT dropped. The duplex model does repeat itself when frames stop
            # changing, but suppressing that here hides two different behaviours behind one silence:
            # a model that correctly re-narrates an unchanged scene at the next GT onset, and a model
            # stuck in a loop. Both belong in the score -- ``merge_emits`` collapses fires inside
            # ``temporal.merge_gap_sec`` and keeps the count as ``redundancies``, and anything
            # further out is a real fire that either matches a later GT or counts as a false alarm.
            return Emit(
                time_sec=at, response_type=infer_response_type(turn), reply_text=turn, raw=raw
            )
        return None

    def end(self) -> None:
        self._spec = None
        self._buf = ""
        self._buf_t = None

    def session_stats(self) -> dict:
        return {"adapter_version": "minicpm-duplex-audio-clock-v2",
                "audio_chunks_encoded": self._audio_chunks,
                "audio_padding_samples": self._audio_padded_samples}

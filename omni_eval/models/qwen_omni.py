"""Qwen omni adapters — Qwen2.5-Omni-7B and Qwen3-Omni-30B-A3B (audio+vision, text-out).

Recipe adapted from OmniPro (github.com/RuixiangZhao/OmniPro, ``models/qwen2_5_omni.py`` /
``models/qwen3_omni.py``): the omni ForConditionalGeneration model + its processor +
``qwen_omni_utils.process_mm_info``, with the talker (audio-generation head) disabled and
``return_audio=False`` so we only decode text.

Adaptations to this harness:
- Frames are pre-decoded, so the ``video`` content item is the sampled PIL frame list (not a path).
- The audio track is supplied as a separate ``audio`` waveform item (mono 16 kHz np array) rather
  than muxed into the video, so we pass ``use_audio_in_video=False`` consistently. Audio arrives
  from the Offline Probing driver (``ProbeContext.audio_waveform``) or the Online Rolling ticks.

Proactivity (ABSTAIN offline / sliding-window online) is inherited from ``PromptedProactiveModel``.
Runs in the "qwen" isolated env (scripts/install_qwen.sh); ``OMNI_HF_MOCK=1`` skips torch/weights.

VERIFIED on real weights (probe, ``--input va``): both models load, generate Chinese replies, and
keep the ``<ui:...>`` / ``<call:...>`` markup intact through ``infer_response_type``. On
audio-dependent interactions ``v`` fires 0 times while ``va`` fires and quotes what it heard, so the
native-audio branch is genuinely wired.

Qwen3-Omni logs ``Unused or unrecognized kwargs: images.`` once per processor call. It is BENIGN and
not ours: ``Qwen3OmniMoeProcessor.__call__`` itself calls ``self.video_processor(images=None, ...)``,
and the video processor's ``validate_kwargs`` warns about that vestigial ``images``. Frames still
arrive — the resulting batch carries both ``pixel_values_videos``/``video_grid_thw`` and
``input_features``. Do not "fix" it by dropping ``images=`` from our call.
"""
from __future__ import annotations

from typing import List, Optional

from ..data.media import Frame
from .base import Capabilities
from .prompted_base import (
    PromptedProactiveModel,
    build_prompt,
    frames_to_pil_timed,
    true_frame_fps,
)

_INSTALL_HINT = (
    "Qwen omni models require transformers with the Qwen*Omni* classes, qwen-omni-utils, flash-attn "
    "(see scripts/install_qwen.sh) and the weights. Set OMNI_HF_MOCK=1 to run without them."
)


class _QwenOmniBase(PromptedProactiveModel):
    """Shared omni recipe. Subclasses set ``_hf_model_cls`` / ``_hf_proc_cls`` / ``_dtype_kwarg``."""

    capabilities = Capabilities(
        modalities_in={"text", "image", "video", "audio"}, supports_streaming=True, native_fps=1.0
    )
    _consumes_audio = True

    # subclass fills these in _ensure_model via _load_classes()
    def _load_classes(self):  # pragma: no cover - needs weights
        raise NotImplementedError(_INSTALL_HINT)

    def __init__(
        self,
        model_path: str = "",
        fps: float = 1.0,
        max_pixels: int = 360 * 420,
        **kw,
    ):
        super().__init__(model_path=model_path, **kw)
        self.fps = fps
        self.max_pixels = max_pixels

    def _ensure_model(self) -> None:  # pragma: no cover - needs weights
        if self._model is not None:
            return
        model_cls, proc_cls, dtype_kwargs = self._load_classes()
        self._model = model_cls.from_pretrained(
            self.model_path,
            device_map=self.device,
            attn_implementation="flash_attention_2",
            **dtype_kwargs,
        )
        # Suppress the audio-generation head (we only score text).
        if hasattr(self._model, "disable_talker"):
            self._model.disable_talker()
        self._proc = proc_cls.from_pretrained(self.model_path)

    def _generate(
        self,
        query_text: str,
        audio_ctx: str,
        frames: List[Frame],
        query_image: Optional[str] = None,
        query_audio: Optional[str] = None,
        audio_waveform=None,
    ) -> str:  # pragma: no cover - needs weights
        from qwen_omni_utils import process_mm_info  # type: ignore

        pil, times = frames_to_pil_timed(frames)
        # The rate the frames were really taken at, not the nominal sample_fps: both Omni
        # processors turn this into TMRoPE's ``video_second_per_grid`` (= temporal_patch_size / fps),
        # i.e. it is how the model learns how much time the clip covers. Left at the nominal 1.0 it
        # is wrong twice over -- Qwen2.5-Omni's own default is fps=2.0 (video_second_per_grid 1.0,
        # so 2x off from what this harness feeds), and neither model can know that max_frames
        # clamping stretched the real gap to a 1.73 s median / 11.21 s p90. Unlike Qwen3-VL, no
        # frames are *lost* here: both video processors default to do_sample_frames=False, so this
        # is a timing correction only. See prompted_base.true_frame_fps.
        vid_fps = true_frame_fps(len(pil), times, fallback=self.fps)
        content: list = []
        if query_image:
            content.append({"type": "image", "image": query_image})
        if pil:
            content.append(
                {"type": "video", "video": pil, "max_pixels": self.max_pixels, "fps": vid_fps}
            )
        # Native audio: spoken query file and/or the video's audio-track waveform.
        if self.use_audio and query_audio:
            content.append({"type": "audio", "audio": query_audio})
        if self.use_audio and audio_waveform is not None:
            content.append({"type": "audio", "audio": audio_waveform})
        content.append({"type": "text", "text": build_prompt(query_text, audio_ctx)})
        conversation = [{"role": "user", "content": content}]

        text = self._proc.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        # Frames/audio are supplied separately (not muxed) -> use_audio_in_video=False.
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
        inputs = self._proc(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False,
            # Scalar, not a list: processing_qwen2_5_omni computes
            # ``temporal_patch_size / fps`` directly and a list raises TypeError.
            **({"fps": vid_fps} if pil else {}),
        )
        inputs = inputs.to(self._model.device).to(self._model.dtype)

        gen_out = self._model.generate(
            **inputs,
            **self._gen_kwargs(use_audio_in_video=False, return_audio=False),
        )
        text_ids = gen_out[0] if isinstance(gen_out, tuple) else gen_out
        out = self._proc.batch_decode(
            text_ids[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return (out[0] if out else "").strip()


class Qwen25OmniModel(_QwenOmniBase):
    _display_name = "qwen2.5-omni-7b"

    def __init__(self, model_path: str = "Qwen/Qwen2.5-Omni-7B", **kw):
        super().__init__(model_path=model_path, **kw)

    def _load_classes(self):  # pragma: no cover - needs weights
        try:
            from transformers import (  # type: ignore
                Qwen2_5OmniForConditionalGeneration,
                Qwen2_5OmniProcessor,
            )
        except Exception as e:
            raise NotImplementedError(_INSTALL_HINT) from e
        return Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor, {"torch_dtype": "auto"}


class Qwen3OmniModel(_QwenOmniBase):
    _display_name = "qwen3-omni-30b-a3b"

    def __init__(self, model_path: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct", **kw):
        super().__init__(model_path=model_path, **kw)

    def _load_classes(self):  # pragma: no cover - needs weights
        try:
            from transformers import (  # type: ignore
                Qwen3OmniMoeForConditionalGeneration,
                Qwen3OmniMoeProcessor,
            )
        except Exception as e:
            raise NotImplementedError(_INSTALL_HINT) from e
        return Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor, {"dtype": "auto"}

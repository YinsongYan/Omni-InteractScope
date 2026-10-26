"""Qwen3-VL-8B adapter (vision+text, NO audio).

Recipe adapted from OmniPro (github.com/RuixiangZhao/OmniPro, ``models/qwen3_vl.py``):
``AutoModelForImageTextToText`` + ``AutoProcessor`` + ``qwen_vl_utils.process_vision_info``.
Adaptation: this harness pre-decodes frames, so instead of handing the model a video *path* we
build a ``video`` message whose value is the sampled PIL frame list (no fps re-decode needed).

Proactivity (ABSTAIN offline / sliding-window online) comes from ``PromptedProactiveModel``.
Runs in the "qwen" isolated env (see scripts/install_qwen.sh); set ``OMNI_HF_MOCK=1`` to run the
code path without torch/weights.
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
    "Qwen3-VL requires transformers>=4.57, qwen-vl-utils, flash-attn (see scripts/install_qwen.sh) "
    "and the Qwen3-VL-8B weights. Set OMNI_HF_MOCK=1 to exercise the pipeline without them."
)


class Qwen3VLModel(PromptedProactiveModel):
    capabilities = Capabilities(
        modalities_in={"text", "image", "video"}, supports_streaming=True, native_fps=1.0
    )
    _consumes_audio = False
    _display_name = "qwen3-vl-8b"

    def __init__(
        self,
        model_path: str = "Qwen/Qwen3-VL-8B-Instruct",
        fps: float = 1.0,
        max_pixels: Optional[int] = None,
        **kw,
    ):
        super().__init__(model_path=model_path, **kw)
        self.fps = fps
        # Per-frame pixel budget, honoured by qwen_vl_utils.process_vision_info's smart_resize.
        # None = the processor's own default (size.longest_edge = 25165824), which is what this
        # adapter shipped with. It is exposed because it is the ONE knob that is not aligned across
        # the four Offline Probing models: both Qwen-Omni configs cap at 151200 (360*420) and
        # InternVL is effectively 448*448 = 200704 per tile, so leaving Qwen3-VL uncapped gives it
        # several times the visual detail of the models it is tabled against -- and makes it the
        # most expensive of the four (measured at max_frames=32: 3.13 s/probe vs 1.31 for
        # qwen2.5-omni-7b, on the same 95-probe smoke). Set it in configs/models/qwen3-vl-8b.yaml
        # to align the comparison; see docs/throughput.md 1.0.
        self.max_pixels = int(max_pixels) if max_pixels else None

    def _ensure_model(self) -> None:  # pragma: no cover - needs weights
        if self._model is not None:
            return
        try:
            import torch  # type: ignore
            from transformers import AutoModelForImageTextToText, AutoProcessor  # type: ignore
        except Exception as e:
            raise NotImplementedError(_INSTALL_HINT) from e
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=self.device,
        )
        self._proc = AutoProcessor.from_pretrained(self.model_path)

    def _video_kwargs(self, pil: list, times: list) -> dict:  # pragma: no cover - needs weights
        """Tell ``Qwen3VLVideoProcessor`` that ``pil`` *is* the frame list, and when each was taken.

        Without this the adapter silently loses most of its frames and mis-times the rest, because
        the processor defaults to ``do_sample_frames=True`` / ``fps=2`` / ``min_frames=4`` and, with
        no metadata to go on, assumes the frames it was handed were decoded at 24 fps. It then
        *re-samples* them: 16 frames -> 16/24*2 = 1.3 -> clamped up to ``min_frames=4`` -> a
        ``grid_thw`` of ``[2, 36, 64]``. Measured on Qwen3-VL-8B, 8, 16 and 32 frames all collapse to
        the same 1152 visual tokens; only 64 frames (5.3 -> 6) reaches ``[3, 36, 64]``. So the whole
        ``offline_probing.max_frames`` / ``model_args.max_frames`` budget was being spent decoding
        frames the model never saw, and raising it would have changed almost nothing.

        The mis-timing is the worse half. Qwen3-VL interleaves per-patch timestamps into the prompt
        ("Qwen3VL requires frame timestamps to construct prompts"), derived from the same phantom
        24 fps -- so a 60 s cumulative prefix was announced to the model as spanning **0.5 s**. On a
        benchmark whose whole question is *when* to speak, that is not a rounding error.

        Both are fixed by handing over the real numbers: ``do_sample_frames=False`` (these are
        already the sampled frames) plus a ``VideoMetadata`` whose fps comes from the frames' own
        ``time_sec``. ``process_vision_info`` is still used for the resize, so this changes nothing
        else about the recipe. Frames at 0,4,..,60 s now stamp as 2.0/10.0/18.0 s -- each the
        midpoint of its 2-frame temporal patch -- and all 16 frames survive as ``[8, 36, 64]``.

        Falls back to ``self.fps`` when the span is degenerate (one frame, or a synthetic clip whose
        frames share a timestamp), and to no kwargs at all on an older transformers without
        ``VideoMetadata`` -- the old lossy behaviour, but never a crash mid-campaign.
        """
        if not pil:
            return {}
        try:
            from transformers.video_utils import VideoMetadata  # type: ignore
        except Exception:
            return {}
        fps = true_frame_fps(len(pil), times, fallback=self.fps)
        md = VideoMetadata(
            total_num_frames=len(pil),
            fps=fps,
            duration=len(pil) / max(fps, 1e-6),
            video_backend="decord",
            frames_indices=list(range(len(pil))),
        )
        return {"video_metadata": [md], "do_sample_frames": False}

    def _generate(
        self,
        query_text: str,
        audio_ctx: str,
        frames: List[Frame],
        query_image: Optional[str] = None,
        query_audio: Optional[str] = None,
        audio_waveform=None,
    ) -> str:  # pragma: no cover - needs weights
        from qwen_vl_utils import process_vision_info  # type: ignore

        pil, times = frames_to_pil_timed(frames)
        content: list = []
        if query_image:
            content.append({"type": "image", "image": query_image})
        if pil:
            # Pre-sampled frames as a video sequence (adaptation of the path+fps recipe).
            vid: dict = {"type": "video", "video": pil}
            if self.max_pixels:
                vid["max_pixels"] = self.max_pixels
            content.append(vid)
        content.append({"type": "text", "text": build_prompt(query_text, audio_ctx)})
        messages = [{"role": "user", "content": content}]

        text = self._proc.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos = process_vision_info(messages)
        kw = self._video_kwargs(pil, times)
        inputs = self._proc(
            text=[text], images=images, videos=videos, return_tensors="pt", padding=True, **kw
        ).to(self._model.device)

        generated = self._model.generate(**inputs, **self._gen_kwargs())
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
        out = self._proc.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return (out[0] if out else "").strip()

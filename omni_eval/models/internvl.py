"""InternVL3.5-8B adapter (vision+text, NO audio).

Recipe adapted from OmniPro (github.com/RuixiangZhao/OmniPro, ``models/internvl3.py``): the
official InternVL ``trust_remote_code`` + ``model.chat()`` API with dynamic-tiling preprocessing.
Adaptation: OmniPro samples 16 segments from a video *path* with decord; this harness already hands
us sampled frames, so we run the same 448×448 ImageNet transform (1 tile/frame) over the given
frames and build the ``FrameN: <image>`` prefix that InternVL's chat template expects.

Proactivity (ABSTAIN offline / sliding-window online) is inherited from ``PromptedProactiveModel``.
Runs in the "internvl" isolated env (scripts/install_internvl.sh); ``OMNI_HF_MOCK=1`` skips
torch/weights.
"""
from __future__ import annotations

from typing import List, Optional

from ..data.media import Frame
from .base import Capabilities
from .prompted_base import PromptedProactiveModel, build_prompt, frames_to_pil

_INSTALL_HINT = (
    "InternVL3.5 requires transformers>=4.52.1, flash-attn, torchvision, decord "
    "(see scripts/install_internvl.sh) and the InternVL3_5-8B weights. "
    "Set OMNI_HF_MOCK=1 to run without them."
)

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class InternVL35Model(PromptedProactiveModel):
    capabilities = Capabilities(
        modalities_in={"text", "image", "video"}, supports_streaming=True, native_fps=1.0
    )
    _consumes_audio = False
    _display_name = "internvl3.5-8b"

    def __init__(
        self,
        model_path: str = "OpenGVLab/InternVL3_5-8B",
        input_size: int = 448,
        max_new_tokens: int = 512,
        **kw,
    ):
        super().__init__(model_path=model_path, max_new_tokens=max_new_tokens, **kw)
        self.input_size = input_size
        self._transform = None

    def _ensure_model(self) -> None:  # pragma: no cover - needs weights
        if self._model is not None:
            return
        try:
            import torch  # type: ignore
            import torchvision.transforms as T  # type: ignore
            from torchvision.transforms.functional import InterpolationMode  # type: ignore
            from transformers import AutoModel, AutoTokenizer  # type: ignore
        except Exception as e:
            raise NotImplementedError(_INSTALL_HINT) from e
        self._model = (
            AutoModel.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                use_flash_attn=True,
                trust_remote_code=True,
            )
            .eval()
            .cuda()
        )
        self._tok = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True, use_fast=False
        )
        self._transform = T.Compose(
            [
                T.Resize((self.input_size, self.input_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ]
        )

    def _generate(
        self,
        query_text: str,
        audio_ctx: str,
        frames: List[Frame],
        query_image: Optional[str] = None,
        query_audio: Optional[str] = None,
        audio_waveform=None,
    ) -> str:  # pragma: no cover - needs weights
        import torch  # type: ignore

        pil = frames_to_pil(frames)
        if query_image:
            try:
                from PIL import Image  # type: ignore

                pil = [Image.open(query_image).convert("RGB")] + pil
            except Exception:
                pass
        instruction = build_prompt(query_text, audio_ctx)
        if not pil:
            # No decodable frames: text-only chat (keeps mock/synthetic paths alive).
            response = self._model.chat(
                self._tok, None, instruction, self._gen_kwargs()
            )
            return (response or "").strip()

        tiles = [self._transform(img) for img in pil]  # 1 tile/frame (max_num=1)
        pixel_values = torch.stack(tiles).to(torch.bfloat16).cuda()
        num_patches_list = [1] * len(tiles)
        prefix = "".join(f"Frame{i + 1}: <image>\n" for i in range(len(num_patches_list)))
        question = prefix + instruction

        response = self._model.chat(
            self._tok,
            pixel_values,
            question,
            self._gen_kwargs(),
            num_patches_list=num_patches_list,
            history=None,
            return_history=False,
        )
        return (response or "").strip()

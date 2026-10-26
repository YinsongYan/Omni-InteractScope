"""MMDuet2 (3B) adapter — native proactive streaming.

Paper: "MMDuet2: Enhancing Proactive Interaction of Video MLLMs with Multi-Turn Reinforcement
Learning" (ICLR 2026). Repo: https://github.com/yellow-binary-tree/MMDuet2 — base model
Qwen2.5-VL-3B-Instruct, so **vision+text only, no audio** (cell ``va`` is not offered).

MMDuet2 is one of the few checkpoints in the matrix with a *trained* notion of when to speak: it
is asked at every frame and answers ``NO REPLY`` until the video warrants a response. That makes
it ``native_streaming=True`` -> routed to Online Rolling.

Mechanism (ported from upstream's ``ProactiveInferenceClient``):

- one user turn per tick, holding that tick's frame; the standing request rides on the first turn
- ``past_key_values`` is carried across ticks, so each frame is vision-encoded exactly once
- the ``NO REPLY`` sentinel maps to standby (this harness's ABSTAIN); anything else is an emit
- KV is dropped and the session re-seeded with a digest of past replies past
  ``max_context_tokens`` — a long clip at 1 fps otherwise outgrows the context

**Why the streaming tick does its own prefill instead of calling ``generate()``.** Stock
``Qwen2_5_VLForConditionalGeneration.prepare_inputs_for_generation`` (transformers 4.49,
``modeling_qwen2_5_vl.py``) contains::

    if cache_position[0] != 0:
        pixel_values = None
        pixel_values_videos = None

With a carried-over cache, ``cache_position[0]`` is the cache length, so from tick 2 onwards the
new frame's pixel values are dropped *silently*: the ``<|image_pad|>`` tokens keep their plain text
embedding, the model is effectively blindfolded, and the replies collapse into gibberish (measured
before this was fixed: HTML/JSON fragments and bare numbers like ``9.5`` from tick ~30 on). No
exception is raised — the shape check that would catch it only runs when ``pixel_values`` is not
``None``. So ``_phase1_timing`` builds the suffix embeddings itself (``visual()`` on the new frame,
``masked_scatter`` into ``embed_tokens``), computes explicit mrope ``position_ids`` via
``get_rope_index`` over the full id stream, and runs one forward + a greedy decode loop against the
cache. Token ids are tracked incrementally (``self._ids``) rather than re-rendered from the chat
template each tick, because anything that does not match the cache byte-for-byte desyncs positions
the same way, just less visibly.

Upstream also tracks ``video_time += num_frames * frame_interval`` itself; here the driver owns
the clock and hands us ``t_sec``, so we timestamp emits from it rather than re-deriving it.

**Two-phase inference** (``two_phase: true``, the default). Phase 1 asks MMDuet2 *when* to speak
under its own ``NO REPLY`` prompt; phase 2 then re-generates the reply single-turn under this
harness's shared proactive prompt, so the answer carries the same ``<call:tool>`` / ``<ui:...>``
markup contract as every other model in the table. Timing stays MMDuet2's — if phase 2 declines,
we emit phase 1's own text rather than dropping the tick. Set ``two_phase: false`` for the pure
upstream baseline (timing and content both from the ``NO REPLY`` prompt), which scores worse on
format-sensitive subtasks but is the unmodified model.

Setup: use the isolated env from ``scripts/install_mmduet2.sh``. ``OMNI_HF_MOCK=1`` runs the whole
pipeline with no torch/weights (every tick becomes standby).

The upstream patched ``qwen2_5_vl`` (``third_party/``) is NOT used, and no clone is needed. Its
reason to exist is DTD frame *dropping*, which we deliberately disable anyway (we measure the
model's own timing, unaided), and the only other thing it provided — a prefill that keeps the visual
buffer alongside a carried cache — is what ``_phase1_timing`` now does directly. The
``wangyueqian/MMDuet2`` checkpoint is 825 stock Qwen2.5-VL tensors with no auxiliary head and no
``auto_map``, so stock ``transformers`` loads exactly the same weights. Report in the paper's setup
notes that frame dropping is off.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from ..data.media import Frame
from ..data.response_parser import infer_response_type, strip_reasoning
from .base import Capabilities, Emit, SessionSpec
from .prompted_base import PromptedProactiveModel, build_prompt, frames_to_pil

_INSTALL_HINT = (
    "MMDuet2 needs transformers==4.49 with Qwen2_5_VLForConditionalGeneration, qwen-vl-utils and "
    "flash-attn>=2.3 (see scripts/install_mmduet2.sh) plus the wangyueqian/MMDuet2 weights. No "
    "third_party clone is required (see this module's docstring). "
    "Set OMNI_HF_MOCK=1 to exercise the pipeline without any of them."
)

_NO_REPLY = "NO REPLY"

# Upstream's official proactive prompt. Kept verbatim: the NO REPLY behaviour is RL-trained
# against this exact wording, so paraphrasing it silently changes what we are measuring.
_TIMING_SYSTEM = (
    "You are a helpful assistant. Your task is to answer questions based on continuously "
    "incoming video frames. Your responses should include information from the video since your "
    'last reply (if any). If the information in this segment of the video cannot answer the '
    'question, output "NO REPLY".'
)

# ChatML fragments, rendered by hand so the tokens we append to the cache are exactly the tokens we
# record in ``self._ids``. Equivalent to what apply_chat_template emits for Qwen2.5-VL.
_SYS_TURN = "<|im_start|>system\n{sys}<|im_end|>\n"
_TURN_CLOSE = "<|im_end|>\n"
_VISION_START, _VISION_END = "<|vision_start|>", "<|vision_end|>"


class MMDuet2Model(PromptedProactiveModel):
    capabilities = Capabilities(
        modalities_in={"text", "image", "video"},   # vision-only: no ``va`` cell
        supports_streaming=True,
        native_streaming=True,                      # -> routed to Online Rolling
        native_fps=1.0,
    )
    _consumes_audio = False
    _display_name = "mmduet2"

    def __init__(
        self,
        model_path: str = "wangyueqian/MMDuet2",
        two_phase: bool = True,
        max_new_tokens: int = 128,
        max_context_tokens: int = 100_000,
        attn_implementation: str = "flash_attention_2",
        device: str = "cuda:0",
        **kw,
    ):
        super().__init__(
            model_path=model_path, max_new_tokens=max_new_tokens, device=device, **kw
        )
        self.two_phase = two_phase
        self.max_context_tokens = max_context_tokens
        self.attn_implementation = attn_implementation
        self._process_vision_info = None
        self._eos: set = set()
        # streaming state (per sample; reset by begin/end)
        self._instruction = ""
        self._pending_text: Optional[str] = None
        # Text that must open the NEXT tick's suffix: the system turn on the first tick, then the
        # <|im_end|> that closes the assistant turn we just generated (the model's own EOS is never
        # fed back, so the separator has to enter the cache with the following tick).
        self._turn_prefix = ""
        self._past = None
        self._ids = None        # every token currently in the KV cache, (1, L)
        self._grid_thw = None   # one row per frame so far; get_rope_index needs them all
        self._n_tokens = 0
        self._said: List[Tuple[float, str]] = []

    # ---- loading ---------------------------------------------------------------------
    def _ensure_model(self) -> None:  # pragma: no cover - needs weights
        if self._model is not None:
            return
        try:
            import torch  # type: ignore
            from transformers import AutoProcessor  # type: ignore
            from transformers import Qwen2_5_VLForConditionalGeneration  # type: ignore
            from qwen_vl_utils import process_vision_info  # type: ignore
        except Exception as e:
            raise NotImplementedError(_INSTALL_HINT) from e

        self._model = (
            Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation=self.attn_implementation,
            )
            .eval()
            .to(self.device)
        )
        self._proc = AutoProcessor.from_pretrained(self.model_path)
        self._process_vision_info = process_vision_info
        tok = self._proc.tokenizer
        self._eos = {i for i in (tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")) if i is not None}

    # ---- Offline Probing (comparison only) + phase-2 content -------------------------
    def _generate(
        self,
        query_text: str,
        audio_ctx: str,
        frames: List[Frame],
        query_image: Optional[str] = None,
        query_audio: Optional[str] = None,
        audio_waveform=None,
    ) -> str:  # pragma: no cover - needs weights
        """One single-turn forward over ``frames`` under the shared proactive prompt.

        Used by ``probe()`` (Offline Probing, for the protocol-comparison row) and by phase 2 of
        the streaming path. Deliberately stateless: no KV reuse. Safe to interleave with a live
        streaming session because the streaming path passes explicit ``position_ids`` and keeps its
        own cache, so nothing this call leaves on the module (``rope_deltas``) can affect it.
        """
        pil = frames_to_pil(frames)
        content: list = []
        if query_image:
            content.append({"type": "image", "image": query_image})
        if pil:
            content.append({"type": "video", "video": pil, "fps": self.capabilities.native_fps})
        content.append({"type": "text", "text": build_prompt(query_text, audio_ctx)})
        messages = [{"role": "user", "content": content}]

        text = self._proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = self._process_vision_info(messages)
        inputs = self._proc(
            text=[text], images=images, videos=videos, return_tensors="pt", padding=True
        ).to(self.device)
        out = self._model.generate(**inputs, **self._gen_kwargs())
        trimmed = out[:, inputs.input_ids.size(1):]
        decoded = self._proc.batch_decode(trimmed, skip_special_tokens=True)
        return (decoded[0] if decoded else "").strip()

    # ---- Online Rolling (native) -----------------------------------------------------
    def begin(self, spec: SessionSpec) -> None:
        self._reset_session()
        self._spec = spec
        self._instruction = (spec.query_text or "").strip()
        # The standing request rides on the first tick's user turn, unprefixed: phase 1 must stay
        # under upstream's wording or the NO REPLY sentinel stops being reliable.
        self._pending_text = build_prompt(self._instruction, "", with_system=False)
        self._turn_prefix = _SYS_TURN.format(sys=_TIMING_SYSTEM)
        if not self.mock:
            self._ensure_model()  # pragma: no cover - needs weights

    def observe(self, frame: Frame, t_sec: float, audio_chunk=None) -> Optional[Emit]:
        if self.mock:
            return None
        img = self._frame_to_pil(frame)
        if img is None:
            return None
        speak, raw = self._phase1_timing(img)
        if not speak:
            return None
        # Timing is MMDuet2's call; phase 2 only reshapes the wording. If it declines, keep the
        # tick and fall back to phase 1's own text rather than silently losing a correct trigger.
        text = (self._phase2_content(img) if self.two_phase else "") or raw
        self._said.append((t_sec, text))
        return Emit(
            time_sec=t_sec, response_type=infer_response_type(text), reply_text=text, raw=raw
        )

    def end(self) -> None:
        self._reset_session()
        self._spec = None

    @staticmethod
    def should_speak(reply: str) -> bool:
        """``NO REPLY`` (upstream's standby sentinel) and empty output mean "keep watching".

        Matched on the full sentinel, not on ``"NO"``: real replies start with "No" often enough
        ("Now the water is boiling") that the short prefix would eat correct triggers.
        """
        t = (reply or "").strip().upper()
        return bool(t) and not t.startswith(_NO_REPLY)

    @staticmethod
    def _frame_to_pil(frame: Frame):
        pil = frames_to_pil([frame])
        return pil[0] if pil else None

    def _phase1_timing(self, img) -> Tuple[bool, str]:  # pragma: no cover - needs weights
        """Ask MMDuet2 whether *now* is the moment, reusing the KV cache across ticks.

        One tick = append this frame's user turn to the cache and greedily decode the answer. See
        the module docstring for why this is hand-rolled rather than a ``generate()`` call.
        """
        if self._n_tokens > self.max_context_tokens:
            self._segment_reset()

        pixel_values, grid = self._encode_frame(img)
        merge = self._proc.image_processor.merge_size ** 2
        n_pads = int(grid.prod().item()) // merge

        text = self._turn_prefix + "<|im_start|>user\n"
        text += _VISION_START + self._proc.image_token * n_pads + _VISION_END
        if self._pending_text:
            text += self._pending_text
            self._pending_text = None
        text += _TURN_CLOSE + "<|im_start|>assistant\n"
        # The assistant turn stays open: its closing <|im_end|> opens the next tick's suffix.
        self._turn_prefix = _TURN_CLOSE

        suffix = self._proc.tokenizer(
            text, add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(self.device)
        gen = self._advance(suffix, pixel_values, grid)
        # strip_reasoning is defensive here (Qwen2.5-VL-3B has no thinking mode), but should_speak
        # matches NO REPLY on a *prefix*, so a reasoning block ahead of it would fake a trigger.
        reply = (
            strip_reasoning(self._proc.tokenizer.decode(gen, skip_special_tokens=True))
            if gen
            else ""
        )
        return self.should_speak(reply), reply

    def _encode_frame(self, img):  # pragma: no cover - needs weights
        """Vision-preprocess one frame, via upstream's ``process_vision_info`` for parity."""
        turn = {"role": "user", "content": [{"type": "image", "image": img}]}
        images, _ = self._process_vision_info([turn])
        res = self._proc.image_processor(
            images=images or [img], videos=None, return_tensors="pt"
        )
        return res["pixel_values"], res["image_grid_thw"]

    def _advance(self, suffix_ids, pixel_values, grid) -> List[int]:  # pragma: no cover
        """Prefill ``suffix_ids`` against the carried cache, then greedy-decode the reply.

        Returns the generated token ids. Both the suffix and the tokens actually fed back during
        decoding are recorded in ``self._ids`` — the final (EOS) token never enters the cache, so it
        must not be recorded either.
        """
        import torch  # type: ignore

        model = self._model
        # get_rope_index needs every frame's grid, because mrope positions of the new tokens depend
        # on how much of the position space the earlier frames consumed.
        self._grid_thw = grid if self._grid_thw is None else torch.cat([self._grid_thw, grid], 0)
        full_ids = suffix_ids if self._ids is None else torch.cat([self._ids, suffix_ids], dim=1)
        n_suffix = int(suffix_ids.size(1))
        mask = torch.ones((1, int(full_ids.size(1))), dtype=torch.long, device=self.device)

        with torch.inference_mode():
            pos_full, _ = model.get_rope_index(
                full_ids, self._grid_thw.to(self.device), None, None, mask
            )
            pos = pos_full[:, :, -n_suffix:]

            embeds = model.model.embed_tokens(suffix_ids)
            img_embeds = model.visual(
                pixel_values.to(self.device, model.visual.dtype), grid_thw=grid.to(self.device)
            )
            img_mask = suffix_ids == model.config.image_token_id
            n_tok, n_feat = int(img_mask.sum()), int(img_embeds.size(0))
            if n_tok != n_feat:
                # Would be a silent blindfold otherwise — the exact failure this path exists to fix.
                raise RuntimeError(f"image tokens/features mismatch: {n_tok} vs {n_feat}")
            embeds = embeds.masked_scatter(
                img_mask.unsqueeze(-1).expand_as(embeds), img_embeds.to(embeds.dtype)
            )

            out = model(
                inputs_embeds=embeds,
                attention_mask=mask,
                position_ids=pos,
                past_key_values=self._past,
                use_cache=True,
            )
            self._past = out.past_key_values
            tok = int(out.logits[0, -1].argmax())

            gen: List[int] = []
            last = pos[:, :, -1:]
            while len(gen) < self.max_new_tokens and tok not in self._eos:
                gen.append(tok)
                last = last + 1
                mask = torch.cat(
                    [mask, torch.ones((1, 1), dtype=mask.dtype, device=mask.device)], dim=1
                )
                step = torch.tensor([[tok]], dtype=torch.long, device=self.device)
                out = model(
                    inputs_embeds=model.model.embed_tokens(step),
                    attention_mask=mask,
                    position_ids=last,
                    past_key_values=self._past,
                    use_cache=True,
                )
                self._past = out.past_key_values
                tok = int(out.logits[0, -1].argmax())

        if gen:
            full_ids = torch.cat(
                [full_ids, torch.tensor([gen], dtype=torch.long, device=self.device)], dim=1
            )
        self._ids = full_ids
        self._n_tokens = int(full_ids.size(1))
        return gen

    def _phase2_content(self, img) -> str:  # pragma: no cover - needs weights
        """Re-generate the reply single-turn under the shared prompt, for a comparable format."""
        out = strip_reasoning(self._run(self._instruction, "", [Frame(time_sec=0.0, image=img)]))
        return "" if self._is_abstain(out) else out

    # ---- state plumbing --------------------------------------------------------------
    def _segment_reset(self) -> None:  # pragma: no cover - needs weights
        """Drop the KV cache at the context limit and re-seed with a digest of what was said."""
        self._past = None
        self._ids = None
        self._grid_thw = None
        self._n_tokens = 0
        # Cache is empty again, so the next suffix has to re-open with the system turn.
        self._turn_prefix = _SYS_TURN.format(sys=_TIMING_SYSTEM)
        said = "\n".join(f"[{t:.0f}s] {txt}" for t, txt in self._said[-10:]) or "(none)"
        self._pending_text = (
            f"{build_prompt(self._instruction, '', with_system=False)}\n\n"
            f"Already said:\n{said}\n\nContinue watching from here."
        )

    def _reset_session(self) -> None:
        self._instruction = ""
        self._pending_text = None
        self._turn_prefix = ""
        self._past = None
        self._ids = None
        self._grid_thw = None
        self._n_tokens = 0
        self._said = []

"""LiveCC-7B-Instruct adapter — native streaming video commentary.

Paper: "LiveCC: Learning Video LLM with Streaming Speech Transcription at Scale" (CVPR 2025).
Checkpoint: ``chenjoya/LiveCC-7B-Instruct`` — a ``Qwen2VLForConditionalGeneration`` on
Qwen/Qwen2-VL-7B, so **vision+text only, no audio**: cell ``v`` is the only legal cell here
(``va`` has no pathway, and ``v+asr`` is not offered online at all — see
``docs/input_modality.md`` §3 and the ``for_online`` guard in ``protocols/base.prepare_inputs``).

Like MMDuet2 and MOSS it has a *trained* notion of when to stay quiet, so it is
``native_streaming=True`` -> routed to Online Rolling. Its standby signal is not a word but a
punctuation mark: upstream reads it off the tokenizer as
``streaming_eos_token_id = tokenizer(' ...').input_ids[-1]``, i.e. a tick the model has nothing to
say about comes back as ``' ...'``. :meth:`LiveCCModel.should_speak` is that predicate.

Mechanism (ported from the model card's ``live_cc`` recipe, which is the reference implementation):

- one user turn per tick, holding that tick's frame(s) plus a ``Time=<start>-<stop>s`` label
- the standing request rides on the *first* tick's turn, appended **after** the video part
  (upstream's ``state['query']`` guard: it is sent once, then never again)
- ``past_key_values`` + ``past_ids`` are carried across ticks, so each frame is vision-encoded once
- from tick 2 the chat template's system prefix is replaced by ``'<|im_end|>\\n'``
  (``system_prompt_offset``), because the system turn is already in the cache
- KV is dropped and the session re-seeded with a digest of past replies past
  ``max_context_tokens`` — the checkpoint's context is 32768 and this release's online scan has a
  500 s tail, so :meth:`_segment_reset` is load-bearing, not defensive

**The three mandatory upstream actions, and why each one matters.**

1. ``apply_liger_kernel_to_qwen2_vl()`` *before* ``from_pretrained``. The model card says
   "important. our model is trained with this. keep consistency" — the RMSNorm/SwiGLU kernels are
   numerically slightly different from stock, and the streaming standby signal is a single token
   decided on a logit margin.
2. ``AutoProcessor.from_pretrained(..., use_fast=False)``. The slow/fast image processors disagree
   on resize kernels; upstream trained and evaluated against the slow one.
3. ``model.prepare_inputs_for_generation = functools.partial(
   prepare_multiturn_multimodal_inputs_for_generation, model)``. **This is the upstream fix for the
   silent-blindfold bug documented at length in ``mmduet2.py``**: stock
   ``prepare_inputs_for_generation`` contains ``if cache_position[0] != 0: pixel_values_videos =
   None``, so with a carried cache every tick from the second onwards drops its frame's pixels
   *without raising* — the ``<|video_pad|>`` tokens keep their plain text embedding and the replies
   degenerate. Because the failure is silent, the patch is asserted at load time
   (:meth:`_ensure_model`) and its effect is re-checked every tick
   (:meth:`_assert_sees_frames`, which raises).

**Deviations from upstream, all four disclosed in docs/paper_design_decisions.md.**

- **1 frame/tick (1 fps) instead of upstream's 2 fps / 2-frame clips.** Forced:
  ``online_rolling.fps`` is inside ``run_fingerprint._CONFIG_KEYS``, so changing it invalidates the
  four completed online runs, and FAR is only comparable across offline/online *because* both grids
  are 1 s (see the ``n_probes`` comment in ``protocols/online_rolling.py``). This costs less than it
  looks: ``Qwen2VLImageProcessor`` pads the temporal dimension up to ``temporal_patch_size = 2``, so
  one frame and upstream's two frames both come out as ``grid_t = 1`` — **identical visual token
  cost per tick**. What is lost is intra-second motion detail, not context.
- **Greedy** (``temperature: 0.0`` -> ``do_sample=False``), where the card's demo uses
  ``do_sample=True``. Same reproducibility reason as every local Instruct checkpoint here; see
  ``prompted_base._gen_kwargs``.
- **``max_new_tokens: 128``** (upstream's streaming call passes none and inherits the
  generation config). A proactive reply that needs more than 128 tokens is not what this benchmark
  scores, and an unbounded commentary model at 57k ticks is a cost problem.
- **``max_pixels: 151200``** rather than the demo's ``384*28*28 = 301056``, so a LiveCC tick costs
  the same visual tokens as every other ``v``-cell run in the matrix: the same ceiling
  (151200/28² = 193) on the same 28-px lattice. Measured, not assumed — a 1920x1080 frame lands on
  504x280 = **180 tokens** (301056 would give 728x392 = 364). The card notes more tokens give
  "better performance but shorter working duration"; both halves of that trade are measured once in
  ``docs/adapter_verification.md`` §2.10.

**Two properties of this run that are not deviations but must be read into its numbers.**

- **It is the only run in the matrix whose prompt carries an absolute clock.** ``Time=12.0-13.0s``
  is LiveCC's trained streaming format and cannot be removed without changing what the checkpoint
  is. The ``n/a†`` footnote for ``time_awareness`` in ``scripts/make_tables.py`` rests on "no
  column's prompt contains a clock"; that stops being true for this one column, so its T5 number is
  measured under a different condition than the other ten.
- **It was trained on English only**, while 1244 targets in this release come from zh videos. The
  query text is sent exactly as the release holds it. Expect a high speak rate and a high FAR; the
  card itself warns about repetition collapse.

``two_phase`` defaults to **False**, the opposite of MMDuet2. LiveCC's own streaming output is the
object under measurement, it speaks often, and phase 2 would re-pay a 32-frame forward per emit —
plausibly doubling the run's cost. ``model_args`` enters the run fingerprint, so this is decided
before the run, not after.

:meth:`session_stats` is a small new contract read by ``protocols/online_rolling.py``: until it
existed, a generation failure inside an online tick was indistinguishable from a deliberate silence
and ``MAX_GEN_ERR_RATE`` in ``run_campaign.sh`` passed every online run unconditionally
(``docs/throughput.md`` §6.5). LiveCC is the adapter that most needs the gate: new venv, patched
``generate``, and a known silent-failure mode.

Setup: ``scripts/install_livecc.sh`` + ``source envs/livecc/env.sh``. ``OMNI_HF_MOCK=1`` runs the
whole driver with no torch/weights (every tick becomes standby).
"""
from __future__ import annotations

import re
from typing import List, Optional

from ..data.media import Frame
from ..data.response_parser import infer_response_type, strip_reasoning
from .base import GENERATE_ERROR_MARK, Capabilities, Emit, SessionSpec
from .prompted_base import (
    PromptedProactiveModel,
    build_prompt,
    frames_to_pil,
    frames_to_pil_timed,
)

_INSTALL_HINT = (
    "LiveCC needs transformers==4.51 with Qwen2VLForConditionalGeneration plus livecc-utils, "
    "liger_kernel and qwen-vl-utils (see scripts/install_livecc.sh) and the "
    "chenjoya/LiveCC-7B-Instruct weights. Set OMNI_HF_MOCK=1 to exercise the pipeline without "
    "any of them."
)

# The per-tick time label, upstream's exact format. ``stop`` is forward-looking from the last
# frame's own timestamp (``timestamps[-1] + frame_time_interval`` in the model card), so a tick
# holding the frame at t covers [t, t+step) -- it never needs clamping at the start of a session.
# This string is the one absolute clock anywhere in this benchmark's prompts; a reword must
# invalidate the run, which is why livecc.py is in run_fingerprint._ADAPTER_PROMPTS.
_TICK_LABEL = "Time={start:.1f}-{stop:.1f}s"

# Replaces the chat template's system prefix once the system turn is already in the KV cache.
_TURN_CLOSE = "<|im_end|>\n"

# Re-seed text after a context reset, same shape as MMDuet2's: the cache is gone but the model must
# not re-narrate what it already said.
_RESEED = "{standing}\n\nAlready said:\n{said}\n\nContinue watching from here."

# Standby. The checkpoint was trained to answer ' ...' when a tick warrants no comment, so what
# counts as silence is "nothing but dots and whitespace" -- this catches ' ...', '…', '', '... ...'
# and the repetition-collapse variant '.........'. Matched on the WHOLE string, never as a prefix:
# a real reply that merely *starts* with an ellipsis is still a reply.
_STANDBY_ONLY = re.compile(r"^[.…\s]*$")


class LiveCCModel(PromptedProactiveModel):
    capabilities = Capabilities(
        modalities_in={"text", "image", "video"},   # vision-only: only cell ``v`` exists
        supports_streaming=True,
        native_streaming=True,                      # -> routed to Online Rolling
        native_fps=1.0,                             # upstream is 2.0; driven at 1 frame/tick
    )
    _consumes_audio = False
    _display_name = "livecc-7b"

    def __init__(
        self,
        model_path: str = "chenjoya/LiveCC-7B-Instruct",
        two_phase: bool = False,
        max_new_tokens: int = 128,
        max_context_tokens: int = 28_000,
        max_pixels: Optional[int] = 360 * 420,
        repetition_penalty: float = 1.05,
        attn_implementation: str = "flash_attention_2",
        device: str = "cuda:0",
        **kw,
    ):
        super().__init__(
            model_path=model_path, max_new_tokens=max_new_tokens, device=device, **kw
        )
        self.two_phase = two_phase
        self.max_context_tokens = max_context_tokens
        self.max_pixels = int(max_pixels) if max_pixels else None
        # Upstream's own value. Left configurable because the model card names it as the first knob
        # to reach for when a commentary model collapses into a repeat pattern, but NOT tuned: a
        # decoding parameter chosen by looking at this benchmark's outputs would be fitting to it.
        self.repetition_penalty = repetition_penalty
        self.attn_implementation = attn_implementation

        self._process_vision_info = None
        self._system_prompt_offset = 0
        self._standby_id: Optional[int] = None   # logged only; no logits interference (see below)
        self._video_token_id: Optional[int] = None
        self._merge = 4

        # ---- per-session state (reset by begin/end) ----
        self._instruction = ""
        self._pending_text: Optional[str] = None
        self._step = 1.0
        self._past = None            # KV cache carried across ticks
        self._past_ids = None        # every token currently in that cache, (1, L)
        self._n_tokens = 0
        self._said: List[str] = []

        # ---- per-session counters, published by session_stats() ----
        # Reset in begin(), deliberately NOT in end(): the driver calls end() in a finally and reads
        # session_stats() afterwards, so the numbers have to survive teardown.
        self._n_generate_errors = 0
        self._generate_error_sample = ""
        self._n_emits = 0
        self._n_context_resets = 0
        self._n_error_resets = 0
        self._n_visual_ticks = 0

    # ---- loading ---------------------------------------------------------------------
    def _ensure_model(self) -> None:  # pragma: no cover - needs weights
        if self._model is not None:
            return
        try:
            import functools

            import torch  # type: ignore
            from liger_kernel.transformers import apply_liger_kernel_to_qwen2_vl  # type: ignore
            from livecc_utils import (  # type: ignore
                prepare_multiturn_multimodal_inputs_for_generation,
            )
            from qwen_vl_utils import process_vision_info  # type: ignore
            from transformers import AutoProcessor  # type: ignore
            from transformers import Qwen2VLForConditionalGeneration  # type: ignore
        except Exception as e:
            raise NotImplementedError(_INSTALL_HINT) from e

        # Must precede from_pretrained: it swaps the module classes the weights are then loaded into.
        apply_liger_kernel_to_qwen2_vl()
        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype="auto",             # upstream's choice; the config pins bfloat16
            device_map=self.device,
            attn_implementation=self.attn_implementation,
            revision=self.revision,
        ).eval()
        # max_pixels is forwarded to the image processor here as well as per-message, so the ceiling
        # holds even on a path that bypasses process_vision_info.
        self._proc = AutoProcessor.from_pretrained(
            self.model_path, use_fast=False, revision=self.revision, max_pixels=self.max_pixels
        )
        self._process_vision_info = process_vision_info

        ip = self._proc.image_processor
        if self.max_pixels:
            ip.max_pixels = self.max_pixels
            if isinstance(getattr(ip, "size", None), dict) and "max_pixels" in ip.size:
                ip.size["max_pixels"] = self.max_pixels
        self._merge = int(getattr(ip, "merge_size", 2)) ** 2

        tok = self._proc.tokenizer
        # Upstream's trained standby token. Recorded for the log and for should_speak's docstring to
        # point at; we do NOT install a logits processor, because the card's own streaming example
        # passes streaming_eos_base_threshold=0.0 / step=0 (neutral) and the thresholding processor
        # lives in the GitHub demo, not in the checkpoint or in livecc-utils.
        self._standby_id = tok(" ...").input_ids[-1]
        self._video_token_id = getattr(self._model.config, "video_token_id", None)
        if self._video_token_id is None:
            self._video_token_id = tok.convert_tokens_to_ids("<|video_pad|>")

        # The system-prefix offset, measured off the template exactly as upstream does it (no
        # add_generation_prompt), so tick 2's spliced text matches what is already in the cache.
        probe = {"role": "user", "content": [{"type": "text", "text": "livecc"}]}
        texts = self._proc.apply_chat_template([probe], tokenize=False)
        self._system_prompt_offset = texts.index("<|im_start|>user")

        self._model.prepare_inputs_for_generation = functools.partial(
            prepare_multiturn_multimodal_inputs_for_generation, self._model
        )
        # Assert the patch landed. Without it every tick after the first is silently blindfolded
        # (see the module docstring), and a wrong transformers version that renamed or inlined
        # prepare_inputs_for_generation would fail exactly that way -- quietly.
        patched = self._model.prepare_inputs_for_generation
        if getattr(patched, "func", None) is not prepare_multiturn_multimodal_inputs_for_generation:
            raise RuntimeError(
                "prepare_inputs_for_generation patch did not stick; multi-turn KV reuse would drop "
                "every frame after the first without raising. Check the transformers/livecc-utils "
                "versions from scripts/install_livecc.sh."
            )

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
        """One stateless single-turn forward over ``frames`` under the shared proactive prompt.

        Not on the measured path: this run is Online Rolling only, so this serves the ``two_phase``
        knob (off by default) and any future offline arm. Stateless by construction — it never
        touches ``self._past``/``self._past_ids``, so it is safe to interleave with a live session.
        """
        pil, _times = frames_to_pil_timed(frames)
        content: list = []
        if query_image:
            content.append({"type": "image", "image": query_image})
        if pil:
            vid: dict = {"type": "video", "video": pil}
            if self.max_pixels:
                vid["max_pixels"] = self.max_pixels
            content.append(vid)
        content.append({"type": "text", "text": build_prompt(query_text, audio_ctx)})
        messages = [{"role": "user", "content": content}]

        text = self._proc.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos = self._process_vision_info(messages)
        inputs = self._proc(
            text=[text], images=images, videos=videos, return_tensors="pt", padding=True
        ).to(self._model.device)
        out = self._model.generate(**inputs, **self._gen_kwargs())
        trimmed = out[:, inputs.input_ids.size(1):]
        decoded = self._proc.batch_decode(trimmed, skip_special_tokens=True)
        return (decoded[0] if decoded else "").strip()

    # ---- Online Rolling (native) -----------------------------------------------------
    def begin(self, spec: SessionSpec) -> None:
        self._release_session()
        self._reset_counters()
        self._spec = spec
        self._instruction = (spec.query_text or "").strip()
        # ``with_system=False, with_tools=True``: the timing prompt is the checkpoint's own trained
        # format, so it must not be given our ABSTAIN sentinel, but the reply is still scored against
        # our markup contract -- the same setting MOSS uses. See build_prompt's docstring.
        self._pending_text = build_prompt(
            self._instruction, "", with_system=False, with_tools=True
        )
        # The label's width comes from the driver's own grid, not from a constant, so it can never
        # claim a span the frames do not cover.
        self._step = 1.0 / spec.fps if getattr(spec, "fps", 0) else 1.0
        if not self.mock:
            self._ensure_model()  # pragma: no cover - needs weights

    def observe(self, frame: Frame, t_sec: float, audio_chunk=None) -> Optional[Emit]:
        if self.mock:
            return None
        img = self._frame_to_pil(frame)
        if img is None:
            return None
        try:
            raw = self._tick(img, t_sec)  # pragma: no cover - needs weights
        except Exception as e:
            # Counted, not raised, and not silently turned into standby: `n_generate_errors` reaches
            # the record via session_stats() and so reaches MAX_GEN_ERR_RATE. The cache is dropped
            # because a failure mid-generate can leave it desynced from ``_past_ids``, and every
            # later tick would then fail for a reason that has nothing to do with the model.
            self._n_generate_errors += 1
            if not self._generate_error_sample:
                self._generate_error_sample = (
                    f"{GENERATE_ERROR_MARK} {type(e).__name__}: {e}"[:300]
                )
            self._segment_reset(error=True)
            return None
        text = self._visible(raw)
        if not self.should_speak(text):
            return None
        self._said.append(f"[{t_sec:.0f}s] {text}")
        self._n_emits += 1
        return Emit(
            time_sec=t_sec, response_type=infer_response_type(text), reply_text=text, raw=raw
        )

    def end(self) -> None:
        # Frees the KV cache (a 28k-token cache is GPU-resident) but leaves the counters alone --
        # online_rolling.run calls this in a ``finally`` and reads session_stats() next.
        self._release_session()
        self._spec = None

    def session_stats(self) -> dict:
        """Per-session health counters, merged into the record's ``meta`` by the online driver.

        ``setdefault`` there means driver keys always win, so this can never overwrite ``n_ticks`` /
        ``n_probes``. ``n_emits`` duplicates ``len(rec["emits"])`` on purpose: if the two ever
        disagree, emits are being lost between the adapter and the record.
        """
        stats = {
            "n_generate_errors": self._n_generate_errors,
            "n_emits": self._n_emits,
            "n_context_resets": self._n_context_resets,
            "n_error_resets": self._n_error_resets,
            "n_visual_ticks": self._n_visual_ticks,
        }
        if self._generate_error_sample:
            stats["generate_error_sample"] = self._generate_error_sample
        return stats

    @staticmethod
    def should_speak(reply: str) -> bool:
        """False when the tick came back as LiveCC's trained standby answer.

        The signal is ``' ...'`` (upstream's ``streaming_eos_token_id``), so anything that reduces to
        dots and whitespace is silence — including ``''``, ``'…'``, ``'... ...'`` and the
        repetition-collapse ``'.....'``. The match is anchored over the whole string rather than a
        prefix, which is the lesson from MMDuet2's ``NO REPLY``: a prefix test eats real replies.
        A reply that merely *ends* in a full stop is therefore still a reply.
        """
        return not bool(_STANDBY_ONLY.match(strip_reasoning(reply or "")))

    @staticmethod
    def _frame_to_pil(frame: Frame):
        pil = frames_to_pil([frame])
        return pil[0] if pil else None

    def _tick(self, img, t_sec: float) -> str:  # pragma: no cover - needs weights
        """One streaming turn: append this frame against the carried cache and decode the answer."""
        import torch  # type: ignore

        if self._n_tokens > self.max_context_tokens:
            self._segment_reset()

        # Upstream's content order, which is load-bearing: the time label, then the clip, then --
        # only on the turn that introduces it -- the standing request.
        content: list = [
            {
                "type": "text",
                "text": _TICK_LABEL.format(start=t_sec, stop=t_sec + self._step),
            }
        ]
        vid: dict = {"type": "video", "video": [img]}
        if self.max_pixels:
            vid["max_pixels"] = self.max_pixels
        content.append(vid)
        if self._pending_text:
            content.append({"type": "text", "text": self._pending_text})
            self._pending_text = None
        message = {"role": "user", "content": content}

        texts = self._proc.apply_chat_template(
            [message], tokenize=False, add_generation_prompt=True
        )
        if self._past_ids is not None:
            # The system turn is already in the cache; re-sending it would duplicate it and desync
            # every position after it.
            texts = _TURN_CLOSE + texts[self._system_prompt_offset:]

        # process_vision_info applies max_pixels on the 28-px lattice and pads the frame count up to
        # FRAME_FACTOR, which is what upstream's get_smart_resized_clip does -- the same arithmetic
        # every other Qwen adapter in this repo relies on rather than re-implementing smart_resize.
        _images, videos = self._process_vision_info([message])
        inputs = self._proc(
            text=texts,
            images=None,
            videos=videos,
            return_tensors="pt",
            # No attention mask on purpose (upstream does the same): we splice ``past_ids`` in below,
            # and a mask sized for the new tokens only would silently mask out the whole cache.
            return_attention_mask=False,
        )
        self._assert_sees_frames(inputs)
        inputs = inputs.to(self._model.device)
        if self._past_ids is not None:
            inputs["input_ids"] = torch.cat([self._past_ids, inputs.input_ids], dim=1)

        n_prompt = int(inputs["input_ids"].size(1))
        with torch.inference_mode():
            outputs = self._model.generate(
                **inputs,
                past_key_values=self._past,
                return_dict_in_generate=True,
                repetition_penalty=self.repetition_penalty,
                **self._gen_kwargs(),
            )
        self._past = outputs.past_key_values
        # ``[:, :-1]``, upstream's slice: the final (EOS) token is never in the cache, so recording
        # it would offset every subsequent position by one.
        self._past_ids = outputs.sequences[:, :-1]
        self._n_tokens = int(self._past_ids.size(1))
        self._n_visual_ticks += 1
        return self._proc.decode(outputs.sequences[0, n_prompt:], skip_special_tokens=True)

    def _assert_sees_frames(self, inputs) -> None:  # pragma: no cover - needs weights
        """Raise unless this tick's frame really reached the model as pixels.

        Two independent checks, both on the tick's OWN ids (before ``past_ids`` is spliced in):
        the video tensor is present and non-empty, and the number of ``<|video_pad|>`` placeholders
        in the text matches the number of visual tokens the grid will produce. Either mismatch is
        the silent-blindfold failure this adapter's patch exists to prevent, so it must be an
        exception -- the alternative is a run of fluent text that never saw the video.
        """
        pv = inputs.get("pixel_values_videos", None)
        grid = inputs.get("video_grid_thw", None)
        if pv is None or grid is None or int(getattr(pv, "numel", lambda: 0)()) == 0:
            raise RuntimeError(
                "tick carries no pixel_values_videos: the frame would be dropped silently"
            )
        n_pads = int((inputs["input_ids"] == self._video_token_id).sum())
        expected = int(grid.prod().item()) // self._merge
        if n_pads != expected:
            raise RuntimeError(
                f"video pad tokens/features mismatch: {n_pads} vs {expected}. "
                "Would be a silent blindfold otherwise."
            )

    # ---- state plumbing --------------------------------------------------------------
    def _segment_reset(self, error: bool = False) -> None:
        """Drop the KV cache and re-seed with a digest of what was already said.

        Reached two ways: at ``max_context_tokens`` (the expected path — a 32768-token context at
        ~200 tokens/tick covers roughly 150 s, and this release's online scans run longer), and
        after a failed tick, where the cache may be desynced from ``_past_ids``.
        """
        self._past = None
        self._past_ids = None
        self._n_tokens = 0
        if error:
            self._n_error_resets += 1
        else:
            self._n_context_resets += 1
        said = "\n".join(self._said[-10:]) or "(none)"
        self._pending_text = _RESEED.format(
            standing=build_prompt(self._instruction, "", with_system=False, with_tools=True),
            said=said,
        )

    def _release_session(self) -> None:
        self._instruction = ""
        self._pending_text = None
        self._step = 1.0
        self._past = None
        self._past_ids = None
        self._n_tokens = 0
        self._said = []

    def _reset_counters(self) -> None:
        self._n_generate_errors = 0
        self._generate_error_sample = ""
        self._n_emits = 0
        self._n_context_resets = 0
        self._n_error_resets = 0
        self._n_visual_ticks = 0

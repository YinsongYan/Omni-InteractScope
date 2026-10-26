"""MOSS-VL-Realtime adapter — native proactive streaming.

MOSS-VL-Realtime (OpenMOSS, HF ``OpenMOSS-Team/MOSS-VL-Realtime``, 11B) is a **vision+text,
no-audio, text-out** streaming model: cross-attention VLM, 256K context, timestamp-aware frames,
and a *trained* ``<|silence|>`` idle token that is exactly this benchmark's notion of "keep
watching". That makes it ``native_streaming=True`` -> routed to Online Rolling, and vision-only ->
the ``v`` cell only, never ``va``.

Isolated env: scripts/install_moss_vl.sh. ``OMNI_HF_MOCK=1`` runs the pipeline as a pure standby.

The session API below is READ OFF the checkpoint's ``trust_remote_code`` module
(``modeling_moss_vl.py``: ``MossVLForConditionalGeneration.create_realtime_session`` ->
``MossVLRealtimeSession``), not off the model card::

    sess = model.create_realtime_session(proc, initial_prompt=..., system_prompt=...,
                                         frame_queue_size=256, max_tokens_per_turn=12,
                                         max_new_tokens=..., do_sample=False)
    sess.start()
    sess.push_frame(pil, timestamp=t)      # timestamps must be NON-DECREASING within a session
    chunk = sess.poll_output(timeout=...)  # Optional[str], INCREMENTAL text piece (or None)
    sess.close()

Three properties of that API drive the design here, and getting any of them wrong silently ruins
the numbers. All three were measured against the real session
(``scripts/debug_moss_chunks.py`` prints the chunk-by-chunk trace):

1. **Generation is asynchronous.** ``start()`` spawns a worker thread; ``push_frame`` only enqueues.
   Polling with ``timeout=0`` right after a push therefore almost always returns ``None``, which
   would make the model look permanently silent. We poll with a real timeout instead.
2. **``poll_output`` yields ONE CHUNK PER TOKEN, and an utterance spans many ticks.** Upstream's
   ``_real_time_sample`` queues each decoded token and paces itself to ``max_tokens_per_turn``
   tokens **per second** (``wait = 1.0 / max_tokens_per_turn - cost``) — the name says "per turn"
   but it is a rate, and it is the only thing that knob does. Measured: ~0.08–0.13 s between
   fragments inside a turn, and a 24-token reply spread over 4 consecutive 1 fps ticks::

       t=22.8  '目前' '只' '看到' '一种' '商品' '，' '是'
       t=23.8  'D' 'ove' '多' '芬' '密集' '修' '护'
       t=24.8  '氨基酸' '发' '膜' '。' '其他' '商品' '还没有'
       t=25.8  '进入' '视野' '。' '<|silence|>'

3. **``<|silence|>`` is the ONLY end-of-turn signal.** It arrives as its own chunk
   (``output_text_queue.put("<|silence|>")``) and idle ticks produce exactly one of them. An absence
   of chunks means nothing: there is a ~0.5 s gap at every tick boundary (the push + one forward),
   which is indistinguishable from "stopped talking". An earlier version flushed after N empty
   polls and therefore cut every reply at its first tick boundary — the run emitted the single word
   "目前" instead of the sentence above. ``<|response|>`` is NOT a start marker either: it recurs
   mid-utterance whenever new frames are drained.

So fragments accumulate ACROSS ticks, and an emit is cut at a **sentence end** (``。！？!?``) or at
``<|silence|>``, whichever comes first.

Sentence ends matter because ``<|silence|>`` can be far away: measured, one silence-bounded turn ran
for ~90 s and six sentences ("现在能看到两种商品…" → "目前一共分装了五种商品…", the count climbing as
new frames arrived), eventually degenerating into repetition. Flushing only on silence would stamp
all of it — including a sentence spoken at t≈130 s — at the tick where the turn began (t=42.8 s),
which is a temporal metric measuring the wrong thing. One sentence = one proactive response,
stamped when that sentence started, is both closer to what the user would hear and consistent with
the per-tick emits of every other adapter in the table.

The emit is timestamped at ``self._buf_t`` — the tick at which the FIRST fragment of the utterance
appeared, i.e. when the model decided to speak — not at the tick where the last fragment landed.
The temporal metric scores *when* the model spoke, so the decision moment is the honest anchor.

Consecutive identical sentences are NOT de-duped. The model does repeat itself verbatim when frames
stop changing, and an earlier version dropped those repeats — but suppressing them in the adapter
hides two different behaviours behind one silence: a model that correctly re-narrates an unchanged
scene at the next GT onset, and a model stuck in a loop. Both belong in the score. Redundancy is
handled downstream, where it is measurable: ``merge_emits`` collapses fires inside
``temporal.merge_gap_sec`` and reports the collapsed count as ``redundancies``, and a repeat further
out than that is a real fire that either matches a later GT or counts as a false alarm.

Because the driver stops ticking shortly after the last GT time, an utterance that starts near the
end of the window would otherwise be dropped with the buffer — and that is exactly where the GT
moment usually is. So the final tick drains for up to ``final_budget_sec`` and flushes whatever it
has, stamped at its start. A partial trailing reply is still evidence the model spoke on time.

Control tokens are stripped before scoring. Keep sample_fps=1 / playback_speed=1 / do_sample=False
to reproduce paper conditions.

**Frames must be pixel-budgeted (``max_pixels``), and that is not optional here.** The checkpoint's
``preprocessor_config.json`` leaves resolution effectively uncapped (``size.longest_edge`` =
16777216), so a raw 720x1280 benchmark frame becomes ~900 vision tokens against the ~148 the three
Qwen adapters get from their shared ``max_pixels: 151200`` -- a 6x modality advantage that would
make any score gap unattributable. It also does not merely cost memory, it *runs out* of it: this
is a cross-attention VLM whose vision KV accumulates over every pushed frame, and upstream routes
cross-attention through SDPA even when ``flash_attention_2`` is requested
(``modeling_moss_vl.py``: "如果是flash attention，走sdpa_attention_forward"). SDPA can only use its
native GQA path when ``attention_mask is None``, and the frame-causal ``cross_attention_mask`` is
never None, so ``repeat_kv`` materialises the KV expanded 4x (32 heads / 8 kv heads) on every
layer. The 2026-09-02 campaign died exactly there: all 8 shards hit ``torch.OutOfMemoryError`` on a
single 5.03 GiB ``repeat_kv`` reshape, each on its first item past ~235 s of scan span, after
completing 597/1200 between them. Budgeting the frames shrinks that tensor by the same 6x and
removes the cliff; it is the aligned setting, not a workaround.

**``revision`` must be pinned.** The session semantics documented above live in the checkpoint's
``trust_remote_code`` modules, i.e. in a *mutable* HF repo -- not in this codebase and not in
transformers. Between the 2026-09-02 campaign (``06b0676``) and the 2026-09-03 smoke, ``main`` moved
to ``25e81cb`` and HF silently re-downloaded ``modeling_moss_vl.py`` (27 lines) and
``processing_moss_vl.py`` (7): among them a mid-string U+FFFD drop in the very queue this adapter
reads, which changes reply *text*. An unpinned revision therefore makes two runs of "the same model"
incomparable with nothing in the fingerprint to show it, so the revision rides in ``model_args``
(where the fingerprint does see it) rather than being left to whatever ``main`` points at today.
"""
from __future__ import annotations

import os
import re
from typing import List, Optional

from ..data.media import Frame
from ..data.response_parser import infer_response_type, strip_reasoning
from .base import Capabilities, Emit, ProbeContext, ProbeResult, SessionSpec, StreamingModel
from .prompted_base import build_prompt, pixel_budget

_INSTALL_HINT = (
    "MOSS-VL-Realtime requires its isolated env + weights (see scripts/install_moss_vl.sh) and "
    "trust_remote_code=True. OMNI_HF_MOCK=1 runs this adapter as a standby instead."
)

_SILENCE = "<|silence|>"

_CONTROL_TOKENS = re.compile(
    r"<\|silence\|>|<\|round_start\|>|<\|round_end\|>|<\|response\|>|<\|assistant\|>"
    r"|<\|im_start\|>|<\|im_end\|>|<go>"
)

# Sentence end = utterance boundary. Chinese full stop and both question/exclamation forms; the
# ASCII "." is deliberately excluded, since it also appears in decimals ("3.5") and would cut a
# reply mid-number.
_SENTENCE_END = "。！？!?…"

# patch_size 16 * merge_size 2 (preprocessor_config.json): one vision token covers a 32x32 block,
# and both sides must land on that grid or the processor pads and the token count stops matching
# the budget.
_PATCH_GRID = 32


def strip_control_tokens(text: str) -> str:
    """Remove session control tokens *and* any thinking block, leaving the spoken answer."""
    return strip_reasoning(_CONTROL_TOKENS.sub("", text or ""))


class MossVLRealtimeModel(StreamingModel):
    capabilities = Capabilities(
        # Vision-only (no audio encoder) -> cell ``v`` only, never ``va``.
        modalities_in={"text", "image", "video"},
        supports_streaming=True,
        # A real realtime session with a trained <|silence|> idle token -> must be ROUTED to
        # Online Rolling by --protocol auto.
        native_streaming=True,
        native_fps=1.0,
    )

    def __init__(
        self,
        model_path: str = "OpenMOSS-Team/MOSS-VL-Realtime",
        max_tokens_per_turn: int = 12,
        max_new_tokens: int = 512,
        frame_queue_size: int = 256,
        poll_timeout_sec: float = 0.25,
        poll_budget_sec: float = 1.0,
        final_budget_sec: float = 6.0,
        max_pixels: Optional[int] = None,
        revision: Optional[str] = None,
        attn_implementation: str = "flash_attention_2",
        **_,
    ):
        self.model_path = model_path
        # Pin the trust_remote_code snapshot: the session API is upstream code, not ours (see the
        # module docstring). None means "whatever main points at today", which is not reproducible.
        self.revision = revision or None
        # Per-frame pixel budget, shared with the Qwen adapters (see the module docstring for why
        # leaving it unset is both a comparability defect and an OOM).
        self.max_pixels = int(max_pixels) if max_pixels else None
        # Upstream misnomer: this is a tokens-per-SECOND pace, not a per-turn cap (see docstring).
        self.max_tokens_per_turn = max_tokens_per_turn
        self.max_new_tokens = max_new_tokens
        self.frame_queue_size = frame_queue_size
        # How long a single poll_output blocks. Fragments arrive ~0.1 s apart inside a turn, so
        # 0.25 s costs nothing while a turn is running and bounds the wait on an idle tick.
        self.poll_timeout_sec = poll_timeout_sec
        # Wall-clock a single tick may spend draining. 1.0 s matches native_fps=1.0: a reply that
        # takes longer keeps streaming into the following ticks instead of stalling the clock. The
        # queue is FIFO and unbounded, so draining less than the worker produces loses nothing.
        self.poll_budget_sec = poll_budget_sec
        # The last tick has no successor to continue into, so it waits longer for <|silence|> and
        # then flushes regardless. ~6 s covers a full reply at the measured 0.1 s/token.
        self.final_budget_sec = final_budget_sec
        self.attn_implementation = attn_implementation
        # Without this, OMNI_HF_MOCK=1 still reaches from_pretrained and hangs on the HF fetch.
        self.mock = os.environ.get("OMNI_HF_MOCK", "") == "1"
        self._model = None
        self._proc = None
        self._session = None
        # utterance buffer (per session)
        self._buf: List[str] = []
        self._buf_t: Optional[float] = None
        # last tick of the scan window, so the final drain knows not to defer to a next tick
        self._last_tick_t: Optional[float] = None

    def name(self) -> str:
        return "moss-vl-realtime"

    def _ensure_model(self):
        if self.mock or self._model is not None:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor  # type: ignore
            import torch  # type: ignore
        except Exception as e:  # pragma: no cover
            raise NotImplementedError(_INSTALL_HINT) from e
        # frame_extract_num_threads=1: we hand over pre-decoded frames, so the processor's own
        # extractor pool would only add threads that never do work.
        # ``code_revision`` is what actually pins the trust_remote_code modules; ``revision`` alone
        # pins the weights/config and lets the .py files track main. Both, same sha.
        pin = {"revision": self.revision, "code_revision": self.revision} if self.revision else {}
        self._proc = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=True, frame_extract_num_threads=1, **pin
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation=self.attn_implementation,
            **pin,
        )
        self._model.eval()

    def _budget(self, img):
        """Downscale to ``max_pixels`` on the 32-px vision grid, preserving aspect ratio.

        Never upscales: a frame already inside the budget is handed over untouched, so the budget
        is a ceiling rather than a normalisation. Shared with the Gemini adapter (which needs the
        same arithmetic before JPEG-encoding), hence the delegation — same numbers, one copy.
        """
        return pixel_budget(img, self.max_pixels, grid=_PATCH_GRID)

    def _to_pil(self, frame: Frame):
        """Frame.image is a PIL.Image, an RGB ndarray, or None for a synthetic frame."""
        img = frame.image
        if img is None:
            return None
        if hasattr(img, "convert"):  # already PIL
            return self._budget(img.convert("RGB"))
        from PIL import Image  # type: ignore

        import numpy as np  # type: ignore

        return self._budget(Image.fromarray(np.asarray(img).astype("uint8"), mode="RGB"))

    # ---- Offline Probing (protocol-comparison row only) --------------------------------
    def probe(self, ctx: ProbeContext) -> ProbeResult:  # pragma: no cover - needs weights
        """One offline forward over the cumulative frames, via ``offline_batch_generate``.

        Frames are pre-decoded here, so they go in as individual ``image`` content items rather
        than a ``video`` path. MOSS-VL is an online model in this benchmark; this exists only for
        the probe-vs-online row.
        """
        if self.mock:
            return ProbeResult(fired=False)
        self._ensure_model()
        images = [p for p in (self._to_pil(f) for f in ctx.frames) if p is not None]
        content = [{"type": "image", "image": im} for im in images]
        content.append(
            {"type": "text", "text": build_prompt(ctx.query_text, ctx.audio_context_text)}
        )
        out = self._model.offline_batch_generate(
            self._proc, [{"messages": [{"role": "user", "content": content}]}]
        )
        results = out.get("results") or []
        raw = (results[0].get("text") if results else "") or ""
        text = strip_control_tokens(raw)
        if not text or _SILENCE in raw:
            return ProbeResult(fired=False, raw=raw)
        return ProbeResult(
            fired=True, response_type=infer_response_type(text), reply_text=text, raw=raw
        )

    # ---- Online Rolling (native) -------------------------------------------------------
    def begin(self, spec: SessionSpec) -> None:  # pragma: no cover - needs weights
        self._ensure_model()
        self._buf, self._buf_t = [], None
        # The driver ticks while ``t <= scan_until``, so the last tick is the one whose successor
        # would fall past it. Knowing that lets the final tick flush a reply that is still
        # streaming instead of dropping it with the buffer (see the docstring).
        step = 1.0 / spec.fps if getattr(spec, "fps", 0) else 1.0
        scan_until = getattr(spec, "scan_until_sec", None)
        self._last_tick_t = None if scan_until is None else float(scan_until) - step
        if self.mock:
            self._session = None
            return
        # MOSS-VL keeps its OWN system prompt (that is what the <|silence|> timing was trained
        # against; passing ``system_prompt=None`` selects upstream's built-in one); our shared
        # proactive text rides on initial_prompt with_system=False so the reply format matches every
        # other model in the table without touching its timing.
        #
        # ``with_tools=True`` (2026-09-03) adds the markup + tool vocabulary to that user turn, and
        # nothing else — no ABSTAIN sentinel, which would compete with <|silence|>. Before this the
        # model was scored on ``<call:NAME>`` names it had never been shown, so every
        # ``simultaneous_tool_calls_*`` item was a structural zero. Its realtime session has no
        # ``tools=`` argument, so the user turn is the only channel.
        self._session = self._model.create_realtime_session(
            self._proc,
            initial_prompt=build_prompt(spec.query_text, "", with_system=False, with_tools=True),
            frame_queue_size=self.frame_queue_size,
            max_tokens_per_turn=self.max_tokens_per_turn,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        self._session.start()

    def observe(self, frame: Frame, t_sec: float, audio_chunk=None) -> Optional[Emit]:  # pragma: no cover
        if self.mock:
            return None  # pure standby: exercises the tick loop without a model
        img = self._to_pil(frame)
        if img is not None:
            self._session.push_frame(img, timestamp=t_sec)
        final = self._last_tick_t is not None and t_sec >= self._last_tick_t - 1e-6
        return self._drain(t_sec, final=final)

    def end(self) -> None:  # pragma: no cover - needs weights
        if self._session is not None:
            self._session.close(timeout=5.0)
            self._session = None
        self._buf, self._buf_t = [], None
        self._last_tick_t = None

    # ---- utterance assembly ------------------------------------------------------------
    def _drain(self, t_sec: float, final: bool = False) -> Optional[Emit]:  # pragma: no cover
        """Collect fragments; return an Emit when the turn ends.

        ``<|silence|>`` is the only end-of-turn signal (see the docstring): empty polls carry no
        information, because every tick boundary has a ~0.5 s gap that looks identical to "stopped
        talking". So a poll that returns ``None`` is simply ignored, and fragments carry over to the
        next tick until the sentinel arrives.

        Each tick spends at most ``poll_budget_sec`` here so the driver's clock keeps moving; the
        output queue is FIFO and unbounded, so anything not drained this tick is drained next tick,
        in order. ``final=True`` marks the last tick of the scan window — it waits longer and then
        flushes whatever it has, since there is no next tick to continue into.
        """
        import time

        deadline = time.monotonic() + (self.final_budget_sec if final else self.poll_budget_sec)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            chunk = self._session.poll_output(timeout=min(self.poll_timeout_sec, remaining))
            if chunk is None:
                continue
            if _SILENCE in chunk:
                # Text may share the chunk with the sentinel; keep the text part.
                rest = chunk.replace(_SILENCE, "")
                if rest.strip():
                    self._note(rest, t_sec)
                return self._flush()   # end of turn -> one emit (None if it was an idle tick)
            self._note(chunk, t_sec)
            if chunk.rstrip().endswith(tuple(_SENTENCE_END)):
                emit = self._flush()   # sentence end: one response, stamped where it started
                if emit is not None:
                    return emit

        return self._flush() if final else None

    def _note(self, chunk: str, t_sec: float) -> None:
        if self._buf_t is None:
            self._buf_t = t_sec   # the moment the model chose to speak
        self._buf.append(chunk)

    def _flush(self) -> Optional[Emit]:
        raw = "".join(self._buf)
        at = self._buf_t if self._buf_t is not None else 0.0
        self._buf, self._buf_t = [], None
        text = strip_control_tokens(raw)
        if not text:
            return None   # empty: an idle tick, no utterance to report
        # Verbatim repeats are NOT dropped -- see the module docstring.
        return Emit(
            time_sec=at, response_type=infer_response_type(text), reply_text=text, raw=raw
        )

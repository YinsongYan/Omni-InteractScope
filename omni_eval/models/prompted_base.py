"""Shared base for prompted (turn-based) HF models made proactive.

None of the local checkpoints wired here — Qwen3-VL, Qwen2.5/3-Omni, InternVL3.5, MiniCPM-o —
is natively "proactive"; they answer one turn given media + a prompt. Turning a turn-based model
into a proactive one is *protocol*-level logic, not model-specific:

    Offline Probing : at each probe feed cumulative media + the standing query, and require the
        model to answer OR emit the sentinel ``ABSTAIN`` (silence).
    Online Rolling  : approximate streaming with a **sliding-window re-prompt** — every
        ``stride_sec`` re-prompt the recent window and emit only when the answer is new.

This module factors that out so each concrete adapter only implements the HF recipe:
``_ensure_model()`` (lazy, isolated-env import) and ``_generate(...)`` (one forward → text).

Graceful degradation: ``OMNI_HF_MOCK=1`` (or ``mock=True``) short-circuits every generation to
``ABSTAIN`` so the full inference→scoring pipeline and the test-suite run with no torch / weights.

Response markup convention (``<call:tool>`` / ``<ui:component>``) is shared with the GT and parsed
by ``omni_eval.data.response_parser.infer_response_type`` — no per-model special-casing.
"""
from __future__ import annotations

import math
import os
import re
from abc import abstractmethod
from typing import List, Optional

from ..config import disclosed_tools
from ..data.media import Frame
from ..data.response_parser import infer_response_type, strip_reasoning
from .base import (
    GENERATE_ERROR_MARK,
    Capabilities,
    Emit,
    ProbeContext,
    ProbeResult,
    SessionSpec,
    StreamingModel,
)

_ABSTAIN = "ABSTAIN"

# The sentinel alone on a line, possibly with punctuation around it ("**ABSTAIN**", "ABSTAIN."").
# Observed: InternVL3.5 answered "<call:count>\n\nABSTAIN" — a real response with our standby word
# tacked on. Matching only a whole line keeps two other cases intact: prose that happens to contain
# the word, and the "ABSTAIN  # generate-error: ..." fallback from ``_run`` (both have other word
# characters on the line, so they fall through to the prefix check in ``_is_abstain``).
_SENTINEL_LINE_RE = re.compile(rf"(?im)^\W*{_ABSTAIN}\W*$")

# The tool vocabulary, disclosed to every model — read from ``configs/tools.yaml``, which is the
# single source of truth (it also defines each name's retrieval input modality and the
# acceptable-substitute sets the strict/lenient tool scores use). This list used to be six names
# hard-coded here, one of three uncoordinated copies.
#
# Without disclosure the ``simultaneous_tool_calls_*`` subtask degenerates into guessing a string —
# ``metrics/content.py`` compares tool-name *sets* with ``tool_f1_pass: 1.0`` (exact equality),
# and models guessed <call:拍照> / <call:count> / <call:take_a_photo> / <call:audio>, so
# ``tool_ok`` was structurally unreachable for ~25% of the eval. Naming the action space is what a
# real tool-using agent gets; it turns the subtask back into "should I call a tool, and when".
#
# One clause of documentation per tool is disclosed as well, because the vocabulary is split by
# *retrieval input modality* (text / frame / audio, plus one action): with bare names, choosing
# between ``web_search`` and ``image_search`` would be a guess at our naming convention rather than
# a routing decision. Frequencies are NOT disclosed — that would leak the label distribution.
#
# All-English by construction: a name that varied with the sample's QA language would be
# unscoreable in one of them, since the metric compares the set exactly.
_TOOL_SPECS = disclosed_tools()
_TOOLS = tuple(_TOOL_SPECS)
_TOOL_DOC = "\n".join(
    f"  - {name}({', '.join(spec.get('args') or [])}): "
    + " ".join(str(spec.get("desc") or "").split())
    for name, spec in _TOOL_SPECS.items()
)

# The response-format half of the contract: markup schema + declared action space. Factored out of
# ``_SYSTEM`` (which it still composes, byte for byte) because the three native-streaming adapters
# need exactly this and none of the rest: they keep their own trained standby signal (MMDuet2's
# ``NO REPLY``, MOSS-VL's ``<|silence|>``, MiniCPM-o's duplex listen/speak head), so handing them
# the ABSTAIN sentinel would fight their timing, but withholding the tool vocabulary makes
# ``tool_ok`` structurally unreachable for them — the same defect the disclosure above exists to
# fix, and neither streaming runtime accepts a native ``tools=`` schema (verified 2026-09-03:
# ``MiniCPM-o.as_duplex().prepare()`` takes only ``prefix_system_prompt``/``ref_audio``/
# ``prompt_wav_path``, ``MossVL.create_realtime_session()`` only ``initial_prompt``/
# ``system_prompt``/pacing; the tool-aware chat templates they inherit from their LM backbones are
# unreachable dead code on the streaming path). Free text in the prompt is the only channel there is.
#
# The markup spec must read as a *schema*, not as an example. Written as "use the markup
# <call:tool_name>", InternVL3.5 emitted the literal string ``<call:tool_name>`` — our own
# placeholder, parroted back, which scores 0 on the tool-name set-F1 through no fault of the model.
# Capitalised NAME plus an explicit "don't output the placeholder" fixes that.
# "no closing tag" is not pedantry: InternVL3.5 answered `<call>统计</call>`, XML-pair style, which
# the shared parser does not recognise as a tool call at all (it requires the colon form).
_TOOL_BLOCK = (
    "When a tool call or UI element is warranted, mark it up inline as <call:NAME> / <ui:NAME> — "
    "the name goes right after the colon, there is no closing tag, and the reply text follows the "
    "marker. NAME is a placeholder: never output it literally.\n"
    # Declared action space. See _TOOLS for why this is disclosed and what is deliberately not.
    "- The available tools are exactly these, and no others. Use a name verbatim after <call: — do "
    f"not invent or translate one.\n{_TOOL_DOC}\n"
    # Merely mentioning the markup makes models sprinkle it everywhere: Qwen2.5/3-Omni answered
    # "<call:audio>" (grabbing the nearest modality label in context) and InternVL "<call:拍照>" on
    # items whose GT is plain text. The judge sees ``pred_reply`` verbatim on text-scored tasks
    # (metrics/content.py), so stray markup is noise inside the judged text; stating that markup is
    # opt-in costs nothing and discloses no distribution.
    "- Do not add markup unless you would really invoke a tool or render a UI element; plain text "
    "on its own is a complete response.\n"
)

# ONE wording/markup contract for every prompted adapter, so a score gap is the model's and not
# the prompt's. The native-streaming adapters take ``tool_disclosure()`` only (see above).
_SYSTEM = (
    "You are a proactive, always-on streaming video assistant. You are shown video frames "
    "sampled up to the CURRENT moment, plus the user's standing request. Decide whether now "
    "is the right moment to respond.\n"
    f"- If the event/condition the user asked about has NOT happened yet, reply with exactly: {_ABSTAIN}\n"
    "- If it HAS happened by now, give the appropriate response. " + _TOOL_BLOCK +
    "Be concise and do not restate the question."
)


def tool_disclosure() -> str:
    """The markup schema + tool vocabulary as a standalone bulleted block.

    For adapters that own their proactive/timing prompt and must not be given the ABSTAIN
    sentinel, but still have to be told the action space they are scored against.
    """
    return "- " + _TOOL_BLOCK


def frames_to_pil_timed(frames: List[Frame]) -> tuple:
    """Like :func:`frames_to_pil`, but also returns each surviving frame's ``time_sec``.

    Adapters that must tell the processor *when* the frames were taken need the times of the
    frames that actually survived the conversion, not of the input list — a dropped synthetic or
    undecodable frame would otherwise shift every subsequent timestamp.
    """
    out: list = []
    times: List[float] = []
    try:
        from PIL import Image  # type: ignore
        import numpy as np
    except Exception:
        return out, times
    for fr in frames:
        img = getattr(fr, "image", None)
        if img is None:
            continue
        try:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img.astype("uint8"))
            out.append(img.convert("RGB"))
            times.append(float(getattr(fr, "time_sec", 0.0) or 0.0))
        except Exception:
            continue
    return out, times


def frames_to_pil(frames: List[Frame]) -> list:
    """Convert decoded frames to PIL images, dropping synthetic / empty ones.

    Returns ``[]`` when nothing is decodable (e.g. synthetic media in tests); adapters should
    handle an empty frame list gracefully.
    """
    return frames_to_pil_timed(frames)[0]


def pixel_budget(img, max_pixels: Optional[int], grid: int = 32):
    """Downscale ``img`` to ``max_pixels`` on a ``grid``-px lattice, preserving aspect ratio.

    A **ceiling, not a normalisation**: a frame already inside the budget is returned untouched,
    never upscaled. ``max_pixels`` falsy is a no-op.

    The pixel budget is a comparability knob, not a performance one. Every Qwen adapter passes its
    ``max_pixels: 151200`` (360*420) into the processor, so an adapter that hands over native-
    resolution frames gives its model several times the visual detail on the same probe — the
    ablation would then read as a model gap. This lived only in ``moss_vl_realtime._budget``
    (where the checkpoint's own processor applies no cap at all); the Gemini adapter needs the same
    arithmetic before JPEG-encoding, so it is one function here rather than two copies.
    """
    if not max_pixels:
        return img
    w, h = img.size
    if w * h <= max_pixels:
        return img
    scale = math.sqrt(max_pixels / float(w * h))
    dims = tuple(max(grid, int(round(side * scale / grid)) * grid) for side in (w, h))
    from PIL import Image  # type: ignore

    return img.resize(dims, Image.BICUBIC)


def true_frame_fps(n_frames: int, times: List[float], fallback: float = 1.0) -> float:
    """The rate the sampled frames were *actually* taken at, from their own timestamps.

    Not the same number as ``offline_probing.sample_fps``, and the difference is not small.
    ``MediaSource.sample_frames`` asks for ``sample_fps`` frames per second but then clamps the
    count to ``max_frames`` and ``linspace``s over the whole prefix, so once a probe's cumulative
    context exceeds ``max_frames`` seconds the real gap stretches: on this release's probe grid the
    median inter-frame gap is 1.73 s and the p90 is 11.21 s at ``max_frames = 16``, and only 40.5%
    of probes are genuinely 1 fps.

    Every Qwen processor needs this number to place the frames in time (Qwen3-VL stamps the prompt
    with it, Qwen-Omni turns it into TMRoPE's ``video_second_per_grid = temporal_patch_size / fps``),
    so handing over the nominal 1.0 tells a model watching a 168 s prefix that it spans 16 s.

    ``fallback`` is returned when the span is degenerate -- a single frame, or synthetic media whose
    frames share a timestamp -- where no rate is derivable.
    """
    if n_frames < 2 or len(times) != n_frames:
        return float(fallback)
    span = times[-1] - times[0]
    return (n_frames - 1) / span if span > 1e-6 else float(fallback)


def build_prompt(
    query_text: str, audio_ctx: str, with_system: bool = True, with_tools: bool = False
) -> str:
    """Combine the proactive framing, the standing query and any ASR/text audio context.

    ``_SYSTEM`` is folded into the single user message rather than sent as a ``system`` turn:
    the four HF adapters have four different chat templates, and one shared text block keeps the
    wording contract identical across them.

    Without it the ``ABSTAIN`` sentinel is unreachable — the model is never told the sentinel
    exists, so it answers every probe and every item scores ``premature``. ``with_system=False``
    is for adapters that own their proactive prompt (MMDuet2's trained ``NO REPLY``).

    ``with_tools=True`` adds :func:`tool_disclosure` *without* the sentinel — the setting for an
    adapter whose timing is trained but whose output format is still scored against our markup.
    Ignored when ``with_system`` is set, since ``_SYSTEM`` already contains the block.
    """
    prompt = (query_text or "").strip()
    # T8 carries an explicit assistance policy, including its timing contract. Applying
    # the legacy event-request system here would turn query-free A into an empty question.
    from ..t8.policy import PREFIX as T8_PREFIX
    t8 = prompt.startswith(T8_PREFIX.rstrip())
    if audio_ctx:
        prompt = (prompt + "\n\n" + audio_ctx).strip()
    if t8:
        return prompt
    if with_system:
        prompt = (_SYSTEM + "\n\n[Standing request] " + prompt).strip()
    elif with_tools:
        prompt = (tool_disclosure() + "\n[Standing request] " + prompt).strip()
    return prompt


class PromptedProactiveModel(StreamingModel):
    """Turn-based HF model + ABSTAIN(offline) / sliding-window(online) proactivity.

    Subclasses set the class attributes ``capabilities``, ``_consumes_audio`` and
    ``_display_name``, and implement ``_ensure_model`` + ``_generate``.
    """

    capabilities = Capabilities(
        modalities_in={"text", "image", "video"}, supports_streaming=True, native_fps=1.0
    )
    _consumes_audio: bool = False
    _display_name: str = "prompted-model"

    def __init__(
        self,
        model_path: str = "",
        max_frames: int = 16,
        stride_sec: float = 2.0,
        recent_window_sec: float = 8.0,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        use_audio: bool = True,
        device: str = "auto",
        revision: Optional[str] = None,
        **_,
    ):
        self.model_path = model_path
        # Hub revision to pin. Only meaningful for adapters that load ``trust_remote_code`` code out
        # of a mutable hub repo -- there the checkpoint ships behaviour, not just weights, and `main`
        # moving between two runs makes them incomparable with nothing in the fingerprint to show it
        # (it already happened once: see the moss-vl-realtime docstring). Named here rather than
        # per-adapter so a ``revision:`` in configs/models/*.yaml is never silently swallowed by
        # ``**_``; each concrete ``_ensure_model`` decides whether to forward it.
        self.revision = revision or None
        self.max_frames = max_frames
        self.stride_sec = stride_sec
        self.recent_window_sec = recent_window_sec
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        # ``use_audio`` only bites when the concrete adapter is audio-capable.
        self.use_audio = use_audio and self._consumes_audio
        self.device = device
        self.mock = os.environ.get("OMNI_HF_MOCK", "") == "1"

        self._model = None
        self._proc = None
        self._tok = None
        # streaming state
        self._spec: Optional[SessionSpec] = None
        self._frame_buf: List[Frame] = []
        self._audio_buf: List[tuple] = []  # (t_sec, np.ndarray) when _consumes_audio
        self._last_probe_t: float = -1e9

    def name(self) -> str:
        return self._display_name

    # ---- to be implemented per checkpoint --------------------------------------------
    @abstractmethod
    def _ensure_model(self) -> None:
        """Lazily load model/processor. Raise a clear install hint if deps are missing."""

    @abstractmethod
    def _generate(
        self,
        query_text: str,
        audio_ctx: str,
        frames: List[Frame],
        query_image: Optional[str] = None,
        query_audio: Optional[str] = None,
        audio_waveform=None,
    ) -> str:
        """Run one forward pass over ``frames`` (+audio for omni) and return the decoded text."""

    # ---- shared helpers --------------------------------------------------------------
    @staticmethod
    def _is_abstain(text: str) -> bool:
        t = (text or "").strip()
        return (not t) or t.upper().startswith(_ABSTAIN)

    @staticmethod
    def _visible(raw: str) -> str:
        """Strip our own control vocabulary, leaving what the model actually answered.

        Two things are ours, not the model's, and neither should reach the judge: the reasoning
        block of a thinking model, and the ABSTAIN sentinel. Stripping both *before* the standby
        check is what makes the check order-independent — "ABSTAIN\\n\\n<call:x>" and
        "<call:x>\\n\\nABSTAIN" are the same self-contradictory output and must be read the same
        way (as the response, with the stray sentinel dropped). A remainder that is empty means the
        model said nothing but the sentinel, i.e. standby.
        """
        return _SENTINEL_LINE_RE.sub("", strip_reasoning(raw)).strip()

    def _gen_kwargs(self, **extra) -> dict:
        """Decoding args every adapter must pass: greedy unless a temperature was configured.

        Leaving them out hands control to the checkpoint's ``generation_config``, and the Instruct
        checkpoints ship ``do_sample: true`` — so two runs of the same probe disagree. Measured on
        Qwen3-Omni-30B-A3B, same inputs, same code: one run answered 6 of 9 probes with a full
        audio-grounded reply, the next answered 1 of 9 with a single word. A benchmark number has
        to be reproducible, so decoding is pinned here (one place) rather than per adapter.
        """
        kw = dict(max_new_tokens=self.max_new_tokens, do_sample=self.temperature > 0)
        if self.temperature > 0:
            kw["temperature"] = self.temperature
        kw.update(extra)
        return kw

    def _run(
        self,
        query_text: str,
        audio_ctx: str,
        frames: List[Frame],
        query_image: Optional[str] = None,
        query_audio: Optional[str] = None,
        audio_waveform=None,
    ) -> str:
        if self.mock:
            return _ABSTAIN
        self._ensure_model()
        used = frames[-self.max_frames :] if len(frames) > self.max_frames else frames
        try:
            return self._generate(
                query_text=query_text,
                audio_ctx=audio_ctx,
                frames=used,
                query_image=query_image,
                query_audio=query_audio,
                audio_waveform=audio_waveform if self._consumes_audio else None,
            )
        except Exception as e:  # keep the run alive; treat a generation failure as abstain
            # The marker is what the protocol drivers count, so a crashing adapter cannot pass for
            # a conservative one (see base.GENERATE_ERROR_MARK).
            return f"{_ABSTAIN}  {GENERATE_ERROR_MARK} {e}"

    def _recent_audio(self, t_sec: float):
        """Concatenate buffered audio chunks within the recent window into one 16 kHz array."""
        if not (self._consumes_audio and self.use_audio and self._audio_buf):
            return None
        try:
            import numpy as np
        except Exception:
            return None
        lo = t_sec - self.recent_window_sec
        chunks = [c for (ct, c) in self._audio_buf if c is not None and ct >= lo]
        if not chunks:
            return None
        try:
            return np.concatenate([np.asarray(c, dtype="float32").reshape(-1) for c in chunks])
        except Exception:
            return None

    # ---- Offline Probing -------------------------------------------------------------
    def probe(self, ctx: ProbeContext) -> ProbeResult:
        raw = self._run(
            ctx.query_text,
            ctx.audio_context_text,
            ctx.frames,
            query_image=ctx.query_image,
            query_audio=ctx.query_audio,
            audio_waveform=getattr(ctx, "audio_waveform", None),
        )
        # Thinking models put their deliberation ahead of the answer; strip it BEFORE the sentinel
        # check (a prefix match would never see an ABSTAIN that follows a <think> block) and score
        # only the visible answer. ``raw`` keeps the full output.
        out = self._visible(raw)
        if self._is_abstain(out):
            return ProbeResult(fired=False, raw=raw)
        return ProbeResult(
            fired=True, response_type=infer_response_type(out), reply_text=out, raw=raw
        )

    # ---- Online Rolling (sliding-window approximation) -------------------------------
    def begin(self, spec: SessionSpec) -> None:
        self._spec = spec
        self._frame_buf = []
        self._audio_buf = []
        self._last_probe_t = -1e9

    def observe(self, frame: Frame, t_sec: float, audio_chunk=None) -> Optional[Emit]:
        self._frame_buf.append(frame)
        if self._consumes_audio and audio_chunk is not None:
            self._audio_buf.append((t_sec, audio_chunk))
        if t_sec - self._last_probe_t < self.stride_sec:
            return None
        self._last_probe_t = t_sec

        recent = [f for f in self._frame_buf if f.time_sec >= t_sec - self.recent_window_sec]
        q = self._spec.query_text if self._spec else ""
        q_img = self._spec.query_image if self._spec else None
        q_aud = self._spec.query_audio if self._spec else None
        raw = self._run(
            q,
            "",
            recent,
            query_image=q_img,
            query_audio=q_aud,
            audio_waveform=self._recent_audio(t_sec),
        )
        out = self._visible(raw)
        if self._is_abstain(out):
            return None
        # Verbatim repeats are NOT dropped (2026-09-01; same change as the native streaming
        # adapters). Re-prompting every tick does make this path repeat itself when the recent
        # window stops changing, but suppressing that here hides two behaviours behind one silence:
        # a model that correctly re-narrates at the next GT onset, and a model stuck in a loop.
        # ``merge_emits`` collapses fires inside ``temporal.merge_gap_sec`` (2.0 s, i.e. two ticks on
        # the 1 s grid, so consecutive-tick repeats still collapse) and keeps the count as
        # ``redundancies``.
        return Emit(time_sec=t_sec, response_type=infer_response_type(out), reply_text=out, raw=raw)

    def end(self) -> None:
        self._spec = None
        self._frame_buf = []
        self._audio_buf = []

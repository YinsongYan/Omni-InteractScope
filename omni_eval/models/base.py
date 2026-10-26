"""Model adapter contract shared by both protocols.

A **ProbeModel** answers one probe: given cumulative media up to a probe time plus the query,
decide fire-or-abstain and (if firing) produce a reply. This is all Offline Probing needs and
works for *any* model, streaming or not.

A **StreamingModel** additionally supports the Online Rolling loop: ``begin`` once, then
``observe`` per tick returning an ``Emit`` or ``None`` (standby), then ``end``.

Adapters declare ``Capabilities`` so the harness can check them against the run's input cell
(``omni_eval.adaptation.input_config``) — e.g. refuse ``--input va`` for a vision-only model, and
route query audio through ASR for everyone in cell ``v``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

from ..data.media import Frame
from ..schema import Interaction


@dataclass
class Capabilities:
    modalities_in: set = field(default_factory=lambda: {"text", "image", "video"})
    supports_streaming: bool = False   # can drive the Online Rolling loop at all (begin/observe/end)
    # True only for models with a *real* streaming mechanism (MiniCPM-o duplex, MMDuet2's
    # per-frame decision head). These are ROUTED to Online Rolling; prompted VL/omni models keep
    # ``supports_streaming=True`` but ``native_streaming=False`` and are routed to Offline Probing.
    native_streaming: bool = False
    native_fps: float = 1.0
    # True only when ``probe()`` may be called concurrently from several threads — i.e. the
    # adapter holds no per-call state and talks to a server (vLLM / OpenAI-style). Local HF
    # adapters keep this False; scale those with ``--shard`` + one process per GPU instead.
    thread_safe_probe: bool = False


@dataclass
class ProbeContext:
    interaction: Interaction
    probe_time_sec: float
    frames: List[Frame]
    query_text: str = ""                 # may be ASR-substituted
    query_image: Optional[str] = None
    query_audio: Optional[str] = None    # only set when model consumes audio natively
    audio_context_text: str = ""         # ASR track transcript (asr-substituted samples)
    # Raw audio track waveform (mono 16 kHz np.ndarray) over [origin, probe_time], set by the
    # Offline Probing driver only for audio-native models so omni models hear audio events too.
    audio_waveform: Optional[object] = None
    modality_tag: str = "native"


# Adapters keep a run alive by degrading a generation crash into an abstain (see
# ``prompted_base._run``). That is the right call for a 10-hour run, but it makes a systematically
# broken adapter byte-identical to a conservative model in the prediction file: ``probe_log`` stores
# only ``fired``, so a task_type at 0% fire rate cannot be told from one that never generated. The
# protocol drivers grep ``raw`` for this marker and report the count in ``meta``, so the difference
# is visible without keeping every raw string on disk.
GENERATE_ERROR_MARK = "# generate-error:"


@dataclass
class ProbeResult:
    fired: bool
    response_type: str = "text"
    reply_text: str = ""
    raw: str = ""


@dataclass
class SessionSpec:
    interaction: Interaction
    duration_sec: float
    fps: float = 1.0
    query_text: str = ""
    query_image: Optional[str] = None
    query_audio: Optional[str] = None
    audio_context_text: str = ""
    modality_tag: str = "native"
    # Path to the source video. Native-streaming adapters (AURA) need the file itself to build
    # 1 s mp4 chunks; frame-based adapters ignore it. None when media is synthetic/unavailable.
    video_path: Optional[str] = None
    # Bound for the streaming scan: the driver stops ticking past this time (max GT + margin) to
    # keep real-streaming cost finite. None = scan to clip end.
    scan_until_sec: Optional[float] = None


@dataclass
class Emit:
    time_sec: float
    response_type: str = "text"
    reply_text: str = ""
    raw: str = ""


class ProbeModel(ABC):
    capabilities: Capabilities = Capabilities()

    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def probe(self, ctx: ProbeContext) -> ProbeResult:
        """Decide fire/abstain for a single probe."""


class StreamingModel(ProbeModel):
    @abstractmethod
    def begin(self, spec: SessionSpec) -> None: ...

    @abstractmethod
    def observe(self, frame: Frame, t_sec: float, audio_chunk=None) -> Optional[Emit]:
        """Return an Emit if the model chooses to speak at this tick, else None (standby)."""

    @abstractmethod
    def end(self) -> None: ...

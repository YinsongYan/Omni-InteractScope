"""Legacy query-relative rolling protocol for the separately reported T5 column."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..adaptation.asr import ASR
from ..adaptation.input_config import resolve_input_config
from ..adaptation.modality import ModalityDecision
from ..config import get
from ..data.loader import interaction_duration
from ..data.media import MediaSource
from ..models.base import SessionSpec, StreamingModel
from ..schema import Interaction
from .base import PreparedInputs, new_record, prepare_inputs
from ..t8.protocol import public_interaction


AUDIO_CLOCK_VERSION = "online-rolling-causal-audio-v2"


def _max_gt_time(it: Interaction) -> Optional[float]:
    """Latest ground-truth time across an interaction's responses (window end or trigger point).

    None when no response carries a time (e.g. a pure should-stay-silent negative), meaning the
    scan is not bounded and runs to the clip end.
    """
    times: List[float] = []
    for r in it.responses:
        if r.valid_response_window_sec:
            times.append(r.valid_response_window_sec[1])
        if r.notify_trigger_sec is not None:
            times.append(r.notify_trigger_sec)
    return max(times) if times else None


def run(
    it: Interaction,
    model: StreamingModel,
    media: MediaSource,
    decision: ModalityDecision,
    cfg: Dict[str, Any],
    asr: Optional[ASR] = None,
    prepared: Optional[PreparedInputs] = None,
    video_meta: Optional[dict] = None,
) -> Dict[str, Any]:
    rec = new_record(it, "online_rolling", model.name(), decision)
    if not isinstance(model, StreamingModel):
        raise TypeError(f"{model.name()} is not a StreamingModel; cannot run Online Rolling")
    if not decision.supported:
        rec["meta"] = {"skipped": "unsupported"}
        return rec

    if prepared is None:
        prepared = prepare_inputs(it, model.capabilities, decision, media, cfg, asr, for_online=True)

    fps = get(cfg, "online_rolling.fps", 1.0)
    duration = interaction_duration(it, media, video_meta)
    start_t = max(0.0, it.query_time_sec or 0.0)
    # Only cell ``va`` streams raw audio chunks. Cell ``v+asr`` is NOT offered online (mid-stream
    # transcript injection is out of scope this round — see docs/input_modality.md), so online
    # runs use ``v`` or ``va``.
    ic = resolve_input_config(decision.input_config or None)
    feed_audio = ic.track_native
    if feed_audio:
        rec["protocol_version"] = AUDIO_CLOCK_VERSION

    q_text = prepared.query_text
    if prepared.query_audio_context and not feed_audio:
        q_text = (q_text + "\n" + prepared.query_audio_context).strip()

    step = 1.0 / fps if fps > 0 else 1.0
    # Bound real-streaming cost: stop ticking past the last GT time + margin (== tolerance_sec, the
    # acceptable-window half-width). Unbounded (scan to clip end) only for time-less negatives.
    tol = get(cfg, "temporal.tolerance_sec", 2.0)
    max_gt = _max_gt_time(it)
    scan_until = duration if max_gt is None else min(duration, max_gt + tol + step)
    video_path = media.path if getattr(media, "exists", False) else None

    spec = SessionSpec(
        interaction=public_interaction(it),
        duration_sec=duration,
        fps=fps,
        query_text=q_text,
        query_image=prepared.query_image,
        query_audio=prepared.query_audio if feed_audio else None,
        audio_context_text="",
        modality_tag=decision.tag,
        video_path=video_path,
        scan_until_sec=scan_until,
    )
    model.begin(spec)
    emits: List[dict] = []
    t = start_t
    previous_t = start_t
    n_ticks = 0
    try:
        while t <= scan_until + 1e-6:
            frame = media.frame_at(t)
            # audio_chunk(start, duration) reads [start, start + duration).
            # Reading (t, step) here exposed the next tick's audio at time t.
            audio_chunk = (
                media.audio_chunk(previous_t, t - previous_t)
                if feed_audio and t > previous_t else None
            )
            emit = model.observe(frame, t, audio_chunk)
            if emit is not None:
                emits.append(
                    {
                        "time_sec": emit.time_sec,
                        "response_type": emit.response_type,
                        "reply_text": emit.reply_text,
                        "raw": emit.raw,
                    }
                )
            n_ticks += 1
            previous_t = t
            t += step
    finally:
        model.end()

    rec["emits"] = emits
    rec["meta"] = {
        "n_ticks": n_ticks,
        # ``n_probes`` is the denominator of ``false_alarm_rate`` (metrics/aggregate.py). Offline
        # Probing sets it to its grid size; online it is the tick count, which is the same quantity:
        # the number of moments at which the model got to choose between speaking and staying quiet.
        # Without it FAR was None for every online run, so the protocol-comparison table had a hole
        # exactly where it needed a number. The two are commensurable only because both grids run at
        # 1 s (offline_probing.probe_interval_sec == 1.0, online_rolling.fps == 1.0) -- FAR is a
        # per-decision rate, so changing either rate changes the denominator and the offline/online
        # columns stop being comparable. It is a diagnostic either way and never enters headline
        # accuracy.
        "n_probes": n_ticks,
        "duration_sec": duration,
        "fps": fps,
        "scan_until_sec": scan_until,
        "start_sec": start_t,
    }
    # Optional per-session health counters from the adapter (``n_generate_errors`` above all).
    # Without them a generation failure inside a tick is indistinguishable from a deliberate
    # silence -- ``observe`` just returns None and the marked text goes with it -- so
    # ``MAX_GEN_ERR_RATE`` in run_campaign.sh passed every online run unconditionally, which is
    # exactly the blind spot docs/throughput.md 6.5 described. ``setdefault``, not ``update``: the
    # driver owns ``n_ticks``/``n_probes`` and an adapter must not be able to redefine the FAR
    # denominator. Adapters without the method are unaffected, byte for byte.
    stats = getattr(model, "session_stats", None)
    if callable(stats):
        try:
            for k, v in (stats() or {}).items():
                rec["meta"].setdefault(k, v)
        except Exception as exc:  # a counter must never cost us the predictions
            rec["meta"]["session_stats_error"] = f"{type(exc).__name__}: {exc}"
    return rec

"""Common causal clock, matched context availability, and optional exact suffix saving.

The model never receives target annotations or the scoring cutoff. Query-conditioned
history is encoded at request arrival; outputs from this initialization are not decisions.
"""
from dataclasses import replace
import math

from ..t8.protocol import clock_grid, public_interaction
from ..t8.utterances import scoring_emits
from ..models.base import GENERATE_ERROR_MARK, ProbeContext, SessionSpec
from .base import new_record, transcript_up_to

VERSION = 'table2-common-clock-v1'


def plan(it, duration, first_frame=0.0, step=1.0, arm='continuous', cutoff=False):
    query = max(0.0, it.query_time_sec or 0.0)
    start = max(query, first_frame)
    if start > duration:
        raise ValueError('Request begins after the available clip')
    full = clock_grid(duration, start, step)
    if arm != 'continuous' or cutoff:
        raise ValueError('Release evaluation uses the full continuous clock')
    decisions = full
    if not decisions:
        raise ValueError('No causal decision points')
    return decisions, full


def run(it, model, media, decision, prepared, arm='continuous', cutoff=False,
        step=1.0, heartbeat=None):
    native = bool(model.capabilities.native_streaming)
    if native and arm != 'continuous':
        raise ValueError('The local arm is a same-checkpoint prompted control')
    if not decision.supported:
        raise ValueError('Unsupported configuration')
    from ..adaptation.input_config import resolve_input_config
    ic = resolve_input_config(decision.input_config)
    if native and ic.track_asr:
        raise ValueError('This version does not implement native ASR injection')
    duration = float(media.duration)
    first_frame = float(media.pts[0]) if hasattr(media, 'pts') else 0.0
    grid, full = plan(it, duration, first_frame, step, arm, cutoff)
    query_time = max(0.0, it.query_time_sec or 0.0)
    pub = public_interaction(it)
    query_text = prepared.query_text
    if prepared.query_audio_context:
        query_text = (query_text + '\n' + prepared.query_audio_context).strip()
    rec = new_record(it, 'time_aligned_native' if native else 'time_aligned_probe', model.name(), decision)
    rec.update(protocol_version=VERSION, arm=arm)
    emits, logs, warmup = [], [], []

    def observe_log(t, frame, fired, **extra):
        if frame.synthetic or frame.time_sec > t + 1e-7:
            raise ValueError('Noncausal or synthetic frame')
        entry = dict(time_sec=t, last_frame_sec=frame.time_sec, fired=bool(fired), **extra)
        logs.append(entry)
        if heartbeat:
            heartbeat(t, len(logs), len(grid))

    if native:
        # duration and scan_until are the actual clip end, never the GT-based cutoff.
        spec = SessionSpec(interaction=pub, duration_sec=duration, fps=1.0 / step,
            query_text=query_text, query_image=prepared.query_image,
            query_audio=prepared.query_audio, modality_tag=decision.tag,
            video_path=media.path, scan_until_sec=duration)
        history = [t for t in clock_grid(duration, first_frame, step) if t < grid[0]]
        if hasattr(media, 'prepare_clock'):
            media.prepare_clock(history + grid, native=True)
        model.begin(spec)
        prior = 0.0
        try:
            for t in history + grid:
                frame = media.frame_at(t)
                if frame.synthetic or frame.time_sec > t + 1e-7:
                    raise ValueError('Noncausal or synthetic native frame')
                audio = media.audio_chunk(prior, t - prior) if ic.track_native and t > prior else None
                emit = model.observe(frame, t, audio)
                audio_interval = [prior, t] if audio is not None else None
                prior = t
                if t < grid[0]:
                    warmup.append(dict(time_sec=t, last_frame_sec=frame.time_sec,
                                       emitted=emit is not None))
                    if heartbeat:
                        heartbeat(t, 0, len(grid))
                    continue
                observe_log(t, frame, emit is not None, audio_interval=audio_interval)
                if emit is not None:
                    if emit.time_sec > t + 1e-7:
                        raise ValueError('Future-dated native response')
                    if GENERATE_ERROR_MARK in (emit.raw or ''):
                        raise RuntimeError('Native generation failed')
                    # Credit only when the full returned content is available.
                    emits.append(dict(time_sec=t, adapter_time_sec=emit.time_sec,
                        response_type=emit.response_type, reply_text=emit.reply_text, raw=emit.raw))
        finally:
            model.end()
        stats = getattr(model, 'session_stats', lambda: {})() or {}
        if stats.get('n_generate_errors', 0) or stats.get('n_error_resets', 0):
            raise RuntimeError('Native model reported a generation error')
        rec['meta']['session_stats'] = stats
    else:
        if hasattr(media, 'prepare_clock'):
            media.prepare_clock(grid, native=False)
        for t in grid:
            frames = media.sample_frames(0.0, t, max_frames=32, sample_fps=1.0)
            if not frames or any(f.synthetic or f.time_sec > t + 1e-7 for f in frames):
                raise ValueError('Noncausal or empty prefix')
            ctx = ProbeContext(interaction=pub, probe_time_sec=t, frames=frames,
                query_text=prepared.query_text, query_image=prepared.query_image,
                query_audio=prepared.query_audio,
                audio_context_text='\n'.join(x for x in [prepared.query_audio_context,
                    transcript_up_to(prepared.transcript_segments, t)] if x),
                audio_waveform=media.audio_chunk(0.0, t) if ic.track_native and t > 0 else None,
                modality_tag=decision.tag)
            result = model.probe(ctx)
            if GENERATE_ERROR_MARK in (result.raw or ''):
                raise RuntimeError('Prompted generation failed: ' + (result.raw or '')[:250])
            observe_log(t, frames[-1], result.fired, n_frames=len(frames),
                        audio_interval=[0.0, t] if ic.track_native and t > 0 else None)
            if result.fired:
                emits.append(dict(time_sec=t, response_type=result.response_type,
                                  reply_text=result.reply_text, raw=result.raw))
    rec['emits'] = emits
    rec['meta'].update(n_probes=len(logs), probe_log=logs, duration_sec=duration,
        query_time_sec=query_time, start_sec=grid[0], last_decision_sec=grid[-1],
        origin_sec=0.0, step_sec=step, full_clock_decisions=len(full),
        execution_prefix_only=cutoff, context_initialization='query-conditioned causal history replay at request arrival',
        warmup=warmup, independent_probes=not native, ground_truth_visible_to_model=False)
    if model.name() == 'livecc-7b':
        rec['raw_emits'] = emits
        # Retain complete utterances only. End-of-file must not turn a partial
        # fragment into an answer, which would break suffix invariance.
        rec['emits'] = [e for e in scoring_emits(rec) if e.get('utterance_complete', True)]
        rec['meta']['output_processing'] = 'native continuation boundary; complete content timestamp'
    return rec

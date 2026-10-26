"""T8 full-clip causal probing and native streaming."""
from dataclasses import replace
import math
from ..adaptation.input_config import resolve_input_config
from ..models.base import GENERATE_ERROR_MARK, ProbeContext, SessionSpec
from ..protocols.base import new_record, transcript_up_to
from .policy import request_text

VERSION = 't8-causal-v1'


def clock_grid(duration, start=0.0, step=1.0):
    if not math.isfinite(duration) or duration <= 0 or step <= 0:
        raise ValueError('invalid clock bounds')
    if start < 0 or start > duration:
        raise ValueError('query outside clip')
    pts = {round(start, 6), round(duration, 6)}
    pts.update(round(k * step, 6) for k in range(math.ceil(start / step), math.floor(duration / step) + 1))
    return sorted(pts)


def public_interaction(it):
    """Adapters get no targets, trigger reasons, annotation-dependent spans or source names."""
    return replace(it, responses=[], raw={}, video_name='', raw_interaction_id='',
                   interaction_id='', video_key='', video='',
                   video_url='', query_audio_url='', query_image_url='')


def run(it, model, media, decision, prepared, step=1.0):
    native = model.capabilities.native_streaming
    if not decision.supported:
        raise ValueError('unsupported input cell')
    query = request_text(it, native=native)
    ic = resolve_input_config(decision.input_config)
    start = it.query_time_sec or 0.0
    duration = media.duration
    first_frame = float(media.pts[0]) if hasattr(media, 'pts') else 0.0
    grid = clock_grid(duration, max(start, first_frame), step)
    pub = public_interaction(it)
    rec = new_record(it, 't8_native' if native else 't8_probe', model.name(), decision)
    rec.update(condition="main", target_group=it.target_group, qa_language=it.qa_language,
               video_key=it.video_key, protocol_version=VERSION)
    logs, emits = [], []
    if native:
        spec = SessionSpec(interaction=pub, duration_sec=duration, fps=1.0 / step,
                           query_text=query, query_audio=prepared.query_audio,
                           video_path=media.path, scan_until_sec=duration, modality_tag=decision.tag)
        model.begin(spec)
        previous = 0.0
        try:
            # Replay the same pre-query history used by prefix models. The query is known
            # during this warmup, but output before query_time is not an eligible decision.
            native_grid = sorted(set(clock_grid(duration, first_frame, step) + [max(start, first_frame)]))
            if hasattr(media, 'prepare_clock'):
                media.prepare_clock(native_grid, native=True)
            for t in native_grid:
                fr = media.frame_at(t)
                if fr.synthetic:
                    raise ValueError('synthetic frame in real T8 run')
                audio = media.audio_chunk(previous, t - previous) if ic.track_native and t > previous else None
                emit = model.observe(fr, t, audio)
                previous = t
                if t < start:
                    continue
                logs.append(dict(time_sec=t, fired=emit is not None, last_frame_sec=fr.time_sec))
                if emit is not None and GENERATE_ERROR_MARK in (emit.raw or ''):
                    raise RuntimeError('Native generation failed')
                if emit is not None and emit.time_sec >= start:
                    if emit.time_sec > t + 1e-6:
                        raise ValueError('adapter returned future-dated emit')
                    # Some adapters stamp a completed sentence at its first token.
                    # Credit only when its entire content has actually become available.
                    emits.append(dict(time_sec=t, adapter_time_sec=emit.time_sec, reply_text=emit.reply_text,
                                      raw=emit.raw, response_type=emit.response_type))
        finally:
            model.end()
        stats = getattr(model, 'session_stats', lambda: {})() or {}
        if stats.get('n_generate_errors', 0) or stats.get('n_error_resets', 0):
            raise RuntimeError('native model generation errors')
        rec['meta']['session_stats'] = stats
    else:
        if hasattr(media, 'prepare_clock'):
            media.prepare_clock(grid)
        for t in grid:
            frames = media.sample_frames(0.0, t, max_frames=32, sample_fps=1.0)
            if any(f.synthetic or f.time_sec > t + 1e-6 for f in frames):
                raise ValueError('non-causal/synthetic frame')
            ctx = ProbeContext(interaction=pub, probe_time_sec=t, frames=frames, query_text=query,
                               query_audio=prepared.query_audio, query_image=prepared.query_image,
                               audio_context_text=transcript_up_to(prepared.transcript_segments, t),
                               audio_waveform=media.audio_chunk(0, t) if ic.track_native and t > 0 else None,
                               modality_tag=decision.tag)
            res = model.probe(ctx)
            if GENERATE_ERROR_MARK in (res.raw or ''):
                raise RuntimeError('probe generation failed; no scoreable prediction written')
            logs.append(dict(time_sec=t, fired=bool(res.fired), n_frames=len(frames),
                             last_frame_sec=max(f.time_sec for f in frames)))
            if res.fired:
                emits.append(dict(time_sec=t, reply_text=res.reply_text, response_type=res.response_type, raw=res.raw))
    rec['emits'] = emits
    rec['meta'].update(n_probes=len(logs), probe_log=logs, duration_sec=duration,
                       scan_until_sec=duration, origin_sec=0.0, query_time_sec=it.query_time_sec,
                       query_audio_policy='native_or_annotated_text',
                       independent_probes=not native, mock=False)
    return rec

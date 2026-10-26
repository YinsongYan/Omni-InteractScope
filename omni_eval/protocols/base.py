"""Shared protocol plumbing: per-interaction input prep (incl. modality adaptation) and the
standardized prediction record consumed by scoring."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..adaptation.asr import ASR, TranscriptSegment, format_transcript
from ..adaptation.input_config import resolve_input_config
from ..adaptation.modality import ModalityDecision
from ..data.media import MediaSource
from ..models.base import Capabilities
from ..schema import Interaction


@dataclass
class PreparedInputs:
    query_text: str = ""
    query_image: Optional[str] = None
    query_audio: Optional[str] = None
    transcript_segments: List[TranscriptSegment] = field(default_factory=list)  # offline track ASR
    query_audio_context: str = ""  # ASR of the audio query (online path)


def media_origin_sec(it: Interaction) -> float:
    """Start of the cumulative media context for probing.

    long_memory=yes -> earliest long_memory_start, or the clip start when the annotation omits
    it (the release data leaves ``long_memory_start`` null on all 637 long-memory responses, and
    starting at the query time there would throw away exactly the history the item tests).
    Otherwise the query time, else 0.
    """
    long_memory = [r for r in it.responses if r.long_memory == "yes"]
    starts = [r.long_memory_start_sec for r in long_memory if r.long_memory_start_sec is not None]
    if starts:
        return max(0.0, min(starts))
    if long_memory:
        return 0.0
    return max(0.0, it.query_time_sec or 0.0)


def track_asr_window(it: Interaction, cfg: Dict[str, Any]) -> tuple:
    """The ``[start, end]`` span of the audio track that cell ``v+asr`` transcribes.

    Factored out of :func:`prepare_inputs` so ``scripts/build_asr_transcripts.py`` derives the
    *same* span — the ASR disk cache is keyed on it, so a second definition here would mean
    every precomputed transcript missed at run time.
    """
    if it.task_type == "impairment_assistance" and cfg.get("t8", {}).get("full_horizon"):
        return 0.0, it.video_duration_sec
    origin = media_origin_sec(it)
    end = 0.0
    for r in it.responses:
        if r.notify_trigger_sec is not None:
            end = max(end, r.notify_trigger_sec)
        if r.valid_response_window_sec is not None:
            end = max(end, r.valid_response_window_sec[1])
    from ..config import get

    return origin, end + get(cfg, "offline_probing.grid_span_sec", 4.0)


def query_audio_path(it: Interaction, cfg: Dict[str, Any]) -> str:
    """Absolute path of the audio-query attachment this item feeds, or "" when it has none."""
    from ..config import get

    if it.query_type != "audio" or not it.query_audio:
        return ""
    audio_dir = get(cfg, "paths.audio", "data/audio")
    return it.query_audio if os.path.isabs(it.query_audio) else os.path.join(audio_dir, it.query_audio)


def prepare_inputs(
    it: Interaction,
    caps: Capabilities,
    decision: ModalityDecision,
    media: MediaSource,
    cfg: Dict[str, Any],
    asr: Optional[ASR] = None,
    for_online: bool = False,
) -> PreparedInputs:
    from ..config import get

    ic = resolve_input_config(decision.input_config or None)
    images_dir = get(cfg, "paths.images", "data/images")
    out = PreparedInputs(query_text=it.query_text)
    t8 = it.task_type == "impairment_assistance"
    if t8:
        from ..t8.policy import request_text
        out.query_text = request_text(it, native=for_online)

    # image query
    if it.query_type == "text_image" and it.query_image:
        p = it.query_image if os.path.isabs(it.query_image) else os.path.join(images_dir, it.query_image)
        out.query_image = p

    # audio query. Only cell ``va`` feeds it natively; ``v`` and ``v+asr`` transcribe it for
    # every model (omni included) so the spoken instruction still reaches the model and the
    # cells stay comparable across model families.
    apath = query_audio_path(it, cfg)
    if apath:
        if ic.query_audio_native and "audio" in caps.modalities_in:
            out.query_audio = apath  # feed natively
        elif t8 and get(cfg, "t8.query_audio_policy") == "native_or_annotated_text":
            if not it.query_text:
                raise ValueError("T8 annotated query text missing")
            # The text is already supplied to every model, including audio-native models.
            # Explicitly version this policy; do not pretend it is a machine ASR transcript.
        elif asr is not None:  # ASR-substitute the spoken query into text
            txt = asr.transcribe_file(apath, language=it.language)
            if txt:
                out.query_audio_context = f'User said (transcribed): "{txt}"'
                if not out.query_text:
                    out.query_text = txt

    # audio-triggered track transcript (Offline Probing only; see README on the online limitation)
    if (decision.use_asr or (t8 and ic.track_asr)) and asr is not None and not for_online:
        start, end = (0.0, media.duration) if t8 else track_asr_window(it, cfg)
        out.transcript_segments = asr.transcribe_timeline(media, start, end, language=it.language)

    return out


def transcript_up_to(segs: List[TranscriptSegment], t_sec: float) -> str:
    """Format only the segments that have **ended** by ``t_sec``.

    Gating on ``end_sec``, not ``start_sec``: a segment spanning ``[12, 27]`` contains words
    spoken at t=26, so revealing it at t=12 hands the model up to a whole segment of *future*
    audio — and for an audio-triggered item the trigger sound is exactly what the segment
    containing it gives away. With ``tolerance_sec = 2.0`` a leak of that size decides the score.

    The cost is lag: the transcript trails the probe by up to one segment length, so the ASR
    segmentation granularity (``modality.asr_window_sec`` for Qwen3-ASR) is what bounds how
    fresh the text context can be. See ``adaptation/asr._qwen3_timeline``.
    """
    return format_transcript([s for s in segs if s.end_sec <= t_sec])


def new_record(it: Interaction, protocol: str, model_name: str, decision: ModalityDecision) -> Dict[str, Any]:
    return {
        "interaction_id": it.interaction_id,
        "task_type": it.task_type,
        "sub_task_type": it.sub_task_type,
        "protocol": protocol,
        "model": model_name,
        "modality_tag": decision.tag,
        "supported": decision.supported,
        "input_config": decision.input_config,
        "audio_starved": decision.audio_starved,
        "emits": [],
        "meta": {},
    }

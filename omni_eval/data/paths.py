"""Resolve media references in the source annotation formats. Release inference validates all files explicitly."""
from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

from ..schema import Interaction
from ..utils.io import read_json
from .media import MediaSource

VIDEO_EXTS = (".mp4", ".mkv", ".mov", ".webm", ".avi", ".flv", ".m4v")
AUDIO_EXTS = (".m4a", ".mp3", ".wav", ".aac", ".flac", ".ogg")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def load_video_meta(path: str) -> Dict[str, dict]:
    if path and os.path.exists(path):
        return read_json(path)
    return {}


def _find_with_exts(directory: str, name: str, exts: Tuple[str, ...]) -> str:
    """Locate ``name`` in ``directory``, tolerating a missing/different extension."""
    if not name:
        return ""
    direct = os.path.join(directory, name)
    if os.path.exists(direct):
        return direct
    stem = os.path.splitext(os.path.basename(name))[0]
    for ext in exts:
        cand = os.path.join(directory, stem + ext)
        if os.path.exists(cand):
            return cand
    return ""


def resolve_video_path(
    it: Interaction, videos_dir: str, video_meta: Optional[Dict[str, dict]] = None
) -> str:
    """Local path for an interaction's clip.

    Tolerant on purpose: the release annotations name videos inconsistently (``video_name`` is
    often extension-less), so an exact hit, a stem match against known video extensions, and a
    ``video_meta.json`` entry are all tried before giving up. The non-existent join is still
    returned last so ``MediaSource`` can fall back to synthetic media.
    """
    cand = it.video or it.video_name or ""
    if not cand:
        return ""
    if os.path.isabs(cand) or os.path.exists(cand):
        return cand
    hit = _find_with_exts(videos_dir, cand, VIDEO_EXTS)
    if hit:
        return hit
    stem = os.path.splitext(os.path.basename(cand))[0]
    for key in (cand, stem, it.video_key):
        meta = (video_meta or {}).get(key) if key else None
        if meta and meta.get("path") and os.path.exists(meta["path"]):
            return str(meta["path"])
    if it.video_key:
        hit = _find_with_exts(videos_dir, it.video_key, VIDEO_EXTS)
        if hit:
            return hit
    return os.path.join(videos_dir, cand)


def resolve_attachment_path(name: str, directory: str, kind: str = "audio") -> str:
    """Local path for a query attachment (``query_audio`` / ``query_image``)."""
    if not name:
        return ""
    exts = AUDIO_EXTS if kind == "audio" else IMAGE_EXTS
    return _find_with_exts(directory, name, exts) or os.path.join(directory, name)


def _meta_for(it: Interaction, path: str, video_meta: Optional[Dict[str, dict]]) -> dict:
    vm = video_meta or {}
    for key in (it.video, os.path.basename(path), it.video_key, it.video_name):
        if key and key in vm:
            return vm[key] or {}
    return {}


def make_media_source(
    it: Interaction,
    videos_dir: str,
    video_meta: Optional[Dict[str, dict]] = None,
    frame_cache_dir: str = "",
) -> MediaSource:
    """Build the frame/audio accessor for one interaction.

    Duration priority is **probed > annotated**: a probed value (``video_meta.json`` or decord)
    describes the actual file, while the annotated ``source_end_ms - source_start_ms`` is the
    fallback that lets the protocols bound their scan window even with no video downloaded.

    ``frame_cache_dir`` (optional) turns on an on-disk JPEG frame cache, worth it when several
    models/input cells probe the same grid; see docs/throughput.md.
    """
    path = resolve_video_path(it, videos_dir, video_meta)
    meta = _meta_for(it, path, video_meta)
    media = MediaSource(
        path,
        duration=meta.get("duration"),
        fps=meta.get("fps") or it.video_fps,
        frame_cache_dir=frame_cache_dir,
    )
    if not media.duration and it.video_duration_sec:
        media.duration = float(it.video_duration_sec)
    return media


def interaction_duration(
    it: Interaction,
    media: Optional[MediaSource] = None,
    video_meta: Optional[Dict[str, dict]] = None,
) -> float:
    """Best-effort clip duration in seconds for bounding the Online Rolling loop.

    Priority: media (decoded/probed) -> video_meta -> annotated clip length -> last GT + margin.
    ``media=None`` answers from the annotations alone (no file needed).
    """
    if media is not None and media.duration:
        return media.duration
    meta = _meta_for(it, media.path if media is not None else "", video_meta)
    if meta.get("duration"):
        return float(meta["duration"])
    if it.video_duration_sec and it.video_duration_sec > 0:
        return float(it.video_duration_sec)
    last = 0.0
    for r in it.responses:
        if r.notify_trigger_sec is not None:
            last = max(last, r.notify_trigger_sec)
        if r.valid_response_window_sec is not None:
            last = max(last, r.valid_response_window_sec[1])
    return last + 5.0

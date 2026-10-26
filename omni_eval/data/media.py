"""Media access: frame sampling + audio slicing, with a synthetic fallback.

Real decoding uses ``decord`` (video) and ``ffmpeg`` (audio → 16 kHz mono). When the media
file or those libraries are unavailable, a **synthetic** source is returned: frames carry
their timestamp but ``image=None`` and ``synthetic=True``. This lets the protocol drivers,
dummy models, and the whole test-suite run end-to-end without any real videos.

Annotation timestamps are seconds throughout (already converted from ms in the loader). fps
here is used only for decoding / sampling, never to interpret annotation times.
"""
from __future__ import annotations

import json
import os
import subprocess
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

_DEFAULT_FPS = 30.0
# Frames kept in memory per MediaSource. An Offline-Probing grid re-reads the same early frames
# at every probe (cumulative context ``[origin, t_p]``), so a small cache covers the overlap;
# at 720p a frame is ~2.7 MB, hence the deliberately low default.
_FRAME_CACHE_SIZE = 32


@dataclass
class VideoMeta:
    video_id: str
    path: str
    fps: float = _DEFAULT_FPS
    duration: float = 0.0
    n_frames: int = 0
    exists: bool = False


@dataclass
class Frame:
    time_sec: float
    image: Optional[object] = None      # PIL.Image / np.ndarray / None (synthetic)
    synthetic: bool = False


def ffprobe_meta(path: str) -> Optional[VideoMeta]:
    """Read fps/duration via ffprobe. Returns None if ffprobe or the file is unavailable."""
    if not os.path.exists(path):
        return None
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate,duration,nb_frames",
                "-show_entries", "format=duration",
                "-of", "json", path,
            ],
            capture_output=True, text=True, timeout=60,
        )
        info = json.loads(out.stdout or "{}")
        stream = (info.get("streams") or [{}])[0]
        rate = stream.get("r_frame_rate", "30/1")
        num, _, den = rate.partition("/")
        fps = float(num) / float(den or 1) if den else float(num)
        duration = float(stream.get("duration") or info.get("format", {}).get("duration") or 0.0)
        nb = stream.get("nb_frames")
        n_frames = int(nb) if nb and str(nb).isdigit() else int(duration * fps)
        return VideoMeta(os.path.basename(path), path, fps or _DEFAULT_FPS, duration, n_frames, True)
    except Exception:
        return None


def get_video_duration(path: str) -> float:
    """Video duration in seconds (0.0 if the file / ffprobe is unavailable)."""
    meta = ffprobe_meta(path)
    return meta.duration if meta else 0.0


class MediaSource:
    """Frame + audio access for one interaction's video.

    ``duration`` may be provided (from a prebuilt video_meta.json) to avoid re-probing.
    """

    def __init__(
        self,
        path: str,
        duration: Optional[float] = None,
        fps: Optional[float] = None,
        cache_frames: int = _FRAME_CACHE_SIZE,
        frame_cache_dir: str = "",
        audio_only: bool = False,
    ):
        self.path = path
        self.exists = bool(path) and os.path.exists(path)
        self._reader = None
        self._decord_fps = fps or _DEFAULT_FPS
        self.duration = duration or 0.0
        # index -> decoded frame; bounded LRU, shared by frame_at/sample_frames
        self._cache: "OrderedDict[int, object]" = OrderedDict()
        self._cache_size = max(0, int(cache_frames))
        self._disk_dir = ""
        if frame_cache_dir and self.exists:
            stem = os.path.splitext(os.path.basename(path))[0]
            self._disk_dir = os.path.join(frame_cache_dir, stem)
        self.cache_hits = 0
        self.cache_misses = 0
        # ``audio_only``: skip the decord open. The ASR precompute reads audio through ffmpeg and
        # never touches a frame, and opening a VideoReader per clip over NFS dominated its
        # runtime. ``decodable`` then stays False, which is honest — this source cannot decode.
        if self.exists and not audio_only:
            self._try_open()

    def _try_open(self) -> None:
        try:
            import decord  # type: ignore

            self._reader = decord.VideoReader(self.path)
            self._decord_fps = float(self._reader.get_avg_fps()) or self._decord_fps
            if not self.duration:
                self.duration = len(self._reader) / self._decord_fps
        except Exception:
            self._reader = None  # fall back to synthetic

    @property
    def decodable(self) -> bool:
        """True when frames come from the real file.

        Strictly stronger than ``exists``: a file that is present but truncated, or in a codec the
        installed decoder cannot open, leaves ``exists=True`` while every ``frame_at`` still returns
        a synthetic ``image=None``. Callers that must not silently score synthetic media (see
        ``run_inference``'s ``--allow-synthetic``) have to test this, not ``exists``.
        """
        return self._reader is not None

    # ---- frames -----------------------------------------------------------------
    def _remember(self, idx: int, img) -> None:
        if self._cache_size <= 0:
            return
        self._cache[idx] = img
        self._cache.move_to_end(idx)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    def _disk_path(self, idx: int) -> str:
        return os.path.join(self._disk_dir, f"{idx:08d}.jpg") if self._disk_dir else ""

    def _read_disk(self, idx: int):
        p = self._disk_path(idx)
        if not p or not os.path.exists(p):
            return None
        try:
            from PIL import Image  # type: ignore

            return np.asarray(Image.open(p).convert("RGB"))
        except Exception:
            return None

    def _write_disk(self, idx: int, img) -> None:
        p = self._disk_path(idx)
        if not p or os.path.exists(p):
            return
        try:
            from PIL import Image  # type: ignore

            os.makedirs(self._disk_dir, exist_ok=True)
            Image.fromarray(np.asarray(img)).save(p, quality=92)
        except Exception:
            pass  # a cold cache is only a speed problem, never a correctness one

    def _decode(self, idx: int):
        """One frame by index, through the in-memory LRU then the optional on-disk cache."""
        if idx in self._cache:
            self._cache.move_to_end(idx)
            self.cache_hits += 1
            return self._cache[idx]
        img = self._read_disk(idx)
        if img is not None:
            self.cache_hits += 1
            self._remember(idx, img)
            return img
        img = self._reader[idx].asnumpy()
        self.cache_misses += 1
        self._remember(idx, img)
        self._write_disk(idx, img)
        return img

    def frame_at(self, t_sec: float) -> Frame:
        if self._reader is not None:
            try:
                idx = int(round(t_sec * self._decord_fps))
                idx = max(0, min(idx, len(self._reader) - 1))
                return Frame(time_sec=t_sec, image=self._decode(idx), synthetic=False)
            except Exception:
                pass
        return Frame(time_sec=t_sec, image=None, synthetic=True)

    def sample_frames(
        self,
        start_sec: float,
        end_sec: float,
        max_frames: int = 32,
        sample_fps: float = 1.0,
    ) -> List[Frame]:
        """Uniformly sample frames from [start_sec, end_sec] (inclusive), capped at max_frames."""
        start_sec = max(0.0, start_sec)
        if end_sec <= start_sec:
            return [self.frame_at(start_sec)]
        n = int((end_sec - start_sec) * sample_fps) + 1
        n = max(1, min(n, max_frames))
        times = np.linspace(start_sec, end_sec, n)
        return [self.frame_at(float(t)) for t in times]

    # ---- audio ------------------------------------------------------------------
    def audio_chunk(self, t_sec: float, dur: float = 1.0, sr: int = 16000) -> Optional[np.ndarray]:
        """Return a mono 16 kHz chunk [t, t+dur) as float32, or None if unavailable."""
        if not self.exists:
            return None
        try:
            out = subprocess.run(
                [
                    "ffmpeg", "-v", "error", "-ss", f"{t_sec:.3f}", "-t", f"{dur:.3f}",
                    "-i", self.path, "-f", "f32le", "-ac", "1", "-ar", str(sr), "pipe:1",
                ],
                capture_output=True, timeout=60,
            )
            if out.returncode != 0 or not out.stdout:
                return None
            return np.frombuffer(out.stdout, dtype=np.float32).copy()
        except Exception:
            return None


def decode_audio_file(path: str, sr: int = 16000) -> Optional[np.ndarray]:
    """Decode any audio/video file to a mono float32 waveform at ``sr``, or None on failure.

    Module-level because audio-query attachments (``data/audio/*.m4a|mp3``) are standalone files,
    not a :class:`MediaSource`.
    """
    if not path or not os.path.exists(path):
        return None
    try:
        out = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", "1", "-ar", str(sr), "pipe:1"],
            capture_output=True, timeout=300,
        )
        if out.returncode != 0 or not out.stdout:
            return None
        return np.frombuffer(out.stdout, dtype=np.float32).copy()
    except Exception:
        return None

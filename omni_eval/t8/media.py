"""PTS-based causal video sampling and sample-accurate audio prefixes."""
import bisect
from fractions import Fraction
import json
import subprocess
import numpy as np
from ..data.media import MediaSource, Frame


def video_start_offset(path):
    """Restore the stream origin omitted by decord 0.6.0 timestamps.

    Audio already uses the container clock (-copyts). Keep that clock and the
    annotations unchanged. Source time-base integers avoid ffprobe's decimal
    start_time rounding. Decord's existing float32 precision is retained.
    """
    probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=start_pts,time_base', '-of', 'json', str(path)],
        capture_output=True, text=True, timeout=30, check=True)
    stream, = json.loads(probe.stdout)['streams']
    return float(int(stream['start_pts']) * Fraction(stream['time_base']))


class T8Media(MediaSource):
    def _try_open(self):
        import decord
        # Bound decoder threads per episode when API review workers run concurrently.
        self._reader = decord.VideoReader(self.path, num_threads=2)
        self._decord_fps = float(self._reader.get_avg_fps())

    def __init__(self, path, lossless_cache_dir='', **kwargs):
        self.lossless_cache_dir = lossless_cache_dir
        self._lossless = None
        super().__init__(path, **kwargs)
        if not self.decodable:
            raise ValueError('T8 requires decodable real video')
        # Average FPS does not locate frames reliably in VFR clips. Never round a future
        # frame down to the requested clock: select the latest actual PTS <= that clock.
        self.pts = np.asarray(self._reader.get_frame_timestamp(range(len(self._reader))), dtype=np.float64)[:, 0]
        self.video_start_offset_sec = video_start_offset(path)
        # A zero offset is deliberately an exact no-op, permitting checked reuse
        # of unchanged episodes. Positive offsets must not expose the first
        # video frame at zero while the audio clock still starts at zero.
        if self.video_start_offset_sec:
            self.pts += self.video_start_offset_sec
        if np.any(np.diff(self.pts) < 0):
            raise ValueError('non-monotone frame PTS')
        # Keep source timestamps; pad/trim to a clip-relative zero. A plain decode
        # discards stream start offsets and can expose up to 21 ms of future audio.
        decoded = subprocess.run(['ffmpeg', '-v', 'error', '-copyts', '-i', path,
            '-map', '0:a:0', '-af', 'aresample=16000:async=1:first_pts=0',
            '-ac', '1', '-f', 'f32le', 'pipe:1'], capture_output=True, timeout=90, check=True)
        self._waveform = np.frombuffer(decoded.stdout, dtype=np.float32)
        if not len(self._waveform):
            raise ValueError('T8 requires a decodable original audio track')
        self.duration = float(self._reader.get_frame_timestamp(len(self._reader) - 1)[1])
        if self.video_start_offset_sec:
            self.duration += self.video_start_offset_sec

    def prepare_clock(self, grid, native=False):
        """Decode the exact union of selected indices once, with no sampling changes."""
        if not self.lossless_cache_dir:
            return
        from .frame_cache import LosslessFrameCache
        times = []
        for end in grid:
            start = max(0.0, float(self.pts[0]))
            if native or end <= start:
                times.append(end if native else start)
            else:
                n = max(1, min(int(end - start) + 1, 32))
                times.extend(np.linspace(start, end, n))
        indices = {bisect.bisect_right(self.pts, float(t) + 1e-9) - 1 for t in times}
        if min(indices) < 0:
            raise ValueError('cache plan requests a synthetic frame')
        shape = self._reader[0].shape
        self._lossless = LosslessFrameCache(self.lossless_cache_dir, self.path, self.pts)
        # Free the random-access decoder before creating the sequential decoder.
        self._reader = None
        self._lossless.prepare(indices, self.path, self.pts, shape)

    def _decode(self, idx):
        if self._lossless is None:
            return super()._decode(idx)
        if idx in self._cache:
            self._cache.move_to_end(idx)
            self.cache_hits += 1
            return self._cache[idx]
        rgb = self._lossless.read(idx, self.pts[idx])
        if rgb is None:
            raise ValueError('frame absent from exact cache plan')
        self.cache_misses += 1
        self._remember(idx, rgb)
        return rgb

    def close(self):
        self._reader = None
        self._cache.clear()
        if self._lossless is not None:
            self._lossless.close()

    def frame_at(self, t_sec):
        idx = bisect.bisect_right(self.pts, t_sec + 1e-9) - 1
        if idx < 0:
            return Frame(time_sec=t_sec, image=None, synthetic=True)
        return Frame(time_sec=float(self.pts[idx]), image=self._decode(idx), synthetic=False)

    def sample_frames(self, start_sec, end_sec, max_frames=32, sample_fps=1.0):
        # A clip can start a few milliseconds after zero. Never request a synthetic
        # zero frame as part of every later prefix of such a clip.
        start_sec = max(float(self.pts[0]), start_sec)
        return super().sample_frames(start_sec, end_sec, max_frames, sample_fps)

    def audio_chunk(self, t_sec, dur=1.0, sr=16000):
        if sr != 16000:
            raise ValueError('T8 waveform clock is 16000 Hz')
        lo = max(0, int(t_sec * sr))
        hi = max(lo, int((t_sec + dur) * sr))
        return self._waveform[lo:hi].copy()

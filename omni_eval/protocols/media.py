"""Causal PTS video with lazy, sample-aligned original audio."""
import subprocess
import numpy as np
from ..t8.media import T8Media, video_start_offset
from ..data.media import MediaSource


class CausalMedia(T8Media):
    def __init__(self, path, **kwargs):
        self.lossless_cache_dir = kwargs.pop('frame_cache_dir', '')
        self._lossless = None
        MediaSource.__init__(self, path, **kwargs)
        if not self.decodable:
            raise ValueError('Undecodable source video')
        self.pts = np.asarray(self._reader.get_frame_timestamp(range(len(self._reader))), dtype=np.float64)[:, 0]
        self.video_start_offset_sec = video_start_offset(path)
        if self.video_start_offset_sec:
            self.pts += self.video_start_offset_sec
        if np.any(np.diff(self.pts) < 0):
            raise ValueError('Nonmonotone PTS')
        self.duration = float(self._reader.get_frame_timestamp(len(self._reader) - 1)[1])
        self.duration += self.video_start_offset_sec
        self._waveform = None

    def audio_chunk(self, t_sec, dur=1.0, sr=16000):
        if self._waveform is None:
            result = subprocess.run(['ffmpeg', '-v', 'error', '-copyts', '-i', self.path,
                '-map', '0:a:0', '-af', 'aresample=16000:async=1:first_pts=0',
                '-ac', '1', '-f', 'f32le', 'pipe:1'], capture_output=True, timeout=90, check=True)
            self._waveform = np.frombuffer(result.stdout, dtype=np.float32)
            if not len(self._waveform):
                raise ValueError('Empty native audio')
        return super().audio_chunk(t_sec, dur, sr)

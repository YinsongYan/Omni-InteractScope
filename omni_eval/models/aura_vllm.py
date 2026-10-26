"""Pinned AURA native silent-token client, using causally cut 1-second videos.

The official streaming benchmark format is retained: up to 30 video turns,
silent assistant placeholders, a query on the final video, temperature zero.
This adapter never reads reference answers or trigger timestamps. Benchmark
drivers determine the clock; errors raise instead of becoming model silence.
"""
from __future__ import annotations

import base64
from collections import OrderedDict
from fractions import Fraction
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import time
from urllib.parse import urlparse

from .base import Capabilities, Emit, ProbeContext, SessionSpec, StreamingModel
from .prompted_base import tool_disclosure
from ..data.response_parser import infer_response_type, strip_reasoning

SYSTEM_PROMPT = (
    "You are receiving a live video stream where the final frame is the present moment. "
    "Respond only when a response is needed based on the user's message or the visual context. "
    "Otherwise, output `<|silent|>` to signify silence."
)
SILENT = '<|silent|>'
VERSION = 'aura-causal-chunks-v3-official-query'


def chunk_bounds(t, max_segments=30):
    if not math.isfinite(t) or t < 0:
        raise ValueError('invalid observation timestamp')
    if not isinstance(max_segments, int) or max_segments < 1:
        raise ValueError('max_segments must be a positive integer')
    n = min(max_segments, int(math.floor(t)))
    if n < 1:
        return [(0.0, t)]
    return [(round(t - n + i, 6), round(t - n + i + 1, 6)) for i in range(n)]


def spoken_text(raw):
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError('empty generation is not a native silence token')
    text = strip_reasoning(raw).replace(SILENT, '').strip()
    text = re.sub(r'<\|(?:im_start|im_end|endoftext)\|>', '', text).strip()
    if not text and SILENT not in raw:
        raise ValueError('generation contains neither speech nor native silence')
    return text


class CausalChunks:
    """Seek/decode with source PTS filtering BEFORE encoding or model input.

    Cache identities include source metadata and exact endpoints. Short/zero
    prefixes duplicate only an already visible frame, never a future frame.
    Every encoded frame's original timestamp is retained in the sidecar.
    """
    def __init__(self, source, cache_dir, max_pixels):
        import av
        self.av = av
        self.path = Path(source).resolve()
        st = self.path.stat()
        key = json.dumps([VERSION, str(self.path), st.st_size, st.st_mtime_ns, max_pixels])
        self.root = Path(cache_dir) / hashlib.sha256(key.encode()).hexdigest()
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_pixels = max_pixels
        self.reader = av.open(str(self.path))
        self.stream = self.reader.streams.video[0]
        self.stream.thread_type = 'NONE'
        self.stream.codec_context.thread_count = 1
        self._cache = OrderedDict()

    def close(self):
        self.reader.close()

    def get(self, lo, hi):
        # Independent interaction shards can share source clips and their cache.
        with (self.root / '.cache.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            return self._get_locked(lo, hi)

    def _get_locked(self, lo, hi):
        if not (math.isfinite(lo) and math.isfinite(hi) and 0 <= lo <= hi):
            raise ValueError('invalid chunk bounds')
        key = f'{lo:.6f}_{hi:.6f}'
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        p = self.root / (key + '.mp4')
        meta_path = p.with_suffix('.json')
        if p.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta['source_max_pts'] > hi + 1e-9 or meta['source_min_pts'] < lo - 1e-9:
                raise ValueError('cached video chunk violates source PTS bounds')
            b = p.read_bytes()
            if hashlib.sha256(b).hexdigest() != meta['sha256']:
                raise ValueError('cached video integrity failure')
        else:
            self.reader.seek(max(0, int(lo / self.stream.time_base)), stream=self.stream, backward=True)
            frames, pts = [], []
            for fr in self.reader.decode(self.stream):
                if fr.pts is None:
                    raise ValueError('source video frame has no PTS')
                t = float(fr.pts * fr.time_base)
                if t > hi + 1e-9:
                    break
                if t >= lo - 1e-9:
                    frames.append(fr)
                    pts.append(t)
            if not frames:
                raise ValueError(f'no causal video frame in [{lo}, {hi}]')
            # Retain original image sequence. Spatial cap matches the Qwen/MOSS
            # baseline setting and is separately reported from official 602112.
            w, h = frames[0].width, frames[0].height
            scale = min(1.0, math.sqrt(self.max_pixels / (w * h)))
            w, h = max(2, int(w * scale) // 2 * 2), max(2, int(h * scale) // 2 * 2)
            tmp = p.with_suffix('.partial.mp4')
            with self.av.open(str(tmp), 'w') as enc:
                rate = self.stream.average_rate or Fraction(30, 1)
                video = enc.add_stream('libx264', rate=rate)
                video.width, video.height, video.pix_fmt = w, h, 'yuv420p'
                video.codec_context.thread_count = 1
                video.options = {'crf': '18', 'preset': 'ultrafast'}
                # Preserve source frame spacing, including variable frame rate.
                # A singleton is repeated only to satisfy the video decoder.
                video.time_base = Fraction(1, 1000000)
                video.codec_context.time_base = Fraction(1, 1000000)
                encoded = frames if len(frames) > 1 else frames * 2
                output_pts = ([round((t - pts[0]) * 1000000) for t in pts]
                              if len(pts) > 1 else [0, round(1000000 / float(rate))])
                for fr, timestamp in zip(encoded, output_pts):
                    # Make an owned frame: reformat can return the source itself.
                    f = self.av.VideoFrame.from_ndarray(fr.to_ndarray(format='rgb24'), format='rgb24')
                    f = f.reformat(width=w, height=h, format='yuv420p')
                    f.pts = timestamp
                    f.time_base = Fraction(1, 1000000)
                    for packet in video.encode(f):
                        enc.mux(packet)
                for packet in video.encode():
                    enc.mux(packet)
            b = tmp.read_bytes()
            meta = dict(source_min_pts=min(pts), source_max_pts=max(pts),
                        source_pts=pts, requested_bounds=[lo, hi],
                        sha256=hashlib.sha256(b).hexdigest(), encoded_frames=len(encoded),
                        size=[w, h], output_pts_us=output_pts, version=VERSION)
            tmp.replace(p)
            meta_path.write_text(json.dumps(meta))
        result = ('data:video/mp4;base64,' + base64.b64encode(b).decode(), meta)
        self._cache[key] = result
        while len(self._cache) > 64:
            self._cache.popitem(last=False)
        return result


class AURAModel(StreamingModel):
    capabilities = Capabilities(modalities_in={'text', 'image', 'video'},
                                supports_streaming=True, native_streaming=True, native_fps=1.0)

    def __init__(self, base_url='http://127.0.0.1:8028/v1', served_model='AURA',
                 max_segments=30, max_tokens=512, max_pixels=151200,
                 chunk_cache_dir='../chunk_cache', revision='', **_):
        if urlparse(base_url).hostname not in ('localhost', '127.0.0.1', '::1'):
            raise ValueError('AURA client is restricted to the local model server')
        from openai import OpenAI
        self.client = OpenAI(api_key='EMPTY', base_url=base_url, timeout=3600, max_retries=2)
        self.served_model = served_model
        self.max_segments, self.max_tokens, self.max_pixels = max_segments, max_tokens, max_pixels
        self.chunk_cache_dir, self.revision = chunk_cache_dir, revision
        self._chunks = None
        self._stats = {}

    def name(self):
        return 'aura'

    def probe(self, ctx: ProbeContext):
        raise NotImplementedError('AURA must use its native streaming protocol')

    def begin(self, spec: SessionSpec):
        if not spec.video_path:
            raise ValueError('AURA requires a real source video')
        if spec.query_audio:
            raise ValueError('AURA requires the benchmark ASR/text query path')
        self._spec = spec
        self._chunks = CausalChunks(spec.video_path, self.chunk_cache_dir, self.max_pixels)
        self._last_t = -1.0
        self._stats = dict(n_generate_errors=0, n_calls=0, n_silent=0, chunk_trace=[],
                           method=VERSION, checkpoint_revision=self.revision,
                           max_segments=self.max_segments, max_pixels=self.max_pixels)

    def messages(self, chunks):
        system = SYSTEM_PROMPT + '\n\n' + tool_disclosure()
        messages = [{'role': 'system', 'content': system}]
        # Official AURA StreamingBench/OVO runners attach the request to the
        # latest video turn. Do not synthesize an acknowledgement before the
        # stream: that external infer_e format suppressed even a positive
        # non-benchmark reading control. Reapply the same visible request at
        # each decision tick; this is explicitly a per-prefix adaptation.
        for url, _ in chunks[:-1]:
            messages.append({'role': 'user', 'content': [{'type': 'video_url', 'video_url': {'url': url}}]})
            messages.append({'role': 'assistant', 'content': SILENT})
        content = [{'type': 'video_url', 'video_url': {'url': chunks[-1][0]}}]
        if self._spec.query_image:
            p = Path(self._spec.query_image)
            mime = 'image/png' if p.suffix.lower() == '.png' else 'image/jpeg'
            content.append({'type': 'image_url', 'image_url': {
                'url': f'data:{mime};base64,' + base64.b64encode(p.read_bytes()).decode()}})
        content.append({'type': 'text', 'text': self._spec.query_text or 'Monitor the video and assist when needed.'})
        messages.append({'role': 'user', 'content': content})
        return messages

    def observe(self, frame, t_sec, audio_chunk=None):
        if t_sec < self._last_t:
            raise ValueError('nonmonotone observation clock')
        self._last_t = t_sec
        try:
            chunks = [self._chunks.get(lo, hi) for lo, hi in chunk_bounds(t_sec, self.max_segments)]
            start = time.monotonic()
            res = self.client.chat.completions.create(
                model=self.served_model, messages=self.messages(chunks),
                temperature=0, max_tokens=self.max_tokens,
                extra_body={'skip_special_tokens': False, 'mm_processor_kwargs': {
                    'fps': 1.0, 'size': {'shortest_edge': 4096, 'longest_edge': self.max_pixels}}})
            choice = res.choices[0]
            raw = choice.message.content
            text = spoken_text(raw)
            self._stats['n_calls'] += 1
            self._stats['n_silent'] += int(not text)
            usage = res.usage
            self._stats['chunk_trace'].append(dict(
                time_sec=t_sec, n_chunks=len(chunks),
                source_min_pts=min(m['source_min_pts'] for _, m in chunks),
                source_max_pts=max(m['source_max_pts'] for _, m in chunks),
                prompt_tokens=getattr(usage, 'prompt_tokens', None),
                completion_tokens=getattr(usage, 'completion_tokens', None),
                finish_reason=choice.finish_reason, wall_sec=time.monotonic() - start))
            if not text:
                return None
            return Emit(time_sec=t_sec, reply_text=text, raw=raw, response_type=infer_response_type(text))
        except Exception:
            self._stats['n_generate_errors'] += 1
            raise

    def end(self):
        if self._chunks:
            self._chunks.close()
            self._chunks = None

    def session_stats(self):
        return self._stats

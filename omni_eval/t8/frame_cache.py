"""Bounded, process-shared lossless RGB cache; original PTS remain authoritative.

Only requested source-frame indices are stored. The sequential decoder may read ahead,
but cached frames reach a model only through T8Media's causal frame selection.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np


def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, sort_keys=True))
    tmp.replace(path)


class LosslessFrameCache:
    def __init__(self, root, source, pts, max_bytes=256 * 1024**3):
        import decord
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self.identity = dict(source_sha256=file_sha(source),
            pts_sha256=hashlib.sha256(pts.tobytes()).hexdigest(),
            decoder='decord-' + decord.__version__, format='numpy-rgb-uint8-v1')
        key = hashlib.sha256(json.dumps(self.identity, sort_keys=True).encode()).hexdigest()
        self.path = self.root / key
        self.guard = (self.root / 'budget.lock').open('a+b')
        with self._budget_lock():
            self.path.mkdir(exist_ok=True)
            self.lease = (self.path / 'lease.lock').open('a+b')
            fcntl.flock(self.lease, fcntl.LOCK_SH)
        self.builder = (self.path / 'builder.lock').open('a+b')
        self.manifest = {}
        self.closed = False

    def _budget_lock(self):
        from contextlib import contextmanager
        @contextmanager
        def locked():
            fcntl.flock(self.guard, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(self.guard, fcntl.LOCK_UN)
        return locked()

    def _reserve(self, size):
        with self._budget_lock():
            paths = sorted((p for p in self.root.iterdir() if p.is_dir()),
                           key=lambda p: p.stat().st_mtime)
            sizes = {p: int((p / 'reserved').read_text()) if (p / 'reserved').exists() else 0
                     for p in paths}
            total = sum(sizes.values()) - sizes.get(self.path, 0) + size
            for path in paths:
                if total <= self.max_bytes and shutil.disk_usage(self.root).free > size + 20 * 1024**3:
                    break
                if path == self.path:
                    continue
                with (path / 'lease.lock').open('a+b') as lease:
                    try:
                        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        continue
                    shutil.rmtree(path)
                    total -= sizes[path]
            if total > self.max_bytes or shutil.disk_usage(self.root).free < size + 20 * 1024**3:
                raise RuntimeError('lossless frame cache budget exhausted by active videos')
            (self.path / 'reserved').write_text(str(size))

    def prepare(self, indices, source, pts, shape):
        import decord
        indices = sorted(set(map(int, indices)))
        fcntl.flock(self.builder, fcntl.LOCK_EX)
        try:
            path = self.path / 'manifest.json'
            doc = json.loads(path.read_text()) if path.exists() else dict(identity=self.identity, frames={})
            if doc['identity'] != self.identity:
                raise ValueError('lossless cache identity mismatch')
            frames = doc['frames']
            missing = [i for i in indices if str(i) not in frames]
            count = len(set(frames) | set(map(str, indices)))
            self._reserve(count * (int(np.prod(shape)) + 256))
            if missing:
                reader = decord.VideoReader(source, num_threads=2)
                previous = -1
                for idx in missing:
                    if idx > previous + 1:
                        reader.skip_frames(idx - previous - 1)
                    rgb = reader.next().asnumpy()
                    previous = idx
                    if rgb.dtype != np.uint8 or tuple(rgb.shape) != tuple(shape):
                        raise ValueError('unexpected decoded RGB format')
                    target = self.path / f'{idx}.npy'
                    temp = target.with_suffix('.tmp')
                    with temp.open('wb') as stream:
                        np.save(stream, rgb, allow_pickle=False)
                    temp.replace(target)
                    frames[str(idx)] = dict(pts=float(pts[idx]), shape=list(shape),
                        rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest())
                del reader
                atomic_json(path, doc)
            self.manifest = frames
            os.utime(self.path, None)
        finally:
            fcntl.flock(self.builder, fcntl.LOCK_UN)

    def read(self, idx, pts):
        record = self.manifest.get(str(idx))
        if record is None:
            return None
        if record['pts'] != float(pts):
            raise ValueError('lossless cache PTS mismatch')
        rgb = np.load(self.path / f'{idx}.npy', allow_pickle=False)
        if (rgb.dtype != np.uint8 or list(rgb.shape) != record['shape'] or
                hashlib.sha256(rgb.tobytes()).hexdigest() != record['rgb_sha256']):
            raise ValueError('lossless cache pixel integrity failure')
        return rgb

    def close(self):
        if not self.closed:
            self.closed = True
            self.builder.close()
            self.lease.close()
            self.guard.close()

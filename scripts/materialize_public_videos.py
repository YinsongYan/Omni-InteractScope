"""Restore the external platform videos listed in the current dataset manifest.

The self-recorded videos in the current Git LFS release are never replaced.
Uses only the standard library; run from any working directory.
"""
import argparse
import errno
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def materialize(source, dataset, *, checksums=False, copy=False, dry_run=False):
    source, dataset = Path(source).resolve(), Path(dataset).resolve()
    video_source = source/'videos' if (source/'videos').is_dir() else source
    manifest = dataset/'external_public_videos.jsonl'
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    jobs, seen = [], set()
    # Validate the complete plan before creating any destination files.
    for row in rows:
        rel = PurePosixPath(row['path'])
        if (len(rel.parts) != 2 or rel.parts[0] != 'videos' or rel.suffix != '.mp4'
                or rel.name in {'..', '.'} or row['path'] in seen):
            raise ValueError('Invalid/duplicate public-video path: ' + row['path'])
        seen.add(row['path'])
        src, dest = (video_source/rel.name).resolve(), dataset/row['path']
        if dest.is_symlink() or not dest.resolve().is_relative_to(dataset):
            raise ValueError('Destination must stay inside the dataset without external symlinks')
        candidate = dest if dest.exists() else src
        if not candidate.is_file() or candidate.stat().st_size != row['bytes']:
            raise ValueError('Missing or unexpected video size: ' + str(candidate))
        if checksums and sha256(candidate) != row['sha256']:
            raise ValueError('Checksum mismatch: ' + str(candidate))
        jobs.append((src, dest))
    stats = {'listed': len(jobs), 'existing': 0, 'linked': 0, 'copied': 0, 'planned': 0}
    for src, dest in jobs:
        if dest.exists():
            stats['existing'] += 1
            continue
        stats['planned'] += 1
        if dry_run:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not copy:
            try:
                os.link(src, dest)
                stats['linked'] += 1
                continue
            except OSError as e:
                if e.errno != errno.EXDEV:
                    raise
        # Copy completely before exposing the final filename; never overwrite a
        # file that another process created while this plan was being applied.
        fd, temporary = tempfile.mkstemp(prefix='.public-video-', dir=dest.parent)
        os.close(fd)
        try:
            shutil.copyfile(src, temporary)
            os.link(temporary, dest)
        finally:
            os.unlink(temporary)
        stats['copied'] += 1
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True,
                        help='Existing full dataset root or its videos directory')
    parser.add_argument('--dataset', type=Path, default=Path('data/Omni-InteractScope'))
    parser.add_argument('--checksums', action='store_true', help='Also verify SHA-256; reads all listed videos')
    parser.add_argument('--copy', action='store_true', help='Copy instead of hard-linking when on the same filesystem')
    parser.add_argument('--dry-run', action='store_true', help='Validate and report without writing files')
    args = parser.parse_args()
    print(json.dumps(materialize(args.source, args.dataset, checksums=args.checksums,
                                 copy=args.copy, dry_run=args.dry_run), indent=2))


if __name__ == '__main__':
    main()

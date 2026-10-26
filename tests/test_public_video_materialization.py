import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('materialize_public',
    Path(__file__).resolve().parents[1]/'scripts/materialize_public_videos.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def layout(tmp_path):
    source, dataset = tmp_path/'source', tmp_path/'dataset'
    (source/'videos').mkdir(parents=True)
    (dataset/'videos').mkdir(parents=True)
    payload = b'public fixture'
    (source/'videos/public.mp4').write_bytes(payload)
    row = {'path':'videos/public.mp4','bytes':len(payload),
           'sha256':hashlib.sha256(payload).hexdigest()}
    (dataset/'external_public_videos.jsonl').write_text(json.dumps(row)+'\n')
    return source, dataset


def test_restores_only_public_assets_and_preserves_processed_self_recording(tmp_path):
    source, dataset = layout(tmp_path)
    (source/'videos/self.mp4').write_bytes(b'old unprocessed self recording')
    (dataset/'videos/self.mp4').write_bytes(b'current processed self recording')
    result = mod.materialize(source, dataset, checksums=True)
    assert result['linked'] == 1
    assert (dataset/'videos/public.mp4').read_bytes() == b'public fixture'
    assert (dataset/'videos/self.mp4').read_bytes() == b'current processed self recording'
    assert mod.materialize(source, dataset)['existing'] == 1


def test_conflicting_destination_is_never_overwritten(tmp_path):
    source, dataset = layout(tmp_path)
    dest = dataset/'videos/public.mp4'
    dest.write_bytes(b'conflicting content')
    with pytest.raises(ValueError, match='unexpected video size'):
        mod.materialize(source, dataset)
    assert dest.read_bytes() == b'conflicting content'


def test_checksum_mode_rejects_same_length_corruption_before_copy(tmp_path):
    source, dataset = layout(tmp_path)
    (source/'videos/public.mp4').write_bytes(b'x'*len(b'public fixture'))
    with pytest.raises(ValueError, match='Checksum mismatch'):
        mod.materialize(source, dataset, checksums=True)
    assert not (dataset/'videos/public.mp4').exists()


def test_manifest_cannot_escape_dataset(tmp_path):
    source, dataset = layout(tmp_path)
    (dataset/'external_public_videos.jsonl').write_text(json.dumps(
        {'path':'videos/../../outside.mp4','bytes':1,'sha256':'0'*64})+'\n')
    with pytest.raises(ValueError, match='Invalid/duplicate'):
        mod.materialize(source, dataset)


def test_copy_and_dry_run(tmp_path):
    source, dataset = layout(tmp_path)
    assert mod.materialize(source, dataset, dry_run=True)['planned'] == 1
    assert not (dataset/'videos/public.mp4').exists()
    assert mod.materialize(source, dataset, copy=True)['copied'] == 1
    assert not (dataset/'videos/public.mp4').samefile(source/'videos/public.mp4')

from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from omni_eval.data.hub import download_dataset


def stub_snapshot(monkeypatch, snapshot):
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(snapshot)

    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(snapshot_download=download))
    return calls


def test_combined_snapshot_restores_dataset_without_code_or_nested_prefix(tmp_path, monkeypatch):
    snapshot = tmp_path / 'snapshot'
    source = snapshot / 'data' / 'Omni-InteractScope'
    (source / 'videos').mkdir(parents=True)
    (source / 'annotations.json').write_text('{}')
    (source / 'README.md').write_text('Dataset guide')
    (snapshot / 'README.md').write_text('Repository guide')
    blob = tmp_path / 'cached-video'
    blob.write_bytes(b'video payload')
    (source / 'videos' / 'example.mp4').symlink_to(blob)
    calls = stub_snapshot(monkeypatch, snapshot)

    destination = tmp_path / 'custom-data'
    assert download_dataset('owner/repo', 'pinned-commit', destination) == destination
    assert (destination / 'annotations.json').read_text() == '{}'
    assert (destination / 'README.md').read_text() == 'Dataset guide'
    assert (destination / 'videos/example.mp4').read_bytes() == b'video payload'
    assert not (destination / 'videos/example.mp4').is_symlink()
    assert not (destination / 'data').exists()
    assert calls == [{'repo_id': 'owner/repo', 'repo_type': 'dataset', 'revision': 'pinned-commit'}]


def test_legacy_flat_snapshot_and_existing_destination(tmp_path, monkeypatch):
    source = tmp_path / 'old-snapshot'
    source.mkdir()
    (source / 'annotations.json').write_text('{"release": 1}')
    stub_snapshot(monkeypatch, source)
    destination = tmp_path / 'dataset'
    destination.mkdir()
    (destination / 'local-note.txt').write_text('Keep this unrelated file')
    download_dataset('owner/repo', 'old-revision', destination)
    assert (destination / 'annotations.json').read_text() == '{"release": 1}'
    assert (destination / 'local-note.txt').is_file()


def test_revision_without_dataset_does_not_copy_repository(tmp_path, monkeypatch):
    source = tmp_path / 'code-only'
    source.mkdir()
    (source / 'README.md').write_text('No dataset')
    stub_snapshot(monkeypatch, source)
    with pytest.raises(ValueError, match='does not contain'):
        download_dataset('owner/repo', 'revision', tmp_path / 'dataset')
    assert not (tmp_path / 'dataset').exists()


def test_destination_inside_snapshot_is_rejected(tmp_path, monkeypatch):
    source = tmp_path / 'snapshot'
    source.mkdir()
    (source / 'annotations.json').write_text('{}')
    stub_snapshot(monkeypatch, source)
    with pytest.raises(ValueError, match='outside'):
        download_dataset('owner/repo', 'revision', source / 'nested-destination')


def test_cli_download_verifies_the_requested_dataset_directory(tmp_path, monkeypatch):
    from omni_eval import cli

    snapshot = tmp_path / 'snapshot'
    source = snapshot / 'data' / 'Omni-InteractScope'
    source.mkdir(parents=True)
    (source / 'annotations.json').write_text('{}')
    stub_snapshot(monkeypatch, snapshot)
    destination = tmp_path / 'custom-dataset'
    checked = []

    def verify(path):
        assert (Path(path) / 'annotations.json').is_file()
        checked.append(path)
        return {'verified': True}

    monkeypatch.setattr(cli, 'verify', verify)
    cli.main(['download', '--repo-id', 'owner/repo', '--revision', 'commit', '--dataset', str(destination)])
    assert checked == [str(destination)]

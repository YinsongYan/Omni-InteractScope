import json
from pathlib import Path

import pytest

from omni_eval.provenance import contract_fingerprint, load_prediction_files


def run_file(root, name='one', ids=('a',), **changes):
    contract = dict(split='table2', annotations_sha256='dataset-version-a', model='model',
                    input='v', shard=name, expected_ids=list(ids), config={'step': 1})
    contract.update(changes)
    fingerprint = contract_fingerprint(contract)
    path = root / (name + '.jsonl')
    records = [dict(interaction_id=i, model=contract['model'], input_config=contract['input'],
                    run_fingerprint=fingerprint) for i in ids]
    path.write_text(''.join(json.dumps(r) + '\n' for r in records))
    Path(str(path) + '.run.json').write_text(json.dumps(dict(contract=contract, fingerprint=fingerprint)))
    return path


def load(paths, **kwargs):
    return load_prediction_files(paths, split='table2', annotations_sha256='dataset-version-a', **kwargs)


def test_single_file_requires_matching_annotation_version(tmp_path):
    path = run_file(tmp_path, annotations_sha256='dataset-version-b')
    with pytest.raises(ValueError, match='different split or annotations'):
        load([path])


def test_single_file_requires_matching_split(tmp_path):
    path = run_file(tmp_path, split='table3')
    with pytest.raises(ValueError, match='different split or annotations'):
        load([path])


def test_sidecar_cannot_be_edited_without_invalidating_fingerprint(tmp_path):
    path = run_file(tmp_path)
    sidecar = Path(str(path) + '.run.json')
    receipt = json.loads(sidecar.read_text())
    receipt['contract']['annotations_sha256'] = 'edited'
    sidecar.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='fingerprint'):
        load([path])


@pytest.mark.parametrize('field,value', [('model', 'other'), ('input_config', 'va'),
                                        ('run_fingerprint', 'other'), ('interaction_id', 'other')])
def test_prediction_must_belong_to_its_recorded_run(tmp_path, field, value):
    path = run_file(tmp_path)
    rec = json.loads(path.read_text())
    rec[field] = value
    path.write_text(json.dumps(rec) + '\n')
    with pytest.raises(ValueError):
        load([path])


def test_shards_can_differ_only_in_shard_and_selected_ids(tmp_path):
    a = run_file(tmp_path, '0/2'.replace('/', '-'), ('a',))
    b = run_file(tmp_path, '1/2'.replace('/', '-'), ('b',))
    records = load([a, b])
    assert [r['interaction_id'] for r in records] == ['a', 'b']
    assert len({r['run_fingerprint'] for r in records}) == 1
    b = run_file(tmp_path, '1-2', ('b',), config={'step': 2})
    with pytest.raises(ValueError, match='different evaluation contracts'):
        load([a, b])


def test_partial_predictions_require_explicit_diagnostic_mode(tmp_path):
    path = run_file(tmp_path, ids=('a', 'b'))
    path.write_text(path.read_text().splitlines()[0] + '\n')
    with pytest.raises(ValueError, match='do not cover'):
        load([path])
    assert len(load([path], allow_partial=True)) == 1


def test_predictions_without_sidecar_fail_before_judging(tmp_path):
    path = run_file(tmp_path)
    Path(str(path) + '.run.json').unlink()
    with pytest.raises(ValueError, match='Missing inference run sidecar'):
        load([path])

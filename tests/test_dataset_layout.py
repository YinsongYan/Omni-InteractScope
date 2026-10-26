"""The compact distribution must preserve fixed populations without duplicate files."""
import hashlib
import json

import pytest

from omni_eval import dataset


@pytest.fixture
def compact_dataset(tmp_path, monkeypatch):
    records = []
    for name in ('table2', 'table3', 't5', 'excluded'):
        record = dict(schema_version=dataset.SCHEMA, interaction_id=name,
                      query_time_ms=1000, video='videos/example.mp4',
                      evaluation_splits=[] if name == 'excluded' else [name],
                      responses=[dict(response_id=1, notify_trigger_time_ms=2000,
                                      valid_response_window_ms=None, long_memory_start_ms=None)])
        if name == 'excluded':
            record['evaluation_exclusion'] = 'Target outside usable media'
        records.append(record)
    counts = {'all': (4, 4), 'table2': (1, 1), 'table3': (1, 1), 't5': (1, 1)}
    digests = {}
    for split in counts:
        ids = [r['interaction_id'] for r in records if split == 'all' or split in r['evaluation_splits']]
        digests[split] = hashlib.sha256(json.dumps(ids, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    monkeypatch.setattr(dataset, 'SPLITS', counts)
    monkeypatch.setattr(dataset, 'SPLIT_ID_SHA256', digests)
    doc = dict(schema_version=dataset.SCHEMA, interactions=records)
    (tmp_path / 'annotations.json').write_text(json.dumps(doc))
    return tmp_path, doc


def test_one_file_loads_every_split_and_keeps_excluded_records(compact_dataset):
    root, _ = compact_dataset
    assert [p.name for p in root.iterdir()] == ['annotations.json']
    assert len(dataset.load_split(root, 'all')) == 4
    for split in ('table2', 'table3', 't5'):
        items = dataset.load_split(root, split)
        assert [it.interaction_id for it in items] == [split]
        assert items[0].responses[0].notify_trigger_sec == 2


def test_swapped_membership_cannot_pass_by_preserving_counts(compact_dataset):
    root, doc = compact_dataset
    doc['interactions'][0]['evaluation_splits'] = ['table3']
    doc['interactions'][1]['evaluation_splits'] = ['table2']
    (root / 'annotations.json').write_text(json.dumps(doc))
    with pytest.raises(ValueError, match='split IDs'):
        dataset.load_split(root, 'table2')


def test_reordered_population_is_rejected(compact_dataset):
    root, doc = compact_dataset
    doc['interactions'].reverse()
    (root / 'annotations.json').write_text(json.dumps(doc))
    with pytest.raises(ValueError, match='ordering'):
        dataset.load_split(root, 'all')


@pytest.mark.parametrize('membership', [['all'], ['unknown'], ['table2', 'table3'], 'table2'])
def test_invalid_membership_is_rejected(compact_dataset, membership):
    root, doc = compact_dataset
    doc['interactions'][0]['evaluation_splits'] = membership
    (root / 'annotations.json').write_text(json.dumps(doc))
    with pytest.raises(ValueError, match='evaluation_splits'):
        dataset.load_split(root, 'all')


def test_exclusion_requires_a_reason(compact_dataset):
    root, doc = compact_dataset
    del doc['interactions'][-1]['evaluation_exclusion']
    (root / 'annotations.json').write_text(json.dumps(doc))
    with pytest.raises(ValueError, match='evaluation_exclusion'):
        dataset.load_split(root, 'all')


def test_old_package_schema_requires_matching_code(compact_dataset):
    root, doc = compact_dataset
    doc['schema_version'] = 'omni-interactscope-1.0'
    (root / 'annotations.json').write_text(json.dumps(doc))
    with pytest.raises(ValueError, match='single-file 1.1'):
        dataset.load_split(root, 'all')

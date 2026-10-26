"""Bind predictions to their recorded dataset, protocol and model configuration."""
import hashlib
import json
from pathlib import Path

from .utils.io import read_jsonl


def contract_fingerprint(contract):
    return hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()


def load_prediction_files(paths, *, split, annotations_sha256, allow_partial=False):
    records, contracts = [], []
    if not paths:
        raise ValueError('No prediction files')
    for path in paths:
        sidecar = Path(str(path) + '.run.json')
        if not sidecar.is_file():
            raise ValueError('Missing inference run sidecar: ' + str(sidecar))
        receipt = json.loads(sidecar.read_text())
        contract = receipt['contract']
        fingerprint = receipt['fingerprint']
        if contract_fingerprint(contract) != fingerprint:
            raise ValueError('Run sidecar fingerprint does not match its contract')
        if contract.get('split') != split or contract.get('annotations_sha256') != annotations_sha256:
            raise ValueError('Predictions were collected for a different split or annotations file')
        expected = contract['expected_ids']
        if not expected or len(expected) != len(set(expected)):
            raise ValueError('Invalid expected IDs in run sidecar')
        part = read_jsonl(str(path))
        actual = [r['interaction_id'] for r in part]
        if len(actual) != len(set(actual)) or not set(actual) <= set(expected):
            raise ValueError('Predictions contain duplicate IDs or IDs outside their run')
        if not allow_partial and set(actual) != set(expected):
            raise ValueError('Predictions do not cover their recorded inference run')
        for record in part:
            if record.get('run_fingerprint') != fingerprint:
                raise ValueError('Prediction fingerprint differs from its run sidecar')
            if (record.get('model') != contract.get('model')
                    or record.get('input_config') != contract.get('input')):
                raise ValueError('Prediction model/input differs from its run sidecar')
        shared = {k: v for k, v in contract.items() if k not in ('shard', 'expected_ids')}
        contracts.append(shared)
        records.extend(part)
    if any(c != contracts[0] for c in contracts[1:]):
        raise ValueError('Shards use different evaluation contracts')
    if len(paths) > 1:
        common = contract_fingerprint(contracts[0])
        records = [dict(r, run_fingerprint=common) for r in records]
    return records

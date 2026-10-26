"""Portable release records, split loading and local asset validation."""
from dataclasses import fields
import hashlib
import json
from pathlib import Path
from .schema import Interaction, Response, validate_interactions

SCHEMA = 'omni-interactscope-1.1'
ANNOTATIONS = 'annotations.json'
SPLITS = {'table2': (1063, 1313), 'table3': (299, 360), 't5': (115, 115), 'all': (1479, 1790)}
# Fixed ordered populations, without distributing duplicate annotations or ID lists.
SPLIT_ID_SHA256 = {
    'all': 'ee440b2cb2b831af3902ed3966ab1cf1d6720f4fb679be338ebeeed4502d2278',
    'table2': '274fce2b9ac64c4677e9acf2d96e0250c93dcfae0231472f24ad1f58f700ac96',
    'table3': '5724592cb901f6da9d06bbb3a32ac26ef0446b519dcbee07d955d682b7baf700',
    't5': 'd34b6c6f8f3ab3aaf427a28a68d400d3cb77f8d10b9245b5e5e225130619b639',
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def local_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not relative or Path(relative).is_absolute() or not path.is_relative_to(root):
        raise ValueError('Dataset paths must stay inside the dataset root')
    return path


def from_release(record):
    if record.get('schema_version') != SCHEMA:
        raise ValueError('Unsupported dataset schema')
    kwargs = {f.name: record[f.name] for f in fields(Interaction) if f.name in record and f.name != 'responses'}
    kwargs['query_time'] = record['query_time_ms']
    responses = []
    for source in record['responses']:
        r = dict(source)
        for key in ('notify_trigger_time', 'valid_response_window', 'long_memory_start'):
            r[key] = r.pop(key + '_ms')
        responses.append(Response.from_dict(r, strict=True))
    kwargs['responses'] = responses
    return Interaction(**kwargs)


def _validate_population(items, split):
    validate_interactions(items, strict=True)
    if (len(items), sum(len(i.responses) for i in items)) != SPLITS[split]:
        raise ValueError('The split population differs from the release')
    ids = [i.interaction_id for i in items]
    digest = hashlib.sha256(json.dumps(ids, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    if digest != SPLIT_ID_SHA256[split]:
        raise ValueError('The split IDs or ordering differ from the release')


def load_split(root, split):
    """Read the sole annotation file; select a fixed evaluation population in memory."""
    if split not in SPLITS:
        raise ValueError('Unknown split')
    doc = json.loads((Path(root) / ANNOTATIONS).read_text())
    if doc.get('schema_version') != SCHEMA:
        raise ValueError('Unsupported dataset schema; expected the single-file 1.1 release')
    records = doc['interactions']
    for record in records:
        membership = record.get('evaluation_splits')
        if (not isinstance(membership, list) or len(membership) > 1
                or any(s not in SPLITS or s == 'all' for s in membership)):
            raise ValueError('Invalid evaluation_splits')
        if not membership and not record.get('evaluation_exclusion'):
            raise ValueError('Excluded interactions must state their evaluation_exclusion')
    all_items = [from_release(r) for r in records]
    _validate_population(all_items, 'all')
    if split == 'all':
        return all_items
    items = [it for it, r in zip(all_items, records) if split in r['evaluation_splits']]
    _validate_population(items, split)
    return items


def verify(root, checksums=False):
    root = Path(root)
    items = load_split(root, 'all')
    for split in ('table2', 'table3', 't5'):
        load_split(root, split)
    manifest = [json.loads(s) for s in (root / 'file_manifest.jsonl').read_text().splitlines() if s.strip()]
    indexed = {r['path']: r for r in manifest}
    if len(indexed) != len(manifest):
        raise ValueError('Duplicate file manifest entries')
    if not {ANNOTATIONS, 'media_metadata.json', 'README.md'} <= indexed.keys():
        raise ValueError('Required dataset files are absent from the manifest')
    for row in manifest:
        path = local_path(root, row['path'])
        if not path.is_file() or path.stat().st_size != row['bytes']:
            raise ValueError('Missing/truncated dataset file: ' + row['path'])
        # Annotation integrity is always checked; large media checksums are opt-in.
        if checksums or path.suffix in ('.json', '.jsonl', '.md'):
            if sha256(path) != row['sha256']:
                raise ValueError('Checksum mismatch: ' + row['path'])
    assets = set()
    for it in items:
        for rel in (it.video, it.query_image, it.query_audio):
            if rel:
                if rel not in indexed:
                    raise ValueError('Referenced asset absent from manifest: ' + rel)
                assets.add(rel)
    metadata = json.loads((root / 'media_metadata.json').read_text())
    for it in load_split(root, 't5'):
        info = metadata.get(it.video, {})
        if info.get('duration', 0) <= 0 or info.get('fps', 0) <= 0:
            raise ValueError('Missing/invalid T5 media metadata: ' + it.video)
    return {'interactions': len(items), 'targets': sum(len(it.responses) for it in items),
            'files': len(manifest), 'referenced_assets': len(assets), 'full_checksums': checksums}

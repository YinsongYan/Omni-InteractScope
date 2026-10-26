"""Command-line entry points for the release evaluation workflow."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from dataclasses import replace
import json
import os
from pathlib import Path
import random
import re
import sys

from .config import get, load_config, resolve_config_path
from .dataset import ANNOTATIONS, SPLITS, load_split, local_path, sha256, verify
from .provenance import contract_fingerprint, load_prediction_files
from .utils.io import append_jsonl, read_jsonl, write_json


def expand_env(value):
    if isinstance(value, dict):
        return {k: expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env(v) for v in value]
    if isinstance(value, str):
        def sub(match):
            name = match.group(1)
            if not os.environ.get(name):
                raise ValueError('Set environment variable ' + name)
            return os.environ[name]
        return re.sub(r'\$\{([A-Z_][A-Z_0-9]*)\}', sub, value)
    return value


def config_for(args):
    cfg = load_config(args.config)
    root = Path(args.dataset).resolve()
    cfg['paths'].update(videos=str(root), audio=str(root), images=str(root),
                        asr_cache=str(Path(args.asr_cache).resolve()) if args.asr_cache else str(root / 'asr_cache' / ('table3' if args.split == 'table3' else 'table2')),
                        frame_cache=str(Path(args.cache_dir).resolve() / 'frames'))
    if args.split == 'table3':
        cfg['modality'].update(asr_timestamps='window', asr_window_sec=1.0)
    return cfg


def asr_for(cfg, backend='cache'):
    from .adaptation.asr import ASR
    m = cfg['modality']
    return ASR(backend=backend, model_name=m['asr_model'], window_sec=m['asr_window_sec'],
               context_chars=m['asr_context_chars'], device=m['asr_device'],
               cache_dir=cfg['paths']['asr_cache'], timestamps=m['asr_timestamps'],
               aligner_model=m['asr_aligner_model'], segment_max_sec=m['asr_segment_max_sec'],
               force_language=m['asr_force_language'], mock=False)


def open_media(it, args, cfg):
    from .t8.media import T8Media
    from .protocols.media import CausalMedia
    path = str(local_path(args.dataset, it.video))
    if not Path(path).is_file():
        raise FileNotFoundError('Missing dataset video: ' + it.video)
    for asset in (it.query_audio, it.query_image):
        if asset and not local_path(args.dataset, asset).is_file():
            raise FileNotFoundError('Missing query attachment: ' + asset)
    if args.split == 't5':
        from .data.media import MediaSource
        metadata = json.loads((Path(args.dataset) / 'media_metadata.json').read_text())[it.video]
        media = MediaSource(path, duration=metadata['duration'], fps=metadata['fps'])
        if not media.decodable:
            raise ValueError('Undecodable T5 video')
        return media
    if args.split == 'table3':
        return T8Media(path, lossless_cache_dir=cfg['paths']['frame_cache'])
    return CausalMedia(path, frame_cache_dir=cfg['paths']['frame_cache'])


def infer(args):
    from .adaptation.modality import decide_adaptation
    from .models.registry import build_model
    from .protocols.base import prepare_inputs
    from .protocols.time_aligned import run as historical_run
    from .t8.protocol import run as assistance_run
    from .protocols.online_rolling import run as legacy_run
    from .evaluation import validate_records
    if any(os.environ.get(k) == '1' for k in ('OMNI_HF_MOCK', 'OMNI_ASR_MOCK', 'OMNI_JUDGE_MOCK')):
        raise ValueError('Mock environment flags are forbidden for benchmark inference')
    if args.workers < 1 or args.limit < 0:
        raise ValueError('Invalid worker count or limit')
    items = load_split(args.dataset, args.split)
    if args.limit:
        items = items[:args.limit]
    idx, shards = map(int, args.shard.split('/'))
    if not 0 <= idx < shards:
        raise ValueError('Shard must be I/N with 0 <= I < N')
    items = [it for k, it in enumerate(items) if k % shards == idx]
    if not items:
        raise ValueError('Empty shard')
    cfg = config_for(args)
    model_cfg = load_config(args.model_config or f'configs/models/{args.model}.yaml')
    model_name = args.model or model_cfg['model']
    if model_cfg.get('model', model_name) != model_name:
        raise ValueError('Model name and model config disagree')
    input_config = args.input or model_cfg['input']
    kwargs = expand_env(model_cfg['model_args'])
    if kwargs.get('mock') or 'api_key' in kwargs or 'access_key' in kwargs:
        raise ValueError('Use environment credentials; mock models and literal credentials are forbidden')
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    # A single writer per shard. Thread workers use the locked JSONL append helper.
    import fcntl
    with Path(str(output) + '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        code = {str(p.relative_to(Path(__file__).parent)): sha256(p) for p in sorted(Path(__file__).parent.rglob('*.py'))}
        import importlib.metadata
        environment = {}
        for package in ('numpy', 'torch', 'transformers', 'decord', 'openai', 'Pillow', 'av', 'qwen-vl-utils', 'qwen-omni-utils', 'livecc-utils', 'minicpmo-utils'):
            try:
                environment[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                pass
        contract = dict(environment=environment, split=args.split, annotations_sha256=sha256(Path(args.dataset) / ANNOTATIONS),
                        model=model_name, model_args=kwargs, input=input_config, config=cfg,
                        code=code, shard=args.shard, limit=args.limit,
                        expected_ids=[it.interaction_id for it in items], seed=20260917)
        fp = contract_fingerprint(contract)
        sidecar = Path(str(output) + '.run.json')
        if sidecar.exists() and json.loads(sidecar.read_text())['fingerprint'] != fp:
            raise ValueError('Resume fingerprint changed; use a new output path')
        prior = read_jsonl(str(output))
        if prior:
            validate_records(items, prior, args.split, allow_partial=True)
            if any(r.get('run_fingerprint') != fp for r in prior):
                raise ValueError('Existing predictions have different provenance')
        write_json(str(sidecar), dict(fingerprint=fp, contract=contract))
        if len(prior) == len(items):
            print('Already complete:', output)
            return
        if model_name == 'aura':
            kwargs['chunk_cache_dir'] = str(Path(args.cache_dir).resolve() / 'aura_chunks')
        if model_name in ('gemini-3.1-pro', 'gemini-3.5-flash', 'qwen3.5-omni-plus', 'openai-compatible'):
            kwargs['cache_path'] = str(Path(args.cache_dir).resolve() / 'api' / (fp + '.jsonl'))
        random.seed(20260917)
        import numpy as np
        np.random.seed(20260917)
        model = build_model(model_name, **kwargs)
        if getattr(model, 'mock', False):
            raise ValueError('Mock model forbidden')
        native = model.capabilities.native_streaming
        if not model.capabilities.thread_safe_probe:
            import torch
            torch.manual_seed(20260917)
        if args.workers > 1 and (native or not model.capabilities.thread_safe_probe):
            raise ValueError('This model requires workers=1; use separate GPU shards')
        if native and input_config == 'v+asr':
            raise ValueError('Native scene-ASR injection is not part of this protocol')
        if args.split == 't5' and not native:
            raise ValueError('T5 is only reported for native rolling models')
        asr = asr_for(cfg)
        done = {r['interaction_id'] for r in prior}
        def one(it):
            media = None
            try:
                media = open_media(it, args, cfg)
                decision = decide_adaptation(it, model.capabilities.modalities_in, input_config)
                if not decision.supported:
                    raise ValueError('Unsupported input configuration')
                prepared = prepare_inputs(it, model.capabilities, decision, media, cfg, asr, for_online=native)
                if args.split == 'table3':
                    rec = assistance_run(it, model, media, decision, prepared)
                elif args.split == 't5':
                    rec = legacy_run(it, model, media, decision, cfg, asr, prepared)
                else:
                    rec = historical_run(it, model, media, decision, prepared)
                rec['meta']['inference_complete'] = True
                rec['meta']['duration_sec'] = media.duration
                stats = getattr(model, 'session_stats', lambda: {})() or {}
                if stats.get('n_generate_errors') or stats.get('n_error_resets'):
                    raise RuntimeError('Native adapter error')
                rec.update(run_fingerprint=fp, scope='smoke' if args.limit else 'benchmark')
                validate_records([it], [rec], args.split)
                append_jsonl(str(output), rec)
                return True
            except Exception as exc:
                # Keep response bodies, prompts and credentials out of exception logs.
                append_jsonl(str(output) + '.errors.jsonl', dict(interaction_id=it.interaction_id,
                             error_type=type(exc).__name__, run_fingerprint=fp))
                print('Failed:', it.interaction_id, type(exc).__name__, file=sys.stderr, flush=True)
                return False
            finally:
                if media is not None:
                    close = getattr(media, 'close', None)
                    if close is not None:
                        close()
        todo = [it for it in items if it.interaction_id not in done]
        failures = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for n, ok in enumerate(pool.map(one, todo), 1):
                failures += not ok
                print(f'{n}/{len(todo)} completed; failures={failures}', flush=True)
        if failures:
            raise RuntimeError(f'{failures} interactions failed; rerun the same command to retry missing IDs')
        validate_records(items, read_jsonl(str(output)), args.split)
        write_json(str(output) + '.complete.json', dict(fingerprint=fp, interactions=len(items),
                   predictions_sha256=sha256(output), model_stats=getattr(model, 'run_stats', lambda: {})()))


def score(args):
    from .evaluation import score_records, validate_records
    from .metrics.reliable_judge import ReliableLLMJudge
    if os.environ.get('OMNI_JUDGE_MOCK') == '1':
        raise ValueError('Mock judging is forbidden')
    items = load_split(args.dataset, args.split)
    annotations_digest = sha256(Path(args.dataset) / ANNOTATIONS)
    records = load_prediction_files(args.predictions, split=args.split,
                                    annotations_sha256=annotations_digest,
                                    allow_partial=args.allow_partial)
    validate_records(items, records, args.split, args.allow_partial)
    judge = ReliableLLMJudge(model='gpt-4o-2024-08-06', mode='binary', mock=False,
                            temperature=0.0, max_retries=6, cache_path=args.judge_cache)
    if args.cache_only:
        def missing(prompt):
            raise RuntimeError('Required valid judge verdict is absent from cache')
        judge._call_api_binary = missing
    result = score_records(items, records, args.split, judge)
    result.update(scope='benchmark' if len(records) == len(items) else 'partial',
                  predictions_sha256={str(p): sha256(p) for p in args.predictions},
                  annotations_sha256=annotations_digest,
                  judge_model=judge.model, judge_mode='binary')
    write_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != 'items'}, indent=2))


def tables(args):
    rows = [json.loads(Path(p).read_text()) for p in args.scores]
    if any(r['scope'] != 'benchmark' for r in rows):
        raise ValueError('Partial scores cannot appear in a benchmark table')
    seen = set()
    t5 = {(r['model'], r['input']): r['by_task']['T5']['percentage'] for r in rows if r['split'] == 't5'}
    order = ['gemini-3.1-pro','gemini-3.5-flash','qwen3.5-omni-plus','internvl3.5-8b','qwen3-vl-8b','qwen2.5-omni-7b','qwen3-omni-30b-a3b','livecc-7b','aura','mmduet2','moss-vl-realtime','minicpm-o']
    folder = Path(args.output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    for split in ('table2', 'table3', 't5'):
        part = sorted([r for r in rows if r['split'] == split], key=lambda r: order.index(r['model']) if r['model'] in order else len(order))
        if not part:
            continue
        columns = ['model', 'input'] + (['T1', 'T2', 'T3', 'T4', 'T5*', 'T6', 'T7', 'Avg.'] if split == 'table2' else ['Overall', 'T8-A', 'T8-B', 'T8-C', 'Complete T8-C'] if split == 'table3' else ['T5'])
        values = []
        for r in part:
            key = (split, r['model'], r['input'])
            if key in seen:
                raise ValueError('Duplicate result configuration')
            seen.add(key)
            if split == 'table3':
                nums = [r['overall']['percentage']] + [r['by_task'][k]['percentage'] for k in 'ABC']
                line = [r['model'], r['input']] + [f'{n:.2f}' for n in nums] + [f"{r['complete_c']['hits']}/{r['complete_c']['total']}"]
            else:
                nums = [r['by_task'][k]['percentage'] for k in ('T1','T2','T3','T4','T6','T7')] + [r['macro_pct']] if split == 'table2' else [r['by_task']['T5']['percentage']]
                cells = [f'{n:.2f}' for n in nums]
                if split == 'table2':
                    legacy = t5.get((r['model'],r['input']))
                    cells.insert(4, f'{legacy:.2f}' if legacy is not None else 'N/A' if r['model'] in order[:7] else 'not run')
                line = [r['model'], r['input']] + cells
            values.append(line)
        with (folder / (split + '.csv')).open('w', newline='') as f:
            writer = csv.writer(f); writer.writerow(columns); writer.writerows(values)
        text = ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |']
        text += ['| ' + ' | '.join(v) + ' |' for v in values]
        (folder / (split + '.md')).write_text('\n'.join(text) + '\n')
        print(folder / (split + '.csv'))


def precompute(args):
    from .adaptation.modality import decide_adaptation
    from .models.base import Capabilities
    from .protocols.base import prepare_inputs
    cfg = config_for(args)
    asr = asr_for(cfg, 'qwen3-asr')
    for n, it in enumerate(load_split(args.dataset, args.split), 1):
        media = open_media(it, args, cfg)
        try:
            decision = decide_adaptation(it, {'text', 'image'}, 'v+asr')
            prepare_inputs(it, Capabilities(modalities_in={'text','image'}), decision, media, cfg, asr)
        finally:
            close = getattr(media, 'close', None)
            if close is not None:
                close()
        print(n, it.interaction_id, flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description='Omni-InteractScope Table 2/3 evaluation')
    subs = ap.add_subparsers(dest='command', required=True)
    p = subs.add_parser('verify', help='check local dataset counts, references and hashes')
    p.add_argument('--dataset', default='data/Omni-InteractScope'); p.add_argument('--checksums', action='store_true')
    p = subs.add_parser('download', help='download a published Hugging Face dataset snapshot')
    p.add_argument('--repo-id', required=True); p.add_argument('--revision', required=True)
    p.add_argument('--dataset', default='data/Omni-InteractScope')
    for name in ('infer', 'precompute-asr'):
        p = subs.add_parser(name)
        p.add_argument('--dataset', default='data/Omni-InteractScope')
        p.add_argument('--split', choices=['table2','table3','t5'], required=True)
        p.add_argument('--config', default='configs/eval.yaml'); p.add_argument('--asr-cache')
        p.add_argument('--cache-dir', default='.cache/omni-interactscope')
        if name == 'infer':
            p.add_argument('--model'); p.add_argument('--model-config'); p.add_argument('--input', choices=['v','v+asr','va'])
            p.add_argument('--output', required=True); p.add_argument('--workers', type=int, default=1)
            p.add_argument('--shard', default='0/1'); p.add_argument('--limit', type=int, default=0)
    p = subs.add_parser('score')
    p.add_argument('--dataset', default='data/Omni-InteractScope'); p.add_argument('--split', choices=['table2','table3','t5'], required=True)
    p.add_argument('--predictions', nargs='+', required=True); p.add_argument('--output', required=True)
    p.add_argument('--judge-cache', default='.cache/judge.jsonl'); p.add_argument('--cache-only', action='store_true')
    p.add_argument('--allow-partial', action='store_true')
    p = subs.add_parser('tables'); p.add_argument('--scores', nargs='+', required=True); p.add_argument('--output-dir', default='outputs/tables')
    args = ap.parse_args(argv)
    if args.command == 'verify':
        print(json.dumps(verify(args.dataset, args.checksums), indent=2))
    elif args.command == 'download':
        from .data.hub import download_dataset
        download_dataset(args.repo_id, args.revision, args.dataset)
        print(json.dumps(verify(args.dataset), indent=2))
    elif args.command == 'infer':
        if not args.model and not args.model_config:
            ap.error('infer requires --model or --model-config')
        infer(args)
    elif args.command == 'score':
        score(args)
    elif args.command == 'tables':
        tables(args)
    else:
        precompute(args)


if __name__ == '__main__':
    main()

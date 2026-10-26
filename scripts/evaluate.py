#!/usr/bin/env python3
"""Run the paper model configurations through inference, scoring and table export."""
import argparse
from pathlib import Path
import shlex
import subprocess
import sys


API_MODELS = ('gemini-3.1-pro', 'gemini-3.5-flash', 'qwen3.5-omni-plus')
PROMPTED_MODELS = API_MODELS + (
    'internvl3.5-8b', 'qwen3-vl-8b', 'qwen2.5-omni-7b', 'qwen3-omni-30b-a3b',
)
STREAMING_MODELS = ('livecc-7b', 'aura', 'mmduet2', 'moss-vl-realtime', 'minicpm-o')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, choices=PROMPTED_MODELS + STREAMING_MODELS)
    parser.add_argument('--dataset', default='data/Omni-InteractScope')
    parser.add_argument('--output-dir', default='outputs', help='writes under OUTPUT_DIR/MODEL/')
    parser.add_argument('--stage', choices=('all', 'infer', 'score'), default='all',
                        help='score also exports CSV and Markdown tables')
    parser.add_argument('--splits', nargs='+', choices=('table2', 'table3', 't5'),
                        help='default: table2 table3, plus t5 for streaming models')
    parser.add_argument('--workers', type=int, default=1, help='API inference concurrency')
    parser.add_argument('--judge-cache', default='.cache/judge.jsonl')
    parser.add_argument('--dry-run', action='store_true', help='print commands without running them')
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error('--workers must be positive')
    if args.stage != 'score' and args.workers > 1 and args.model not in API_MODELS:
        parser.error('local and streaming models require --workers 1')
    splits = args.splits or ['table2', 'table3'] + (['t5'] if args.model in STREAMING_MODELS else [])
    if len(splits) != len(set(splits)):
        parser.error('--splits must not contain duplicates')
    if 't5' in splits and args.model not in STREAMING_MODELS:
        parser.error('T5 is N/A for prompted models; use table2 and table3')

    folder = Path(args.output_dir) / args.model
    cli = [sys.executable, '-m', 'omni_eval.cli']
    commands = []
    scores = []
    for split in splits:
        predictions = str(folder / f'{split}.jsonl')
        score = str(folder / 'scores' / f'{split}.json')
        if args.stage in ('all', 'infer'):
            commands.append(cli + ['infer', '--model', args.model, '--dataset', args.dataset,
                                   '--split', split, '--output', predictions,
                                   '--workers', str(args.workers)])
        if args.stage in ('all', 'score'):
            commands.append(cli + ['score', '--dataset', args.dataset, '--split', split,
                                   '--predictions', predictions, '--output', score,
                                   '--judge-cache', args.judge_cache])
            scores.append(score)
    if scores:
        commands.append(cli + ['tables', '--scores', *scores,
                               '--output-dir', str(folder / 'tables')])
    for command in commands:
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            try:
                subprocess.run(command, check=True)
            except subprocess.CalledProcessError as exc:
                print('Evaluation stopped. Fix the reported error and rerun the same command.',
                      file=sys.stderr)
                return exc.returncode if exc.returncode > 0 else 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

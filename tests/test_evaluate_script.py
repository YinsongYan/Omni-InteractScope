"""Check task selection and subprocess failure handling without inference/API calls."""
import importlib.util
from pathlib import Path
import shlex
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('evaluate_script', ROOT / 'scripts/evaluate.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def calls(monkeypatch):
    recorded = []
    monkeypatch.setattr(runner.subprocess, 'run', lambda argv, **kwargs: recorded.append(argv))
    return recorded


@pytest.mark.parametrize('model', runner.PROMPTED_MODELS + runner.STREAMING_MODELS)
def test_default_tasks_and_export_include_only_applicable_scores(model, monkeypatch):
    commands = calls(monkeypatch)
    assert runner.main(['--model', model, '--dataset', '/data with spaces/bench',
                        '--output-dir', '/results with spaces']) == 0
    expected = ['table2', 'table3'] + (['t5'] if model in runner.STREAMING_MODELS else [])
    for stage in ('infer', 'score'):
        selected = [c for c in commands if c[3] == stage]
        assert [c[c.index('--split') + 1] for c in selected] == expected
        assert all(c[c.index('--dataset') + 1] == '/data with spaces/bench' for c in selected)
    scores = [c[c.index('--output') + 1] for c in commands if c[3] == 'score']
    export = commands[-1]
    assert export[3:5] == ['tables', '--scores']
    assert export[5:export.index('--output-dir')] == scores
    assert len(commands) == len(expected) * 2 + 1


@pytest.mark.parametrize('stage,expected', [('infer', ['infer']), ('score', ['score', 'tables'])])
def test_single_split_and_separate_stages(stage, expected, monkeypatch):
    commands = calls(monkeypatch)
    runner.main(['--model', 'minicpm-o', '--splits', 't5', '--stage', stage])
    assert [c[3] for c in commands] == expected


def test_failure_stops_before_scoring_or_later_splits(monkeypatch):
    commands = []
    def fail(argv, **kwargs):
        commands.append(argv)
        assert kwargs['check'] is True
        raise subprocess.CalledProcessError(7, argv)
    monkeypatch.setattr(runner.subprocess, 'run', fail)
    assert runner.main(['--model', 'livecc-7b']) == 7
    assert len(commands) == 1 and commands[0][3] == 'infer'


def test_dry_run_has_no_subprocesses_and_quotes_paths(monkeypatch, capsys):
    commands = calls(monkeypatch)
    assert runner.main(['--model', 'qwen3-vl-8b', '--dataset', '/data with spaces', '--dry-run']) == 0
    assert commands == []
    printed = [shlex.split(line) for line in capsys.readouterr().out.splitlines()]
    assert len(printed) == 5
    assert printed[0][printed[0].index('--dataset') + 1] == '/data with spaces'


@pytest.mark.parametrize('argv', [
    ['--model', 'qwen3-vl-8b', '--splits', 't5'],
    ['--model', 'minicpm-o', '--workers', '2'],
    ['--model', 'internvl3.5-8b', '--workers', '2'],
    ['--model', 'gemini-3.1-pro', '--workers', '0'],
    ['--model', 'aura', '--splits', 'table2', 'table2'],
])
def test_invalid_selections_fail_before_work(argv, monkeypatch):
    commands = calls(monkeypatch)
    with pytest.raises(SystemExit) as exc:
        runner.main(argv)
    assert exc.value.code == 2 and not commands


def test_api_workers_are_forwarded_to_inference_only(monkeypatch):
    commands = calls(monkeypatch)
    runner.main(['--model', 'gemini-3.1-pro', '--workers', '4'])
    for c in commands:
        if c[3] == 'infer':
            assert c[c.index('--workers') + 1] == '4'
        else:
            assert '--workers' not in c

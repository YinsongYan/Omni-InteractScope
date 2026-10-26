"""Table 2 first-response accuracy and Table 3 ordered target recall."""
from collections import Counter
import math
from .config import load_task_overrides, resolve_config_path
from .metrics.aggregate import _response_successes
from .metrics.temporal import EmitItem, merge_emits
from .taxonomy import TaxonomyRouter
from .paper_taxonomy import PaperTaxonomy
from .t8.scoring import score as assistance_score
from .protocols.livecc_onset import merge_record

TASKS = ('T1', 'T2', 'T3', 'T4', 'T6', 'T7')


def table2_record(record):
    """Apply the accepted onset conversion exactly once, retaining raw timestamps."""
    rec = dict(record, meta=dict(record.get('meta', {})))
    if rec.get('protocol') == 'time_aligned_native_onset':
        return rec
    if rec.get('model') == 'livecc-7b':
        rec = merge_record(dict(rec, emits=rec.get('raw_emits', rec['emits'])), 2.0)[0]
    elif rec.get('protocol') == 'time_aligned_native':
        emits = []
        for e in rec['emits']:
            onset = e['adapter_time_sec']
            if onset > e['time_sec'] + 1e-7:
                raise ValueError('Onset occurs after completion')
            if onset < rec['meta']['query_time_sec'] - 1e-7:
                continue
            emits.append(dict(e, time_sec=onset, completion_sec=e['time_sec']))
        rec['emits'] = emits
    if rec.get('protocol') == 'time_aligned_native':
        rec['protocol'] = 'time_aligned_native_onset'
    return rec


def validate_records(items, records, split, allow_partial=False):
    expected = {it.interaction_id for it in items}
    ids = [r['interaction_id'] for r in records]
    if not records or len(ids) != len(set(ids)) or not set(ids) <= expected:
        raise ValueError('Empty predictions, duplicate IDs, or unknown interactions')
    if set(ids) != expected and not allow_partial:
        raise ValueError(f'Incomplete population: {len(ids)}/{len(expected)}; missing records are not silence')
    if len({(r.get('model'), r.get('input_config'), r.get('run_fingerprint')) for r in records}) != 1:
        raise ValueError('Predictions mix model, input configuration, or run fingerprint')
    # Also accept historical T8 trace labels when scoring previously generated predictions.
    permitted = {'table2': {'time_aligned_probe', 'time_aligned_native', 'time_aligned_native_onset'},
                 'table3': {'t8_probe', 't8_native', 'cat10_probe', 'cat10_native'}, 't5': {'online_rolling'}}[split]
    for rec in records:
        meta = rec.get('meta', {})
        if rec.get('supported') is False or meta.get('mock') or meta.get('n_generate_errors') or meta.get('session_stats', {}).get('n_generate_errors'):
            raise ValueError('Unsupported/mock/error predictions cannot be scored')
        if rec.get('protocol') not in permitted:
            raise ValueError('Wrong protocol for this split')
        if rec.get('status') == 'terminal_failure':
            if not rec.get('failure_reason') or rec.get('emits'):
                raise ValueError('Terminal failures require a reason and zero invented emits')
            continue
        if meta.get('execution_prefix_only') or not meta.get('inference_complete'):
            raise ValueError('Public scoring requires completed full-clock traces')
        if meta.get('session_stats', {}).get('n_error_resets'):
            raise ValueError('An adapter error reset is unresolved')
        last = -1.0
        for e in rec['emits']:
            t = e.get('time_sec')
            if not isinstance(t, (int, float)) or not math.isfinite(t) or t < 0 or t < last:
                raise ValueError('Invalid or unordered emission timestamp')
            if t > meta['duration_sec'] + 1e-6:
                raise ValueError('Emission beyond clip end')
            if '# generate-error:' in e.get('raw', ''):
                raise ValueError('Generation error is not a valid response')
            last = t


def score_records(items, records, split, judge):
    by_id = {it.interaction_id: it for it in items}
    router = TaxonomyRouter(load_task_overrides(resolve_config_path('configs/tasks.yaml')))
    tax = PaperTaxonomy()
    results = []
    for original in records:
        it = by_id[original['interaction_id']]
        failure = original.get('status') == 'terminal_failure'
        if split == 'table3':
            row = assistance_score(it, original, judge, 2.0)
        else:
            rec = table2_record(original)
            emits = merge_emits([EmitItem(e['time_sec'], e['reply_text'], e.get('response_type', 'text')) for e in rec['emits']], 2.0)
            outcomes = _response_successes(it, emits, router, judge, 2.0, 4.0, 1.0, 6)
            if len(outcomes) != len(it.responses):
                raise ValueError('Scorer lost a target')
            row = dict(interaction_id=it.interaction_id, task=tax.class_of(it.task_type, it.sub_task_type),
                       n_targets=len(outcomes), hit_targets=sum(r['success'] for r in outcomes),
                       successes=outcomes, statuses=dict(Counter(r['status'] for r in outcomes)))
        row['terminal_failure'] = failure
        results.append(row)
    def pool(part):
        n = sum(r['n_targets'] for r in part)
        hit = sum(r['hit_targets'] for r in part)
        return dict(interactions=len(part), targets=n, hits=hit, percentage=100 * hit / n if n else None)
    result = dict(split=split, model=records[0]['model'], input=records[0]['input_config'],
                  overall=pool(results), terminal_failures=sum(r['terminal_failure'] for r in results), items=results)
    if split == 'table3':
        result['by_task'] = {s: pool([r for r in results if r['sub_task_type'] == s]) for s in 'ABC'}
        guidance = [r for r in results if r['sub_task_type'] == 'C']
        result['complete_c'] = {'hits': sum(r['complete_interaction'] for r in guidance), 'total': len(guidance)}
    else:
        tasks = TASKS if split == 'table2' else ('T5',)
        result['by_task'] = {s: pool([r for r in results if r['task'] == s]) for s in tasks}
        pcts = [r['percentage'] for r in result['by_task'].values()]
        result['macro_pct'] = sum(pcts) / len(pcts) if all(v is not None for v in pcts) else None
    return result

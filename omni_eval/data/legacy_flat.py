"""The original flat annotation layout: a JSON list of ``Interaction`` dicts.

**Frozen.** ``data/fixtures/`` is written in this layout and the tests read it, so it stays
loadable exactly as before; new annotation batches arrive in the release layout
(:mod:`omni_eval.data.release`). The only thing this module has to keep in sync is the
evaluability rules — a filter that applies to one layout must apply to both, or the two layouts
would disagree about the item set.
"""
from __future__ import annotations

from typing import Any, List, Optional

from ..schema import Interaction
from .filters import Filters, LoadReport
from .formats import records_of
from .normalize import normalize_task_type


def load_flat(
    data: Any,
    strict: bool = False,
    filters: Optional[Filters] = None,
    report: Optional[LoadReport] = None,
) -> List[Interaction]:
    """Parse flat records into ``Interaction`` objects and apply ``filters``."""
    rep = report or LoadReport()
    rep.fmt = rep.fmt or "flat"
    records = records_of(data)
    rep.n_interactions_raw += len(records)
    items = [Interaction.from_dict(d, strict=strict) for d in records]
    items = apply_flat_filters(items, filters, rep)
    rep.n_interactions_kept += len(items)
    for it in items:
        rep.task_type_dist[it.task_type] += 1
        rep.n_responses_kept += len(it.responses)
    return items


def apply_flat_filters(
    items: List[Interaction], filters: Optional[Filters], rep: LoadReport
) -> List[Interaction]:
    """Apply the same evaluability rules to the flat layout (no-op when ``filters is None``)."""
    if filters is None:
        for it in items:
            rep.n_responses_raw += len(it.responses)
        return items
    kept: List[Interaction] = []
    for it in items:
        rep.n_responses_raw += len(it.responses)
        task = normalize_task_type(it.task_type) or it.task_type
        if not filters.keeps_task(task):
            rep.dropped_interactions[f"excluded_task:{task}"] += 1
            rep.dropped_responses[f"excluded_task:{task}"] += len(it.responses)
            continue
        responses = []
        for r in it.responses:
            if r.is_silence and not filters.include_silence_negatives:
                rep.dropped_responses["silence_negative"] += 1
                continue
            if not r.has_gt_time and not r.is_silence:
                rep.dropped_responses["no_gt_time"] += 1
                continue
            responses.append(r)
        if not responses:
            rep.dropped_interactions["no_evaluable_response"] += 1
            continue
        it.responses = responses
        kept.append(it)
    return kept

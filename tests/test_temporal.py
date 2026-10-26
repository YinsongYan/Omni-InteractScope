from omni_eval.metrics.temporal import (
    EmitItem,
    make_gt,
    match,
    merge_emits,
    prf,
    window_iou,
)


def _emits(times):
    return [EmitItem(time_sec=t, reply_text="x") for t in times]


def test_merge_within_gap():
    merged = merge_emits(_emits([10.0, 10.5, 11.0, 20.0]), gap=2.0)
    assert len(merged) == 2
    assert merged[0].time_sec == 10.0 and merged[0].redundant == 2
    assert merged[1].time_sec == 20.0


def test_point_match_in_and_out_of_window():
    gts = [make_gt(0, center=10.0, tol=2.0)]
    # in-window fire
    r = match(_emits([11.0]), gts, tol=2.0)
    assert r.counts() == (1, 0, 0)
    # out-of-window fire -> FP + FN
    r = match(_emits([5.0]), gts, tol=2.0)
    assert r.counts() == (0, 1, 1)


def test_greedy_one_to_one():
    gts = [make_gt(0, center=10.0, tol=2.0), make_gt(1, center=20.0, tol=2.0)]
    r = match(_emits([10.2, 19.8, 30.0]), gts, tol=2.0)
    tp, fp, fn = r.counts()
    assert (tp, fp, fn) == (2, 1, 0)  # 30.0 is an unmatched FP


def test_window_many_to_one_not_penalized():
    gts = [make_gt(0, window=[8.0, 16.0], tol=2.0)]
    # multiple in-window fires -> one window TP, no FP
    r = match(_emits([8.0, 12.0, 16.0]), gts, tol=2.0, merge_gap=0.0)
    tp, fp, fn = r.counts()
    assert tp == 1 and fp == 0 and fn == 0
    iou = window_iou(r.matched[0])
    assert iou is not None and iou > 0.9  # predicted span [8,16] == GT window


def test_prf():
    p, r, f1 = prf(3, 1, 1)
    assert round(p, 3) == 0.75 and round(r, 3) == 0.75 and round(f1, 3) == 0.75
    assert prf(0, 0, 0) == (0.0, 0.0, 0.0)


def test_latency():
    gts = [make_gt(0, center=10.0, tol=2.0)]
    r = match(_emits([11.5]), gts, tol=2.0)
    assert abs(r.matched[0].latency - 1.5) < 1e-6

"""Proactive Turn-taking helpers (unifies OmniMMI PT + PA).

A benchmark stream mixes **should-speak** opportunities (point/window GTs, scored by the
temporal matcher) and **should-stay-silent** noise events (responses with
``expected_behavior == "silence"``). This module scores the silence side and exposes counts so
``aggregate`` can combine both into a single turn-taking F1:

    P = TP / (TP + FP_time + FP_noise)     # fired when it should + didn't over-fire / speak to noise
    R = TP / (TP + FN)
    silence_accuracy = TN_noise / (TN_noise + FP_noise)

A "spoke-to-noise" is any emit falling inside a noise region ``[c - w, c + w]`` (w =
``noise_window_sec``), where ``c`` is the silence response's ``notify_trigger_time``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class NoiseResult:
    n_regions: int = 0
    fp: int = 0                       # regions where the model wrongly spoke
    tn: int = 0                       # regions where the model correctly stayed silent
    consumed_emit_idx: set = field(default_factory=set)  # emit indices used up by noise regions


def build_noise_regions(responses, noise_window: float) -> List[Tuple[float, float]]:
    regions = []
    for r in responses:
        if not r.is_silence:
            continue
        c = r.notify_trigger_sec
        if c is None and r.valid_response_window_sec is not None:
            lo, hi = r.valid_response_window_sec
            regions.append((lo - noise_window, hi + noise_window))
        elif c is not None:
            regions.append((c - noise_window, c + noise_window))
        else:
            regions.append((0.0, float("inf")))  # unknown time -> whole stream must stay silent
    return regions


def score_noise(emit_times: List[float], regions: List[Tuple[float, float]]) -> NoiseResult:
    res = NoiseResult(n_regions=len(regions))
    for lo, hi in regions:
        hit_idx = [i for i, t in enumerate(emit_times) if lo <= t <= hi]
        if hit_idx:
            res.fp += 1
            res.consumed_emit_idx.update(hit_idx)
        else:
            res.tn += 1
    return res


def latency_decay(latency: float, decay_sec: float) -> float:
    """Full credit (1.0) at the window start, linearly decaying to 0 over ``decay_sec``."""
    if decay_sec <= 0:
        return 1.0 if latency <= 0 else 0.0
    return max(0.0, 1.0 - latency / decay_sec)

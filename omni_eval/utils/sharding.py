"""Deterministic item sharding for multi-GPU / multi-process runs.

``--shard I/N`` splits the item list N ways and keeps slice I. Assignment is by MD5 of the
``interaction_id``, not by list position, so a shard's membership is stable even if the item
set changes between invocations (a data refresh, a different ``--limit``) and independent of
the order the loader happens to return.

Orthogonal to ``--resume``: shard first, then skip ids already in the output file.
``utils.io.append_jsonl`` takes an exclusive advisory lock, so every shard may append to the
same prediction file; use per-shard ``--output`` plus ``scripts/merge_shards.py`` when the
shards run on machines that do not share a filesystem.
"""
from __future__ import annotations

import hashlib
from typing import List, Sequence, Tuple, TypeVar

T = TypeVar("T")


def parse_shard(spec: str) -> Tuple[int, int]:
    """Parse ``"I/N"`` into ``(index, total)``. ``""`` means no sharding -> ``(0, 1)``."""
    spec = (spec or "").strip()
    if not spec:
        return (0, 1)
    if "/" not in spec:
        raise ValueError(f"--shard expects 'I/N' (e.g. 0/8), got {spec!r}")
    i_raw, n_raw = spec.split("/", 1)
    try:
        index, total = int(i_raw), int(n_raw)
    except ValueError:
        raise ValueError(f"--shard expects integers in 'I/N', got {spec!r}") from None
    if total < 1:
        raise ValueError(f"--shard total must be >= 1, got {total}")
    if not 0 <= index < total:
        raise ValueError(f"--shard index must be in [0, {total}), got {index}")
    return (index, total)


def shard_of(key: str, total: int) -> int:
    if total <= 1:
        return 0
    digest = hashlib.md5(str(key).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % total


def shard_items(items: Sequence[T], spec: str, key_attr: str = "interaction_id") -> List[T]:
    """Keep only the items belonging to shard ``I`` of ``N``."""
    index, total = parse_shard(spec)
    if total <= 1:
        return list(items)
    return [x for x in items if shard_of(getattr(x, key_attr), total) == index]

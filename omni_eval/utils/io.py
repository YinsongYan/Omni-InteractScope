"""IO helpers: JSON / JSONL read+write with an advisory file lock for multi-GPU sharded
prediction writing, plus resume support (skip already-completed interaction ids)."""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from typing import Any, Dict, Iterable, Iterator, List, Optional

try:  # POSIX advisory locking; optional (single-process runs work without it)
    import fcntl  # type: ignore

    _HAVE_FCNTL = True
except Exception:  # pragma: no cover - non-POSIX
    _HAVE_FCNTL = False


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, obj: Any, indent: int = 2) -> None:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


@contextmanager
def _locked(fh) -> Iterator[None]:
    if _HAVE_FCNTL:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
    try:
        yield
    finally:
        if _HAVE_FCNTL:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def append_jsonl(path: str, record: Dict[str, Any]) -> None:
    """Append one record to a JSONL file under an exclusive advisory lock so multiple
    inference workers (one per GPU) can write the same prediction file safely."""
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "a", encoding="utf-8") as f:
        with _locked(f):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()


def completed_ids(path: str, key: str = "interaction_id") -> set:
    """Return the set of ids already present in a JSONL prediction file (for resume).

    Tolerates — and repairs — a truncated *final* line. ``append_jsonl`` flushes one line per
    interaction, so a run killed mid-write (SIGKILL, a dead node, a full disk) can leave half a
    JSON object at the end of the file. Reading that with ``read_jsonl`` raises, which would make
    the resume fail, then the merge fail, and the only way out would be deleting hours of work —
    the exact loss resume exists to prevent. So drop the partial tail and truncate the file, since
    every later reader (merge_shards, validate_predictions, run_eval) is strict by design and would
    otherwise still choke on it. A malformed line anywhere *else* is real corruption, not an
    interrupted write, and still raises.
    """
    if not os.path.exists(path):
        return set()
    ids, good_bytes = set(), 0
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        last = i == len(lines) - 1
        # No trailing newline on the last line means the write was cut short too, even if what
        # landed happens to parse: appending to it would glue the next record onto the same line.
        # Treating it as incomplete costs one recomputed interaction and keeps the file at a record
        # boundary, which is the invariant every strict reader downstream depends on.
        if last and not line.endswith("\n"):
            break
        stripped = line.strip()
        if stripped:
            try:
                rec = json.loads(stripped)
            except ValueError:
                if not last:
                    raise
                break  # truncated tail; good_bytes is where the file should end
            if key in rec:
                ids.add(rec[key])
        good_bytes += len(line.encode("utf-8"))
    if good_bytes < os.path.getsize(path):
        print(f"[resume] {path}: dropping a truncated trailing line "
              f"({os.path.getsize(path) - good_bytes} bytes) left by an interrupted write")
        with open(path, "r+b") as f:
            f.truncate(good_bytes)
    return ids

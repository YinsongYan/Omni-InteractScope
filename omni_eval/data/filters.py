"""What counts as evaluable (``Filters``) and what the loader had to fix or drop (``LoadReport``).

``Filters`` is built once from ``configs/eval.yaml`` ``filters:`` and passed to both runners, so
inference and scoring can never disagree about the item set. ``filters=None`` means "keep
everything" — the direct ``load_annotations(dir)`` calls in the tests rely on that.

``LoadReport`` exists because this data is hand-annotated: a dirty batch should produce a printed
summary (and a machine-readable ``--load-report``), not silently-wrong numbers.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple

from .normalize import NormalizeReport, normalize_task_type


@dataclass
class Filters:
    """Which annotations count as evaluable.

    Defaults mirror ``configs/eval.yaml``. No task type is excluded: ``simultaneous_speech`` used to
    be, on the belief that its multi-turn ground truth had been collapsed into one turn, which is
    true of the old 611-video release but not of the current pool — that one carries real
    multi-round window GT and is the benchmark's only duplex-discriminative class (see the NOTE in
    ``configs/eval.yaml``). ``<silence>`` negatives stay excluded because the agreed protocol scores
    only should-speak targets (flip ``include_silence_negatives`` to score turn-taking too).
    """

    exclude_task_types: Tuple[str, ...] = ()
    include_task_types: Tuple[str, ...] = ()
    include_silence_negatives: bool = False
    drop_invalid_duration: bool = True
    drop_unknown_task_type: bool = True
    require_available: bool = False
    require_publishable: bool = False

    @classmethod
    def from_config(cls, cfg: Optional[Dict[str, Any]]) -> "Filters":
        f = ((cfg or {}).get("filters") or {}) if isinstance(cfg, dict) else {}
        d = cls()
        return cls(
            exclude_task_types=cls._norm_tasks(f.get("exclude_task_types", d.exclude_task_types)),
            include_task_types=cls._norm_tasks(f.get("include_task_types", d.include_task_types)),
            include_silence_negatives=bool(
                f.get("include_silence_negatives", d.include_silence_negatives)
            ),
            drop_invalid_duration=bool(f.get("drop_invalid_duration", d.drop_invalid_duration)),
            drop_unknown_task_type=bool(f.get("drop_unknown_task_type", d.drop_unknown_task_type)),
            require_available=bool(f.get("require_available", d.require_available)),
            require_publishable=bool(f.get("require_publishable", d.require_publishable)),
        )

    @staticmethod
    def _norm_tasks(raw: Any) -> Tuple[str, ...]:
        """Task-type filters are matched post-normalization, so normalise them too."""
        if not raw:
            return ()
        if isinstance(raw, str):
            raw = [x for x in raw.split(",") if x.strip()]
        out = []
        for t in raw:
            out.append(normalize_task_type(t) or str(t).strip())
        return tuple(out)

    def keeps_task(self, task_type: str) -> bool:
        if self.include_task_types and task_type not in self.include_task_types:
            return False
        return task_type not in self.exclude_task_types

    def with_cli(self, tasks: str = "", exclude_tasks: str = "") -> "Filters":
        """Layer ``--tasks`` / ``--exclude-tasks`` on top of the config filters.

        Both are matched against *normalized* task types, so ``--tasks "proactive _notifying"``
        and ``--tasks proactive_notifying`` select the same items.
        """
        include = self._norm_tasks(tasks) or self.include_task_types
        exclude = tuple(dict.fromkeys(self.exclude_task_types + self._norm_tasks(exclude_tasks)))
        return replace(self, include_task_types=include, exclude_task_types=exclude)


def filters_from_cli(
    cfg: Optional[Dict[str, Any]], tasks: str = "", exclude_tasks: str = ""
) -> Filters:
    """The single place both runners build their filter set, so their item sets always match."""
    return Filters.from_config(cfg).with_cli(tasks=tasks, exclude_tasks=exclude_tasks)


@dataclass
class LoadReport:
    """What the loader saw, rewrote, and threw away."""

    paths: List[str] = field(default_factory=list)
    fmt: str = ""
    n_videos: int = 0
    n_videos_kept: int = 0
    n_interactions_raw: int = 0
    n_interactions_kept: int = 0
    n_responses_raw: int = 0
    n_responses_kept: int = 0
    dropped_videos: Counter = field(default_factory=Counter)
    dropped_interactions: Counter = field(default_factory=Counter)
    dropped_responses: Counter = field(default_factory=Counter)
    normalize: NormalizeReport = field(default_factory=NormalizeReport)
    id_collisions: Counter = field(default_factory=Counter)
    suspect_times: List[str] = field(default_factory=list)
    task_type_dist: Counter = field(default_factory=Counter)
    tagged_unavailable: int = 0
    tagged_unpublishable: int = 0
    missing_media: int = 0
    validation_issues: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "paths": self.paths,
            "format": self.fmt,
            "videos": {"total": self.n_videos, "kept": self.n_videos_kept,
                       "dropped": dict(self.dropped_videos)},
            "interactions": {"total": self.n_interactions_raw, "kept": self.n_interactions_kept,
                             "dropped": dict(self.dropped_interactions)},
            "responses": {"total": self.n_responses_raw, "kept": self.n_responses_kept,
                          "dropped": dict(self.dropped_responses)},
            "task_type_dist": dict(self.task_type_dist.most_common()),
            "normalize": self.normalize.as_dict(),
            "id_collisions": dict(self.id_collisions),
            "suspect_times": self.suspect_times,
            # Counted in interactions (not videos) so the numbers compare with `interactions.kept`.
            "tagged": {"unavailable_interactions": self.tagged_unavailable,
                       "unpublishable_interactions": self.tagged_unpublishable},
            "missing_media": self.missing_media,
            "validation_issues": self.validation_issues,
        }

    def summary(self) -> str:
        lines = [
            f"[loader] format={self.fmt or 'flat'} files={len(self.paths)}",
            f"[loader] videos {self.n_videos_kept}/{self.n_videos} kept, "
            f"interactions {self.n_interactions_kept}/{self.n_interactions_raw} kept, "
            f"responses {self.n_responses_kept}/{self.n_responses_raw} kept",
        ]
        for label, counter in (
            ("videos", self.dropped_videos),
            ("interactions", self.dropped_interactions),
            ("responses", self.dropped_responses),
        ):
            if counter:
                detail = ", ".join(f"{k}={v}" for k, v in counter.most_common())
                lines.append(f"[loader] dropped {label}: {detail}")
        if self.task_type_dist:
            detail = ", ".join(f"{k}={v}" for k, v in self.task_type_dist.most_common())
            lines.append(f"[loader] task_type: {detail}")
        if self.normalize.n_rewritten or self.normalize.n_unresolved:
            lines.append(
                f"[loader] normalized {self.normalize.n_rewritten} dirty enum values, "
                f"{self.normalize.n_unresolved} unresolvable "
                f"(unresolvable -> field default; see load-report for the raw values)"
            )
        if self.normalize.missing:
            detail = ", ".join(f"{k}={v}" for k, v in self.normalize.missing.most_common())
            lines.append(f"[loader] blank enum fields (defaulted): {detail}")
        if self.id_collisions:
            lines.append(f"[loader] id collisions disambiguated: {sum(self.id_collisions.values())}")
        if self.suspect_times:
            lines.append(f"[loader] timestamps past clip end ({len(self.suspect_times)} shown):")
            lines.extend(f"[loader]   {s}" for s in self.suspect_times)
        if self.tagged_unavailable or self.tagged_unpublishable:
            lines.append(
                f"[loader] tagged but kept: {self.tagged_unavailable} interactions from "
                f"is_available=false videos, {self.tagged_unpublishable} from unpublishable ones"
            )
        for msg in self.validation_issues[:10]:
            lines.append(f"[loader][warn] {msg}")
        if len(self.validation_issues) > 10:
            lines.append(f"[loader][warn] ... {len(self.validation_issues) - 10} more issues")
        return "\n".join(lines)

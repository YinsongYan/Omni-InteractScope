"""Map historical annotation categories to the seven explicit-request task labels."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Tuple

from .config import load_paper_classes

# A cell is the annotator guideline's own coordinate: "<task_type>/<LETTER>".
Cell = str


def _expand_cells(spec: str) -> List[Cell]:
    """``"visual_hints/ACDE"`` -> the four cells; ``"proactive_notifying"`` -> ``[]`` (whole task).

    The compact multi-letter form exists because two classes take a *subset* of one task type's
    letters (``visual_hints`` splits across T2 and T6), and writing it out cell by cell in
    ``members:`` would bury the one fact that matters — that the split is by letter.
    """
    task, _, letters = spec.partition("/")
    letters = letters.strip()
    if not letters:
        return []
    return [f"{task.strip()}/{ch}" for ch in letters]


@dataclass(frozen=True)
class Leaf:
    name: str
    cells: FrozenSet[Cell] = frozenset()
    tool_modality: Optional[str] = None
    crosslingual: Optional[bool] = None

    @property
    def by_cells(self) -> bool:
        return bool(self.cells)


@dataclass(frozen=True)
class PaperClass:
    cid: str                        # stable id, T1..T7 — never rename
    name: str                       # display text
    predicate: str
    capability: str
    member_tasks: FrozenSet[str]    # task types belonging wholly to this class
    member_cells: FrozenSet[Cell]   # individual cells (a task type split across classes)
    leaves: Tuple[Leaf, ...]
    sampling_stratum: str = "leaf"  # "leaf" | "class" (see T1's `sampling_stratum: class`)


class PaperTaxonomy:
    """Resolver over ``configs/tasks.yaml::paper_classes``.

    ``class_of`` raises on an unknown task type rather than returning a catch-all: an annotation
    category missing from the paper's taxonomy must fail loudly at build time, not vanish from the
    main table.
    """

    def __init__(self, spec: Optional[Dict[str, Any]] = None):
        raw = spec if spec is not None else load_paper_classes()
        self.classes: Dict[str, PaperClass] = {}
        for cid, body in raw.items():
            members = body.get("members") or []
            tasks, cells = set(), set()
            for m in members:
                expanded = _expand_cells(str(m))
                if expanded:
                    cells.update(expanded)
                else:
                    tasks.add(str(m).strip())
            leaves = []
            for lf in body.get("leaves") or []:
                rule = lf.get("rule") or {}
                leaves.append(Leaf(
                    name=str(lf["name"]),
                    cells=frozenset(str(c).strip() for c in (rule.get("cells") or [])),
                    tool_modality=rule.get("tool_modality"),
                    crosslingual=rule.get("crosslingual"),
                ))
            self.classes[str(cid)] = PaperClass(
                cid=str(cid),
                name=str(body.get("name") or cid),
                predicate=str(body.get("predicate") or ""),
                capability=str(body.get("capability") or ""),
                member_tasks=frozenset(tasks),
                member_cells=frozenset(cells),
                leaves=tuple(leaves),
                sampling_stratum=str(body.get("sampling_stratum") or "leaf"),
            )
        self._validate()

    # ---------------------------------------------------------------- construction checks
    def _validate(self) -> None:
        """Catch the two ways this config can silently mis-report: a task type claimed by two
        classes, and a leaf that can never be reached."""
        owner: Dict[str, str] = {}
        for c in self.classes.values():
            for key in list(c.member_tasks) + list(c.member_cells):
                if key in owner:
                    raise ValueError(
                        f"paper_classes: {key!r} is claimed by both {owner[key]} and {c.cid}; "
                        "classes must be mutually exclusive"
                    )
                owner[key] = c.cid
        for c in self.classes.values():
            if not c.leaves:
                raise ValueError(f"paper_classes: {c.cid} has no leaves")
            if c.sampling_stratum not in ("leaf", "class"):
                raise ValueError(
                    f"paper_classes: {c.cid} sampling_stratum={c.sampling_stratum!r} "
                    "(expected 'leaf' or 'class')"
                )
            for lf in c.leaves:
                if not (lf.cells or lf.tool_modality or lf.crosslingual is not None):
                    raise ValueError(f"paper_classes: {c.cid} leaf {lf.name!r} has an empty rule")

    # ---------------------------------------------------------------- lookups
    def class_of(self, task_type: str, sub_task_type: str = "") -> str:
        """Most specific first: an explicit cell beats a whole-task membership."""
        task = (task_type or "").strip()
        letter = (sub_task_type or "").strip()
        cell = f"{task}/{letter}"
        for c in self.classes.values():
            if cell in c.member_cells:
                return c.cid
        for c in self.classes.values():
            if task in c.member_tasks:
                return c.cid
        raise KeyError(
            f"task_type {task!r} (sub {letter!r}) belongs to no paper class. Add it to "
            "configs/tasks.yaml::paper_classes — an unmapped category would be missing from the "
            "main table."
        )

    def leaf_of(
        self,
        task_type: str,
        sub_task_type: str = "",
        *,
        tool_modality: Optional[str] = None,
        crosslingual: Optional[bool] = None,
    ) -> Optional[str]:
        """Leaf name, or ``None`` when the discriminant this class needs is unavailable.

        ``None`` is a real answer, not an error: T1's leaves are keyed on the retrieval input
        modality, which only exists after ``scripts/relabel_tools.py`` has run on the selected
        items, so during sampling every T1 candidate is legitimately leaf-less.

        ``cells`` rules resolve first so that a class mixing rule kinds (T7) stays deterministic
        regardless of the order its leaves are listed in.
        """
        c = self.classes[self.class_of(task_type, sub_task_type)]
        cell = f"{(task_type or '').strip()}/{(sub_task_type or '').strip()}"
        for lf in c.leaves:
            if lf.by_cells and cell in lf.cells:
                return lf.name
        for lf in c.leaves:
            if lf.tool_modality is not None and tool_modality == lf.tool_modality:
                return lf.name
            if lf.crosslingual is not None and crosslingual is not None \
                    and bool(crosslingual) == bool(lf.crosslingual):
                return lf.name
        return None

    def stratum_of(
        self,
        task_type: str,
        sub_task_type: str = "",
        *,
        tool_modality: Optional[str] = None,
        crosslingual: Optional[bool] = None,
    ) -> str:
        """Sampling key. ``"T2/agent-action onset"`` normally; bare ``"T1"`` for a class whose
        leaves are deferred, and bare ``"<cid>"`` as the fallback when a leaf cannot be resolved —
        an unresolvable item is still sampled, it just does not constrain the leaf balance."""
        cid = self.class_of(task_type, sub_task_type)
        if self.classes[cid].sampling_stratum == "class":
            return cid
        leaf = self.leaf_of(task_type, sub_task_type,
                            tool_modality=tool_modality, crosslingual=crosslingual)
        return f"{cid}/{leaf}" if leaf else cid

    # ---------------------------------------------------------------- display helpers
    @property
    def class_ids(self) -> List[str]:
        return list(self.classes.keys())

    def name_of(self, cid: str) -> str:
        return self.classes[cid].name

    def leaf_names(self, cid: str) -> List[str]:
        return [lf.name for lf in self.classes[cid].leaves]

    def known_cells(self) -> FrozenSet[Cell]:
        """Every cell named anywhere in the config — the letters the taxonomy actually covers."""
        out: set = set()
        for c in self.classes.values():
            out |= set(c.member_cells)
            for lf in c.leaves:
                out |= set(lf.cells)
        return frozenset(out)


def tool_modality_of(tool_calls: Optional[Iterable[dict]]) -> Optional[str]:
    """Retrieval input modality from a GT ``tool_calls`` block (T1's leaf discriminant).

    ``modality`` is written into the annotation by ``scripts/relabel_tools.py``, copied from
    ``configs/tools.yaml``; reading it back rather than re-deriving it from ``primary`` keeps this
    resolver independent of the registry. Multiple calls on one response would be ambiguous, so
    only the first is used — the current release has exactly one call per tool response.
    """
    for tc in tool_calls or []:
        if isinstance(tc, dict) and tc.get("modality"):
            return str(tc["modality"]).strip()
    return None

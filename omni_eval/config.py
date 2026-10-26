"""Central configuration: one YAML file is the single source of truth.

``configs/eval.yaml`` holds every default; there is no second copy in Python. A duplicated
``DEFAULTS`` dict used to live here, which meant three places could define the same knob
(DEFAULTS / eval.yaml / configs/models/*.yaml) and nobody could tell which one was in effect.
Now a missing config file is an error rather than a silent fallback to stale coded values.

- ``load_config(path)`` reads the YAML (``path=None`` -> ``configs/eval.yaml``). A relative path is
  tried against the cwd first, then against the repo root, so running from a subdirectory still
  finds the shipped config.
- ``get(cfg, "a.b.c", default)`` reads nested keys; the ``default`` still covers keys a config
  legitimately omits.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional

# <repo>/omni_eval/config.py -> <repo>
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_CONFIG = "configs/eval.yaml"


def resolve_config_path(path: Optional[str] = None) -> str:
    """Locate a config file, trying the cwd then the repo root. Raise if it is nowhere."""
    rel = path or _DEFAULT_CONFIG
    for cand in (rel, os.path.join(_REPO_ROOT, rel),
                 os.path.join(sys.prefix, "share", "omni-interactscope", rel)):
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(
        f"config file not found: {rel!r} (looked in {os.getcwd()!r} and {_REPO_ROOT!r}). "
        "There is no built-in fallback — configs/eval.yaml is the source of truth."
    )


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    import yaml  # lazy: keeps `import omni_eval.config` cheap for callers that never load

    with open(resolve_config_path(path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get(cfg: Dict[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def load_tool_registry(path: Optional[str] = None) -> Dict[str, Any]:
    """``configs/tools.yaml`` — the tool vocabulary, split by retrieval input modality.

    Single source of truth for three consumers that used to each carry their own copy: the names
    disclosed to every model (``models/prompted_base``), the ``<call:NAME>`` markers in the
    annotation (``scripts/relabel_tools``), and the acceptable-substitute sets behind the
    strict/lenient tool scores. Missing file is an error, not a fallback: a stale hard-coded
    vocabulary is exactly the failure this file exists to remove. See that file's header.
    """
    import yaml

    with open(resolve_config_path(path or "configs/tools.yaml"), "r", encoding="utf-8") as f:
        reg = yaml.safe_load(f) or {}
    if not reg.get("tools"):
        raise ValueError(f"{path or 'configs/tools.yaml'}: no `tools:` block")
    return reg


def disclosed_tools(path: Optional[str] = None) -> Dict[str, dict]:
    """Every tool a model is told about, in registry order (GT-eligible ones and distractors).

    ``retired:`` names are deliberately not included — see the registry header.
    """
    return dict(load_tool_registry(path).get("tools") or {})


def load_paper_classes(path: Optional[str] = None) -> Dict[str, Any]:
    """``configs/tasks.yaml`` -> the T1–T7 explicit-request groups.

    T8 is handled by the assistance scorer using its A/B/C subtypes.

    Read by ``omni_eval.paper_taxonomy``, which is the only consumer that should parse it. It is a
    *display and stratification* layer: ``load_task_overrides`` deliberately returns only
    ``overrides:``, so nothing here can reach ``TaxonomyRouter`` or ``metrics/``.
    """
    import yaml

    with open(resolve_config_path(path or "configs/tasks.yaml"), "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    classes = data.get("paper_classes")
    if not classes:
        raise ValueError(f"{path or 'configs/tasks.yaml'}: no `paper_classes:` block")
    return classes


def load_task_overrides(path: Optional[str]) -> Dict[str, dict]:
    if path and os.path.exists(path):
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data.get("overrides", data) or {}
    return {}

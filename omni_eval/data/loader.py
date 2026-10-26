"""Load annotation files and resolve their media — the one import site for both.

This module is the **facade**: it owns format dispatch and the public entry points, and re-exports
everything callers used to import from here. The pieces live next door, one concern per file:

* :mod:`omni_eval.data.formats`     — which layout is this file (``detect_format``)
* :mod:`omni_eval.data.release`     — flatten the ``video_grouped`` annotation release
* :mod:`omni_eval.data.legacy_flat` — the original flat layout (frozen)
* :mod:`omni_eval.data.filters`     — ``Filters`` (what is evaluable) + ``LoadReport``
* :mod:`omni_eval.data.paths`       — annotation -> files on disk -> ``MediaSource``

``filters=None`` means "keep everything" — that is what the direct ``load_annotations(dir)`` calls
in the tests rely on. Runners pass ``Filters.from_config(cfg)`` so inference and scoring always
see an identical item set.
"""
from __future__ import annotations

import glob
import os
from typing import List, Optional, Tuple

from ..schema import Interaction, validate_interactions
from ..utils.io import read_json
from .filters import Filters, LoadReport, filters_from_cli
from .formats import detect_format, records_of
from .legacy_flat import apply_flat_filters, load_flat
from .media import MediaSource, VideoMeta
from .paths import (
    AUDIO_EXTS,
    IMAGE_EXTS,
    VIDEO_EXTS,
    interaction_duration,
    load_video_meta,
    make_media_source,
    resolve_attachment_path,
    resolve_video_path,
)
from .release import load_video_grouped, url_basename, video_filename_of, video_key_of

__all__ = [
    # entry points
    "load_annotations",
    "load_annotations_with_report",
    "load_annotation_file",
    # per-layout loaders
    "detect_format",
    "records_of",
    "load_video_grouped",
    "load_flat",
    "apply_flat_filters",
    # evaluability + reporting
    "Filters",
    "LoadReport",
    "filters_from_cli",
    # media / paths
    "MediaSource",
    "VideoMeta",
    "load_video_meta",
    "make_media_source",
    "resolve_video_path",
    "resolve_attachment_path",
    "interaction_duration",
    "VIDEO_EXTS",
    "AUDIO_EXTS",
    "IMAGE_EXTS",
    # release-layout naming helpers (used by scripts/fetch_media.py)
    "url_basename",
    "video_key_of",
    "video_filename_of",
]


def load_annotation_file(
    path: str,
    strict: bool = False,
    filters: Optional[Filters] = None,
    report: Optional[LoadReport] = None,
) -> List[Interaction]:
    """Load one JSON file, dispatching on its detected layout."""
    data = read_json(path)
    rep = report or LoadReport()
    rep.paths.append(path)
    if detect_format(data) == "video_grouped":
        items, _ = load_video_grouped(data, filters=filters, report=rep)
        return items
    return load_flat(data, strict=strict, filters=filters, report=rep)


def load_annotations(
    source: str,
    strict: bool = False,
    filters: Optional[Filters] = None,
) -> List[Interaction]:
    """Load interactions from a single JSON file or every ``*.json`` in a directory."""
    items, _ = load_annotations_with_report(source, strict=strict, filters=filters)
    return items


def load_annotations_with_report(
    source: str,
    strict: bool = False,
    filters: Optional[Filters] = None,
) -> Tuple[List[Interaction], LoadReport]:
    """Same as ``load_annotations`` but also returns the data-quality ``LoadReport``."""
    if os.path.isdir(source):
        paths = sorted(glob.glob(os.path.join(source, "*.json")))
    else:
        paths = [source]
    rep = LoadReport()
    items: List[Interaction] = []
    for p in paths:
        items.extend(load_annotation_file(p, strict=strict, filters=filters, report=rep))
    rep.validation_issues = validate_interactions(items, strict=strict)
    return items, rep

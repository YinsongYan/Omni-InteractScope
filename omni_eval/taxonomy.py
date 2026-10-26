"""Metric routing: map each ground-truth response to a *metric profile* describing how it is
scored. Routing is data-driven (survives new task categories) with optional per-subtask
overrides from ``configs/tasks.yaml``.

matching_mode:
    point   -> notify_trigger_time ± tolerance (default)
    window  -> valid_response_window (± tolerance) present  (narration / interpretation)
    noise   -> expected_behavior == "silence" (turn-taking negative; model must stay silent)

content_scorer:
    judge       -> binary LLM judge (Correct/Incorrect) on the reply text (text / ui / multimodal)
    translation -> binary judge with a translation-quality rubric (cross-lingual)
    tool_set    -> set-F1 over parsed tool-call names (response_type == toolcall)
    tool_judge  -> tool-name set-F1 AND a binary judge on the marker-stripped body; both must
                   pass (the tool-call tasks, whose GT mixes ``<call:..>`` with spoken text)
    integer     -> exact integer match (counting subtasks; opt-in via tasks.yaml)
    state       -> exact normalized-string match (state tracking; opt-in)
    none        -> time-only (any in-window fire is correct; no content check)

``metrics/content.py`` also implements a 9-region grounding matcher, but it is not routable: no
annotation in the release carries region ground truth. Re-add ``"region"`` below when it does.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from .data.response_parser import parse_reply
from .schema import Interaction, Response

MATCHING_MODES = {"point", "window", "noise"}
CONTENT_SCORERS = {
    # `tool_name` is T1's scorer; `tool_judge` is its retired predecessor, still accepted so an
    # archived config can be re-run (see metrics/content.py's header).
    "judge", "translation", "tool_set", "tool_name", "tool_judge", "integer", "state", "none",
}


JUDGE_RUBRICS = {"general", "event", "translation"}


@dataclass
class MetricProfile:
    matching_mode: str
    content_scorer: str
    judge_rubric: str = "general"  # which binary judge prompt: general | event | translation

    @property
    def is_time_only(self) -> bool:
        return self.content_scorer == "none"

    @property
    def uses_judge(self) -> bool:
        return self.content_scorer in ("judge", "translation", "tool_judge")


class TaxonomyRouter:
    """Routes (interaction, response) -> MetricProfile.

    ``overrides`` maps a key to a partial profile dict, e.g.::

        {"1.a.i": {"content_scorer": "none"},
         "forward_active_responding/counting": {"content_scorer": "integer"}}

    Keys are matched most-specific first: ``"<task>/<subtask>"`` then ``"<task>"``.
    """

    def __init__(self, overrides: Optional[Dict[str, dict]] = None):
        self.overrides = overrides or {}

    def _override_for(self, it: Interaction) -> dict:
        for key in (f"{it.task_type}/{it.sub_task_type}", it.task_type):
            if key in self.overrides:
                return self.overrides[key]
        return {}

    def route(self, it: Interaction, r: Response) -> MetricProfile:
        # matching mode
        if r.is_silence:
            matching_mode = "noise"
        elif r.is_windowed:
            matching_mode = "window"
        else:
            matching_mode = "point"

        # content scorer (default from response_type)
        gt_tools = parse_reply(r.assistant_reply_text).tool_calls
        if matching_mode == "noise":
            content_scorer = "none"
        elif r.response_type == "toolcall":
            content_scorer = "tool_set"
        elif r.is_crosslingual:
            content_scorer = "translation"
        elif gt_tools and r.response_type in ("ui", "multimodal", ""):
            # GT mixes a tool call with spoken text (or declares a UI card alongside one) ->
            # both halves matter, so gate on the tool set AND judge the body.
            content_scorer = "tool_judge"
        elif r.response_type in ("text", "ui", "multimodal", ""):
            content_scorer = "judge"
        else:
            content_scorer = "judge"

        # judge rubric (binary prompt selection). matching_mode no longer distinguishes
        # proactive vs reactive, so the event prompt is opt-in via tasks.yaml ``judge_rubric``;
        # translation content always uses the translation prompt (re-asserted after the overrides).
        judge_rubric = "translation" if content_scorer == "translation" else "general"

        # apply overrides
        ov = self._override_for(it)
        matching_mode = ov.get("matching_mode", matching_mode)
        content_scorer = ov.get("content_scorer", content_scorer)
        judge_rubric = ov.get("judge_rubric", judge_rubric)
        if ov.get("time_only"):
            content_scorer = "none"
        if content_scorer == "translation":
            # Not a preference an override may express: the translation prompt is the only one that
            # can judge a reply whose content IS a translation. This is what lets a task-level
            # rubric cover a mixed task type -- `simultaneous_speech` is `event` for its narration
            # items and `translation` for its interpretation ones, chosen by the language fields on
            # the response rather than by the annotator's (demonstrably wrong) subtype letter.
            judge_rubric = "translation"

        assert matching_mode in MATCHING_MODES, matching_mode
        assert content_scorer in CONTENT_SCORERS, content_scorer
        assert judge_rubric in JUDGE_RUBRICS, judge_rubric
        return MetricProfile(
            matching_mode=matching_mode, content_scorer=content_scorer, judge_rubric=judge_rubric
        )

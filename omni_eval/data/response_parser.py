"""Shared response parser.

Both ground-truth ``assistant_reply_text`` and model predictions use the same lightweight
markup convention:

    tool call : ``<call:web_search>``           -> tool_calls = ["web_search"]
    UI        : ``<ui:商品信息卡片>``            -> ui = ["商品信息卡片"]
    text      : any residual natural language

A ``multimodal`` reply mixes them, e.g.
    ``<call:weather_fetch>今天早上可能会下雨，出门注意带伞``
    ``<ui:商品信息卡片>这是关于香奈儿轻盈蜜粉的商品信息。``

Parsing GT and prediction with the *same* function lets tool/ui/multimodal scoring work
without per-model special-casing. Per the approved plan, only ``toolcall`` responses are
scored on the parsed tool set (set-F1); ``text`` / ``ui`` / ``multimodal`` are judged on the
full reply text by the LLM judge. The parsed ``tool_calls`` / ``ui`` fields are still emitted
for diagnostics and error analysis.

``extract_tool_calls`` (bottom of the file) is the prediction-side counterpart: same question,
but tolerant of the markup shapes models actually emit instead of the contract. ``parse_reply``
stays strict because it is shared with GT.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# <call:NAME> and <ui:NAME>  (NAME may contain CJK, spaces, underscores, digits, dashes)
_CALL_RE = re.compile(r"<\s*call\s*:\s*([^<>]+?)\s*>", re.IGNORECASE)
_UI_RE = re.compile(r"<\s*ui\s*:\s*([^<>]+?)\s*>", re.IGNORECASE)

# Thinking-mode chain of thought. InternVL3.5 emits `<think>...</think>` spontaneously (observed:
# a 800-char deliberation before a 12-char answer), and Qwen3 / MiniCPM-o have thinking modes too.
# GT never contains one, and leaving it in the scored text breaks two things at once: the LLM judge
# is handed reasoning prose instead of an answer, and the ABSTAIN sentinel becomes unreachable
# because it is matched on a *prefix* — a thinking model could never abstain.
_THINK_BLOCK_RE = re.compile(
    r"<\s*(think|thinking|reasoning)\s*>.*?<\s*/\s*\1\s*>", re.IGNORECASE | re.DOTALL
)
_THINK_CLOSE_RE = re.compile(
    r"^.*<\s*/\s*(?:think|thinking|reasoning)\s*>", re.IGNORECASE | re.DOTALL
)
_THINK_OPEN_RE = re.compile(r"<\s*(?:think|thinking|reasoning)\s*>.*$", re.IGNORECASE | re.DOTALL)


@dataclass
class ParsedResponse:
    text: str = ""
    tool_calls: List[str] = field(default_factory=list)
    ui: List[str] = field(default_factory=list)
    raw: str = ""

    @property
    def has_tool(self) -> bool:
        return len(self.tool_calls) > 0

    @property
    def has_ui(self) -> bool:
        return len(self.ui) > 0


def strip_reasoning(text: str) -> str:
    """Drop a thinking-mode chain of thought, keeping only the user-visible answer.

    Applied by the adapters to model output *before* the ABSTAIN/NO-REPLY check, so the sentinel
    still works when a model thinks first. The unstripped string is kept as the record's ``raw``,
    so nothing is lost for error analysis — only the scored text is cleaned.

    Three shapes, in order:

    - ``<think>…</think>answer``  -> ``answer`` (the normal case)
    - ``…</think>answer``        -> ``answer`` (opener swallowed by the chat template, as InternVL
      does when the template pre-fills ``<think>``); cut at the LAST closing tag
    - ``<think>answer-never-came`` -> ``""`` (generation hit ``max_new_tokens`` mid-thought). Empty
      means abstain, which is the honest reading: the model produced no answer. Do not try to
      salvage a sentence out of the reasoning — that would score the model on prose it never
      addressed to the user.

    Note ``parse_reply`` / ``infer_response_type`` are deliberately NOT changed: they are shared
    with GT scoring in ``metrics/``, and GT has no reasoning to strip.
    """
    s = text or ""
    s = _THINK_BLOCK_RE.sub("", s)
    s = _THINK_CLOSE_RE.sub("", s)
    s = _THINK_OPEN_RE.sub("", s)
    return s.strip()


def parse_reply(reply: str) -> ParsedResponse:
    """Split a reply string into (text, tool_calls, ui)."""
    raw = reply or ""
    tool_calls = [m.strip() for m in _CALL_RE.findall(raw)]
    ui = [m.strip() for m in _UI_RE.findall(raw)]
    text = _UI_RE.sub("", _CALL_RE.sub("", raw)).strip()
    return ParsedResponse(text=text, tool_calls=tool_calls, ui=ui, raw=raw)


# =======================================================================================
# Format-tolerant tool-call extraction — PREDICTIONS ONLY
# =======================================================================================
# ``parse_reply`` above is deliberately left alone: it parses ground truth too, and every one of
# the 209 T1 GT responses carries exactly one clean ``<call:NAME>`` marker (measured: 125
# image_search / 72 web_search / 12 audio_search, 0 without a marker). Loosening the shared parser
# would change what the GT means; the tolerance belongs on the prediction side only.
#
# Why tolerate at all: the prompt asks for ``<call:NAME>``, and a model that picks the RIGHT tool
# but writes it as ``<image_search:frame_0 query:这款发型>`` has made an instruction-following slip,
# not a tool-selection error. Measured over the 11 completed runs, that conflation is worth ~20
# accuracy points on T1 and it REORDERS the models: strict-only says internvl3.5 (.321) > qwen3-vl
# (.292) >> qwen3-omni-30b (.177); format-tolerant says internvl3.5 (.373) ≈ qwen3-omni-30b (.368)
# > qwen3-vl (.301). So the strict number alone would have published a last place that is an
# artefact of markup habits. Both are reported: `shape` is what makes the strict number (and the
# markup-compliance rate) recoverable from the tolerant one — see scripts/tool_report.py.
#
# EVERY shape below is anchored on the tool vocabulary (configs/tools.yaml), never on a free
# identifier. That is the property that keeps this from inventing tool names: a name that is not in
# the vocabulary can never be extracted, so tolerance can only recover calls to tools the model was
# actually told about. The shapes are the ones OBSERVED in 11,656 emits over the T1 items. The only
# speculative one is ``json_name`` (0 occurrences so far, kept because it is the native tool-call
# format of the two streaming checkpoints that had not run yet); the bare ``<NAME>`` arm of
# ``angle_name`` was added on that reasoning too and then turned out to account for 12 real
# recoveries (``<web_search>电钻的优点</web_search>``), all 12 of them correct tool choices.
_SHAPE_LADDER = ("pair_tag", "angle_name", "json_name", "python_call")

# The leading identifier of a malformed marker body: `<call:web_search(query="…")>` -> web_search.
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass
class ToolExtraction:
    """Tool names recovered from a model reply, plus the markup shape they were recovered from.

    ``shape`` is one of:

    ==============  ====================================================  ============  =====
    shape           example                                               obs. on T1    valid
    ==============  ====================================================  ============  =====
    ``exact``       ``<call:web_search>``  (the contract)                        7,441   yes
    ``name_args``   ``<call:image_search(15, "seagulls in the water")>``           463   yes
    ``pair_tag``    ``<call>web_search(ripe durian)</call>``                        18   yes
    ``angle_name``  ``<image_search:frame_0 …>``, ``<web_search>q</web_search>``   ~800   yes
    ``json_name``   ``<tool_call>{"name": "web_search", …}``                         0   yes
    ``python_call`` ``web_search("小银鲳鱼价格")``                                ~1,500   yes
    ``unknown_name````<call:tool_name>`` — call-shaped, name not in vocab             8   no
    ``none``        no call of any recognised shape                                  —   no
    ==============  ====================================================  ============  =====

    A bare in-prose mention (``用 web_search 查一下``, 915 occurrences) is deliberately ``none``:
    mentioning a tool is not calling one, and counting it is worth ≤0.9 accuracy points on any run.
    """

    names: List[str] = field(default_factory=list)
    shape: str = "none"

    @property
    def is_call(self) -> bool:
        """Did the model produce something call-shaped at all (even with an unusable name)?"""
        return self.shape != "none"

    @property
    def markup_compliant(self) -> bool:
        """Did it use the exact markup the prompt asked for? The denominator for the compliance
        rate is ``is_call``, so a model is only judged on markup when it tried to call something."""
        return self.shape == "exact"


def default_tool_vocab(path: Optional[str] = None) -> Tuple[str, ...]:
    """Every tool name that may be recovered from a prediction, from ``configs/tools.yaml``.

    Disclosed tools **plus** ``retired:`` ones. The retired names are not in any prompt, so a model
    emitting one has picked a tool that does not exist — but it picked it with correct markup, and
    folding that into ``unknown_name`` would blame the format for a selection error. Either way the
    name is not the GT primary, so the accuracy is identical; only the diagnosis differs.

    Imported lazily: this module is imported by every adapter, and ``config`` pulls in yaml.
    """
    from ..config import load_tool_registry

    reg = load_tool_registry(path)
    return tuple(dict(reg.get("tools") or {})) + tuple(reg.get("retired") or {})


@lru_cache(maxsize=8)
def _shape_res(vocab: Tuple[str, ...]) -> Dict[str, "re.Pattern"]:
    """Compile the recovery patterns for one vocabulary. Cached: called per scored response."""
    # Longest-first so `web_search` cannot be shadowed by a hypothetical `web` prefix.
    alt = "|".join(re.escape(v) for v in sorted(vocab, key=len, reverse=True))
    return {
        # <call>web_search(ripe durian)</call> — the paired-tag reading of the same contract.
        "pair_tag": re.compile(r"<\s*call\s*>\s*(%s)\b" % alt, re.IGNORECASE),
        # <image_search:frame_0 …> / <image_search(frame_ref, "…")> / <function=web_search> /
        # <web_search> — the tool name promoted into the tag name.
        "angle_name": re.compile(r"<\s*(?:function\s*=\s*)?(%s)\s*(?:[:(]|>)" % alt, re.IGNORECASE),
        # {"name": "web_search", "arguments": {...}} — HF's native tool-call JSON.
        "json_name": re.compile(r"""["']name["']\s*:\s*["'](%s)["']""" % alt, re.IGNORECASE),
        # web_search("小银鲳鱼价格") — no markup at all, just a call written as code. The full-width
        # paren is in there because CJK-heavy decoding produces it.
        "python_call": re.compile(r"(?<![\w.])(%s)\s*[(（]" % alt, re.IGNORECASE),
    }


def _dedup(names: Iterable[str]) -> List[str]:
    out: List[str] = []
    for n in names:
        if n not in out:
            out.append(n)
    return out


def extract_tool_calls(reply: str, vocab: Optional[Sequence[str]] = None) -> ToolExtraction:
    """Recover the tool names a prediction meant to call, tolerating markup slips.

    The shapes are tried in a fixed order and the FIRST one that yields a vocabulary name wins:
    ``exact``/``name_args`` (the ``<call:>`` marker) → ``pair_tag`` → ``angle_name`` → ``json_name``
    → ``python_call``. Short-circuiting matters because the shapes overlap by construction —
    ``<call:image_search(15, "x")>`` also matches ``angle_name``'s inner text and ``python_call`` —
    and ``shape`` has to name the *best* form the model used, or the compliance rate is meaningless.

    ``vocab`` defaults to :func:`default_tool_vocab`. Pass it explicitly in tests and in tooling
    that must not depend on the shipped registry.
    """
    raw = reply or ""
    if vocab is None:
        vocab = default_tool_vocab()
    canon = {v.lower(): v for v in vocab}

    # 1. The contract marker. Present but unusable is still evidence of intent, so a marker whose
    #    body resolves to nothing does NOT return early -- the ladder still gets a turn, and only
    #    then does it fall back to `unknown_name`.
    bodies = [m.strip() for m in _CALL_RE.findall(raw)]
    names, slipped = [], False
    for body in bodies:
        if body.lower() in canon:
            names.append(canon[body.lower()])
            continue
        slipped = True   # anything other than a bare vocabulary name is a markup slip
        head = _IDENT_RE.match(body)
        if head and head.group(0).lower() in canon:
            names.append(canon[head.group(0).lower()])
    if names:
        return ToolExtraction(_dedup(names), "name_args" if slipped else "exact")

    # 2. The recovery ladder, vocabulary-anchored throughout.
    res = _shape_res(tuple(vocab))
    for shape in _SHAPE_LADDER:
        found = [canon[m.lower()] for m in res[shape].findall(raw)]
        if found:
            return ToolExtraction(_dedup(found), shape)

    return ToolExtraction([], "unknown_name" if bodies else "none")


def infer_response_type(reply: str) -> str:
    """Best-effort response_type from a raw reply (used when a model does not declare one)."""
    p = parse_reply(reply)
    has_text = bool(p.text)
    n_kinds = int(p.has_tool) + int(p.has_ui) + int(has_text)
    if n_kinds >= 2:
        return "multimodal"
    if p.has_tool:
        return "toolcall"
    if p.has_ui:
        return "ui"
    return "text"

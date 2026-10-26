import os

import pytest

from omni_eval.data.loader import load_annotations
from omni_eval.data.response_parser import (
    default_tool_vocab,
    extract_tool_calls,
    infer_response_type,
    parse_reply,
)
from omni_eval.schema import Interaction

FIXTURES = "tests/fixtures"


def test_load_and_ms_to_sec():
    items = load_annotations(FIXTURES)
    assert len(items) >= 8
    by_id = {it.interaction_id: it for it in items}

    ev = by_id["c1_event_notify_01"]
    assert ev.responses[0].notify_trigger_time == 10000
    assert ev.responses[0].notify_trigger_sec == 10.0  # ms -> sec

    narr = by_id["c1_live_narration_01"]
    assert narr.responses[0].valid_response_window == [8000, 16000]
    assert narr.responses[0].valid_response_window_sec == [8.0, 16.0]
    assert narr.responses[0].is_windowed


def test_silence_and_audio_flags():
    by_id = {it.interaction_id: it for it in load_annotations(FIXTURES)}
    noise = by_id["c2_turn_taking_noise_01"]
    assert noise.responses[0].is_silence
    # audio query interaction needs audio
    aud = by_id["c3_audio_query_tool_01"]
    assert aud.needs_audio and aud.query_type == "audio"
    # vision_audio trigger needs audio
    acc = by_id["c4_accessibility_reminder_01"]
    assert acc.needs_audio


def test_long_memory_origin():
    by_id = {it.interaction_id: it for it in load_annotations(FIXTURES)}
    wf = by_id["c4_workflow_correction_01"]
    r = wf.responses[0]
    assert r.long_memory == "yes" and r.long_memory_start_sec == 2.0


def test_response_parser():
    p = parse_reply("<call:weather_fetch>It may rain, take an umbrella.")
    assert p.tool_calls == ["weather_fetch"]
    assert "umbrella" in p.text
    assert infer_response_type("<call:web_search>") == "toolcall"
    assert infer_response_type("<ui:card>hello") == "multimodal"
    assert infer_response_type("just text") == "text"
    p2 = parse_reply("<ui:product_card>Here is the info.")
    assert p2.ui == ["product_card"] and p2.text == "Here is the info."


# ---- format-tolerant tool extraction (predictions only) -------------------------------------
# The strings below are REAL, taken from the 11 completed runs' predictions. T1 accuracy moves by up
# to 21 points depending on whether these count, and the strict-only reading reorders the models, so
# each shape is pinned rather than left to a regex nobody re-reads.
@pytest.mark.parametrize("reply,names,shape", [
    # the contract
    ("<call:web_search>正在帮你查一下", ["web_search"], "exact"),
    # arguments stuffed inside the marker -> take the leading identifier (455 of 463 resolve)
    ('<call:image_search(15, "seagulls in the water")>', ["image_search"], "name_args"),
    ('<call:web_search(query="小银鲳鱼 价格")>', ["web_search"], "name_args"),
    # the paired-tag reading of the same contract
    ("<call>web_search(ripe durian)</call>", ["web_search"], "pair_tag"),
    # the tool name promoted into the tag name
    ("<image_search:frame_0000 query:这款发型叫什么>", ["image_search"], "angle_name"),
    ('<image_search(frame_ref, "这是什么花")>', ["image_search"], "angle_name"),
    ("<function=audio_search>", ["audio_search"], "angle_name"),
    # HF's native tool-call JSON (0 occurrences in the 11 runs; the streaming checkpoints may emit it)
    ('<tool_call>\n{"name": "web_search", "arguments": {"query": "durian"}}\n</tool_call>',
     ["web_search"], "json_name"),
    # no markup at all, just the call written as code
    ('web_search("小银鲳鱼价格")', ["web_search"], "python_call"),
    ("image_search（这是什么品种）", ["image_search"], "python_call"),
])
def test_extract_recovers_every_observed_markup_shape(reply, names, shape):
    ex = extract_tool_calls(reply)
    assert (ex.names, ex.shape) == (names, shape)
    assert ex.is_call
    assert ex.markup_compliant is (shape == "exact")


def test_a_bare_mention_in_prose_is_not_a_call():
    """915 emits merely name a tool mid-sentence. Counting them would be worth <=0.9 accuracy points
    and would make "mentioned" indistinguishable from "called", which is the thing being measured."""
    for reply in ("我可以用 web_search 帮你查一下，需要吗？",
                  "建议使用 image_search 这个工具。",
                  "web_search 是一个搜索工具"):
        ex = extract_tool_calls(reply)
        assert (ex.names, ex.shape, ex.is_call) == ([], "none", False), reply


def test_a_name_outside_the_vocabulary_is_never_invented():
    """Every recovery rule is anchored on configs/tools.yaml, so tolerance can only recover calls to
    tools the model was actually told about -- it can never manufacture a tool name."""
    ex = extract_tool_calls("<call:tool_name>")           # observed placeholder leakage
    assert ex.names == [] and ex.shape == "unknown_name"  # call-shaped, so still counts as trying
    assert extract_tool_calls('<call:super_finder(query="x")>').names == []
    assert extract_tool_calls('super_finder("x")').shape == "none"
    assert extract_tool_calls("<maps:frame_0>").names == []
    # ... and a marker that resolves to nothing does not swallow a recoverable call next to it.
    mixed = extract_tool_calls("<call:tool_name> 然后 <image_search:frame_1 query:这是什么>")
    assert (mixed.names, mixed.shape) == (["image_search"], "angle_name")


def test_default_vocab_comes_from_the_registry_and_covers_retired_names():
    vocab = default_tool_vocab()
    assert {"web_search", "image_search", "audio_search", "navigation_tool"} <= set(vocab)
    assert "translate" in vocab, "reachable distractor must stay recoverable"
    # Retired names resolve so that correct markup with a nonexistent tool is diagnosed as a
    # selection error, not a format error. Either way it is not the GT primary, so acc is unchanged.
    assert extract_tool_calls("<call:web_fetch>").shape == "exact"


def test_from_dict_defaults():
    it = Interaction.from_dict({"interaction_id": "x", "responses": [{"response_id": 1}]})
    assert it.query_type == "text"
    assert it.responses[0].expected_behavior == "respond"

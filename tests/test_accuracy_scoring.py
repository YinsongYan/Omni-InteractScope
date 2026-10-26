"""Locked scoring model: every task is a `point` item scored to a binary per-Response success,
rolled up to per-category accuracy.

success = FIRST relevant fire lands in [T-tol, T+tol]  AND  content is correct
          (premature 抢跑 / miss / wrong-content all fail).

Covers: assign_first_fire (first-fire + neighborhood boundary), the five per-item outcomes
(ontime-ok / premature / miss / content-fail / tool-set), per-category accuracy roll-up, and
judge-rubric routing (general | event | translation) end-to-end.
"""
import pytest

from omni_eval.config import get, load_config
from omni_eval.metrics.aggregate import aggregate, score_interaction
from omni_eval.metrics.llm_judge import LLMJudge
from omni_eval.metrics.temporal import EmitItem, assign_first_fire, make_gt
from omni_eval.schema import Interaction
from omni_eval.taxonomy import TaxonomyRouter

CFG = load_config(None)  # temporal.tolerance_sec=2.0, merge_gap_sec=2.0, noise_window_sec=4.0


# ---- fixtures ------------------------------------------------------------------------
def _emits(pairs, rtype="text"):
    return [EmitItem(time_sec=t, reply_text=txt, response_type=rtype) for t, txt in pairs]


def _it(responses, task="t", sub="s", query_text="q"):
    return Interaction.from_dict(
        {"interaction_id": "x", "task_type": task, "sub_task_type": sub,
         "query_text": query_text, "query_time": 0, "responses": responses}
    )


def _resp(rid, trigger_sec, reply, rtype="text", **kw):
    d = {"response_id": rid, "notify_trigger_time": int(trigger_sec * 1000),
         "assistant_reply_text": reply, "response_type": rtype}
    d.update(kw)
    return d


def _rec(pairs, rtype="text"):
    return {"protocol": "online_rolling", "modality_tag": "native", "supported": True,
            "emits": [{"time_sec": t, "response_type": rtype, "reply_text": txt, "raw": txt}
                      for t, txt in pairs], "meta": {}}


def _score(it, rec, judge=None):
    judge = judge or LLMJudge(mock=True, mode="binary")
    return score_interaction(it, rec, TaxonomyRouter({}), judge, CFG)


# ---- assign_first_fire (temporal level) ----------------------------------------------
def test_first_fire_ontime_premature_miss():
    gts = [make_gt(0, center=10.0, tol=2.0)]
    assert assign_first_fire(_emits([(11.0, "x")]), gts, 2.0)[0].status == "ontime"
    assert assign_first_fire(_emits([(5.0, "x")]), gts, 2.0)[0].status == "premature"
    assert assign_first_fire(_emits([(30.0, "x")]), gts, 2.0)[0].status == "miss"
    assert assign_first_fire([], gts, 2.0)[0].status == "miss"


def test_first_fire_neighborhood_boundary():
    """An early blurt fails event#1 but must NOT be mis-charged as event#2's 抢跑: event#2 with
    its own on-time fire still succeeds (lower bound = midpoint between consecutive triggers)."""
    gts = [make_gt(0, center=10.0, tol=2.0), make_gt(1, center=20.0, tol=2.0)]
    out = {o.gt.gid: o.status for o in assign_first_fire(_emits([(5.0, "a"), (11.0, "b"), (19.0, "c")]), gts, 2.0)}
    assert out == {0: "premature", 1: "ontime"}
    # no early blurt -> both on-time
    out2 = {o.gt.gid: o.status for o in assign_first_fire(_emits([(11.0, "b"), (19.0, "c")]), gts, 2.0)}
    assert out2 == {0: "ontime", 1: "ontime"}


# ---- five per-item outcomes (score_interaction) --------------------------------------
def test_point_ontime_correct_is_success():
    it = _it([_resp(1, 10.0, "the cat jumped onto the table")])
    r = _score(it, _rec([(11.0, "the cat jumped onto the table")]))
    assert r["success_hit"] == 1 and r["success_n"] == 1
    assert r["status_counts"] == {"ok": 1}


def test_point_premature_fails():
    it = _it([_resp(1, 10.0, "the cat jumped onto the table")])
    r = _score(it, _rec([(5.0, "the cat jumped onto the table")]))  # correct content, but 抢跑
    assert r["success_hit"] == 0 and r["status_counts"] == {"premature": 1}


def test_shipped_tolerance_is_plus_minus_two_seconds():
    """The locked acceptable-response window is [T-2, T+2] -- 4.0 s wide.

    Pins the shipped configs/eval.yaml, not the scorer: `assign_first_fire` takes tol as an
    argument, so a silent edit of the knob would rescale every published number without failing a
    single unit test. +-1 s was tried on 2026-09-01 and reverted the same day (the annotation's own
    noise floor is 1.62 s median absolute error, wider than a +-1 s half-window) -- see the comment
    in configs/eval.yaml::temporal for the full argument.
    """
    assert get(CFG, "temporal.tolerance_sec") == 2.0
    # merge_gap must stay <= 2*tol, or merging destroys two fires the matcher could tell apart.
    assert get(CFG, "temporal.merge_gap_sec") <= 2 * get(CFG, "temporal.tolerance_sec")

    it = _it([_resp(1, 10.0, "the cat jumped onto the table")])
    at_edge = _score(it, _rec([(12.0, "the cat jumped onto the table")]))
    assert at_edge["status_counts"] == {"ok": 1}, "T+2.0 is inside the window"
    just_out = _score(it, _rec([(12.5, "the cat jumped onto the table")]))
    assert just_out["status_counts"] == {"miss": 1}, "T+2.5 must be outside the window"
    early = _score(it, _rec([(7.5, "the cat jumped onto the table")]))
    assert early["status_counts"] == {"premature": 1}, "T-2.5 must be 抢跑"


def test_point_ontime_wrong_content_fails():
    it = _it([_resp(1, 10.0, "the cat jumped onto the table")])
    r = _score(it, _rec([(11.0, "the weather is sunny today")]))
    assert r["success_hit"] == 0 and r["status_counts"] == {"content_fail": 1}


def test_point_miss_fails():
    it = _it([_resp(1, 10.0, "the cat jumped onto the table")])
    r = _score(it, _rec([(30.0, "the cat jumped onto the table")]))
    assert r["success_hit"] == 0 and r["status_counts"] == {"miss": 1}


# ---- tool call: contains AND only contains the GT tools (exact set) -------------------
def test_toolcall_exact_set_is_success():
    it = _it([_resp(1, 10.0, "<call:web_search>", rtype="toolcall")])
    r = _score(it, _rec([(11.0, "<call:web_search>")], rtype="toolcall"))
    assert r["success_hit"] == 1 and r["status_counts"] == {"ok": 1}


def test_toolcall_extra_tool_fails():
    it = _it([_resp(1, 10.0, "<call:web_search>", rtype="toolcall")])
    r = _score(it, _rec([(11.0, "<call:web_search><call:maps>")], rtype="toolcall"))
    assert r["success_hit"] == 0  # 包含但不只包含 -> precision<1 -> fail


def test_toolcall_wrong_tool_fails():
    it = _it([_resp(1, 10.0, "<call:web_search>", rtype="toolcall")])
    r = _score(it, _rec([(11.0, "<call:maps>")], rtype="toolcall"))
    assert r["success_hit"] == 0


# ---- per-category accuracy roll-up ---------------------------------------------------
def test_per_category_accuracy_rollup():
    good = _score(_it([_resp(1, 10.0, "alpha beta gamma")], sub="A"),
                  _rec([(11.0, "alpha beta gamma")]))
    bad = _score(_it([_resp(1, 10.0, "alpha beta gamma")], sub="B"),
                 _rec([(4.0, "alpha beta gamma")]))  # premature
    agg = aggregate([good, bad], CFG)
    subs = agg["by_category"]["t"]["by_subtask"]
    assert subs["A"]["accuracy"] == 1.0
    assert subs["B"]["accuracy"] == 0.0
    assert agg["overall"]["micro"]["accuracy"] == 0.5  # 1 of 2 items
    assert agg["overall"]["macro"]["accuracy"] == 0.5  # mean over subtasks


# ---- judge-rubric routing (general | event | translation) ----------------------------
def test_rubric_default_general_and_event_override():
    it = _it([_resp(1, 10.0, "someone walked in", rtype="text")])
    r = it.responses[0]
    assert TaxonomyRouter({}).route(it, r).judge_rubric == "general"
    ov = TaxonomyRouter({"t/s": {"judge_rubric": "event"}})
    assert ov.route(it, r).judge_rubric == "event"


def test_rubric_translation_autoselected():
    it = _it([_resp(1, 10.0, "你好", rtype="text", source_language="en", target_language="zh")])
    prof = TaxonomyRouter({}).route(it, it.responses[0])
    assert prof.content_scorer == "translation" and prof.judge_rubric == "translation"


def test_event_rubric_forwarded_to_judge_end_to_end():
    """proactive_notify override -> judge is actually called with rubric='event'."""
    class _RecJudge(LLMJudge):
        def __init__(self):
            super().__init__(mock=True, mode="binary")
            self.seen = []

        def score(self, question, gt, pred, rubric="general"):
            self.seen.append(rubric)
            return super().score(question, gt, pred, rubric)

    it = _it([_resp(1, 10.0, "a dog appears", rtype="text")], sub="proactive_notify")
    router = TaxonomyRouter({"t/proactive_notify": {"judge_rubric": "event"}})
    rj = _RecJudge()
    score_interaction(it, _rec([(11.0, "a dog appears")]), router, rj, CFG)
    assert "event" in rj.seen and set(rj.seen) == {"event"}


# ---- tool_judge routing --------------------------------------------------------------
def test_tool_judge_autoselected_when_gt_mixes_a_call_with_speech():
    it = _it([_resp(1, 10.0, "<call:web_search>正在为你搜索价格", rtype="")])
    assert TaxonomyRouter({}).route(it, it.responses[0]).content_scorer == "tool_judge"
    # a declared toolcall response stays on the tool-set-only scorer
    bare = _it([_resp(1, 10.0, "<call:web_search>", rtype="toolcall")])
    assert TaxonomyRouter({}).route(bare, bare.responses[0]).content_scorer == "tool_set"
    # plain speech is unaffected
    plain = _it([_resp(1, 10.0, "someone walked in", rtype="text")])
    assert TaxonomyRouter({}).route(plain, plain.responses[0]).content_scorer == "judge"


def test_tasks_yaml_pins_tool_name_for_the_search_ui_tasks():
    """The shipped configs/tasks.yaml must keep both T1 task types on `tool_name` — the whole T1
    column of the main table is this one line, and the previous scorer (`tool_judge`) scored 0."""
    from omni_eval.config import load_task_overrides

    ov = load_task_overrides("configs/tasks.yaml")
    for tt in ("simultaneous_tool_calls_search_ui",
               "simultaneous_tool_calls_search_generative_ui"):
        assert ov[tt]["content_scorer"] == "tool_name", tt
    assert ov["simultaneous_speech"]["judge_rubric"] == "event"


# ---- tool_name: T1's scorer = timing gate x strict tool name, tolerant markup -----------
_T1 = "simultaneous_tool_calls_search_ui"


def _score_t1(gt_reply, emits):
    """One T1 interaction scored through the SHIPPED configs/tasks.yaml routing."""
    from omni_eval.config import load_task_overrides

    it = _it([_resp(1, 10.0, gt_reply)], task=_T1, sub="A", query_text="这是什么鱼？")
    router = TaxonomyRouter(load_task_overrides("configs/tasks.yaml"))
    assert router.route(it, it.responses[0]).content_scorer == "tool_name"
    return score_interaction(it, _rec(emits), router, LLMJudge(mock=True, mode="binary"), CFG)


@pytest.mark.parametrize("emitted,ok", [
    ("<call:image_search>这是银鲳鱼", True),                       # the contract
    ('<call:image_search(0, "这是什么鱼")>', True),                # args inside the marker
    ("<image_search:frame_0000 query:这是什么鱼>", True),          # tag-name form
    ('image_search("这是什么鱼")', True),                          # bare code form
    ("<call:web_search>这是银鲳鱼", False),                        # wrong input modality routed
    ("<call:image_search><call:web_search>", False),               # extra tool -> precision < 1
    ("这是一条银鲳鱼，市场价约 30 元一斤。", False),                  # answered instead of calling
])
def test_tool_name_accepts_any_markup_but_only_the_right_tool(emitted, ok):
    r = _score_t1("<call:image_search>这是银鲳鱼，市场价约 30 元一斤", [(11.0, emitted)])
    assert r["success_hit"] == int(ok), r["statuses"] if "statuses" in r else r["status_counts"]


def test_tool_name_does_not_require_the_model_to_produce_the_search_result():
    """The retired `tool_judge` also judged the body, which no model can pass: the GT body holds the
    search RESULT (annotators had a browser), the model body holds the search QUERY, and the harness
    never executes a tool. A right-tool call whose body is the query must score ok."""
    r = _score_t1("<call:web_search>开封后的大米在常温下只能存放 90 天左右",
                  [(11.0, "<call:web_search>大米 开封后 保存期限")])
    assert r["success_hit"] == 1 and r["status_counts"] == {"ok": 1}


def test_tool_name_still_fails_a_premature_call():
    """Content tolerance changes nothing about the timing gate: T1 accuracy stays two-gated."""
    r = _score_t1("<call:image_search>这是银鲳鱼", [(4.0, "<call:image_search>")])
    assert r["success_hit"] == 0 and r["status_counts"] == {"premature": 1}


def test_translation_rubric_survives_a_task_level_event_override():
    """``simultaneous_speech`` holds both narration and interpretation items, and its subtype letter
    cannot tell them apart (wrong for ~13 interactions). So the task-level ``event`` rubric must
    apply to narration while a cross-lingual response still gets the translation prompt."""
    from omni_eval.config import load_task_overrides

    router = TaxonomyRouter(load_task_overrides("configs/tasks.yaml"))
    narration = _it([_resp(1, 10.0, "他正在把面团擀开")], task="simultaneous_speech", sub="B")
    p = router.route(narration, narration.responses[0])
    assert (p.content_scorer, p.judge_rubric) == ("judge", "event")

    interp = _it([_resp(1, 10.0, "Hey, look at you", source_language="zh", target_language="en")],
                 task="simultaneous_speech", sub="B")  # letter says narration; the fields say not
    p = router.route(interp, interp.responses[0])
    assert (p.content_scorer, p.judge_rubric) == ("translation", "translation")

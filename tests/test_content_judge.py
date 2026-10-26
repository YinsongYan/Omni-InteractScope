from omni_eval.metrics.content import score_content, set_f1
from omni_eval.metrics.llm_judge import JudgeRecorder, LLMJudge


def test_tool_set_f1():
    assert set_f1(["web_search"], ["web_search"]) == 1.0
    assert set_f1(["web_search"], ["unrelated"]) == 0.0
    assert 0.0 < set_f1(["a", "b"], ["a"]) < 1.0


def test_content_tool_set_scorer():
    cs = score_content("tool_set", "q", "<call:web_search>", "<call:web_search>")
    assert cs.ok and cs.value == 1.0
    cs = score_content("tool_set", "q", "<call:web_search>", "<call:other>")
    assert not cs.ok


def test_content_integer_scorer():
    cs = score_content("integer", "q", "There are 3 cups.", "I count 3 cups.")
    assert cs.ok
    cs = score_content("integer", "q", "There are 3 cups.", "I see 5 cups.")
    assert not cs.ok


def test_mock_judge_discriminates():
    judge = LLMJudge(mock=True)
    gt = "Someone just walked into the room."
    hi = judge.score("q", gt, gt).score
    lo = judge.score("q", gt, "The weather is sunny today.").score
    assert hi > lo
    assert judge.score("q", gt, "").score == 1


def test_content_judge_path():
    judge = LLMJudge(mock=True)
    gt = "The chef is chopping vegetables."
    cs = score_content("judge", "narrate", gt, gt, judge=judge)
    # The verdict is binary, so there is no 1-10 score to report (``judge10`` stays None) and
    # ``judge_pass`` never enters the decision.
    assert cs.ok and cs.judge10 is None and cs.value == 1.0
    cs = score_content("judge", "narrate", gt, "unrelated text here", judge=judge)
    assert not cs.ok and cs.value == 0.0


def test_graded_judge_mode_is_rejected():
    """A stale ``judge.mode: graded`` config must fail loudly, not score on a different scale."""
    import pytest

    with pytest.raises(ValueError, match="binary"):
        LLMJudge(mock=True, mode="graded")


def test_judge_cache(tmp_path):
    cache = str(tmp_path / "cache.jsonl")
    j1 = LLMJudge(mock=True, cache_path=cache)
    s1 = j1.score("q", "gold", "gold")
    j2 = LLMJudge(mock=True, cache_path=cache)
    s2 = j2.score("q", "gold", "gold")
    assert s2.cached and s2.score == s1.score


# ---- binary judge (locked scoring path) ------------------------------------------------
def test_binary_judge_correct_incorrect():
    judge = LLMJudge(mock=True, mode="binary")
    gt = "Someone just walked into the room."
    hit = judge.score("q", gt, gt)
    miss = judge.score("q", gt, "The weather is sunny today.")
    assert hit.correct is True and hit.mode == "binary"
    assert miss.correct is False
    # empty prediction -> Incorrect
    assert judge.score("q", gt, "").correct is False


def test_binary_content_gating():
    judge = LLMJudge(mock=True, mode="binary")
    gt = "The chef is chopping vegetables."
    cs = score_content("judge", "narrate", gt, gt, judge=judge)
    assert cs.ok and cs.value == 1.0 and cs.judge10 is None
    cs = score_content("judge", "narrate", gt, "unrelated text here", judge=judge)
    assert not cs.ok and cs.value == 0.0


def test_binary_judge_cache(tmp_path):
    cache = str(tmp_path / "cache.jsonl")
    j1 = LLMJudge(mock=True, cache_path=cache, mode="binary")
    s1 = j1.score("q", "gold", "gold")
    j2 = LLMJudge(mock=True, cache_path=cache, mode="binary")
    s2 = j2.score("q", "gold", "gold")
    assert s2.cached and s2.correct == s1.correct


# ---- tool_judge: tool names AND the spoken body must both be right ---------------------
def _tool_judge(gt, pred, judge=None, tool_f1_pass=1.0):
    return score_content("tool_judge", "what is this product?", gt, pred,
                         judge=judge or LLMJudge(mock=True, mode="binary"),
                         tool_f1_pass=tool_f1_pass)


def test_tool_judge_requires_both_halves():
    gt = "<call:web_search>这件商品品牌是多芬，正在为你搜索价格"
    assert _tool_judge(gt, gt).ok
    # right tool, unrelated speech -> fail
    assert not _tool_judge(gt, "<call:web_search>今天天气很好适合出门散步").ok
    # right speech, wrong tool -> fail
    wrong_tool = "<call:image_gen>这件商品品牌是多芬，正在为你搜索价格"
    cs = _tool_judge(gt, wrong_tool)
    assert not cs.ok and cs.kind == "tool_judge"
    # missing tool call entirely -> fail
    assert not _tool_judge(gt, "这件商品品牌是多芬，正在为你搜索价格").ok


def test_tool_judge_value_is_the_mean_of_both_halves():
    gt = "<call:web_search>正在为你搜索这件商品的价格"
    cs = _tool_judge(gt, "<call:web_search>今天天气很好适合出门散步")
    assert not cs.ok and 0.0 < cs.value < 1.0, "tool half passed, judge half failed"
    assert "f1=1.00" in cs.detail


def test_tool_judge_degrades_to_judge_without_a_gt_tool_call():
    gt = "记得把吹风机放回原处"
    ok = _tool_judge(gt, gt)
    assert ok.ok and "no-gt-tool" in ok.detail
    assert not _tool_judge(gt, "unrelated text here").ok


def test_tool_judge_bare_tool_call_scores_on_the_tool_set_only():
    cs = _tool_judge("<call:web_search>", "<call:web_search>")
    assert cs.ok and cs.value == 1.0 and cs.judge10 is None


# ---- prewarm: concurrent judging must not change any verdict ---------------------------
def test_judge_recorder_captures_the_marker_stripped_body():
    """The recorder sees what content.py actually asks for, not the raw reply."""
    judge = LLMJudge(mock=True, mode="binary")
    rec = JudgeRecorder(judge)
    gt = "<call:web_search>这件商品品牌是多芬"
    score_content("tool_judge", "q", gt, gt, judge=rec)
    assert len(rec.batch) == 1
    question, gt_text, pred_text, rubric = rec.batch[0]
    assert "<call:" not in gt_text and "<call:" not in pred_text
    assert gt_text == "这件商品品牌是多芬"


def test_prewarm_then_score_matches_serial(tmp_path):
    cache = str(tmp_path / "cache.jsonl")
    batch = [("q1", "the cat sat on the mat", "the cat sat on the mat", "general"),
             ("q2", "the cat sat on the mat", "a rocket launched at dawn", "general"),
             ("q3", "hola mundo", "hello world", "translation")]
    serial = LLMJudge(mock=True, mode="binary")
    want = [serial.score(*a).correct for a in batch]

    warm = LLMJudge(mock=True, mode="binary", cache_path=cache)
    assert warm.prewarm(batch, workers=4) == len(batch)
    got = [warm.score(*a) for a in batch]
    assert [g.correct for g in got] == want
    assert all(g.cached for g in got), "prewarm should make every later score() a cache hit"
    # a second prewarm has nothing left to call
    assert warm.prewarm(batch, workers=4) == 0


def test_prewarm_dedupes_identical_judgements():
    judge = LLMJudge(mock=True, mode="binary")
    one = ("q", "gold", "gold", "general")
    assert judge.prewarm([one, one, one], workers=4) == 1


def test_a_real_run_never_reuses_a_mock_cache_row(tmp_path):
    """The cache key holds the judge MODEL but not the mock flag, so a mock pass and a real pass
    of the same model collide -- and a mock verdict is pure lexical overlap.

    This is a regression test for a live incident: 44 rows written under
    ``gpt-4o-2024-08-06`` by an early mock pass (21 of them "Correct") were still in
    ``outputs/judge_cache.jsonl`` and were being served to the 2026-09-03 scoring campaign,
    i.e. ~1.5% of each run's judgements were decided by string overlap.
    """
    cache = str(tmp_path / "cache.jsonl")
    mock = LLMJudge(mock=True, mode="binary", cache_path=cache, model="gpt-4o-2024-08-06")
    mock.score("q", "the cat sat on the mat", "the cat sat on the mat", "general")

    real = LLMJudge(mock=False, mode="binary", cache_path=cache, model="gpt-4o-2024-08-06")
    assert real.n_mock_rows_ignored == 1, "the mock row must not enter a real run's cache"
    assert real._cache == {}

    # ... while a mock run still reuses its own rows (the cache is what makes mock runs cheap).
    again = LLMJudge(mock=True, mode="binary", cache_path=cache, model="gpt-4o-2024-08-06")
    assert again.n_mock_rows_ignored == 0
    assert again.score("q", "the cat sat on the mat", "the cat sat on the mat", "general").cached

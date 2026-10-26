from types import SimpleNamespace

import pytest

from omni_eval.metrics.reliable_judge import JudgeUnavailable, ReliableLLMJudge
from omni_eval.utils.io import append_jsonl, read_jsonl


class RateLimit(Exception):
    status_code = 429
    response = SimpleNamespace(headers={"retry-after": "7"})


class Unauthorized(Exception):
    status_code = 401


def make_judge(tmp_path, answers, **kwargs):
    delays, calls = [], []
    judge = ReliableLLMJudge(mock=False, cache_path=str(tmp_path / "cache.jsonl"),
                             sleep=delays.append, **kwargs)
    iterator = iter(answers)

    def create(**request):
        calls.append(request)
        value = next(iterator)
        if isinstance(value, Exception):
            raise value
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=value))])

    judge._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return judge, delays, calls


def test_rate_limit_waits_and_preserves_success_for_restart(tmp_path):
    judge, delays, calls = make_judge(tmp_path, [RateLimit(), "Correct"])
    assert judge.score("question", "reference", "prediction").correct is True
    assert delays == [7.0] and len(calls) == 2
    restarted, _, new_calls = make_judge(tmp_path, [])
    assert restarted.score("question", "reference", "prediction").cached
    assert new_calls == []


@pytest.mark.parametrize("answers", [[RateLimit(), RateLimit()], ["", "gibberish"],
                                    [None, "Correct or Incorrect"]])
def test_exhaustion_never_becomes_a_scientific_score(tmp_path, answers):
    judge, _, _ = make_judge(tmp_path, answers, max_retries=2)
    with pytest.raises(JudgeUnavailable) as caught:
        judge.score("q", "gt", "pred")
    assert caught.value.retryable
    assert judge._cache == {}
    assert read_jsonl(str(tmp_path / "cache.jsonl")) == []


def test_auth_failure_is_not_retried_or_cached(tmp_path):
    judge, delays, calls = make_judge(tmp_path, [Unauthorized()])
    with pytest.raises(JudgeUnavailable) as caught:
        judge.score("q", "gt", "pred")
    assert not caught.value.retryable
    assert len(calls) == 1 and delays == [] and judge._cache == {}


@pytest.mark.parametrize("raw,correct", [("error: rate limit", False), ("", True),
                                       ("Incorrect", True), ("mock:Correct", True)])
def test_invalid_historical_cache_is_rejudged(tmp_path, raw, correct):
    judge, _, _ = make_judge(tmp_path, [])
    append_jsonl(str(tmp_path / "cache.jsonl"), dict(
        key=judge.cache_key_for("q", "gt", "pred"), model=judge.model,
        mode="binary", rubric="general", correct=correct, explanation=raw))
    fresh, _, calls = make_judge(tmp_path, ["Incorrect"])
    result = fresh.score("q", "gt", "pred")
    assert result.correct is False and not result.cached and len(calls) == 1


def test_bad_response_then_valid_verdict_does_not_change_prompt(tmp_path):
    judge, _, calls = make_judge(tmp_path, ["I cannot judge", '"Correct".'])
    assert judge.score("q", "gt", "pred").correct
    assert calls[0] == calls[1]
    assert calls[0]["max_tokens"] == 10 and calls[0]["temperature"] == 0


def test_assistance_verdict_is_reused_with_its_original_rubric(tmp_path):
    from omni_eval.t8 import scoring  # registers the unchanged assistance rubric
    judge, _, calls = make_judge(tmp_path, ['Correct'])
    assert judge.score('profile', 'target', 'utterance', rubric='impairment_assistance').correct
    assert 'accessibility assistance response' in calls[0]['messages'][0]['content']
    restarted, _, new_calls = make_judge(tmp_path, [])
    assert restarted.score('profile', 'target', 'utterance', rubric='impairment_assistance').cached
    assert new_calls == []

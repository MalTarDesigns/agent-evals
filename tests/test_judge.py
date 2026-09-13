"""Tests for the model-as-judge scorer.

Every test mocks the API. None makes a network call, which is enforced by
always injecting a fake client.

The malformed-output tests carry the most weight here. A judge that
guesses a verdict from unparseable text would silently invent scores, and
an invented score in an eval harness is worse than a missing one.
"""

from __future__ import annotations

import json

import anthropic
import httpx2 as httpx
import pytest

from agent_evals.cases import Expectations
from agent_evals.runner import RunResult
from agent_evals.scorers.judge import (
    DEFAULT_JUDGE_MODEL,
    JUDGE_TEMPERATURE,
    JudgeError,
    JudgeScorer,
    build_prompt,
    judge_scorer,
    parse_verdict,
)

RUBRIC = "The draft does not invent a price that was never supplied."


class FakeBlock:
    def __init__(self, text, type="text"):
        self.type = type
        self.text = text


class FakeResponse:
    def __init__(self, text):
        self.content = [FakeBlock(text)]


class FakeMessages:
    def __init__(self, behavior):
        self.behavior = behavior if isinstance(behavior, list) else [behavior]
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        index = min(len(self.calls) - 1, len(self.behavior) - 1)
        outcome = self.behavior[index]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeClient:
    def __init__(self, behavior):
        self.messages = FakeMessages(behavior)


def reply(verdict, justification="The draft marks the cost as [confirm cost]."):
    return FakeResponse(
        json.dumps({"justification": justification, "verdict": verdict})
    )


def result(output="Good Afternoon Marcus, the cost is [confirm cost]."):
    return RunResult(case_id="sample", ok=True, output=output)


def score(behavior, rubric=RUBRIC, res=None):
    client = FakeClient(behavior)
    scorer = JudgeScorer()
    s = scorer.score(
        res or result(),
        Expectations(judge=rubric),
        case_input="Tell him what the booking calendar costs.",
        client=client,
    )
    return s, client


# --- the three verdicts -----------------------------------------------


def test_a_pass_verdict_scores_one():
    s, _ = score(reply("pass"))

    assert s.passed
    assert s.score == 1.0
    assert s.reason.startswith("pass: ")
    assert "[confirm cost]" in s.reason


def test_a_fail_verdict_scores_zero():
    s, _ = score(reply("fail", "The draft quotes $800, which was never given."))

    assert not s.passed
    assert s.score == 0.0
    assert "$800" in s.reason


def test_a_partial_verdict_scores_half_and_does_not_pass():
    """Partial is a real state, and it is not a pass."""
    s, _ = score(reply("partial", "One figure is bracketed, the other is not."))

    assert not s.passed
    assert s.score == 0.5


def test_verdicts_are_case_insensitive():
    s, _ = score(FakeResponse('{"justification": "fine", "verdict": "PASS"}'))

    assert s.passed


# --- the bias mitigations ---------------------------------------------


def test_the_judge_defaults_to_a_different_model_from_the_runner():
    """Self-preference: a model rates its own output more favorably."""
    from agent_evals.runner import DEFAULT_MODEL

    assert DEFAULT_JUDGE_MODEL != DEFAULT_MODEL


def test_temperature_is_zero():
    """Non-determinism: reduced, not eliminated."""
    _, client = score(reply("pass"))

    assert client.messages.calls[0]["temperature"] == 0.0
    assert JUDGE_TEMPERATURE == 0.0


def test_the_system_prompt_tells_the_judge_to_ignore_length():
    """Verbosity bias: judges rate longer answers higher."""
    _, client = score(reply("pass"))
    system = client.messages.calls[0]["system"]

    assert "Ignore length" in system


def test_the_system_prompt_asks_for_justification_before_verdict():
    """Post-hoc rationalization: score-then-explain fits reasoning to a number."""
    _, client = score(reply("pass"))
    system = client.messages.calls[0]["system"]

    assert "BEFORE" in system
    assert system.index('"justification"') < system.index('"verdict"')


def test_the_scale_is_three_points_not_ten():
    """Score clustering: wide scales collapse onto 7 and 8."""
    from agent_evals.scorers.judge import VERDICT_SCORES

    assert set(VERDICT_SCORES) == {"pass", "partial", "fail"}


def test_the_judge_model_is_configurable(monkeypatch):
    monkeypatch.setenv("AGENT_EVALS_JUDGE_MODEL", "claude-haiku-4-5")

    assert JudgeScorer().resolve_model() == "claude-haiku-4-5"


def test_an_explicit_model_argument_wins_over_the_env(monkeypatch):
    monkeypatch.setenv("AGENT_EVALS_JUDGE_MODEL", "claude-haiku-4-5")

    assert JudgeScorer("claude-opus-5").resolve_model() == "claude-opus-5"


# --- the rubric lives in the case -------------------------------------


def test_the_rubric_from_the_case_reaches_the_prompt():
    """Not hardcoded in the scorer. Different cases need different criteria."""
    _, client = score(reply("pass"), rubric="Closes with 'Enjoy your day,'.")
    sent = client.messages.calls[0]["messages"][0]["content"]

    assert "Closes with 'Enjoy your day,'." in sent


def test_the_prompt_carries_the_input_and_the_response():
    prompt = build_prompt("the input", "the response", "the criterion")

    assert "the input" in prompt
    assert "the response" in prompt
    assert "the criterion" in prompt


def test_the_criterion_comes_last_in_the_prompt():
    """Most recent in context when the verdict is formed."""
    prompt = build_prompt("an input", "a response", "a criterion")

    assert prompt.index("a response") < prompt.index("a criterion")


# --- malformed output -------------------------------------------------


def test_malformed_json_retries_once_then_succeeds():
    s, client = score([FakeResponse("I think it looks fine!"), reply("pass")])

    assert s.passed
    assert len(client.messages.calls) == 2


def test_malformed_json_after_the_retry_fails_as_unusable():
    """Never a guessed verdict."""
    s, client = score(FakeResponse("no json at all here"))

    assert not s.passed
    assert s.score == 0.0
    assert "not evaluated" in s.reason
    assert len(client.messages.calls) == 2


def test_json_wrapped_in_prose_is_still_read():
    s, _ = score(
        FakeResponse(
            'Here is my assessment:\n```json\n'
            '{"justification": "looks right", "verdict": "pass"}\n```'
        )
    )

    assert s.passed


def test_an_unknown_verdict_is_refused():
    with pytest.raises(JudgeError, match="unknown verdict"):
        parse_verdict('{"justification": "x", "verdict": "excellent"}')


def test_a_verdict_with_no_justification_is_refused():
    """The instruction that mitigates post-hoc reasoning was not followed."""
    with pytest.raises(JudgeError, match="no justification"):
        parse_verdict('{"justification": "  ", "verdict": "pass"}')


def test_an_empty_judge_reply_is_refused():
    with pytest.raises(JudgeError, match="empty response"):
        parse_verdict("")


def test_invalid_json_is_refused():
    with pytest.raises(JudgeError, match="not valid JSON"):
        parse_verdict('{"verdict": "pass", oops}')


def test_a_json_array_is_refused():
    """No brace-delimited object in the text, so there is nothing to read."""
    with pytest.raises(JudgeError, match="no JSON object"):
        parse_verdict("[1, 2, 3]")


# --- api failures -----------------------------------------------------


def test_an_api_error_is_reported_as_not_evaluated_not_as_a_fail():
    """A failure to evaluate and a failing case are different states."""
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(500, request=request)
    s, _ = score(anthropic.APIStatusError("boom", response=response, body=None))

    assert not s.passed
    assert "judge unavailable" in s.reason
    assert "not evaluated" in s.reason


# --- not applicable ---------------------------------------------------


def test_no_rubric_returns_none():
    s = judge_scorer.score(
        result(),
        Expectations(must_contain=("x",)),
        client=FakeClient(reply("pass")),
    )

    assert s is None


def test_an_errored_run_is_not_judged():
    """A runner failure must not be reported as a judge failure."""
    from agent_evals.runner import Failure

    s = judge_scorer.score(
        RunResult(case_id="x", ok=False, output="", failure=Failure.TIMEOUT),
        Expectations(judge=RUBRIC),
        client=FakeClient(reply("pass")),
    )

    assert s is None


def test_an_empty_output_is_not_judged():
    s = judge_scorer.score(
        RunResult(case_id="x", ok=True, output="   "),
        Expectations(judge=RUBRIC),
        client=FakeClient(reply("pass")),
    )

    assert s is None

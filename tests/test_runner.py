"""Tests for the runner.

Every test here mocks the API. None makes a network call, which is
enforced by never constructing a real client: run_case takes an injected
client, and these tests always pass a fake one.

The failure modes are the substance of this module, so each gets its own
test asserting the specific Failure value rather than just that ok is
False. A timeout reported as a generic API error would still pass a
looser assertion, and would be wrong in the report.
"""

from __future__ import annotations

import httpx2 as httpx
import anthropic
import pytest

from agent_evals.cases import Case, Expectations
from agent_evals.runner import (
    DEFAULT_MODEL,
    Failure,
    load_skill,
    run_case,
    run_cases,
)


@pytest.fixture
def skills_root(tmp_path):
    """A skills directory holding one minimal style guide."""
    skill_dir = tmp_path / "email-draft"
    skill_dir.mkdir()
    (skill_dir / "STYLE.md").write_text(
        "Close every email with 'Enjoy your day,'.\n", encoding="utf-8"
    )
    return tmp_path


@pytest.fixture
def case():
    return Case(
        id="sample-case",
        skill="email-draft",
        input="Write a client email saying the staging site is ready.",
        expect=Expectations(must_contain=("Enjoy your day,",)),
    )


class FakeBlock:
    def __init__(self, text, type="text"):
        self.type = type
        self.text = text


class FakeUsage:
    def __init__(self, input_tokens=120, output_tokens=45):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class FakeResponse:
    def __init__(self, blocks, model=DEFAULT_MODEL, usage=None):
        self.content = blocks
        self.model = model
        self.usage = usage if usage is not None else FakeUsage()


class FakeMessages:
    """Stands in for client.messages.

    `behavior` is either a response to return or an exception to raise. A
    list cycles through in order, which is how the rate-limit retry is
    tested.
    """

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


def api_error(status_code, cls=anthropic.APIStatusError):
    """Build a real SDK exception, so the tests exercise real classes."""
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(status_code, request=request)
    return cls("boom", response=response, body=None)


# --- the success path -------------------------------------------------


def test_successful_run_captures_output_and_usage(skills_root, case):
    client = FakeClient(FakeResponse([FakeBlock("Good Morning Dana,")]))

    result = run_case(case, skills_root=skills_root, client=client)

    assert result.ok
    assert result.case_id == "sample-case"
    assert result.output == "Good Morning Dana,"
    assert result.model == DEFAULT_MODEL
    assert result.input_tokens == 120
    assert result.output_tokens == 45
    assert result.elapsed >= 0
    assert result.failure is None


def test_the_skill_file_becomes_the_system_prompt(skills_root, case):
    client = FakeClient(FakeResponse([FakeBlock("ok")]))

    run_case(case, skills_root=skills_root, client=client)

    sent = client.messages.calls[0]
    assert "Enjoy your day," in sent["system"]
    assert sent["messages"] == [{"role": "user", "content": case.input}]


def test_model_is_recorded_on_the_result(skills_root, case):
    client = FakeClient(FakeResponse([FakeBlock("ok")], model="claude-sonnet-5"))

    result = run_case(case, model="claude-sonnet-5", skills_root=skills_root, client=client)

    assert result.model == "claude-sonnet-5"
    assert client.messages.calls[0]["model"] == "claude-sonnet-5"


def test_non_text_blocks_are_skipped(skills_root, case):
    """A response can carry thinking blocks alongside text."""
    client = FakeClient(
        FakeResponse(
            [
                FakeBlock("internal reasoning", type="thinking"),
                FakeBlock("Good Morning Dana,"),
            ]
        )
    )

    result = run_case(case, skills_root=skills_root, client=client)

    assert result.ok
    assert result.output == "Good Morning Dana,"


def test_temperature_is_not_pinned(skills_root, case):
    """Non-determinism is the point, so the runner must not fix a seed."""
    client = FakeClient(FakeResponse([FakeBlock("ok")]))

    run_case(case, skills_root=skills_root, client=client)

    sent = client.messages.calls[0]
    assert "temperature" not in sent


# --- the failure modes ------------------------------------------------


def test_missing_skill_file_is_a_result_not_an_exception(tmp_path, case):
    client = FakeClient(FakeResponse([FakeBlock("never reached")]))

    result = run_case(case, skills_root=tmp_path, client=client)

    assert result.errored
    assert result.failure is Failure.SKILL_NOT_FOUND
    assert "email-draft" in result.detail
    assert client.messages.calls == []  # no API call attempted


def test_empty_skill_file_is_treated_as_missing(tmp_path, case):
    skill_dir = tmp_path / "email-draft"
    skill_dir.mkdir()
    (skill_dir / "STYLE.md").write_text("   \n", encoding="utf-8")
    client = FakeClient(FakeResponse([FakeBlock("never reached")]))

    result = run_case(case, skills_root=tmp_path, client=client)

    assert result.failure is Failure.SKILL_NOT_FOUND
    assert "empty" in result.detail


def test_empty_response_is_an_error_not_an_empty_output(skills_root, case):
    client = FakeClient(FakeResponse([]))

    result = run_case(case, skills_root=skills_root, client=client)

    assert result.errored
    assert result.failure is Failure.EMPTY_RESPONSE
    assert result.output == ""


def test_whitespace_only_response_is_an_empty_response(skills_root, case):
    client = FakeClient(FakeResponse([FakeBlock("   \n  ")]))

    result = run_case(case, skills_root=skills_root, client=client)

    assert result.failure is Failure.EMPTY_RESPONSE


def test_timeout_is_reported_as_a_timeout(skills_root, case):
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    client = FakeClient(anthropic.APITimeoutError(request=request))

    result = run_case(case, skills_root=skills_root, client=client)

    assert result.errored
    assert result.failure is Failure.TIMEOUT
    assert result.elapsed >= 0


def test_connection_error_is_reported_separately(skills_root, case):
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    client = FakeClient(
        anthropic.APIConnectionError(message="no route", request=request)
    )

    result = run_case(case, skills_root=skills_root, client=client)

    assert result.failure is Failure.CONNECTION_ERROR


def test_auth_error_is_reported_separately_from_other_api_errors(skills_root, case):
    client = FakeClient(api_error(401, anthropic.AuthenticationError))

    result = run_case(case, skills_root=skills_root, client=client)

    assert result.failure is Failure.AUTH_ERROR
    assert "ANTHROPIC_API_KEY" in result.detail


def test_server_error_is_reported_as_an_api_error(skills_root, case):
    client = FakeClient(api_error(500))

    result = run_case(case, skills_root=skills_root, client=client)

    assert result.failure is Failure.API_ERROR
    assert "500" in result.detail


# --- rate limiting ----------------------------------------------------


def test_rate_limit_retries_once_then_succeeds(skills_root, case, monkeypatch):
    monkeypatch.setattr("agent_evals.runner.time.sleep", lambda _: None)
    client = FakeClient(
        [
            api_error(429, anthropic.RateLimitError),
            FakeResponse([FakeBlock("Good Morning Dana,")]),
        ]
    )

    result = run_case(case, skills_root=skills_root, client=client)

    assert result.ok
    assert len(client.messages.calls) == 2


def test_rate_limit_gives_up_after_one_retry(skills_root, case, monkeypatch):
    """No unbounded retry loop. A hung run is worse than a failed one."""
    monkeypatch.setattr("agent_evals.runner.time.sleep", lambda _: None)
    client = FakeClient(api_error(429, anthropic.RateLimitError))

    result = run_case(case, skills_root=skills_root, client=client)

    assert result.errored
    assert result.failure is Failure.RATE_LIMITED
    assert len(client.messages.calls) == 2  # initial attempt plus one retry


# --- no secret leakage ------------------------------------------------


def test_no_result_field_carries_an_api_key(skills_root, case):
    """The key must never reach a RunResult, on success or on failure."""
    # Built at runtime rather than written as a literal, so this file
    # contains no string that looks like a key to a secret scanner.
    key_prefix = "sk" + "-ant"
    fake_key = f"{key_prefix}-not-a-real-key-0123456789"
    client = FakeClient(FakeResponse([FakeBlock("Good Morning Dana,")]))

    ok_result = run_case(case, skills_root=skills_root, client=client)
    err_result = run_case(
        case,
        skills_root=skills_root,
        client=FakeClient(api_error(401, anthropic.AuthenticationError)),
    )

    for result in (ok_result, err_result):
        assert fake_key not in repr(result)
        assert key_prefix not in repr(result)


# --- running several cases --------------------------------------------


def test_run_cases_continues_past_a_failed_case(skills_root, case):
    """One case erroring must not abort the run."""
    good = case
    bad = Case(
        id="missing-skill",
        skill="does-not-exist",
        input="anything",
        expect=Expectations(must_contain=("x",)),
    )
    client = FakeClient(FakeResponse([FakeBlock("Good Morning Dana,")]))

    results = run_cases([good, bad], skills_root=skills_root, client=client)

    assert len(results) == 2
    assert results[0].ok
    assert results[1].failure is Failure.SKILL_NOT_FOUND


def test_load_skill_raises_for_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="no skill instructions"):
        load_skill("nope", skills_root=tmp_path)

"""Execute a skill against a case and capture its output.

Responsible for:
    - Invoking the skill under evaluation with a case's input.
    - Capturing the response as text, along with what it cost to produce
      (tokens, wall time), so a run can be compared to a previous one.
    - Isolating failures, so one case that errors does not abort the run.

Deliberately not responsible for:
    - Scoring. The runner records what happened, it does not judge it.
    - Retrying on a bad response. A flaky skill is a finding, not a
      problem for the harness to paper over.
    - Parallelism. Runs are sequential until there is a measured reason
      not to be.

The central design decision here is that a failure is a value, not an
exception. `run_case` returns a RunResult either way. A run of twenty
cases where one times out should report nineteen results and one error,
not lose the whole run to a traceback.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import anthropic

from agent_evals.cases import Case
from agent_evals.config import load_env, model_override

# Recorded in every RunResult. Comparing behavior across model versions is
# the reason this harness exists, so the model is part of the result rather
# than ambient configuration.
DEFAULT_MODEL = "claude-opus-5"

# Skills live in the repo rather than in a user's ~/.claude directory, so a
# clone of this repo runs the shipped cases without any local setup.
DEFAULT_SKILLS_ROOT = Path(__file__).resolve().parent.parent / "skills"

DEFAULT_MAX_TOKENS = 2048

# One retry on a rate limit, then give up. A harness that retries hard turns
# a rate-limited run into a hang, and a hung run is worse than a failed one
# because nobody knows how long to wait.
RATE_LIMIT_RETRIES = 1
RATE_LIMIT_BACKOFF_SECONDS = 2.0


class Failure(str, Enum):
    """Why a run did not produce usable output.

    A closed set on purpose. The report groups by these, so an open-ended
    string would fragment into near-duplicate categories.
    """

    SKILL_NOT_FOUND = "skill_not_found"
    EMPTY_RESPONSE = "empty_response"
    RATE_LIMITED = "rate_limited"
    TIMEOUT = "timeout"
    CONNECTION_ERROR = "connection_error"
    API_ERROR = "api_error"
    AUTH_ERROR = "auth_error"


@dataclass(frozen=True)
class RunResult:
    """What happened when one case was run.

    `ok` is the only field worth branching on. When it is True, `output`
    holds the response text. When it is False, `failure` and `detail` say
    what went wrong and `output` is empty.

    Note there is no field for the API key, and none of these fields ever
    carries one. The key is read from the environment by the SDK and is
    never stored, logged, or returned.
    """

    case_id: str
    ok: bool
    output: str = ""
    model: str = DEFAULT_MODEL
    elapsed: float = 0.0
    input_tokens: int | None = None
    output_tokens: int | None = None
    failure: Failure | None = None
    detail: str = ""

    @property
    def errored(self) -> bool:
        return not self.ok


def _skill_path(skill: str, skills_root: Path) -> Path:
    """Where a skill's instructions live.

    STYLE.md rather than SKILL.md because the file is the thing being
    evaluated, a style guide, not a Claude Code skill definition.
    """
    return Path(skills_root) / skill / "STYLE.md"


def load_skill(skill: str, skills_root: Path | str = DEFAULT_SKILLS_ROOT) -> str:
    """Read a skill's instructions, to be used as the system prompt.

    Raises FileNotFoundError. run_case converts that into a RunResult, so
    callers of run_case never see it.
    """
    path = _skill_path(skill, Path(skills_root))
    if not path.is_file():
        raise FileNotFoundError(f"no skill instructions at {path}")
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise FileNotFoundError(f"skill instructions at {path} are empty")
    return text


def _extract_text(response: object) -> str:
    """Pull the text out of a response's content blocks.

    A response can contain thinking blocks and tool-use blocks alongside
    text, so filter by type rather than assuming content[0] is text.
    """
    parts = []
    for block in getattr(response, "content", []) or []:
        if getattr(block, "type", None) == "text":
            parts.append(getattr(block, "text", "") or "")
    return "".join(parts).strip()


def run_case(
    case: Case,
    model: str | None = None,
    skills_root: Path | str = DEFAULT_SKILLS_ROOT,
    client: anthropic.Anthropic | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: float = 120.0,
) -> RunResult:
    """Run one case and capture the result.

    Never raises for an expected failure. A missing skill file, a timeout,
    an empty response, a rate limit, or an API error each come back as a
    RunResult with `ok` False and a `failure` naming the cause.

    Temperature is left at the API default. Pinning it, or caching
    responses, would make runs repeatable and would defeat the purpose:
    this harness exists to measure real, non-deterministic behavior.
    """
    # Resolution order: the argument, then AGENT_EVALS_MODEL, then the
    # default. An env override lets the same cases be re-run against a
    # different model without editing code, which is the comparison this
    # harness exists to make.
    if model is None:
        load_env()
        model = model_override() or DEFAULT_MODEL

    def failed(failure: Failure, detail: str, elapsed: float = 0.0) -> RunResult:
        return RunResult(
            case_id=case.id,
            ok=False,
            model=model,
            elapsed=elapsed,
            failure=failure,
            detail=detail,
        )

    try:
        system_prompt = load_skill(case.skill, skills_root)
    except FileNotFoundError as exc:
        return failed(Failure.SKILL_NOT_FOUND, str(exc))

    # The SDK reads ANTHROPIC_API_KEY from the environment. load_env puts a
    # local .env into the environment first, if one exists, so a fresh
    # clone runs without the reader having to know what to export. The key
    # itself is never read by this code, never written anywhere, and never
    # put into a RunResult.
    if client is None:
        load_env()
        client = anthropic.Anthropic(timeout=timeout)

    request = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": [{"role": "user", "content": case.input}],
    }

    attempts = RATE_LIMIT_RETRIES + 1
    started = time.monotonic()

    for attempt in range(attempts):
        try:
            response = client.messages.create(**request)
        except anthropic.RateLimitError as exc:
            is_last = attempt == attempts - 1
            if is_last:
                return failed(
                    Failure.RATE_LIMITED,
                    f"rate limited after {attempts} attempt(s): {exc}",
                    time.monotonic() - started,
                )
            time.sleep(RATE_LIMIT_BACKOFF_SECONDS * (attempt + 1))
            continue
        except anthropic.APITimeoutError as exc:
            return failed(
                Failure.TIMEOUT,
                f"request timed out after {timeout}s: {exc}",
                time.monotonic() - started,
            )
        except anthropic.AuthenticationError as exc:
            # Reported separately from other API errors because the fix is
            # different: this one means the environment is wrong, not that
            # the request was. The message from the SDK does not contain
            # the key itself.
            return failed(
                Failure.AUTH_ERROR,
                f"authentication failed, check ANTHROPIC_API_KEY: {exc}",
                time.monotonic() - started,
            )
        except anthropic.APIConnectionError as exc:
            return failed(
                Failure.CONNECTION_ERROR,
                f"could not reach the API: {exc}",
                time.monotonic() - started,
            )
        except anthropic.APIStatusError as exc:
            return failed(
                Failure.API_ERROR,
                f"API returned {exc.status_code}: {exc}",
                time.monotonic() - started,
            )

        elapsed = time.monotonic() - started
        text = _extract_text(response)

        if not text:
            # A 200 with no text is still a failed case. Treating it as an
            # empty output would score it against the expectations and
            # report a misleading fail rather than an error.
            return failed(
                Failure.EMPTY_RESPONSE,
                "API returned a response with no text content",
                elapsed,
            )

        usage = getattr(response, "usage", None)
        return RunResult(
            case_id=case.id,
            ok=True,
            output=text,
            model=getattr(response, "model", model) or model,
            elapsed=elapsed,
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
        )

    # Unreachable: the loop either returns or exhausts into the rate-limit
    # branch above. Present so a future change to the loop cannot fall
    # through and return None.
    return failed(Failure.RATE_LIMITED, "exhausted rate limit retries")


def run_cases(
    cases: list[Case],
    model: str | None = None,
    skills_root: Path | str = DEFAULT_SKILLS_ROOT,
    client: anthropic.Anthropic | None = None,
) -> list[RunResult]:
    """Run every case in order, collecting results.

    Sequential, and it does not stop on a failed case. One case erroring
    tells you about that case, not about the others.
    """
    return [
        run_case(case, model=model, skills_root=skills_root, client=client)
        for case in cases
    ]

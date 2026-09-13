"""Model-as-judge scoring for properties only a reader can assess.

For the properties that are real but not mechanical: whether a draft
sounds like the person it was written for, whether a response actually
answered the question, whether it drifted into generic register.

Responsible for:
    - Asking a model whether a stated property holds for a response.
    - Returning the verdict together with the model's stated reason.

Deliberately not responsible for:
    - Being treated as ground truth. The judge is another model with its
      own failure modes, and its agreement with a human label is itself
      worth measuring.
    - Open-ended quality opinions. A judged property is specific and
      written down in the case, not "is this good".
    - Hiding its cost. Judge runs cost money and time, and the report
      surfaces both.

## The known failure modes, and what is done about each

Every mitigation below is marked in the code where it takes effect.

    Score clustering   A 1 to 10 scale collapses onto 7 and 8. Fixed by
                       using three discrete verdicts with each one
                       defined in the prompt.
    Post-hoc reasoning Asking for a score first produces a justification
                       written to fit a number already chosen. Fixed by
                       requiring justification before verdict, and by
                       putting justification first in the JSON schema.
    Verbosity bias     Judges rate longer answers higher. Fixed by an
                       explicit instruction to ignore length.
    Self-preference    A model rates its own output more favorably. Fixed
                       by defaulting the judge to a different model from
                       the one under test.
    Non-determinism    The same input scored twice can differ. Reduced,
                       not eliminated, by temperature 0.

None of this makes the judge correct. It makes it less predictably wrong.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import anthropic

from agent_evals.cases import Expectations
from agent_evals.config import judge_model, load_env
from agent_evals.runner import RunResult
from agent_evals.scorers.base import Score

# Defaults to a different model from the runner's claude-opus-5. This is
# the self-preference mitigation: a model judging its own output rates it
# more favorably, so the judge is a different model unless configured
# otherwise. It is also the cheaper of the two, which matters because the
# judge runs once per case per run.
DEFAULT_JUDGE_MODEL = "claude-sonnet-5"

# Temperature 0 reduces run-to-run variance in the verdict. It does not
# eliminate it: the same prompt at temperature 0 can still differ across
# calls. Any claim that judge scores are reproducible would be false.
JUDGE_TEMPERATURE = 0.0

JUDGE_MAX_TOKENS = 1024

# One retry on unparseable output, then give up. The same reasoning as the
# runner's rate-limit retry: a harness that retries hard turns a bad run
# into a hang.
MALFORMED_RETRIES = 1

# Three verdicts, not a 1 to 10 scale. Judges cluster on 7 and 8 with wide
# scales and the signal disappears, so each point here is defined in the
# prompt and the gap between them is meant to be large enough that a judge
# can actually tell them apart.
VERDICT_SCORES = {
    "pass": 1.0,
    "partial": 0.5,
    "fail": 0.0,
}

SYSTEM_PROMPT = """\
You are evaluating one response against one specific criterion. You are \
not judging whether the response is good in general.

Rules:

1. Judge ONLY against the criterion given. Ignore anything the criterion \
does not mention, however much you might otherwise comment on it.

2. Ignore length entirely. A short response that meets the criterion \
passes. A long, polished, well-written response that does not meet it \
fails. Length is not evidence of quality.

3. Write your justification BEFORE deciding your verdict. Reason from the \
response to a conclusion, rather than picking a verdict and explaining it \
afterwards.

4. Quote the specific part of the response that decided it. A \
justification that would fit any response is not useful.

Reply with JSON only, no prose around it, in exactly this shape:

{"justification": "<your reasoning, referring to specific text>", \
"verdict": "pass" | "partial" | "fail"}

The verdicts mean:
  pass     The response clearly meets the criterion.
  partial  The response partly meets it, or meets it in a way that leaves \
real doubt.
  fail     The response does not meet it.\
"""


@dataclass(frozen=True)
class JudgeError(Exception):
    """The judge could not produce a usable verdict."""

    detail: str

    def __str__(self) -> str:
        return self.detail


def build_prompt(case_input: str, output: str, rubric: str) -> str:
    """Assemble the judge's user message.

    The criterion comes last, immediately before the judge answers, so it
    is the most recent thing in context when the verdict is formed.
    """
    return (
        "Here is the input that was given to an assistant:\n"
        "<input>\n"
        f"{case_input}\n"
        "</input>\n\n"
        "Here is the response it produced:\n"
        "<response>\n"
        f"{output}\n"
        "</response>\n\n"
        "Evaluate the response against this criterion, and only this "
        "criterion:\n"
        "<criterion>\n"
        f"{rubric}\n"
        "</criterion>"
    )


def parse_verdict(text: str) -> tuple[str, str]:
    """Pull the verdict and justification out of a judge reply.

    Returns (verdict, justification). Raises JudgeError on anything that
    cannot be read confidently.

    A judge that wraps its JSON in prose or a code fence is common enough
    to be worth handling, so the first JSON object in the text is used.
    Anything beyond that is treated as malformed rather than guessed at:
    inferring a verdict from unparseable output would invent a score,
    which is the one thing a scorer must never do.
    """
    if not text or not text.strip():
        raise JudgeError("judge returned an empty response")

    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise JudgeError(
            f"no JSON object in judge response: {text.strip()[:200]}"
        )

    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise JudgeError(
            f"judge response was not valid JSON: {exc}"
        ) from exc

    # No isinstance check on `data` here: the regex only matches text
    # between braces, so anything that parses successfully is a dict. A
    # guard for that would be dead code.
    verdict = data.get("verdict")
    if not isinstance(verdict, str):
        raise JudgeError("judge response has no 'verdict' field")

    verdict = verdict.strip().lower()
    if verdict not in VERDICT_SCORES:
        allowed = ", ".join(sorted(VERDICT_SCORES))
        raise JudgeError(
            f"judge returned an unknown verdict '{verdict}', expected one "
            f"of: {allowed}"
        )

    justification = data.get("justification")
    if not isinstance(justification, str) or not justification.strip():
        # A verdict with no reasoning is refused rather than accepted with
        # a placeholder. The reason field is the reason this scorer is
        # worth reading, and an empty one means the judge did not follow
        # the instruction that mitigates post-hoc rationalization.
        raise JudgeError("judge gave a verdict with no justification")

    return verdict, justification.strip()


class JudgeScorer:
    """Scores a response against a case's plain-English rubric."""

    name = "judge"

    def __init__(self, model: str | None = None):
        self._model = model

    def resolve_model(self) -> str:
        """Argument, then AGENT_EVALS_JUDGE_MODEL, then the default."""
        if self._model:
            return self._model
        load_env()
        return judge_model() or DEFAULT_JUDGE_MODEL

    def score(
        self,
        result: RunResult,
        expect: Expectations,
        case_input: str = "",
        client: anthropic.Anthropic | None = None,
    ) -> Score | None:
        rubric = expect.judge

        # Nothing for this scorer to check.
        if not rubric:
            return None

        # A response that never arrived cannot be judged. Returning None
        # rather than a failing score keeps a runner failure from being
        # reported as a judge failure, which are different problems.
        if not result.ok or not result.output.strip():
            return None

        model = self.resolve_model()

        if client is None:
            load_env()
            client = anthropic.Anthropic()

        prompt = build_prompt(case_input, result.output, rubric)
        attempts = MALFORMED_RETRIES + 1
        last_error = ""

        for _ in range(attempts):
            try:
                response = client.messages.create(
                    model=model,
                    max_tokens=JUDGE_MAX_TOKENS,
                    temperature=JUDGE_TEMPERATURE,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}],
                )
            except anthropic.APIError as exc:
                # An API failure is not a failing case. It is a failure to
                # evaluate, and the two must not be confused.
                return Score(
                    scorer=self.name,
                    passed=False,
                    score=0.0,
                    reason=f"judge unavailable, not evaluated: {exc}",
                )

            text = "".join(
                getattr(block, "text", "") or ""
                for block in getattr(response, "content", []) or []
                if getattr(block, "type", None) == "text"
            )

            try:
                verdict, justification = parse_verdict(text)
            except JudgeError as exc:
                last_error = str(exc)
                continue

            return Score(
                scorer=self.name,
                passed=verdict == "pass",
                score=VERDICT_SCORES[verdict],
                reason=f"{verdict}: {justification}",
            )

        # Out of retries. Reported as a failed score with an explicit
        # reason rather than a guessed verdict.
        return Score(
            scorer=self.name,
            passed=False,
            score=0.0,
            reason=(
                f"judge output unusable after {attempts} attempts, "
                f"not evaluated: {last_error}"
            ),
        )


judge_scorer = JudgeScorer()

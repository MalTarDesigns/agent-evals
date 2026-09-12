"""Exact, mechanical checks on a response.

For properties with no interpretation involved: a required block is
present, a banned character does not appear, a line matches a pattern.

Responsible for:
    - Substring, absence, and pattern checks against a response.
    - Returning which specific check failed, not just that one did.

Deliberately not responsible for:
    - Anything approximate. If a check needs a threshold, it belongs in
      the fuzzy scorer.
    - Meaning. This scorer cannot tell correct from confidently wrong.
"""

from __future__ import annotations

from agent_evals.cases import Expectations
from agent_evals.runner import RunResult
from agent_evals.scorers.base import Score


def _quote(items) -> str:
    """Render substrings for an error message, quoted and comma separated.

    Quoted because expectations are often whitespace or punctuation, and
    an unquoted missing comma in a message is invisible.
    """
    return ", ".join(f'"{item}"' for item in items)


class ExactScorer:
    """Checks must_contain and must_not_contain.

    Case-insensitive by default, since most expectations are about wording
    rather than capitalization. A case sets `case_sensitive: true` when the
    capitalization is the point.
    """

    name = "exact"

    def score(self, result: RunResult, expect: Expectations) -> Score | None:
        required = expect.must_contain
        forbidden = expect.must_not_contain

        # Nothing for this scorer to check. None rather than a passing
        # score, so the report does not count a skipped check as a win.
        if not required and not forbidden:
            return None

        output = result.output if expect.case_sensitive else result.output.lower()

        def present(needle: str) -> bool:
            probe = needle if expect.case_sensitive else needle.lower()
            return probe in output

        missing = [item for item in required if not present(item)]
        found = [item for item in forbidden if present(item)]

        # Every required substring and every forbidden one is one check,
        # so a case with three required and one forbidden has four checks
        # and each is worth a quarter. Weighting them equally is a choice:
        # a case that cares more about one check should say so by being
        # two cases, rather than by carrying weights that nobody maintains.
        total = len(required) + len(forbidden)
        satisfied = total - len(missing) - len(found)
        value = satisfied / total

        # score and passed are deliberately independent. Three of four
        # required phrases scores 0.75 and still fails, because the case
        # is not satisfied. Keeping the number means a prompt change that
        # moves a case from 0.25 to 0.75 shows as progress in the report
        # even though it fails at both ends.
        passed = not missing and not found

        if passed:
            parts = []
            if required:
                parts.append(f"all {len(required)} required present")
            if forbidden:
                parts.append(f"none of {len(forbidden)} forbidden present")
            reason = ", ".join(parts)
        else:
            problems = []
            if missing:
                problems.append(f"missing {_quote(missing)}")
            if found:
                problems.append(f"forbidden {_quote(found)} present")
            sensitivity = "" if expect.case_sensitive else ", case-insensitive"
            reason = (
                f"{'; '.join(problems)} "
                f"({satisfied}/{total} checks passed{sensitivity})"
            )

        return Score(
            scorer=self.name,
            passed=passed,
            score=value,
            reason=reason,
        )


# A module-level instance, since the scorer holds no state. The report and
# any future runner integration import this rather than constructing one.
exact_scorer = ExactScorer()

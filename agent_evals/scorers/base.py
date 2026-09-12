"""The shape every scorer shares.

One interface so the report can treat exact, fuzzy, and judge results
uniformly, and so adding a scorer does not mean touching the runner.

Responsible for:
    - The Score type every scorer returns.
    - The Scorer protocol every scorer satisfies.

Deliberately not responsible for:
    - Registering or discovering scorers. Three scorers do not justify a
      plugin system, and a dict of name to scorer is enough at this size.
    - Deciding an overall verdict for a case. That is the report's job,
      because how to combine three scores is a presentation decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from agent_evals.cases import Expectations
from agent_evals.runner import RunResult


@dataclass(frozen=True)
class Score:
    """One scorer's verdict on one response.

    `score` and `passed` are separate on purpose, and the distinction is
    the reason this type is not just a bool.

    A case requiring four phrases that finds three scores 0.75 and still
    fails. Both numbers matter and they say different things: `passed`
    answers "is this correct", `score` answers "how far off is it". A run
    that moves from 0.2 to 0.75 across a prompt change is improving even
    though it fails at both ends, and only `score` makes that visible.
    Collapsing the two would hide the trend the harness exists to report.

    `reason` is required. A failing check that does not say what was wrong
    forces the reader back into the code, which is exactly the friction
    that stops people from keeping evals up to date. It is populated on a
    pass too, so a passing report is still readable.
    """

    scorer: str
    passed: bool
    score: float
    reason: str

    def __post_init__(self) -> None:
        # Guarding a value this module controls, because a scorer returning
        # 1.5 would silently corrupt every aggregate the report computes.
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(
                f"{self.scorer}: score must be between 0.0 and 1.0, "
                f"got {self.score}"
            )
        if not self.reason.strip():
            raise ValueError(f"{self.scorer}: a Score must carry a reason")


class Scorer(Protocol):
    """What every scorer implements.

    A Protocol rather than an abstract base class: scorers have no shared
    behavior to inherit, only a shared shape, and a Protocol keeps them
    from needing to import a parent just to be recognized.
    """

    name: str

    def score(self, result: RunResult, expect: Expectations) -> Score | None:
        """Judge one response.

        Returns None when the case defines nothing for this scorer to
        check. None means "not applicable" and is distinct from a score of
        0.0, which means "checked and failed". The report counts them
        differently: a skipped scorer should not drag down a pass rate.
        """
        ...

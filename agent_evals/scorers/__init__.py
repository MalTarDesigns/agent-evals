"""Scorers judge a captured response against a case's labels.

Three scorers, chosen because they fail in different ways and so cover
different classes of regression:

    exact  - binary, mechanical properties. Cheap, deterministic, and
             blind to meaning.
    fuzzy  - approximate similarity to a reference. Tolerates rewording,
             and can be fooled by text that is similar but wrong.
    judge  - a model assessing a stated property. Catches what the other
             two cannot, at the cost of money, latency, and its own
             variance.

Every scorer takes a response and a case, and returns a score with a
reason. The reason is required: a number without an explanation is not
actionable when a run regresses.
"""

from agent_evals.scorers.base import Score, Scorer
from agent_evals.scorers.exact import exact_scorer
from agent_evals.scorers.fuzzy import fuzzy_scorer
from agent_evals.scorers.judge import judge_scorer

# A plain dict, not a registry. Three scorers do not justify dynamic
# loading or a plugin system, and the indirection would cost more in
# readability than it saves. Add a scorer by adding a line here.
SCORERS: dict[str, Scorer] = {
    exact_scorer.name: exact_scorer,
    fuzzy_scorer.name: fuzzy_scorer,
    judge_scorer.name: judge_scorer,
}

__all__ = [
    "Score",
    "Scorer",
    "SCORERS",
    "exact_scorer",
    "fuzzy_scorer",
    "judge_scorer",
]

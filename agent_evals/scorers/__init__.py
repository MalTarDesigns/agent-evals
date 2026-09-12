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

"""Approximate similarity between a response and a reference answer.

For properties that survive rewording. A response that says the same
thing in different words should pass, and one that drifts in substance
should not.

Responsible for:
    - Computing a similarity score against a reference.
    - Applying a per-case threshold, so each case sets its own bar.

Deliberately not responsible for:
    - Choosing the similarity measure for every use. The measure is a
      detail and is expected to change once there is real data.
    - Semantic equivalence in the strong sense. Two texts can score as
      similar and still differ in a way that matters, which is what the
      judge scorer is for.
"""

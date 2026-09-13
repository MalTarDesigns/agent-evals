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

## Why token-level F1

Four options were considered:

    Jaccard overlap      free, instant, but treats a short output and a
                         long reference symmetrically and ignores
                         repeated words
    Edit distance        free, instant, but a reordered paragraph scores
                         badly even when it says the same thing
    Token-level F1       free, instant, handles repetition and asymmetry,
                         still blind to genuine paraphrase
    Embedding cosine     catches real paraphrase, but costs an API call
                         per comparison and adds a dependency

F1 was chosen. It is the best of the free options: precision penalizes a
rambling response that happens to contain the reference, recall penalizes
a response that drops most of it, and the harmonic mean means a response
has to do both to score well. Jaccard cannot express that difference.

The honest cost: this measure does not understand meaning. "The site is
live" and "The site is not live" score nearly identically, because they
share every token but one. That is a real weakness and it is why the
judge scorer exists. Fuzzy scoring here is for catching drift in wording
and length, not for judging correctness.

Embeddings were rejected on cost, not on quality. They would be more
correct. But a scorer whose job is catching regressions across repeated
runs should be free to run, and paying per comparison discourages exactly
the frequent running that makes a regression visible.
"""

from __future__ import annotations

import re
from collections import Counter

from agent_evals.cases import Expectations
from agent_evals.runner import RunResult
from agent_evals.scorers.base import Score

# Default pass threshold. This number is arbitrary and is stated as such:
# it has not been calibrated against human judgment, because that would
# require labeled data this project does not have. It is a starting point
# chosen so that a close paraphrase passes and a loose one does not, and
# every case can override it.
#
# Do not tune this to make current cases pass. A fuzzy scorer that passes
# everything is worse than no scorer at all.
DEFAULT_THRESHOLD = 0.8

_WORD = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> list[str]:
    """Split text into comparable tokens.

    Three normalization choices, each of which silently changes results:

    Lowercase: yes. The exact scorer already handles capitalization when
    a case cares about it, so applying it again here would double-count
    the same property.

    Punctuation stripped: yes. The regex keeps letters, digits, and the
    apostrophe, so "day," and "day" are the same token. Punctuation
    differences are almost never the drift this scorer is looking for,
    and leaving them in makes near-identical text score below 1.0 for a
    reason nobody would call a real difference. The apostrophe is kept so
    "don't" stays one token rather than becoming "don" and "t".

    Stopwords removed: no. Deliberately. Stopword lists are a hidden
    dependency and they change what the score means: for a style guide,
    "let me know if you have any questions" is almost entirely stopwords,
    and removing them would erase the phrase this harness most wants to
    track. A general-purpose eval might strip them. This one must not.
    """
    return _WORD.findall(text.lower())


def f1(output_tokens: list[str], reference_tokens: list[str]) -> float:
    """Token-level F1 between two token lists.

    Counter intersection counts repeated words correctly: a reference
    with "the" three times and an output with it once shares one, not
    three. That is the asymmetry Jaccard cannot express.
    """
    # Guarding the empty cases explicitly rather than relying on a later
    # division. An empty output against a non-empty reference is a
    # genuine zero, and two empty texts are vacuously identical.
    if not output_tokens and not reference_tokens:
        return 1.0
    if not output_tokens or not reference_tokens:
        return 0.0

    shared = sum((Counter(output_tokens) & Counter(reference_tokens)).values())
    if shared == 0:
        return 0.0

    precision = shared / len(output_tokens)
    recall = shared / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


class FuzzyScorer:
    """Scores a response against a case's reference text."""

    name = "fuzzy"

    def score(self, result: RunResult, expect: Expectations) -> Score | None:
        reference = expect.reference

        # Nothing to compare against. None rather than a passing score, so
        # a skipped check does not inflate the pass rate.
        if not reference:
            return None

        output_tokens = tokenize(result.output)
        reference_tokens = tokenize(reference)
        value = f1(output_tokens, reference_tokens)
        threshold = expect.threshold
        passed = value >= threshold

        if not output_tokens:
            reason = (
                f"no output to compare against a {len(reference_tokens)}-token "
                f"reference (threshold {threshold:.2f})"
            )
            return Score(
                scorer=self.name, passed=False, score=0.0, reason=reason
            )

        shared_counts = Counter(output_tokens) & Counter(reference_tokens)
        shared = sum(shared_counts.values())
        missing = sorted((Counter(reference_tokens) - Counter(output_tokens)))

        # The reason names concrete words rather than only a number, so a
        # reader can see what drifted without diffing the texts by hand.
        # Capped, because a long reference would otherwise produce a wall
        # of text in every report line.
        detail = ""
        if missing:
            shown = ", ".join(f'"{word}"' for word in missing[:6])
            more = f", and {len(missing) - 6} more" if len(missing) > 6 else ""
            detail = f". Reference words absent: {shown}{more}"

        reason = (
            f"F1 {value:.2f} against threshold {threshold:.2f}, "
            f"{shared} of {len(reference_tokens)} reference tokens present "
            f"in a {len(output_tokens)}-token response{detail}"
        )

        return Score(
            scorer=self.name, passed=passed, score=value, reason=reason
        )


fuzzy_scorer = FuzzyScorer()

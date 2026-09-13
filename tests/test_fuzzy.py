"""Tests for the fuzzy scorer.

The boundary cases matter more than the middle here. A similarity measure
that returns plausible-looking numbers on ordinary input can still divide
by zero on empty text or score unrelated text suspiciously high, and
those are the failures that make a harness untrustworthy.
"""

from __future__ import annotations

from agent_evals.cases import Expectations
from agent_evals.runner import RunResult
from agent_evals.scorers.fuzzy import DEFAULT_THRESHOLD, f1, fuzzy_scorer, tokenize

REFERENCE = (
    "The staging site is updated with the new gallery. "
    "Please take a look and let me know if everything is good on your end."
)


def result(output):
    return RunResult(case_id="sample", ok=True, output=output)


def score(output, reference=REFERENCE, threshold=DEFAULT_THRESHOLD):
    return fuzzy_scorer.score(
        result(output),
        Expectations(reference=reference, threshold=threshold),
    )


# --- the boundaries ---------------------------------------------------


def test_identical_text_scores_one():
    s = score(REFERENCE)

    assert s.score == 1.0
    assert s.passed


def test_unrelated_text_scores_near_zero():
    s = score("Quantum chromodynamics describes the strong nuclear force.")

    assert s.score < 0.2
    assert not s.passed


def test_empty_output_scores_zero_without_dividing_by_zero():
    s = score("")

    assert s.score == 0.0
    assert not s.passed
    assert "no output" in s.reason


def test_whitespace_only_output_scores_zero():
    s = score("   \n\t  ")

    assert s.score == 0.0
    assert not s.passed


def test_a_paraphrase_scores_between_the_extremes():
    """Reworded but equivalent text should land in the middle.

    This is the case the scorer exists for, and also the case it handles
    least well, since it cannot tell a paraphrase from a text that merely
    shares vocabulary.
    """
    s = score(
        "The staging site now has the new gallery. "
        "Take a look and tell me if it all looks right to you."
    )

    assert 0.3 < s.score < 0.95


# --- the F1 function itself -------------------------------------------


def test_f1_of_two_empty_lists_is_one():
    """Vacuously identical. Guarded explicitly rather than by division."""
    assert f1([], []) == 1.0


def test_f1_with_one_side_empty_is_zero():
    assert f1([], ["a", "b"]) == 0.0
    assert f1(["a", "b"], []) == 0.0


def test_f1_with_no_shared_tokens_is_zero():
    assert f1(["a", "b"], ["c", "d"]) == 0.0


def test_f1_counts_repeated_words_correctly():
    """The asymmetry Jaccard cannot express.

    One "the" in the output against three in the reference shares one
    occurrence, not three.
    """
    value = f1(["the"], ["the", "the", "the"])

    # precision 1/1, recall 1/3, harmonic mean 0.5
    assert value == 0.5


def test_f1_penalizes_a_rambling_response():
    """Precision is why a response cannot pass by containing everything."""
    tight = f1(["alpha", "beta"], ["alpha", "beta"])
    padded = f1(["alpha", "beta"] + ["filler"] * 20, ["alpha", "beta"])

    assert tight == 1.0
    assert padded < 0.25


def test_f1_penalizes_a_response_that_drops_most_of_the_reference():
    """Recall is the other half."""
    value = f1(["alpha"], ["alpha", "beta", "gamma", "delta"])

    assert value < 0.5


# --- normalization ----------------------------------------------------


def test_tokenizing_lowercases():
    assert tokenize("Enjoy Your Day") == ["enjoy", "your", "day"]


def test_tokenizing_strips_punctuation():
    assert tokenize("day, and night.") == ["day", "and", "night"]


def test_tokenizing_keeps_apostrophes_inside_words():
    """So "don't" stays one token rather than becoming "don" and "t"."""
    assert tokenize("don't") == ["don't"]


def test_tokenizing_collapses_whitespace():
    assert tokenize("a\n\n  b\tc") == ["a", "b", "c"]


def test_stopwords_are_not_removed():
    """Deliberate, and load-bearing for this project.

    The phrase the style guide most wants tracked is almost entirely
    stopwords, so removing them would erase it.
    """
    tokens = tokenize("let me know if you have any questions")

    assert "if" in tokens
    assert "you" in tokens
    assert len(tokens) == 8


def test_punctuation_difference_alone_still_scores_one():
    s = score(REFERENCE.replace(".", "!").replace(",", ""))

    assert s.score == 1.0


# --- thresholds and reasons -------------------------------------------


def test_the_case_threshold_is_applied_not_the_default():
    text = "The staging site is updated with the new gallery."

    strict = score(text, threshold=0.95)
    lenient = score(text, threshold=0.3)

    assert not strict.passed
    assert lenient.passed
    assert strict.score == lenient.score  # same measurement, different bar


def test_the_reason_reports_the_score_and_the_threshold():
    s = score(REFERENCE, threshold=0.7)

    assert "F1 1.00" in s.reason
    assert "0.70" in s.reason


def test_the_reason_names_missing_reference_words():
    s = score("The staging site is updated.")

    assert "Reference words absent" in s.reason
    assert '"gallery"' in s.reason


def test_the_reason_caps_a_long_list_of_missing_words():
    s = score("staging", reference=" ".join(f"word{n}" for n in range(40)))

    assert "and 34 more" in s.reason


def test_a_reason_is_present_on_a_pass():
    s = score(REFERENCE)

    assert s.reason.strip()


# --- not applicable ---------------------------------------------------


def test_no_reference_returns_none():
    """None means not applicable, which is not the same as scoring zero."""
    s = fuzzy_scorer.score(
        result("anything at all"),
        Expectations(must_contain=("anything",)),
    )

    assert s is None

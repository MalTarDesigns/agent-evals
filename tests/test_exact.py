"""Tests for the exact-match scorer and the Score contract.

The reason strings are asserted on, not just the numbers. A scorer that
returns the right score with a useless explanation has failed at the job
this project cares about.
"""

from __future__ import annotations

import pytest

from agent_evals.cases import Expectations
from agent_evals.runner import RunResult
from agent_evals.scorers.base import Score
from agent_evals.scorers.exact import exact_scorer

EMAIL = (
    "Good Morning Dana,\n\n"
    "The staging site is updated with the new gallery. You can see it HERE.\n\n"
    "Let me know if you have any questions.\n\n"
    "Enjoy your day,\n"
    "Northwind Studio | hello@northwind.example"
)


def result(output=EMAIL):
    return RunResult(case_id="sample", ok=True, output=output)


# --- passing ----------------------------------------------------------


def test_all_required_present_passes_with_a_full_score():
    score = exact_scorer.score(
        result(),
        Expectations(must_contain=("Good Morning Dana,", "Enjoy your day,")),
    )

    assert score.passed
    assert score.score == 1.0
    assert "all 2 required present" in score.reason


def test_no_forbidden_present_passes():
    score = exact_scorer.score(
        result(),
        Expectations(must_not_contain=("I hope this email finds you well",)),
    )

    assert score.passed
    assert score.score == 1.0
    assert "none of 1 forbidden present" in score.reason


def test_both_kinds_of_check_together():
    score = exact_scorer.score(
        result(),
        Expectations(
            must_contain=("HERE",),
            must_not_contain=("Best regards",),
        ),
    )

    assert score.passed
    assert "all 1 required present" in score.reason
    assert "none of 1 forbidden present" in score.reason


# --- partial credit ---------------------------------------------------


def test_three_of_four_required_scores_point_seven_five_and_still_fails():
    """The decision this scorer exists to demonstrate.

    score and passed are independent: a case can improve measurably and
    still be wrong, and the report needs to show both.
    """
    score = exact_scorer.score(
        result(),
        Expectations(
            must_contain=(
                "Good Morning Dana,",
                "HERE",
                "Enjoy your day,",
                "a phrase that is not there",
            )
        ),
    )

    assert score.score == 0.75
    assert not score.passed
    assert '"a phrase that is not there"' in score.reason
    assert "3/4 checks passed" in score.reason


def test_nothing_matches_scores_zero():
    score = exact_scorer.score(
        result(),
        Expectations(must_contain=("nope", "also nope")),
    )

    assert score.score == 0.0
    assert not score.passed


def test_forbidden_and_required_share_one_denominator():
    """Two required plus two forbidden is four checks, one satisfied."""
    score = exact_scorer.score(
        result(output="Best regards"),
        Expectations(
            must_contain=("Good Morning Dana,", "Enjoy your day,"),
            must_not_contain=("Best regards", "Kind regards"),
        ),
    )

    assert score.score == 0.25
    assert "1/4 checks passed" in score.reason


# --- failure reasons --------------------------------------------------


def test_the_reason_names_the_missing_substrings_not_a_count():
    score = exact_scorer.score(
        result(),
        Expectations(must_contain=("first missing", "second missing")),
    )

    assert '"first missing"' in score.reason
    assert '"second missing"' in score.reason


def test_the_reason_names_a_forbidden_substring_that_appeared():
    score = exact_scorer.score(
        result(output="I hope this email finds you well, Dana."),
        Expectations(must_not_contain=("I hope this email finds you well",)),
    )

    assert not score.passed
    assert "forbidden" in score.reason
    assert '"I hope this email finds you well"' in score.reason


def test_a_reason_is_present_on_a_pass_too():
    score = exact_scorer.score(result(), Expectations(must_contain=("HERE",)))

    assert score.reason.strip()


# --- case sensitivity -------------------------------------------------


def test_case_insensitive_by_default():
    score = exact_scorer.score(
        result(),
        Expectations(must_contain=("good morning dana,",)),
    )

    assert score.passed
    assert "case-insensitive" not in score.reason  # only shown on failure


def test_case_sensitive_when_the_case_asks_for_it():
    """A rule requiring HERE in capitals is a real rule in the style guide."""
    score = exact_scorer.score(
        result(output="You can see it here."),
        Expectations(must_contain=("HERE",), case_sensitive=True),
    )

    assert not score.passed
    assert '"HERE"' in score.reason


def test_case_sensitive_passes_on_an_exact_match():
    score = exact_scorer.score(
        result(),
        Expectations(must_contain=("HERE",), case_sensitive=True),
    )

    assert score.passed


def test_case_insensitive_catches_a_forbidden_phrase_in_any_casing():
    score = exact_scorer.score(
        result(output="BEST REGARDS,"),
        Expectations(must_not_contain=("Best regards",)),
    )

    assert not score.passed


def test_failure_reason_mentions_case_insensitivity():
    """So a reader knows why a differently-cased near-match still failed."""
    score = exact_scorer.score(
        result(),
        Expectations(must_contain=("not in the text",)),
    )

    assert "case-insensitive" in score.reason


# --- not applicable ---------------------------------------------------


def test_no_substring_expectations_returns_none():
    """None means not applicable, which is different from scoring zero."""
    score = exact_scorer.score(
        result(),
        Expectations(judge="Sounds like the person it is written for."),
    )

    assert score is None


def test_empty_output_fails_every_required_check():
    score = exact_scorer.score(
        RunResult(case_id="sample", ok=False, output=""),
        Expectations(must_contain=("anything",)),
    )

    assert score.score == 0.0
    assert not score.passed


# --- the Score contract -----------------------------------------------


def test_a_score_must_carry_a_reason():
    with pytest.raises(ValueError, match="must carry a reason"):
        Score(scorer="exact", passed=True, score=1.0, reason="   ")


def test_a_score_outside_zero_to_one_is_rejected():
    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        Score(scorer="exact", passed=True, score=1.5, reason="impossible")

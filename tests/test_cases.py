"""Tests for case loading and validation.

The validation messages are a feature of this project, so these tests
assert on the text of the error, not just that one was raised. If a
message stops naming the file or the specific problem, that is a
regression worth failing on.
"""

from __future__ import annotations

import textwrap

import pytest

from agent_evals.cases import CaseError, load_case, load_cases

VALID = """\
id: {id}
skill: email-draft
input: |
  Write a client email saying the staging site is ready.
expect:
  must_contain:
    - "Enjoy your day,"
  judge: |
    Gets to the point in the first sentence after the greeting.
"""


def write(tmp_path, name, body):
    """Write a case file named `name` and return its path."""
    path = tmp_path / f"{name}.yaml"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def test_loads_a_valid_case(tmp_path):
    path = write(tmp_path, "good-case", VALID.format(id="good-case"))
    case = load_case(path)

    assert case.id == "good-case"
    assert case.skill == "email-draft"
    assert "staging site" in case.input
    assert case.expect.must_contain == ("Enjoy your day,",)
    assert case.expect.judge is not None
    assert case.expect.threshold == 0.8


def test_rejects_a_case_with_no_expectations(tmp_path):
    path = write(
        tmp_path,
        "empty-expect",
        """
        id: empty-expect
        skill: email-draft
        input: |
          Write a client email.
        expect: {}
        """,
    )
    with pytest.raises(CaseError) as exc:
        load_case(path)

    message = str(exc.value)
    assert "empty-expect.yaml" in message
    assert "no expectations defined" in message
    # The message should tell the reader what to add, not just that
    # something is missing.
    assert "must_contain" in message


def test_rejects_a_case_with_the_expect_block_absent(tmp_path):
    path = write(
        tmp_path,
        "no-expect-key",
        """
        id: no-expect-key
        skill: email-draft
        input: |
          Write a client email.
        """,
    )
    with pytest.raises(CaseError, match="no expectations defined"):
        load_case(path)


def test_rejects_malformed_yaml(tmp_path):
    path = write(
        tmp_path,
        "broken",
        """
        id: broken
        skill: email-draft
          input: badly indented
        expect:
          must_contain: ["x"]
        """,
    )
    with pytest.raises(CaseError) as exc:
        load_case(path)

    message = str(exc.value)
    assert "broken.yaml" in message
    assert "invalid YAML" in message


def test_rejects_a_typo_in_an_expectation_key(tmp_path):
    """A typo must fail loudly rather than dropping the expectation.

    This is the failure mode the harness can least afford: a silently
    ignored expectation makes a case pass for the wrong reason.
    """
    path = write(
        tmp_path,
        "typo-key",
        """
        id: typo-key
        skill: email-draft
        input: |
          Write a client email.
        expect:
          must_countain:
            - "Enjoy your day,"
        """,
    )
    with pytest.raises(CaseError) as exc:
        load_case(path)

    message = str(exc.value)
    assert "must_countain" in message
    assert "must_contain" in message  # the allowed-keys list


def test_rejects_a_string_where_a_list_is_required(tmp_path):
    path = write(
        tmp_path,
        "string-not-list",
        """
        id: string-not-list
        skill: email-draft
        input: |
          Write a client email.
        expect:
          must_contain: "Enjoy your day,"
        """,
    )
    with pytest.raises(CaseError, match="must be a list"):
        load_case(path)


def test_rejects_a_threshold_with_no_reference(tmp_path):
    path = write(
        tmp_path,
        "orphan-threshold",
        """
        id: orphan-threshold
        skill: email-draft
        input: |
          Write a client email.
        expect:
          must_contain: ["x"]
          threshold: 0.9
        """,
    )
    with pytest.raises(CaseError, match="no 'reference' to compare against"):
        load_case(path)


def test_rejects_an_id_that_does_not_match_the_filename(tmp_path):
    path = write(tmp_path, "actual-name", VALID.format(id="different-name"))
    with pytest.raises(CaseError) as exc:
        load_case(path)

    message = str(exc.value)
    assert "different-name" in message
    assert "actual-name" in message


def test_rejects_a_missing_required_field(tmp_path):
    path = write(
        tmp_path,
        "no-skill",
        """
        id: no-skill
        input: |
          Write a client email.
        expect:
          must_contain: ["x"]
        """,
    )
    with pytest.raises(CaseError, match="missing required field 'skill'"):
        load_case(path)


def test_rejects_an_empty_file(tmp_path):
    path = write(tmp_path, "blank", "\n")
    with pytest.raises(CaseError, match="file is empty"):
        load_case(path)


def test_load_cases_reads_a_directory_sorted_by_id(tmp_path):
    write(tmp_path, "zulu", VALID.format(id="zulu"))
    write(tmp_path, "alpha", VALID.format(id="alpha"))
    write(tmp_path, "mike", VALID.format(id="mike"))

    cases = load_cases(tmp_path)

    assert [c.id for c in cases] == ["alpha", "mike", "zulu"]


def test_load_cases_stops_on_the_first_invalid_case(tmp_path):
    write(tmp_path, "fine", VALID.format(id="fine"))
    write(
        tmp_path,
        "bad",
        """
        id: bad
        skill: email-draft
        input: |
          Write a client email.
        expect: {}
        """,
    )
    with pytest.raises(CaseError, match="no expectations defined"):
        load_cases(tmp_path)


def test_load_cases_rejects_an_empty_directory(tmp_path):
    with pytest.raises(CaseError, match="no .yaml case files found"):
        load_cases(tmp_path)


def test_the_real_cases_in_the_repo_all_load():
    """The shipped cases must stay valid as the format changes."""
    cases = load_cases("cases")

    assert len(cases) >= 3
    for case in cases:
        assert not case.expect.is_empty()
        assert case.note, f"{case.id} has no note explaining what it probes"

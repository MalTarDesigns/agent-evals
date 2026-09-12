"""Load and validate labeled evaluation cases.

A case is one input to a skill plus the labels describing what a correct
response looks like. Labels are deliberately per-scorer: a case can
require an exact substring, a fuzzy match against a reference answer, and
a judged property, in any combination.

Responsible for:
    - Reading case files from disk.
    - Validating structure and reporting which file and field is wrong.
    - Returning cases in a stable order, so two runs are comparable.

Deliberately not responsible for:
    - Running anything. Cases are inert data.
    - Deciding whether a response passes. That belongs to the scorers.
    - Generating or inferring labels. Every label is written by a human.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


class CaseError(Exception):
    """A case file is missing, malformed, or fails validation.

    Carries the file path in the message. A person adding a case should be
    able to fix it from the error text alone, without opening this module.
    """


@dataclass(frozen=True)
class Expectations:
    """What a correct response looks like, split by the scorer that checks it.

    The three fields map one-to-one onto the three scorers. That symmetry is
    deliberate: it keeps the format from drifting into a general-purpose
    configuration language, because the only way to add a new kind of
    expectation is to add a scorer that can check it.

    Every field is optional on its own, but a case with all three empty is
    rejected. See Case.validate for why.
    """

    # Checked by scorers.exact.
    must_contain: tuple[str, ...] = ()
    must_not_contain: tuple[str, ...] = ()

    # Substring checks ignore case by default, because most expectations
    # are about wording rather than capitalization. A case sets this True
    # when the capitalization is the point, for example a rule requiring
    # the word HERE in capitals.
    case_sensitive: bool = False

    # Checked by scorers.fuzzy. The threshold lives on the case rather than
    # in the scorer because the right bar differs per case: a formulaic
    # confirmation should match a reference closely, an open-ended reply
    # should not be held to the same number.
    reference: str | None = None
    threshold: float = 0.8

    # Checked by scorers.judge. Plain English on purpose. A rubric written as
    # structured data would be easier to parse and worse to write, and the
    # judge reads prose anyway.
    judge: str | None = None

    def is_empty(self) -> bool:
        """True when no scorer has anything to check."""
        return not (
            self.must_contain
            or self.must_not_contain
            or self.reference
            or self.judge
        )


@dataclass(frozen=True)
class Case:
    """One labeled example: an input to a skill, plus what is expected back."""

    id: str
    skill: str
    input: str
    expect: Expectations
    # Free-text note on what this case is actually probing. Not used for
    # scoring. It exists because a case whose purpose is not written down
    # gets deleted by the next person who sees it fail.
    note: str = ""
    source: Path | None = field(default=None, compare=False)


# Keys allowed at the top level of a case file, and inside `expect`. Unknown
# keys are an error rather than ignored: a typo like `must_countain` would
# otherwise silently drop an expectation and the case would pass for the
# wrong reason, which is the single worst failure mode for a harness whose
# whole job is to be trusted.
_CASE_KEYS = {"id", "skill", "input", "expect", "note"}
_EXPECT_KEYS = {
    "must_contain",
    "must_not_contain",
    "case_sensitive",
    "reference",
    "threshold",
    "judge",
}


def _fail(path: Path, problem: str) -> None:
    raise CaseError(f"{path}: {problem}")


def _require_str(path: Path, data: dict, key: str) -> str:
    value = data.get(key)
    if value is None:
        _fail(path, f"missing required field '{key}'")
    if not isinstance(value, str) or not value.strip():
        _fail(path, f"field '{key}' must be a non-empty string")
    return value


def _string_list(path: Path, raw: object, key: str) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        # A bare string here is almost always a mistake, and silently
        # treating it as a one-item list would hide it. Python would also
        # iterate it character by character, which fails in a confusing way
        # much later, so reject it here where the message can be clear.
        _fail(path, f"expect.{key} must be a list, not a string. Use a YAML list")
    if not isinstance(raw, list):
        _fail(path, f"expect.{key} must be a list")
    for item in raw:
        if not isinstance(item, str) or not item.strip():
            _fail(path, f"expect.{key} contains an empty or non-string entry")
    return tuple(raw)


def _parse_expectations(path: Path, raw: object) -> Expectations:
    if raw is None:
        _fail(path, "no expectations defined. Add an 'expect' block")
    if not isinstance(raw, dict):
        _fail(path, "'expect' must be a block of expectations")

    unknown = set(raw) - _EXPECT_KEYS
    if unknown:
        allowed = ", ".join(sorted(_EXPECT_KEYS))
        _fail(
            path,
            f"unknown key(s) in 'expect': {', '.join(sorted(unknown))}. "
            f"Allowed: {allowed}",
        )

    threshold = raw.get("threshold", 0.8)
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        _fail(path, "expect.threshold must be a number between 0 and 1")
    if not 0 <= threshold <= 1:
        _fail(path, f"expect.threshold must be between 0 and 1, got {threshold}")

    reference = raw.get("reference")
    if reference is not None and not isinstance(reference, str):
        _fail(path, "expect.reference must be a string")
    if "threshold" in raw and reference is None:
        _fail(path, "expect.threshold set but no 'reference' to compare against")

    judge = raw.get("judge")
    if judge is not None and (not isinstance(judge, str) or not judge.strip()):
        _fail(path, "expect.judge must be a non-empty string")

    case_sensitive = raw.get("case_sensitive", False)
    if not isinstance(case_sensitive, bool):
        _fail(path, "expect.case_sensitive must be true or false")
    if "case_sensitive" in raw and not (
        raw.get("must_contain") or raw.get("must_not_contain")
    ):
        _fail(
            path,
            "expect.case_sensitive set but there are no substring checks "
            "for it to apply to",
        )

    expectations = Expectations(
        must_contain=_string_list(path, raw.get("must_contain"), "must_contain"),
        must_not_contain=_string_list(
            path, raw.get("must_not_contain"), "must_not_contain"
        ),
        case_sensitive=case_sensitive,
        reference=reference,
        threshold=float(threshold),
        judge=judge,
    )

    # The rule the format rests on. A case with no expectations would run,
    # score nothing, and report a pass. That is worse than no case at all,
    # because it inflates the pass rate the harness exists to report.
    if expectations.is_empty():
        _fail(
            path,
            "no expectations defined. A case needs at least one of: "
            "must_contain, must_not_contain, reference, judge",
        )
    return expectations


def load_case(path: str | Path) -> Case:
    """Load and validate one case file.

    Raises CaseError with the file path and the specific problem.
    """
    path = Path(path)
    if not path.is_file():
        raise CaseError(f"{path}: no such case file")

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        # PyYAML's message includes line and column, which is more useful
        # than anything this module could reconstruct, so pass it through.
        raise CaseError(f"{path}: invalid YAML. {exc}") from exc

    if data is None:
        _fail(path, "file is empty")
    if not isinstance(data, dict):
        _fail(path, "top level of a case file must be a mapping of fields")

    unknown = set(data) - _CASE_KEYS
    if unknown:
        allowed = ", ".join(sorted(_CASE_KEYS))
        _fail(
            path,
            f"unknown top-level key(s): {', '.join(sorted(unknown))}. "
            f"Allowed: {allowed}",
        )

    case_id = _require_str(path, data, "id")
    if case_id != path.stem:
        # Keeping the id and the filename in step means a failure report can
        # name either one and a person can find the file immediately.
        _fail(path, f"id '{case_id}' does not match the filename '{path.stem}'")

    note = data.get("note", "")
    if not isinstance(note, str):
        _fail(path, "field 'note' must be a string")

    return Case(
        id=case_id,
        skill=_require_str(path, data, "skill"),
        input=_require_str(path, data, "input"),
        expect=_parse_expectations(path, data.get("expect")),
        note=note,
        source=path,
    )


def load_cases(directory: str | Path) -> list[Case]:
    """Load every case in a directory, sorted by id.

    Sorted so that two runs list results in the same order and can be
    compared line by line. Stops at the first invalid case rather than
    collecting errors: a broken case file means the run would be scored
    against an incomplete set, and a partial pass rate is misleading.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise CaseError(f"{directory}: no such case directory")

    paths = sorted(
        p for p in directory.iterdir() if p.suffix in {".yaml", ".yml"}
    )
    if not paths:
        raise CaseError(f"{directory}: no .yaml case files found")

    cases = [load_case(p) for p in paths]

    seen: dict[str, Path] = {}
    for case in cases:
        if case.id in seen:
            raise CaseError(
                f"{case.source}: duplicate id '{case.id}', "
                f"already defined in {seen[case.id]}"
            )
        seen[case.id] = case.source

    return sorted(cases, key=lambda c: c.id)

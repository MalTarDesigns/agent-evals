# agent-evals

An evaluation harness for agent and skill workflows. It runs labeled cases
against a skill, scores the output, and reports what changed between runs.

## The problem

Agent and skill workflows degrade silently. A prompt gets reworded, a model
version changes, a rule gets added to a config file, and the behavior shifts in
a way nobody notices until a user reports it. Traditional tests do not help,
because the output is natural language and there is no single correct string to
assert against.

This measures whether the behavior actually changed, and by how much.

## What this does

Takes a set of labeled cases, runs them against a skill, and scores each result
three ways:

- **Exact**, for the properties that are binary. A required block is present or
  it is not. A banned character appears or it does not.
- **Fuzzy**, for the properties that are approximate. Similarity against a
  reference answer, so a reworded but equivalent response still passes.
- **LLM-as-judge**, for the properties only a reader can assess. Tone, whether
  the response answered the question, whether it follows a style guide.

It reports a pass rate per scorer, and flags regressions by comparing against a
saved baseline from a previous run.

## The first target

The harness is developed against a real skill: an email drafting skill with a
written style guide. That guide makes a good first target because it states
rules at all three levels at once. Some are mechanical and exactly checkable,
such as a required signature block and a ban on em dashes. Some are approximate,
such as a word ceiling and a preferred closing line. Some are judgment calls,
such as whether a draft sounds like the person it is written for, or whether it
has drifted into generic assistant register.

A skill like this is where exact-match testing alone breaks down, which is the
reason this project exists.

The real skill is personal and stays unpublished. What ships here is
`skills/email-draft/STYLE.md`, a style guide for a design agency that does
not exist, written to have the same three-level shape. The cases in `cases/`
run against that.

## The case format

A case is one labeled example: an input to a skill, plus what is expected
back. Cases are YAML, one file per case in `cases/`, and the filename must
match the id.

Cases load and validate today, and the runner executes them. The scorers
that read these expectations are not built yet, so output is captured but
nothing is scored against it so far.

```yaml
id: missing-price-quote        # must match the filename
skill: email-draft             # which skill is under test
note: >                        # what this case probes, for a human
  The note asks for a quote but supplies no rate and no hours, so the
  correct draft is one that refuses to fill in the number.

input: |                       # what gets sent to the skill
  Write a client email from the rough note below.
  Recipient: Marcus
  Note: he asked what it would cost to add the booking calendar.

expect:
  # Checked by the exact scorer. Binary, no interpretation.
  must_contain:
    - "["
  must_not_contain:
    - "I hope this email finds you well"

  # Checked by the fuzzy scorer. Optional. The threshold lives on the case
  # because the right bar differs per case.
  reference: "We can add the booking calendar. It will take about
    [confirm hours] to complete."
  threshold: 0.7

  # Checked by the judge scorer. Plain English, because the judge reads prose.
  judge: |
    The draft does NOT state a specific price or hour count, because none
    was supplied. It marks the missing figure as a visible gap in square
    brackets. A draft that invents a plausible number fails this case.
```

Each expectation block maps to one scorer. All three are individually
optional, and **a case with none of them is rejected at load time**. A case
that scores nothing would report a pass and inflate the pass rate, which is
the one number the harness exists to produce.

Validation errors name the file and the problem:

```
cases/oops.yaml: unknown key(s) in 'expect': must_countain.
Allowed: judge, must_contain, must_not_contain, reference, threshold
```

Unknown keys are an error rather than ignored, because a typo would
otherwise drop an expectation silently and the case would pass for the
wrong reason.

## How it works

The runner calls the Claude API directly. It reads the skill's instructions
from `skills/<name>/STYLE.md`, sends them as the system prompt with the
case input as the user message, and captures the response.

The alternative was to shell out to Claude Code, which would run the skill
the way it actually runs in practice. Calling the API directly won on two
points: it is testable, because the client can be replaced with a fake and
the whole test suite runs with no network and no API key, and the harness
stays usable for anyone who does not have Claude Code installed. The cost
is that the harness evaluates the skill's instructions rather than the full
Claude Code environment those instructions normally run inside.

Two things the runner deliberately does not do:

- **It does not pin temperature or cache responses.** Non-determinism is
  what this measures, so removing it would defeat the purpose.
- **It does not retry beyond a single rate-limit retry.** A harness that
  retries hard turns a rate-limited run into a hang, and a hung run is
  worse than a failed one because nobody knows how long to wait.

Failures are values rather than exceptions. A case whose call times out
comes back as a result marked errored, with the reason, so a run of twenty
cases reports nineteen results and one error instead of losing everything
to a traceback.

The API key is read from `ANTHROPIC_API_KEY` by the SDK. It is never read
from a file in this repo, never written anywhere, and never stored on a
result.

## What this does not do

These are non-goals, not gaps to be filled later:

- **No web UI.** Results go to a terminal and to a file.
- **No hosted service.** It runs locally, in your own environment.
- **No multi-model comparison.** It evaluates one configuration at a time.
- **No CI integration.** No GitHub Action, no reporter plugins.

Each of these is a larger project than the harness itself, and adding them would
make this harder to understand without making it better at the one thing it
does.

## Status

Early, and in active development. The scope above is what it is aiming at, not
a description of a finished tool. Expect the case format to change.

## Example output

<!-- PLACEHOLDER: filled in session 10, once there is real output to show.
     Do not write example output before the harness produces it. -->

## License

MIT

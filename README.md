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

All three scorers are built. Cases load and validate, the runner executes
them, and each expectation block is checked by its scorer. What does not
exist yet is the report: results are returned per case, so there is no
pass rate across a run and no comparison against a saved baseline.

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

## Setup

```bash
git clone https://github.com/MalTarDesigns/agent-evals
cd agent-evals
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

The runner needs an Anthropic API key. Either export it:

```bash
export ANTHROPIC_API_KEY=your-key
```

or copy `.env.example` to `.env` and fill it in. The file is gitignored.
An exported variable wins over the file, so a one-off export does what you
expect.

Then run the tests, which make no API calls and need no key:

```bash
pytest
```

`AGENT_EVALS_MODEL` overrides the model for a run, which is how the same
cases get compared across model versions. It defaults to `claude-opus-5`.
`AGENT_EVALS_JUDGE_MODEL` sets the judge separately, defaulting to
`claude-sonnet-5` so the judge is not grading its own output.

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

## Scoring

Every scorer returns the same shape: a pass flag, a score from 0 to 1, and
a reason. The reason is required, on a pass as well as a failure, because a
result that does not say why is not worth reading.

**The score and the pass flag are separate.** A case requiring four phrases
that finds three scores 0.75 and still fails. Both numbers say something
different: the flag answers "is this correct", the score answers "how far
off is it". A prompt change that moves a case from 0.25 to 0.75 is progress
worth seeing, even though the case fails at both ends, and collapsing the
two into a bool would hide exactly the trend this harness exists to report.

A scorer returns nothing at all when a case gives it no expectations to
check. That is different from scoring zero: a skipped check should not drag
down a pass rate.

Failure reasons name the specific strings, not a count:

```
missing "HERE" (8/9 checks passed)

missing "Enjoy your day,"; forbidden "I hope this email finds you well",
"—" present (2/9 checks passed, case-insensitive)
```

Substring checks ignore case by default, since most expectations are about
wording. A case sets `case_sensitive: true` when the capitalization is the
point, such as the style guide's rule that links are labeled HERE in
capitals.

### Fuzzy similarity

The fuzzy scorer uses token-level F1 against the case's reference text.
Precision penalizes a rambling response that happens to contain the
reference, recall penalizes one that drops most of it, and the harmonic
mean means a response has to do both to score well.

**What it misses.** It does not understand meaning. "The site is live" and
"The site is not live" score almost identically, because they share every
token but one. Embedding similarity would catch that, and was rejected on
cost rather than quality: it would add a dependency and an API call per
comparison, and a scorer meant to run on every change should be free to
run. Judging meaning is the judge scorer's job.

Text is lowercased and stripped of punctuation before comparison.
Stopwords are deliberately **not** removed. For a style guide, "let me know
if you have any questions" is almost entirely stopwords, and removing them
would erase the phrase this harness most wants to track.

The default threshold of 0.8 is arbitrary and is labeled as such in the
code. It has not been calibrated against human judgment, because that
would need labeled data this project does not have. Each case overrides
it, and the shipped case that uses a reference sets 0.35, because a
correct draft carries the same facts in its own wording rather than
reproducing the reference.

### Model as judge

The judge scorer sends the case input, the response, and the case's
plain-English rubric to a model, and asks for a verdict.

**A three-point scale, not 1 to 10.** Judges cluster hard on 7 and 8 with
wide scales, which destroys the signal the score is supposed to carry.
The verdicts are pass, partial, and fail, mapped to 1.0, 0.5, and 0.0,
and only pass counts as passing.

**Justification before verdict.** The judge writes its reasoning first and
the verdict second. Asking for the score first produces post-hoc
rationalization of a number the model already picked.

**Biases mitigated, and how.** Verbosity bias, the judge is instructed to
ignore length and judge only against the rubric. Self-preference, the
judge model is configured separately from the model under test and
defaults to a different one, `claude-sonnet-5` judging `claude-opus-5`.
Non-determinism, temperature is 0, which reduces variance without
eliminating it.

**Malformed output fails as an error, never as a pass.** One retry, then
the case is marked errored with the reason. A score is never guessed from
unparseable output.

**The judge is not ground truth.** It is a proxy for human evaluation, and
this project has not measured its agreement with human labels, so no
claim is made that it is calibrated or validated. Treat a judge verdict
as evidence, not as a verdict.

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

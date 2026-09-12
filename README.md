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

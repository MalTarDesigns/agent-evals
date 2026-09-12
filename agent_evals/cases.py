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

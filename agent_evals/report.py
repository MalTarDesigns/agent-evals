"""Format run results for a terminal, and compare against a baseline.

Responsible for:
    - Rendering per-case and per-scorer results in a readable form.
    - Computing pass rates.
    - Diffing a run against a saved baseline and flagging regressions,
      which is the output the harness exists to produce.

Deliberately not responsible for:
    - Deciding what counts as a pass. Scorers return scores, this module
      presents them.
    - Storage format or history. A baseline is a file, not a database.
    - Any output target other than a terminal and a file. No web UI and
      no CI reporter, per the non-goals in the README.
"""

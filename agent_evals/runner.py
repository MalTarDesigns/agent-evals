"""Execute a skill against a case and capture its output.

Responsible for:
    - Invoking the skill under evaluation with a case's input.
    - Capturing the response as text, along with what it cost to produce
      (tokens, wall time), so a run can be compared to a previous one.
    - Isolating failures, so one case that errors does not abort the run.

Deliberately not responsible for:
    - Scoring. The runner records what happened, it does not judge it.
    - Retrying on a bad response. A flaky skill is a finding, not a
      problem for the harness to paper over.
    - Parallelism. Runs are sequential until there is a measured reason
      not to be.
"""

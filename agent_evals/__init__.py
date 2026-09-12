"""Evaluation harness for agent and skill workflows.

Runs labeled cases against a skill, scores the output with several
independent scorers, and reports pass rates and regressions against a
saved baseline.

The package is organized around a one-way pipeline:

    cases -> runner -> scorers -> report

Each stage is independent. Cases know nothing about scoring, the runner
knows nothing about how output is judged, and the report knows nothing
about how results were produced.
"""

__version__ = "0.0.1"

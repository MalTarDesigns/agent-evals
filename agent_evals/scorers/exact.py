"""Exact, mechanical checks on a response.

For properties with no interpretation involved: a required block is
present, a banned character does not appear, a line matches a pattern.

Responsible for:
    - Substring, absence, and pattern checks against a response.
    - Returning which specific check failed, not just that one did.

Deliberately not responsible for:
    - Anything approximate. If a check needs a threshold, it belongs in
      the fuzzy scorer.
    - Meaning. This scorer cannot tell correct from confidently wrong.
"""

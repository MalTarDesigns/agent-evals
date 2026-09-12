"""Where the harness gets its settings.

One job: make `ANTHROPIC_API_KEY` available to the SDK, whether it comes
from the shell or from a local `.env` file, so a fresh clone runs without
the reader having to guess what to export.

Responsible for:
    - Loading a `.env` file from the project root, if one exists.
    - Reading the optional model override.

Deliberately not responsible for:
    - Holding, returning, or logging the key itself. Nothing here reads
      the key's value. `load_env` puts it in the environment and the
      Anthropic SDK picks it up from there, so the secret never passes
      through this module's return values or this project's code.
    - Any other configuration. Model and skills root are arguments to
      `run_case`, not global state, because a run's settings belong on
      the result that records them.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

_loaded = False


def load_env(path: Path | str | None = None, override: bool = False) -> bool:
    """Load a `.env` file into the environment if one is present.

    Returns True when a file was found and read.

    The environment wins over the file by default. Someone who exports a
    key for one command expects that to take effect, and silently
    preferring a stale file would be a confusing way to lose an hour.

    Safe to call repeatedly. Only the first call does work, unless an
    explicit path is given, which the tests use.
    """
    global _loaded

    if path is None:
        if _loaded:
            return True
        path = ENV_FILE

    env_path = Path(path)
    if not env_path.is_file():
        return False

    load_dotenv(env_path, override=override)
    if path == ENV_FILE:
        _loaded = True
    return True


def has_api_key() -> bool:
    """Whether a key is available, without revealing anything about it.

    Returns a bool rather than the key, so no caller can accidentally log
    or store the value. Callers that need to fail early on a missing key
    check this; the SDK reads the actual value from the environment.
    """
    return bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())


def model_override() -> str | None:
    """The model from AGENT_EVALS_MODEL, or None to use the default.

    An override exists so a run can be repeated against a different model
    without editing code, which is the comparison this harness is for.
    """
    value = os.environ.get("AGENT_EVALS_MODEL", "").strip()
    return value or None

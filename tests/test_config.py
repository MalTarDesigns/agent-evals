"""Tests for environment and .env loading.

These write their own temporary env files rather than touching the
project's real .env, so they run identically on a machine that has one
and a machine that does not.

Nothing here asserts on a key's value. The point of config.py is that the
key passes through the environment to the SDK without this project ever
holding it, and the tests are written to match that.
"""

from __future__ import annotations

import agent_evals.config as config
from agent_evals.config import has_api_key, load_env, model_override


def write_env(tmp_path, body):
    path = tmp_path / ".env"
    path.write_text(body, encoding="utf-8")
    return path


def test_loads_a_key_from_an_env_file(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    path = write_env(tmp_path, "ANTHROPIC_API_KEY=from-the-file\n")

    assert load_env(path) is True
    assert has_api_key()


def test_a_missing_env_file_is_not_an_error(tmp_path, monkeypatch):
    """A clone with no .env must still work off an exported variable."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    assert load_env(tmp_path / "nothing-here") is False
    assert not has_api_key()


def test_the_environment_wins_over_the_file(tmp_path, monkeypatch):
    """Exporting a key for one command must take effect.

    Silently preferring a stale file is a confusing way to lose an hour.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-the-shell")
    path = write_env(tmp_path, "ANTHROPIC_API_KEY=from-the-file\n")

    load_env(path)

    import os

    assert os.environ["ANTHROPIC_API_KEY"] == "from-the-shell"


def test_override_lets_the_file_win_when_asked(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-the-shell")
    path = write_env(tmp_path, "ANTHROPIC_API_KEY=from-the-file\n")

    load_env(path, override=True)

    import os

    assert os.environ["ANTHROPIC_API_KEY"] == "from-the-file"


def test_has_api_key_is_false_for_an_empty_value(monkeypatch):
    """An exported but blank variable is not a usable key."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "   ")

    assert not has_api_key()


def test_has_api_key_returns_a_bool_not_the_key(monkeypatch):
    """The contract that keeps the secret out of callers' hands."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a-secret-value")

    result = has_api_key()

    assert result is True
    assert "a-secret-value" not in repr(result)


def test_model_override_reads_the_env_var(monkeypatch):
    monkeypatch.setenv("AGENT_EVALS_MODEL", "claude-sonnet-5")

    assert model_override() == "claude-sonnet-5"


def test_model_override_is_none_when_unset(monkeypatch):
    monkeypatch.delenv("AGENT_EVALS_MODEL", raising=False)

    assert model_override() is None


def test_model_override_ignores_a_blank_value(monkeypatch):
    monkeypatch.setenv("AGENT_EVALS_MODEL", "  ")

    assert model_override() is None


def test_comments_and_blank_lines_in_an_env_file_are_ignored(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENT_EVALS_MODEL", raising=False)
    path = write_env(
        tmp_path,
        "# a comment\n\nAGENT_EVALS_MODEL=claude-sonnet-5\n",
    )

    load_env(path, override=True)

    assert model_override() == "claude-sonnet-5"


def test_repeated_default_loads_do_not_re_read(tmp_path, monkeypatch):
    """Calling load_env on every run must not cost a file read each time."""
    monkeypatch.setattr(config, "_loaded", False)
    monkeypatch.setattr(config, "ENV_FILE", tmp_path / "absent")

    assert load_env() is False

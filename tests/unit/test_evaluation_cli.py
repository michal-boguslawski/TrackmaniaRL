from __future__ import annotations

from rl_lib import cli


def test_nonnegative_int_prompt_accepts_blank_and_valid_step(monkeypatch):
    answers = iter(["", "2200000"])
    monkeypatch.setattr(cli, "_ask", lambda prompt, default=None: next(answers))

    assert cli._nonnegative_int("Checkpoint step", allow_blank=True) is None
    assert cli._nonnegative_int("Checkpoint step", allow_blank=True) == 2_200_000


def test_nonnegative_int_prompt_retries_invalid_values(monkeypatch):
    answers = iter(["-1", "not-a-step", "2200000"])
    monkeypatch.setattr(cli, "_ask", lambda prompt, default=None: next(answers))
    monkeypatch.setattr(cli.logger, "warning", lambda *args: None)

    assert cli._nonnegative_int("Checkpoint step") == 2_200_000

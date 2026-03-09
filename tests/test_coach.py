"""Tests for Claude CLI coaching module."""

from unittest.mock import MagicMock, patch

from meeting_transcriber.coach import (
  _build_prompt,
  _run_coaching,
)


class TestBuildPrompt:
  def test_includes_transcript(self):
    prompt = _build_prompt("system", "hello world", "")
    assert "hello world" in prompt

  def test_includes_playbook_when_provided(self):
    prompt = _build_prompt("system", "transcript", "playbook content")
    assert "playbook content" in prompt
    assert "Playbook" in prompt

  def test_excludes_playbook_when_empty(self):
    prompt = _build_prompt("system", "transcript", "")
    assert "Playbook" not in prompt

  def test_truncates_long_playbook(self):
    long_playbook = "x" * 5000
    prompt = _build_prompt("system", "transcript", long_playbook)
    assert "x" * 4000 in prompt
    assert "x" * 4001 not in prompt

  def test_includes_system_prompt(self):
    prompt = _build_prompt("custom system", "transcript", "")
    assert "custom system" in prompt

  def test_detects_speaker_labels(self):
    prompt = _build_prompt("sys", "[我方] hello", "")
    assert "已標記說話者" in prompt

  def test_detects_no_speaker_labels(self):
    prompt = _build_prompt("sys", "hello world", "")
    assert "未標記說話者" in prompt


class TestRunCoaching:
  @patch("meeting_transcriber.coach.call_claude_cli")
  def test_callback_receives_result(self, mock_cli):
    mock_cli.return_value = "analysis text"
    callback = MagicMock()
    _run_coaching("transcript", "playbook", callback, "system")
    callback.assert_called_once_with("analysis text")

  @patch("meeting_transcriber.coach.call_claude_cli")
  def test_cli_error_shows_message(self, mock_cli):
    mock_cli.side_effect = RuntimeError("CLI failed")
    callback = MagicMock()
    _run_coaching("transcript", "", callback, "system")
    callback.assert_called_once()
    assert "error" in callback.call_args[0][0].lower()

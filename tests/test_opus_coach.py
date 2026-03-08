"""Tests for Opus deep coaching via Claude Code CLI."""

import subprocess
from unittest.mock import MagicMock, patch

from meeting_transcriber.opus_coach import _build_prompt, _run_claude


class TestBuildPrompt:
  def test_includes_transcript(self):
    prompt = _build_prompt("hello world", "")
    assert "hello world" in prompt

  def test_includes_playbook_when_provided(self):
    prompt = _build_prompt("transcript", "playbook content")
    assert "playbook content" in prompt

  def test_excludes_playbook_section_when_empty(self):
    prompt = _build_prompt("transcript", "")
    assert "## Playbook" not in prompt

  def test_truncates_long_playbook(self):
    long_playbook = "x" * 5000
    prompt = _build_prompt("transcript", long_playbook)
    assert len(prompt) < 5000 + 500


class TestRunClaude:
  @patch("meeting_transcriber.opus_coach.subprocess.run")
  def test_calls_claude_cli(self, mock_run):
    mock_run.return_value = subprocess.CompletedProcess(
      args=[], returncode=0, stdout="Analysis result", stderr=""
    )
    callback = MagicMock()
    _run_claude("transcript", "playbook", callback)
    mock_run.assert_called_once()
    args = mock_run.call_args[0][0]
    assert args[0] == "claude"
    assert "-p" in args

  @patch("meeting_transcriber.opus_coach.subprocess.run")
  def test_callback_receives_output(self, mock_run):
    mock_run.return_value = subprocess.CompletedProcess(
      args=[], returncode=0, stdout="Deep analysis", stderr=""
    )
    callback = MagicMock()
    _run_claude("transcript", "", callback)
    callback.assert_called_once_with("Deep analysis")

  @patch("meeting_transcriber.opus_coach.subprocess.run")
  def test_handles_cli_error(self, mock_run):
    mock_run.return_value = subprocess.CompletedProcess(
      args=[], returncode=1, stdout="", stderr="API error"
    )
    callback = MagicMock()
    _run_claude("transcript", "", callback)
    callback.assert_called_once()
    assert "error" in callback.call_args[0][0].lower()

  @patch("meeting_transcriber.opus_coach.subprocess.run")
  def test_handles_timeout(self, mock_run):
    mock_run.side_effect = subprocess.TimeoutExpired(cmd="claude", timeout=30)
    callback = MagicMock()
    _run_claude("transcript", "", callback)
    callback.assert_called_once()
    assert "timed out" in callback.call_args[0][0].lower()

  @patch("meeting_transcriber.opus_coach.subprocess.run")
  def test_handles_missing_cli(self, mock_run):
    mock_run.side_effect = FileNotFoundError()
    callback = MagicMock()
    _run_claude("transcript", "", callback)
    callback.assert_called_once()
    assert "not found" in callback.call_args[0][0].lower()

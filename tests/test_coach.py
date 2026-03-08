"""Tests for Anthropic SDK coaching module."""

from unittest.mock import MagicMock, patch

from meeting_transcriber.coach import (
  _build_user_message,
  _call_anthropic,
  _run_coaching,
)


class TestBuildUserMessage:
  def test_includes_transcript(self):
    msg = _build_user_message("hello world", "")
    assert "hello world" in msg

  def test_includes_playbook_when_provided(self):
    msg = _build_user_message("transcript", "playbook content")
    assert "playbook content" in msg
    assert "## Playbook" in msg

  def test_excludes_playbook_when_empty(self):
    msg = _build_user_message("transcript", "")
    assert "## Playbook" not in msg

  def test_truncates_long_playbook(self):
    long_playbook = "x" * 5000
    msg = _build_user_message("transcript", long_playbook)
    # Playbook truncated to 4000 chars
    assert "x" * 4000 in msg
    assert "x" * 4001 not in msg


class TestCallAnthropic:
  @patch("meeting_transcriber.coach._get_client")
  def test_calls_api_with_correct_params(self, mock_get_client):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="coaching result")]
    mock_client.messages.create.return_value = mock_response
    mock_get_client.return_value = mock_client

    result = _call_anthropic("system prompt", "user msg", "claude-sonnet-4-20250514")

    mock_client.messages.create.assert_called_once()
    call_kwargs = mock_client.messages.create.call_args[1]
    assert call_kwargs["model"] == "claude-sonnet-4-20250514"
    assert call_kwargs["system"] == "system prompt"
    assert call_kwargs["messages"][0]["content"] == "user msg"
    assert result == "coaching result"


class TestRunCoaching:
  @patch("meeting_transcriber.coach._call_anthropic")
  def test_callback_receives_result(self, mock_call):
    mock_call.return_value = "analysis text"
    callback = MagicMock()
    _run_coaching("transcript", "playbook", callback, "model", "system")
    callback.assert_called_once_with("analysis text")

  @patch("meeting_transcriber.coach._call_anthropic")
  def test_empty_result_shows_message(self, mock_call):
    mock_call.return_value = ""
    callback = MagicMock()
    _run_coaching("transcript", "", callback, "model", "system")
    callback.assert_called_once_with("No analysis generated.")

  @patch("meeting_transcriber.coach._call_anthropic")
  def test_api_error_shows_message(self, mock_call):
    mock_call.side_effect = Exception("API down")
    callback = MagicMock()
    _run_coaching("transcript", "", callback, "model", "system")
    callback.assert_called_once()
    assert "error" in callback.call_args[0][0].lower()

  @patch("meeting_transcriber.coach._call_anthropic")
  def test_uses_provided_model(self, mock_call):
    mock_call.return_value = "ok"
    callback = MagicMock()
    _run_coaching("t", "", callback, "claude-opus-4-20250514", "sys")
    assert mock_call.call_args[0][2] == "claude-opus-4-20250514"

  @patch("meeting_transcriber.coach._call_anthropic")
  def test_uses_provided_system_prompt(self, mock_call):
    mock_call.return_value = "ok"
    callback = MagicMock()
    _run_coaching("t", "", callback, "model", "custom system")
    assert mock_call.call_args[0][0] == "custom system"

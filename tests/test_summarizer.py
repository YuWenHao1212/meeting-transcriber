"""Tests for meeting summarization via Anthropic Claude."""

from unittest.mock import MagicMock, patch

import pytest

from meeting_transcriber.summarizer import summarize


SAMPLE_TRANSCRIPT = (
  "Alice: Let's discuss the Q2 roadmap.\n"
  "Bob: I think we should focus on the API redesign.\n"
  "Alice: Agreed. Bob, can you draft a proposal by Friday?\n"
  "Bob: Sure, I'll have it ready.\n"
)

SAMPLE_PLAYBOOK = (
  "## Pre-Meeting Objectives\n"
  "1. Finalize Q2 roadmap priorities\n"
  "2. Assign API redesign owner\n"
  "3. Discuss hiring timeline\n"
)

CUSTOM_TEMPLATE = (
  "## TL;DR\n"
  "{summary}\n"
  "## Takeaways\n"
  "{takeaways}\n"
)


def _make_mock_response(text: str = "# Meeting Summary\n\nMock summary.") -> MagicMock:
  """Create a mock Anthropic messages.create() response."""
  content_block = MagicMock()
  content_block.text = text
  response = MagicMock()
  response.content = [content_block]
  return response


class TestSummarizeCallsAnthropic:
  """Verify summarize() sends correct requests to the Anthropic API."""

  @patch("meeting_transcriber.summarizer.anthropic.Anthropic")
  def test_calls_anthropic_with_structured_system_prompt(self, mock_cls):
    mock_client = MagicMock()
    mock_cls.return_value = mock_client
    mock_client.messages.create.return_value = _make_mock_response()

    summarize(SAMPLE_TRANSCRIPT)

    call_kwargs = mock_client.messages.create.call_args.kwargs
    system_prompt = call_kwargs["system"]
    assert "Meeting Summary" in system_prompt
    assert "Key Decisions" in system_prompt
    assert "Action Items" in system_prompt
    assert "Key Discussions" in system_prompt
    assert "Follow-ups" in system_prompt

  @patch("meeting_transcriber.summarizer.anthropic.Anthropic")
  def test_sends_transcript_as_user_message(self, mock_cls):
    mock_client = MagicMock()
    mock_cls.return_value = mock_client
    mock_client.messages.create.return_value = _make_mock_response()

    summarize(SAMPLE_TRANSCRIPT)

    call_kwargs = mock_client.messages.create.call_args.kwargs
    user_content = call_kwargs["messages"][0]["content"]
    assert SAMPLE_TRANSCRIPT in user_content


class TestSummarizeTranscriptOnly:
  """Test summarize() with transcript only (no playbook, no template)."""

  @patch("meeting_transcriber.summarizer.anthropic.Anthropic")
  def test_no_cross_reference_instruction(self, mock_cls):
    mock_client = MagicMock()
    mock_cls.return_value = mock_client
    mock_client.messages.create.return_value = _make_mock_response()

    summarize(SAMPLE_TRANSCRIPT)

    call_kwargs = mock_client.messages.create.call_args.kwargs
    system_prompt = call_kwargs["system"]
    assert "cross-reference" not in system_prompt.lower()
    assert "playbook" not in system_prompt.lower()


class TestSummarizeWithPlaybook:
  """Test summarize() with transcript + playbook."""

  @patch("meeting_transcriber.summarizer.anthropic.Anthropic")
  def test_injects_cross_reference_instruction(self, mock_cls):
    mock_client = MagicMock()
    mock_cls.return_value = mock_client
    mock_client.messages.create.return_value = _make_mock_response()

    summarize(SAMPLE_TRANSCRIPT, playbook=SAMPLE_PLAYBOOK)

    call_kwargs = mock_client.messages.create.call_args.kwargs
    system_prompt = call_kwargs["system"]
    assert "cross-reference" in system_prompt.lower()
    assert "objectives" in system_prompt.lower()

  @patch("meeting_transcriber.summarizer.anthropic.Anthropic")
  def test_playbook_included_in_user_message(self, mock_cls):
    mock_client = MagicMock()
    mock_cls.return_value = mock_client
    mock_client.messages.create.return_value = _make_mock_response()

    summarize(SAMPLE_TRANSCRIPT, playbook=SAMPLE_PLAYBOOK)

    call_kwargs = mock_client.messages.create.call_args.kwargs
    user_content = call_kwargs["messages"][0]["content"]
    assert SAMPLE_PLAYBOOK in user_content
    assert SAMPLE_TRANSCRIPT in user_content


class TestSummarizeWithTemplate:
  """Test summarize() with custom template."""

  @patch("meeting_transcriber.summarizer.anthropic.Anthropic")
  def test_custom_template_in_system_prompt(self, mock_cls):
    mock_client = MagicMock()
    mock_cls.return_value = mock_client
    mock_client.messages.create.return_value = _make_mock_response()

    summarize(SAMPLE_TRANSCRIPT, template=CUSTOM_TEMPLATE)

    call_kwargs = mock_client.messages.create.call_args.kwargs
    system_prompt = call_kwargs["system"]
    assert "TL;DR" in system_prompt
    assert "Takeaways" in system_prompt


class TestSummarizeOutput:
  """Test that summarize() returns a markdown string."""

  @patch("meeting_transcriber.summarizer.anthropic.Anthropic")
  def test_returns_markdown_string(self, mock_cls):
    expected = "# Meeting Summary\n\nThis is a summary."
    mock_client = MagicMock()
    mock_cls.return_value = mock_client
    mock_client.messages.create.return_value = _make_mock_response(expected)

    result = summarize(SAMPLE_TRANSCRIPT)

    assert isinstance(result, str)
    assert result == expected


class TestSummarizeModelConfig:
  """Test that the model parameter is configurable."""

  @patch("meeting_transcriber.summarizer.anthropic.Anthropic")
  def test_default_model(self, mock_cls):
    mock_client = MagicMock()
    mock_cls.return_value = mock_client
    mock_client.messages.create.return_value = _make_mock_response()

    summarize(SAMPLE_TRANSCRIPT)

    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert call_kwargs["model"] == "claude-sonnet-4-20250514"

  @patch("meeting_transcriber.summarizer.anthropic.Anthropic")
  def test_custom_model(self, mock_cls):
    mock_client = MagicMock()
    mock_cls.return_value = mock_client
    mock_client.messages.create.return_value = _make_mock_response()

    summarize(SAMPLE_TRANSCRIPT, model="claude-opus-4-20250514")

    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert call_kwargs["model"] == "claude-opus-4-20250514"

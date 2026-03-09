"""Tests for meeting summarization and transcript cleaning via Claude CLI."""

from unittest.mock import patch

from meeting_transcriber.summarizer import (
  _build_clean_prompt,
  _build_summarize_prompt,
  _strip_code_fences,
  clean_transcript,
  summarize,
  summarize_incremental,
)

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

CUSTOM_TEMPLATE = "## TL;DR\n{summary}\n## Takeaways\n{takeaways}\n"


class TestBuildSummarizePrompt:
  def test_includes_transcript(self):
    prompt = _build_summarize_prompt(SAMPLE_TRANSCRIPT)
    assert SAMPLE_TRANSCRIPT in prompt

  def test_includes_default_template(self):
    prompt = _build_summarize_prompt(SAMPLE_TRANSCRIPT)
    assert "Meeting Summary" in prompt
    assert "Key Decisions" in prompt
    assert "Action Items" in prompt

  def test_includes_playbook(self):
    prompt = _build_summarize_prompt(SAMPLE_TRANSCRIPT, playbook=SAMPLE_PLAYBOOK)
    assert SAMPLE_PLAYBOOK in prompt
    assert "Playbook 覆蓋率" in prompt

  def test_no_playbook_coverage_without_playbook(self):
    prompt = _build_summarize_prompt(SAMPLE_TRANSCRIPT)
    assert "Playbook 覆蓋率" not in prompt

  def test_custom_template(self):
    prompt = _build_summarize_prompt(SAMPLE_TRANSCRIPT, template=CUSTOM_TEMPLATE)
    assert "TL;DR" in prompt
    assert "Takeaways" in prompt

  def test_playbook_has_checkmarks(self):
    prompt = _build_summarize_prompt(SAMPLE_TRANSCRIPT, playbook=SAMPLE_PLAYBOOK)
    assert "\u2705" in prompt  # ✅
    assert "\u274c" in prompt  # ❌

  def test_playbook_references_timestamps(self):
    prompt = _build_summarize_prompt(SAMPLE_TRANSCRIPT, playbook=SAMPLE_PLAYBOOK)
    assert "timestamp" in prompt.lower()


class TestSummarize:
  @patch("meeting_transcriber.summarizer.call_claude_cli")
  def test_returns_cli_result(self, mock_cli):
    mock_cli.return_value = "# Summary\n\nMock result."
    result = summarize(SAMPLE_TRANSCRIPT)
    assert result == "# Summary\n\nMock result."
    mock_cli.assert_called_once()

  @patch("meeting_transcriber.summarizer.call_claude_cli")
  def test_passes_prompt_with_transcript(self, mock_cli):
    mock_cli.return_value = "ok"
    summarize(SAMPLE_TRANSCRIPT)
    prompt = mock_cli.call_args[0][0]
    assert SAMPLE_TRANSCRIPT in prompt

  @patch("meeting_transcriber.summarizer.call_claude_cli")
  def test_passes_playbook(self, mock_cli):
    mock_cli.return_value = "ok"
    summarize(SAMPLE_TRANSCRIPT, playbook=SAMPLE_PLAYBOOK)
    prompt = mock_cli.call_args[0][0]
    assert SAMPLE_PLAYBOOK in prompt


class TestSummarizeIncremental:
  @patch("meeting_transcriber.summarizer.call_claude_cli")
  def test_includes_existing_summary(self, mock_cli):
    mock_cli.return_value = "updated summary"
    result = summarize_incremental("new text", "existing summary")
    prompt = mock_cli.call_args[0][0]
    assert "existing summary" in prompt
    assert "new text" in prompt
    assert result == "updated summary"

  @patch("meeting_transcriber.summarizer.call_claude_cli")
  def test_includes_playbook(self, mock_cli):
    mock_cli.return_value = "ok"
    summarize_incremental("new", "existing", playbook=SAMPLE_PLAYBOOK)
    prompt = mock_cli.call_args[0][0]
    assert SAMPLE_PLAYBOOK in prompt


class TestBuildCleanPrompt:
  def test_includes_chunk(self):
    prompt = _build_clean_prompt("some transcript", None)
    assert "some transcript" in prompt

  def test_includes_playbook(self):
    prompt = _build_clean_prompt("transcript", "playbook terms")
    assert "playbook terms" in prompt

  def test_includes_context_before(self):
    prompt = _build_clean_prompt("main", None, context_before="before text")
    assert "before text" in prompt
    assert "CONTEXT" in prompt

  def test_includes_context_after(self):
    prompt = _build_clean_prompt("main", None, context_after="after text")
    assert "after text" in prompt

  def test_speaker_instructions(self):
    prompt = _build_clean_prompt("transcript", None)
    assert "NEVER swap speakers" in prompt


class TestStripCodeFences:
  def test_strips_preamble(self):
    text = "Here is the cleaned transcript:\n[00:01] hello"
    assert _strip_code_fences(text) == "[00:01] hello"

  def test_strips_code_fences(self):
    text = "```\n[00:01] hello\n```"
    assert _strip_code_fences(text) == "[00:01] hello"

  def test_strips_language_code_fence(self):
    text = "```markdown\n[00:01] hello\n```"
    assert _strip_code_fences(text) == "[00:01] hello"

  def test_clean_text_unchanged(self):
    text = "[00:01] [我方] hello"
    assert _strip_code_fences(text) == "[00:01] [我方] hello"


class TestCleanTranscript:
  @patch("meeting_transcriber.summarizer.call_claude_cli")
  def test_short_transcript_single_pass(self, mock_cli):
    mock_cli.return_value = "[00:01] cleaned text"
    result = clean_transcript("line1\nline2\nline3")
    assert result == "[00:01] cleaned text"
    assert mock_cli.call_count == 1

  @patch("meeting_transcriber.summarizer.call_claude_cli")
  def test_long_transcript_chunked(self, mock_cli):
    mock_cli.return_value = "cleaned chunk"
    lines = "\n".join([f"[{i:02d}:00] line {i}" for i in range(150)])
    result = clean_transcript(lines)
    assert mock_cli.call_count == 2  # 150 lines / 100 per chunk = 2

  @patch("meeting_transcriber.summarizer.call_claude_cli")
  def test_passes_playbook(self, mock_cli):
    mock_cli.return_value = "cleaned"
    clean_transcript("line1", playbook="playbook content")
    prompt = mock_cli.call_args[0][0]
    assert "playbook content" in prompt

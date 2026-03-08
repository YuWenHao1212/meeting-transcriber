"""Tests for real-time coaching prompt engine."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from meeting_transcriber.prompter import (
  ActionItem,
  ContextChunk,
  ContextMatch,
  DetectedQuestion,
  detect_action_items,
  detect_questions,
  generate_prompt_card,
  load_context,
  match_context,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def context_file(tmp_path: Path) -> Path:
  """Create a sample context markdown file."""
  p = tmp_path / "playbook.md"
  p.write_text(
    "## Pricing Strategy\n\n"
    "- Base price: $500/month\n"
    "- Enterprise discount: 20%\n\n"
    "## Timeline\n\n"
    "- MVP launch: Q2 2026\n"
    "- Full release: Q4 2026\n",
    encoding="utf-8",
  )
  return p


@pytest.fixture()
def chinese_context_file(tmp_path: Path) -> Path:
  """Create a context file with Chinese content."""
  p = tmp_path / "cheatsheet.md"
  p.write_text(
    "## 報價範圍\n\n"
    "- 基本方案預算：每月五千\n"
    "- 進階方案預算：每月一萬\n\n"
    "## 時程規劃\n\n"
    "- 第一階段：需求確認\n"
    "- 第二階段：開發交付\n",
    encoding="utf-8",
  )
  return p


@pytest.fixture()
def empty_file(tmp_path: Path) -> Path:
  """Create an empty file."""
  p = tmp_path / "empty.md"
  p.write_text("", encoding="utf-8")
  return p


def _make_chat_response(data: list[dict]) -> MagicMock:
  """Create a mock Azure OpenAI chat completion response with JSON."""
  message = MagicMock()
  message.content = json.dumps(data)
  choice = MagicMock()
  choice.message = message
  response = MagicMock()
  response.choices = [choice]
  return response


def _make_chat_response_with_codeblock(data: list[dict]) -> MagicMock:
  """Create a mock response wrapped in markdown code blocks."""
  message = MagicMock()
  message.content = f"```json\n{json.dumps(data)}\n```"
  choice = MagicMock()
  choice.message = message
  response = MagicMock()
  response.choices = [choice]
  return response


# ---------------------------------------------------------------------------
# Context Loading
# ---------------------------------------------------------------------------


class TestLoadContext:
  """Test load_context() file loading and chunking."""

  def test_loads_single_file(self, context_file: Path):
    chunks = load_context([str(context_file)])
    assert len(chunks) > 0
    assert all(isinstance(c, ContextChunk) for c in chunks)

  def test_chunk_has_source_filename(self, context_file: Path):
    chunks = load_context([str(context_file)])
    assert all(c.source == "playbook.md" for c in chunks)

  def test_chunk_has_text_and_keywords(self, context_file: Path):
    chunks = load_context([str(context_file)])
    for chunk in chunks:
      assert chunk.text
      assert isinstance(chunk.keywords, list)

  def test_splits_by_paragraph(self, context_file: Path):
    chunks = load_context([str(context_file)])
    # The file has 4 paragraphs (2 headers + 2 bullet lists)
    assert len(chunks) >= 2

  def test_skips_nonexistent_path(self):
    chunks = load_context(["/nonexistent/file.md"])
    assert chunks == []

  def test_skips_nonexistent_mixed_with_valid(self, context_file: Path):
    chunks = load_context(["/nonexistent/file.md", str(context_file)])
    assert len(chunks) > 0

  def test_empty_file_returns_empty(self, empty_file: Path):
    chunks = load_context([str(empty_file)])
    assert chunks == []

  def test_loads_multiple_files(self, context_file: Path, chinese_context_file: Path):
    chunks = load_context([str(context_file), str(chinese_context_file)])
    sources = {c.source for c in chunks}
    assert "playbook.md" in sources
    assert "cheatsheet.md" in sources

  def test_extracts_chinese_keywords(self, chinese_context_file: Path):
    chunks = load_context([str(chinese_context_file)])
    all_keywords = []
    for chunk in chunks:
      all_keywords.extend(chunk.keywords)
    # Should extract Chinese terms like 報價範圍, 預算, etc.
    keyword_text = " ".join(all_keywords)
    assert any(term in keyword_text for term in ["報價", "預算", "時程"]), (
      f"Expected Chinese keywords, got: {all_keywords}"
    )


# ---------------------------------------------------------------------------
# Question Detection
# ---------------------------------------------------------------------------


class TestDetectQuestions:
  """Test detect_questions() with mocked Azure OpenAI client."""

  @patch("meeting_transcriber.prompter._get_client")
  def test_returns_detected_questions(self, mock_get_client):
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    mock_client.chat.completions.create.return_value = _make_chat_response(
      [
        {"question": "What is the timeline?", "keywords": ["timeline", "deadline"]},
        {"question": "How much is the budget?", "keywords": ["budget", "cost"]},
      ]
    )

    result = detect_questions("Client asked about the timeline and budget.")

    assert len(result) == 2
    assert isinstance(result[0], DetectedQuestion)
    assert result[0].question == "What is the timeline?"
    assert "timeline" in result[0].keywords

  @patch("meeting_transcriber.prompter._get_client")
  def test_handles_code_block_response(self, mock_get_client):
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    mock_client.chat.completions.create.return_value = _make_chat_response_with_codeblock(
      [
        {"question": "What about pricing?", "keywords": ["pricing"]},
      ]
    )

    result = detect_questions("They asked about pricing.")

    assert len(result) == 1
    assert result[0].question == "What about pricing?"

  def test_empty_transcript_returns_empty(self):
    result = detect_questions("")
    assert result == []

  def test_whitespace_only_returns_empty(self):
    result = detect_questions("   \n  ")
    assert result == []

  @patch("meeting_transcriber.prompter._get_client")
  def test_api_error_returns_empty(self, mock_get_client):
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    mock_client.chat.completions.create.side_effect = RuntimeError("API down")

    result = detect_questions("Some transcript text.")

    assert result == []

  @patch("meeting_transcriber.prompter._get_client")
  def test_invalid_json_returns_empty(self, mock_get_client):
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    message = MagicMock()
    message.content = "not valid json"
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    mock_client.chat.completions.create.return_value = response

    result = detect_questions("Some transcript text.")

    assert result == []

  @patch("meeting_transcriber.prompter._get_client")
  def test_uses_nano_model(self, mock_get_client):
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    mock_client.chat.completions.create.return_value = _make_chat_response([])

    detect_questions("Hello")

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert "nano" in call_kwargs["model"]


# ---------------------------------------------------------------------------
# Context Matching
# ---------------------------------------------------------------------------


class TestMatchContext:
  """Test match_context() keyword matching logic."""

  def test_matches_keywords_in_chunk_text(self):
    chunks = [
      ContextChunk(text="Our pricing is $500/month", source="a.md", keywords=["pricing"]),
      ContextChunk(text="The timeline is Q2 2026", source="a.md", keywords=["timeline"]),
    ]
    result = match_context(["pricing"], chunks)
    assert len(result) >= 1
    assert result[0].chunk.text == "Our pricing is $500/month"

  def test_sorted_by_score_descending(self):
    chunks = [
      ContextChunk(text="pricing info", source="a.md", keywords=["pricing"]),
      ContextChunk(
        text="pricing and timeline details",
        source="a.md",
        keywords=["pricing", "timeline"],
      ),
    ]
    result = match_context(["pricing", "timeline"], chunks)
    assert len(result) == 2
    assert result[0].score >= result[1].score

  def test_returns_matched_keywords(self):
    chunks = [
      ContextChunk(text="Budget is $1000", source="a.md", keywords=["budget"]),
    ]
    result = match_context(["budget", "timeline"], chunks)
    assert len(result) == 1
    assert "budget" in result[0].matched_keywords

  def test_case_insensitive(self):
    chunks = [
      ContextChunk(text="Enterprise Pricing Plan", source="a.md", keywords=["Pricing"]),
    ]
    result = match_context(["pricing"], chunks)
    assert len(result) == 1

  def test_chinese_keyword_matching(self):
    chunks = [
      ContextChunk(text="基本方案預算：每月五千", source="a.md", keywords=["預算"]),
    ]
    result = match_context(["預算"], chunks)
    assert len(result) == 1
    assert "預算" in result[0].matched_keywords

  def test_chinese_partial_match_in_text(self):
    chunks = [
      ContextChunk(text="預算範圍是五千到一萬", source="a.md", keywords=[]),
    ]
    result = match_context(["預算"], chunks)
    assert len(result) == 1

  def test_empty_keywords_returns_empty(self):
    chunks = [ContextChunk(text="Some text", source="a.md", keywords=["text"])]
    result = match_context([], chunks)
    assert result == []

  def test_empty_chunks_returns_empty(self):
    result = match_context(["keyword"], [])
    assert result == []

  def test_no_match_returns_empty(self):
    chunks = [ContextChunk(text="Unrelated content", source="a.md", keywords=["other"])]
    result = match_context(["pricing"], chunks)
    assert result == []


# ---------------------------------------------------------------------------
# Prompt Card Generation
# ---------------------------------------------------------------------------


class TestGeneratePromptCard:
  """Test generate_prompt_card() formatting."""

  def test_includes_question(self):
    question = DetectedQuestion(question="What is the price?", keywords=["price"])
    card = generate_prompt_card(question, [])
    assert "What is the price?" in card

  def test_includes_context_excerpts(self):
    question = DetectedQuestion(question="What is the price?", keywords=["price"])
    matches = [
      ContextMatch(
        chunk=ContextChunk(text="Price is $500/month", source="pricing.md", keywords=[]),
        score=1.0,
        matched_keywords=["price"],
      ),
    ]
    card = generate_prompt_card(question, matches)
    assert "pricing.md" in card
    assert "$500/month" in card

  def test_limits_to_max_matches(self):
    question = DetectedQuestion(question="Q?", keywords=["k"])
    matches = [
      ContextMatch(
        chunk=ContextChunk(text=f"Match {i}", source=f"f{i}.md", keywords=[]),
        score=1.0 - i * 0.1,
        matched_keywords=["k"],
      )
      for i in range(5)
    ]
    card = generate_prompt_card(question, matches, max_matches=3)
    assert "f0.md" in card
    assert "f2.md" in card
    assert "f3.md" not in card
    assert "f4.md" not in card

  def test_no_matches_shows_fallback(self):
    question = DetectedQuestion(question="Unknown?", keywords=[])
    card = generate_prompt_card(question, [])
    assert "No matching context found" in card

  def test_truncates_long_context(self):
    long_text = "A" * 300
    question = DetectedQuestion(question="Q?", keywords=["k"])
    matches = [
      ContextMatch(
        chunk=ContextChunk(text=long_text, source="a.md", keywords=[]),
        score=1.0,
        matched_keywords=["k"],
      ),
    ]
    card = generate_prompt_card(question, matches)
    assert "..." in card
    assert len(card) < len(long_text)


# ---------------------------------------------------------------------------
# Action Item Detection
# ---------------------------------------------------------------------------


class TestDetectActionItems:
  """Test detect_action_items() with mocked Azure OpenAI client."""

  @patch("meeting_transcriber.prompter._get_client")
  def test_returns_action_items(self, mock_get_client):
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    mock_client.chat.completions.create.return_value = _make_chat_response(
      [
        {"text": "Draft proposal", "owner": "Bob", "deadline": "Friday"},
        {"text": "Review budget", "owner": None, "deadline": None},
      ]
    )

    result = detect_action_items("Bob will draft a proposal by Friday.")

    assert len(result) == 2
    assert isinstance(result[0], ActionItem)
    assert result[0].text == "Draft proposal"
    assert result[0].owner == "Bob"
    assert result[0].deadline == "Friday"
    assert result[1].owner is None

  def test_empty_transcript_returns_empty(self):
    result = detect_action_items("")
    assert result == []

  @patch("meeting_transcriber.prompter._get_client")
  def test_api_error_returns_empty(self, mock_get_client):
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    mock_client.chat.completions.create.side_effect = RuntimeError("API down")

    result = detect_action_items("Some text")

    assert result == []

  @patch("meeting_transcriber.prompter._get_client")
  def test_handles_code_block_response(self, mock_get_client):
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    mock_client.chat.completions.create.return_value = _make_chat_response_with_codeblock(
      [
        {"text": "Send report", "owner": "Alice", "deadline": "Monday"},
      ]
    )

    result = detect_action_items("Alice will send the report by Monday.")

    assert len(result) == 1
    assert result[0].text == "Send report"

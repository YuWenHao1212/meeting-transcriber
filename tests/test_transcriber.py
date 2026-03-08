"""Tests for transcriber and engine architecture."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from meeting_transcriber.engines.base import BaseEngine
from meeting_transcriber.engines.registry import get_engine, list_engines
from meeting_transcriber.models import Segment, TranscriptResult


# --- Engine registry tests ---


class TestGetEngine:
  def test_returns_openai_engine(self):
    engine = get_engine("openai")
    assert engine.name == "openai"

  def test_returns_qwen_engine(self):
    engine = get_engine("qwen")
    assert engine.name == "qwen"

  def test_returns_soniox_engine(self):
    engine = get_engine("soniox")
    assert engine.name == "soniox"

  def test_returns_groq_engine(self):
    engine = get_engine("groq")
    assert engine.name == "groq"

  def test_unknown_engine_raises_value_error(self):
    with pytest.raises(ValueError, match="Unknown engine 'nonexistent'"):
      get_engine("nonexistent")

  def test_error_message_lists_available_engines(self):
    with pytest.raises(ValueError, match="groq"):
      get_engine("bad")


class TestListEngines:
  def test_returns_list(self):
    engines = list_engines()
    assert isinstance(engines, list)
    assert len(engines) >= 4

  def test_each_entry_has_required_keys(self):
    for entry in list_engines():
      assert "name" in entry
      assert "cost_per_minute" in entry



# --- OpenAI engine tests ---


def _make_mock_response(
  text="Hello world 你好世界",
  duration=30.0,
  segments=None,
):
  response = MagicMock()
  response.text = text
  response.duration = duration
  if segments is None:
    segments = [
      MagicMock(start=0.0, end=5.0, text="Hello world"),
      MagicMock(start=5.0, end=10.0, text="你好世界"),
    ]
  response.segments = segments
  return response


def _make_openai_engine():
  """Create OpenAI engine with mocked client (no real API calls)."""
  engine = get_engine("openai")
  mock_client = MagicMock()
  engine._client = mock_client
  return engine, mock_client


class TestOpenAIEngine:
  def test_calls_openai_api(self, tmp_path):
    engine, mock_client = _make_openai_engine()
    mock_client.audio.transcriptions.create.return_value = _make_mock_response()

    # Create a dummy file so open() works
    dummy = tmp_path / "test.wav"
    dummy.write_bytes(b"fake audio")

    engine.transcribe_file(dummy, language="zh")

    mock_client.audio.transcriptions.create.assert_called_once()
    call_kwargs = mock_client.audio.transcriptions.create.call_args[1]
    assert call_kwargs["model"] == "gpt-4o-transcribe"
    assert call_kwargs["language"] == "zh"
    assert call_kwargs["response_format"] == "verbose_json"

  def test_parses_segments(self, tmp_path):
    engine, mock_client = _make_openai_engine()
    mock_client.audio.transcriptions.create.return_value = _make_mock_response()

    dummy = tmp_path / "test.wav"
    dummy.write_bytes(b"fake audio")
    result = engine.transcribe_file(dummy)

    assert len(result.segments) == 2
    assert result.segments[0].text == "Hello world"
    assert result.segments[1].text == "你好世界"
    assert result.segments[0].start == 0.0
    assert result.segments[1].end == 10.0

  def test_calculates_cost(self, tmp_path):
    engine, mock_client = _make_openai_engine()
    mock_client.audio.transcriptions.create.return_value = _make_mock_response()

    dummy = tmp_path / "test.wav"
    dummy.write_bytes(b"fake audio")
    result = engine.transcribe_file(dummy)

    # 30 seconds = 0.5 minutes * $0.006/min = $0.003
    assert result.cost == pytest.approx(0.003, abs=0.001)

  def test_returns_transcript_result(self, tmp_path):
    engine, mock_client = _make_openai_engine()
    mock_client.audio.transcriptions.create.return_value = _make_mock_response()

    dummy = tmp_path / "test.wav"
    dummy.write_bytes(b"fake audio")
    result = engine.transcribe_file(dummy)

    assert isinstance(result, TranscriptResult)
    assert result.full_text == "Hello world 你好世界"
    assert result.duration == 30.0
    assert result.engine == "openai"

  @patch("meeting_transcriber.engines.openai_engine.time.sleep")
  def test_retry_on_failure(self, mock_sleep, tmp_path):
    engine, mock_client = _make_openai_engine()
    mock_client.audio.transcriptions.create.side_effect = [
      Exception("API error"),
      _make_mock_response(),
    ]

    dummy = tmp_path / "test.wav"
    dummy.write_bytes(b"fake audio")
    result = engine.transcribe_file(dummy)

    assert result.full_text == "Hello world 你好世界"
    assert mock_client.audio.transcriptions.create.call_count == 2

  @patch("meeting_transcriber.engines.openai_engine.time.sleep")
  def test_raises_after_max_retries(self, mock_sleep, tmp_path):
    engine, mock_client = _make_openai_engine()
    mock_client.audio.transcriptions.create.side_effect = Exception("API error")

    dummy = tmp_path / "test.wav"
    dummy.write_bytes(b"fake audio")
    with pytest.raises(RuntimeError, match="failed after 3 attempts"):
      engine.transcribe_file(dummy)


# --- BaseEngine.transcribe_chunks tests ---


class TestTranscribeChunks:
  def test_merges_multiple_chunks(self, tmp_path):
    engine, mock_client = _make_openai_engine()

    response1 = _make_mock_response(
      text="First chunk", duration=30.0,
      segments=[MagicMock(start=0.0, end=30.0, text="First chunk")],
    )
    response2 = _make_mock_response(
      text="Second chunk", duration=30.0,
      segments=[MagicMock(start=0.0, end=30.0, text="Second chunk")],
    )
    mock_client.audio.transcriptions.create.side_effect = [response1, response2]

    c1 = tmp_path / "chunk1.wav"
    c2 = tmp_path / "chunk2.wav"
    c1.write_bytes(b"fake")
    c2.write_bytes(b"fake")

    result = engine.transcribe_chunks([c1, c2], language="zh")

    assert len(result.segments) == 2
    assert result.segments[0].start == 0.0
    assert result.segments[1].start == 30.0  # offset by first chunk duration
    assert result.duration == 60.0
    assert "First chunk" in result.full_text
    assert "Second chunk" in result.full_text

  def test_accumulates_cost(self, tmp_path):
    engine, mock_client = _make_openai_engine()

    response = _make_mock_response(text="chunk", duration=60.0, segments=[])
    mock_client.audio.transcriptions.create.return_value = response

    c1 = tmp_path / "c1.wav"
    c2 = tmp_path / "c2.wav"
    c1.write_bytes(b"fake")
    c2.write_bytes(b"fake")

    result = engine.transcribe_chunks([c1, c2])

    # 2 chunks * 60s each = 120s = 2 min * $0.006 = $0.012
    assert result.cost == pytest.approx(0.012, abs=0.001)


# --- High-level transcribe function tests ---


class TestTranscribeFunction:
  @patch("meeting_transcriber.transcriber.get_engine")
  @patch("meeting_transcriber.transcriber.chunk_audio")
  def test_transcribe_uses_engine_and_chunker(self, mock_chunk, mock_get_engine):
    from meeting_transcriber.transcriber import transcribe

    mock_engine = MagicMock()
    mock_engine.transcribe_chunks.return_value = TranscriptResult(
      full_text="test", engine="openai"
    )
    mock_get_engine.return_value = mock_engine
    mock_chunk.return_value = [Path("c1.wav")]

    result = transcribe(Path("test.wav"), engine_name="openai")

    mock_get_engine.assert_called_once_with("openai")
    mock_chunk.assert_called_once()
    mock_engine.transcribe_chunks.assert_called_once()
    assert result.full_text == "test"

  @patch("meeting_transcriber.transcriber.get_engine")
  def test_transcribe_file_no_chunking(self, mock_get_engine):
    from meeting_transcriber.transcriber import transcribe_file

    mock_engine = MagicMock()
    mock_engine.transcribe_file.return_value = TranscriptResult(
      full_text="direct", engine="openai"
    )
    mock_get_engine.return_value = mock_engine

    result = transcribe_file(Path("test.wav"), engine_name="openai")

    mock_engine.transcribe_file.assert_called_once()
    assert result.full_text == "direct"

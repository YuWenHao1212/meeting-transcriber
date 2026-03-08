"""Tests for Groq STT engine."""

from unittest.mock import MagicMock, patch

import pytest

from meeting_transcriber.engines.groq import GroqEngine
from meeting_transcriber.models import TranscriptResult


def _make_mock_response(
  text="Hello world",
  duration=60.0,
  segments=None,
):
  response = MagicMock()
  response.text = text
  response.duration = duration
  if segments is None:
    segments = [
      MagicMock(start=0.0, end=30.0, text="Hello world"),
      MagicMock(start=30.0, end=60.0, text="How are you"),
    ]
  response.segments = segments
  return response


def _make_groq_engine():
  """Create Groq engine with mocked client (no real API calls)."""
  engine = GroqEngine()
  mock_client = MagicMock()
  engine._client = mock_client
  return engine, mock_client


class TestGroqEngineMetadata:
  def test_name(self):
    engine = GroqEngine()
    assert engine.name == "groq"

  def test_cost_per_minute(self):
    engine = GroqEngine()
    assert engine.cost_per_minute == pytest.approx(0.0007)

  def test_model_default(self):
    engine = GroqEngine()
    assert engine.model == "whisper-large-v3-turbo"


class TestGroqEngineTranscription:
  def test_successful_transcription(self, tmp_path):
    engine, mock_client = _make_groq_engine()
    mock_client.audio.transcriptions.create.return_value = _make_mock_response()

    dummy = tmp_path / "test.wav"
    dummy.write_bytes(b"fake audio")
    result = engine.transcribe_file(dummy, language="en")

    assert isinstance(result, TranscriptResult)
    assert result.full_text == "Hello world"
    assert result.duration == 60.0
    assert result.engine == "groq"

  def test_parses_segments(self, tmp_path):
    engine, mock_client = _make_groq_engine()
    mock_client.audio.transcriptions.create.return_value = _make_mock_response()

    dummy = tmp_path / "test.wav"
    dummy.write_bytes(b"fake audio")
    result = engine.transcribe_file(dummy)

    assert len(result.segments) == 2
    assert result.segments[0].text == "Hello world"
    assert result.segments[0].start == 0.0
    assert result.segments[1].text == "How are you"
    assert result.segments[1].end == 60.0

  def test_cost_calculation(self, tmp_path):
    engine, mock_client = _make_groq_engine()
    mock_client.audio.transcriptions.create.return_value = _make_mock_response(
      duration=120.0,
    )

    dummy = tmp_path / "test.wav"
    dummy.write_bytes(b"fake audio")
    result = engine.transcribe_file(dummy)

    # 120s = 2 min * $0.0007/min = $0.0014
    assert result.cost == pytest.approx(0.0014, abs=0.0001)

  def test_language_parameter_passed(self, tmp_path):
    engine, mock_client = _make_groq_engine()
    mock_client.audio.transcriptions.create.return_value = _make_mock_response()

    dummy = tmp_path / "test.wav"
    dummy.write_bytes(b"fake audio")
    engine.transcribe_file(dummy, language="en")

    call_kwargs = mock_client.audio.transcriptions.create.call_args[1]
    assert call_kwargs["language"] == "en"
    assert call_kwargs["model"] == "whisper-large-v3-turbo"
    assert call_kwargs["response_format"] == "verbose_json"


class TestGroqEngineRetry:
  @patch("meeting_transcriber.engines.groq.time.sleep")
  def test_retry_on_failure(self, mock_sleep, tmp_path):
    engine, mock_client = _make_groq_engine()
    mock_client.audio.transcriptions.create.side_effect = [
      Exception("API rate limit"),
      _make_mock_response(),
    ]

    dummy = tmp_path / "test.wav"
    dummy.write_bytes(b"fake audio")
    result = engine.transcribe_file(dummy)

    assert result.full_text == "Hello world"
    assert mock_client.audio.transcriptions.create.call_count == 2
    mock_sleep.assert_called_once()

  @patch("meeting_transcriber.engines.groq.time.sleep")
  def test_raises_after_max_retries(self, mock_sleep, tmp_path):
    engine, mock_client = _make_groq_engine()
    mock_client.audio.transcriptions.create.side_effect = Exception("API error")

    dummy = tmp_path / "test.wav"
    dummy.write_bytes(b"fake audio")
    with pytest.raises(RuntimeError, match="failed after 3 attempts"):
      engine.transcribe_file(dummy)

    assert mock_client.audio.transcriptions.create.call_count == 3

  @patch("meeting_transcriber.engines.groq.time.sleep")
  def test_backoff_timing(self, mock_sleep, tmp_path):
    engine, mock_client = _make_groq_engine()
    mock_client.audio.transcriptions.create.side_effect = [
      Exception("Error 1"),
      Exception("Error 2"),
      _make_mock_response(),
    ]

    dummy = tmp_path / "test.wav"
    dummy.write_bytes(b"fake audio")
    engine.transcribe_file(dummy)

    # Backoff: 2^0=1s, 2^1=2s
    assert mock_sleep.call_count == 2
    mock_sleep.assert_any_call(1.0)
    mock_sleep.assert_any_call(2.0)


class TestGroqEngineClientInit:
  def test_missing_api_key_raises(self):
    engine = GroqEngine()
    with (
      patch.dict("os.environ", {}, clear=True),
      patch("meeting_transcriber.engines.groq.openai.OpenAI", side_effect=Exception("No API key")),
    ):
      # Access client property without GROQ_API_KEY set
      with pytest.raises(Exception):
        _ = engine.client

  @patch("meeting_transcriber.engines.groq.openai.OpenAI")
  def test_lazy_client_init(self, mock_openai_cls):
    engine = GroqEngine()
    assert engine._client is None

    mock_openai_cls.return_value = MagicMock()
    with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
      client = engine.client

    mock_openai_cls.assert_called_once_with(
      api_key="test-key",
      base_url="https://api.groq.com/openai/v1",
    )
    assert client is not None

  def test_client_reuses_instance(self):
    engine = GroqEngine()
    mock_client = MagicMock()
    engine._client = mock_client

    assert engine.client is mock_client

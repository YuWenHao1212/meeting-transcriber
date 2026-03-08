"""Tests for Qwen ASR engine (DashScope OpenAI-compatible API)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from meeting_transcriber.engines.qwen import QwenEngine
from meeting_transcriber.models import Segment, TranscriptResult


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


def _make_qwen_engine():
  """Create Qwen engine with mocked client (no real API calls)."""
  engine = QwenEngine()
  mock_client = MagicMock()
  engine._client = mock_client
  return engine, mock_client


class TestQwenEngineAttributes:
  def test_name(self):
    engine = QwenEngine()
    assert engine.name == "qwen"

  def test_cost_per_minute(self):
    engine = QwenEngine()
    assert engine.cost_per_minute == pytest.approx(0.0067, abs=0.0001)

  def test_model_default(self):
    engine = QwenEngine()
    assert engine.model == "qwen2-audio-asr"


class TestQwenEngineTranscription:
  def test_calls_dashscope_api(self, tmp_path):
    engine, mock_client = _make_qwen_engine()
    mock_client.audio.transcriptions.create.return_value = _make_mock_response()

    dummy = tmp_path / "test.wav"
    dummy.write_bytes(b"fake audio")

    engine.transcribe_file(dummy, language="zh")

    mock_client.audio.transcriptions.create.assert_called_once()
    call_kwargs = mock_client.audio.transcriptions.create.call_args[1]
    assert call_kwargs["model"] == "qwen2-audio-asr"
    assert call_kwargs["language"] == "zh"
    assert call_kwargs["response_format"] == "verbose_json"

  def test_parses_segments(self, tmp_path):
    engine, mock_client = _make_qwen_engine()
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
    engine, mock_client = _make_qwen_engine()
    mock_client.audio.transcriptions.create.return_value = _make_mock_response()

    dummy = tmp_path / "test.wav"
    dummy.write_bytes(b"fake audio")
    result = engine.transcribe_file(dummy)

    # 30 seconds = 0.5 minutes * $0.0067/min = $0.00335
    assert result.cost == pytest.approx(0.00335, abs=0.001)

  def test_returns_transcript_result(self, tmp_path):
    engine, mock_client = _make_qwen_engine()
    mock_client.audio.transcriptions.create.return_value = _make_mock_response()

    dummy = tmp_path / "test.wav"
    dummy.write_bytes(b"fake audio")
    result = engine.transcribe_file(dummy)

    assert isinstance(result, TranscriptResult)
    assert result.full_text == "Hello world 你好世界"
    assert result.duration == 30.0
    assert result.engine == "qwen"

  def test_passes_language_parameter(self, tmp_path):
    engine, mock_client = _make_qwen_engine()
    mock_client.audio.transcriptions.create.return_value = _make_mock_response()

    dummy = tmp_path / "test.wav"
    dummy.write_bytes(b"fake audio")
    engine.transcribe_file(dummy, language="en")

    call_kwargs = mock_client.audio.transcriptions.create.call_args[1]
    assert call_kwargs["language"] == "en"


class TestQwenEngineRetry:
  @patch("meeting_transcriber.engines.qwen.time.sleep")
  def test_retry_on_failure(self, mock_sleep, tmp_path):
    engine, mock_client = _make_qwen_engine()
    mock_client.audio.transcriptions.create.side_effect = [
      Exception("API error"),
      _make_mock_response(),
    ]

    dummy = tmp_path / "test.wav"
    dummy.write_bytes(b"fake audio")
    result = engine.transcribe_file(dummy)

    assert result.full_text == "Hello world 你好世界"
    assert mock_client.audio.transcriptions.create.call_count == 2
    mock_sleep.assert_called_once()

  @patch("meeting_transcriber.engines.qwen.time.sleep")
  def test_raises_after_max_retries(self, mock_sleep, tmp_path):
    engine, mock_client = _make_qwen_engine()
    mock_client.audio.transcriptions.create.side_effect = Exception("API error")

    dummy = tmp_path / "test.wav"
    dummy.write_bytes(b"fake audio")
    with pytest.raises(RuntimeError, match="failed after 3 attempts"):
      engine.transcribe_file(dummy)

    assert mock_client.audio.transcriptions.create.call_count == 3

  @patch("meeting_transcriber.engines.qwen.time.sleep")
  def test_exponential_backoff_timing(self, mock_sleep, tmp_path):
    engine, mock_client = _make_qwen_engine()
    mock_client.audio.transcriptions.create.side_effect = [
      Exception("err1"),
      Exception("err2"),
      _make_mock_response(),
    ]

    dummy = tmp_path / "test.wav"
    dummy.write_bytes(b"fake audio")
    engine.transcribe_file(dummy)

    # backoff: 2^0=1s, 2^1=2s
    assert mock_sleep.call_count == 2
    mock_sleep.assert_any_call(1.0)
    mock_sleep.assert_any_call(2.0)


class TestQwenEngineMissingApiKey:
  def test_missing_api_key_raises_error(self):
    """Client init should fail when DASHSCOPE_API_KEY is not set."""
    engine = QwenEngine()
    engine._client = None  # ensure no injected client

    with patch.dict("os.environ", {}, clear=True):
      with patch("meeting_transcriber.engines.qwen.openai.OpenAI") as mock_openai:
        mock_openai.side_effect = Exception("API key not found")
        with pytest.raises(Exception, match="API key"):
          _ = engine.client

"""Tests for Qwen3-ASR engine (DashScope International)."""

from unittest.mock import MagicMock, patch

import pytest

from meeting_transcriber.engines.qwen import QwenEngine
from meeting_transcriber.models import TranscriptResult


def _make_mock_response(text="Hello world 你好世界"):
  """Create a mock chat completion response."""
  message = MagicMock()
  message.content = text

  choice = MagicMock()
  choice.message = message
  choice.finish_reason = "stop"

  response = MagicMock()
  response.choices = [choice]
  return response


def _make_qwen_engine():
  """Create Qwen engine with mocked client."""
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
    assert engine.cost_per_minute == pytest.approx(0.0054, abs=0.001)

  def test_model_default(self):
    engine = QwenEngine()
    assert engine.model == "qwen3-asr-flash"


class TestQwenEngineTranscription:
  @patch("meeting_transcriber.engines.qwen.sf")
  def test_calls_chat_completions(self, mock_sf, tmp_path):
    engine, mock_client = _make_qwen_engine()
    mock_client.chat.completions.create.return_value = _make_mock_response()
    mock_sf.info.return_value = MagicMock(duration=30.0)

    dummy = tmp_path / "test.wav"
    dummy.write_bytes(b"fake audio")
    engine.transcribe_file(dummy, language="zh")

    mock_client.chat.completions.create.assert_called_once()
    call_kwargs = mock_client.chat.completions.create.call_args[1]
    assert call_kwargs["model"] == "qwen3-asr-flash"
    assert call_kwargs["stream"] is False

  @patch("meeting_transcriber.engines.qwen.sf")
  def test_sends_base64_audio(self, mock_sf, tmp_path):
    engine, mock_client = _make_qwen_engine()
    mock_client.chat.completions.create.return_value = _make_mock_response()
    mock_sf.info.return_value = MagicMock(duration=10.0)

    dummy = tmp_path / "test.wav"
    dummy.write_bytes(b"fake audio data")
    engine.transcribe_file(dummy)

    call_kwargs = mock_client.chat.completions.create.call_args[1]
    messages = call_kwargs["messages"]
    content = messages[0]["content"][0]
    assert content["type"] == "input_audio"
    assert content["input_audio"]["data"].startswith("data:audio/wav;base64,")

  @patch("meeting_transcriber.engines.qwen.sf")
  def test_passes_language_in_asr_options(self, mock_sf, tmp_path):
    engine, mock_client = _make_qwen_engine()
    mock_client.chat.completions.create.return_value = _make_mock_response()
    mock_sf.info.return_value = MagicMock(duration=10.0)

    dummy = tmp_path / "test.wav"
    dummy.write_bytes(b"fake audio")
    engine.transcribe_file(dummy, language="en")

    call_kwargs = mock_client.chat.completions.create.call_args[1]
    asr_options = call_kwargs["extra_body"]["asr_options"]
    assert asr_options["language"] == "en"

  @patch("meeting_transcriber.engines.qwen.sf")
  def test_returns_transcript_result(self, mock_sf, tmp_path):
    engine, mock_client = _make_qwen_engine()
    mock_client.chat.completions.create.return_value = _make_mock_response()
    mock_sf.info.return_value = MagicMock(duration=30.0)

    dummy = tmp_path / "test.wav"
    dummy.write_bytes(b"fake audio")
    result = engine.transcribe_file(dummy)

    assert isinstance(result, TranscriptResult)
    assert result.full_text == "Hello world 你好世界"
    assert result.duration == 30.0
    assert result.engine == "qwen"

  @patch("meeting_transcriber.engines.qwen.sf")
  def test_calculates_cost(self, mock_sf, tmp_path):
    engine, mock_client = _make_qwen_engine()
    mock_client.chat.completions.create.return_value = _make_mock_response()
    mock_sf.info.return_value = MagicMock(duration=60.0)

    dummy = tmp_path / "test.wav"
    dummy.write_bytes(b"fake audio")
    result = engine.transcribe_file(dummy)

    # 60s = 1 min * $0.0054/min
    assert result.cost == pytest.approx(0.0054, abs=0.001)

  @patch("meeting_transcriber.engines.qwen.sf")
  def test_single_segment_for_full_text(self, mock_sf, tmp_path):
    engine, mock_client = _make_qwen_engine()
    mock_client.chat.completions.create.return_value = _make_mock_response("Some transcribed text")
    mock_sf.info.return_value = MagicMock(duration=15.0)

    dummy = tmp_path / "test.wav"
    dummy.write_bytes(b"fake audio")
    result = engine.transcribe_file(dummy)

    assert len(result.segments) == 1
    assert result.segments[0].start == 0.0
    assert result.segments[0].end == 15.0
    assert result.segments[0].text == "Some transcribed text"

  @patch("meeting_transcriber.engines.qwen.sf")
  def test_empty_response(self, mock_sf, tmp_path):
    engine, mock_client = _make_qwen_engine()
    mock_client.chat.completions.create.return_value = _make_mock_response("")
    mock_sf.info.return_value = MagicMock(duration=5.0)

    dummy = tmp_path / "test.wav"
    dummy.write_bytes(b"fake audio")
    result = engine.transcribe_file(dummy)

    assert result.full_text == ""
    assert len(result.segments) == 0


class TestQwenEngineRetry:
  @patch("meeting_transcriber.engines.qwen.sf")
  @patch("meeting_transcriber.engines.qwen.time.sleep")
  def test_retry_on_failure(self, mock_sleep, mock_sf, tmp_path):
    engine, mock_client = _make_qwen_engine()
    mock_client.chat.completions.create.side_effect = [
      Exception("API error"),
      _make_mock_response(),
    ]
    mock_sf.info.return_value = MagicMock(duration=10.0)

    dummy = tmp_path / "test.wav"
    dummy.write_bytes(b"fake audio")
    result = engine.transcribe_file(dummy)

    assert result.full_text == "Hello world 你好世界"
    assert mock_client.chat.completions.create.call_count == 2
    mock_sleep.assert_called_once()

  @patch("meeting_transcriber.engines.qwen.sf")
  @patch("meeting_transcriber.engines.qwen.time.sleep")
  def test_raises_after_max_retries(self, mock_sleep, mock_sf, tmp_path):
    engine, mock_client = _make_qwen_engine()
    mock_client.chat.completions.create.side_effect = Exception("API error")
    mock_sf.info.return_value = MagicMock(duration=10.0)

    dummy = tmp_path / "test.wav"
    dummy.write_bytes(b"fake audio")
    with pytest.raises(RuntimeError, match="failed after 3 attempts"):
      engine.transcribe_file(dummy)

    assert mock_client.chat.completions.create.call_count == 3

  @patch("meeting_transcriber.engines.qwen.sf")
  @patch("meeting_transcriber.engines.qwen.time.sleep")
  def test_exponential_backoff(self, mock_sleep, mock_sf, tmp_path):
    engine, mock_client = _make_qwen_engine()
    mock_client.chat.completions.create.side_effect = [
      Exception("err1"),
      Exception("err2"),
      _make_mock_response(),
    ]
    mock_sf.info.return_value = MagicMock(duration=10.0)

    dummy = tmp_path / "test.wav"
    dummy.write_bytes(b"fake audio")
    engine.transcribe_file(dummy)

    assert mock_sleep.call_count == 2
    mock_sleep.assert_any_call(1.0)
    mock_sleep.assert_any_call(2.0)


class TestQwenEngineClientInit:
  def test_missing_api_key_raises(self):
    engine = QwenEngine()
    engine._client = None
    with patch.dict("os.environ", {}, clear=True):
      with pytest.raises(RuntimeError, match="DASHSCOPE_API_KEY"):
        _ = engine.client

  @patch("meeting_transcriber.engines.qwen.openai.OpenAI")
  def test_uses_intl_base_url(self, mock_openai_cls):
    engine = QwenEngine()
    mock_openai_cls.return_value = MagicMock()
    with patch.dict("os.environ", {"DASHSCOPE_API_KEY": "test-key"}):
      engine.client

    mock_openai_cls.assert_called_once_with(
      api_key="test-key",
      base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    )

  def test_mime_type_detection(self):
    engine = QwenEngine()
    # Test via _encode_audio with different extensions
    import tempfile

    for ext, expected_mime in [
      (".wav", "audio/wav"),
      (".mp3", "audio/mp3"),
      (".flac", "audio/flac"),
    ]:
      with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
        from pathlib import Path

        p = Path(f.name)
        p.write_bytes(b"fake")
        result = engine._encode_audio(p)
        assert f"data:{expected_mime};base64," in result
        p.unlink()

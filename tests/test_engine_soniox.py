"""Tests for Soniox STT engine."""

import json
from unittest.mock import MagicMock, patch

import pytest

from meeting_transcriber.engines.soniox import SonioxEngine
from meeting_transcriber.models import TranscriptResult


def _make_soniox_response(
  words=None,
  duration_ms=30000,
):
  """Build a fake Soniox API JSON response."""
  if words is None:
    words = [
      {"text": "Hello", "start_ms": 0, "end_ms": 500},
      {"text": " ", "start_ms": 500, "end_ms": 500},
      {"text": "world.", "start_ms": 510, "end_ms": 1200},
      {"text": " ", "start_ms": 1200, "end_ms": 1200},
      {"text": "你好", "start_ms": 2000, "end_ms": 2800},
      {"text": "世界。", "start_ms": 2800, "end_ms": 3500},
    ]
  return json.dumps(
    {
      "words": words,
      "duration_ms": duration_ms,
    }
  ).encode("utf-8")


def _mock_urlopen(response_bytes, status=200):
  """Create a mock for urllib.request.urlopen that returns response_bytes."""
  mock_response = MagicMock()
  mock_response.read.return_value = response_bytes
  mock_response.status = status
  mock_response.__enter__ = MagicMock(return_value=mock_response)
  mock_response.__exit__ = MagicMock(return_value=False)
  return mock_response


class TestSonioxEngineProperties:
  def test_name(self):
    engine = SonioxEngine()
    assert engine.name == "soniox"

  def test_cost_per_minute(self):
    engine = SonioxEngine()
    assert engine.cost_per_minute == pytest.approx(0.002)


class TestSonioxMissingApiKey:
  def test_raises_without_api_key(self, tmp_path):
    engine = SonioxEngine()
    dummy = tmp_path / "test.wav"
    dummy.write_bytes(b"fake audio")

    with patch.dict("os.environ", {}, clear=True):
      with pytest.raises(RuntimeError, match="SONIOX_API_KEY"):
        engine.transcribe_file(dummy)


class TestSonioxTranscription:
  @patch("meeting_transcriber.engines.soniox.time.sleep")
  @patch("meeting_transcriber.engines.soniox.urllib.request.urlopen")
  def test_successful_transcription(self, mock_urlopen_fn, mock_sleep, tmp_path):
    mock_urlopen_fn.return_value = _mock_urlopen(_make_soniox_response())

    engine = SonioxEngine()
    dummy = tmp_path / "test.wav"
    dummy.write_bytes(b"fake audio")

    with patch.dict("os.environ", {"SONIOX_API_KEY": "test-key"}):
      result = engine.transcribe_file(dummy)

    assert isinstance(result, TranscriptResult)
    assert result.engine == "soniox"
    assert result.duration == pytest.approx(30.0)
    assert len(result.segments) >= 1
    # Full text should contain all words
    assert "Hello" in result.full_text
    assert "world" in result.full_text
    assert "你好" in result.full_text
    assert "世界" in result.full_text

  @patch("meeting_transcriber.engines.soniox.time.sleep")
  @patch("meeting_transcriber.engines.soniox.urllib.request.urlopen")
  def test_segments_have_timestamps(self, mock_urlopen_fn, mock_sleep, tmp_path):
    mock_urlopen_fn.return_value = _mock_urlopen(_make_soniox_response())

    engine = SonioxEngine()
    dummy = tmp_path / "test.wav"
    dummy.write_bytes(b"fake audio")

    with patch.dict("os.environ", {"SONIOX_API_KEY": "test-key"}):
      result = engine.transcribe_file(dummy)

    for seg in result.segments:
      assert seg.start >= 0.0
      assert seg.end > seg.start
      assert len(seg.text.strip()) > 0

  @patch("meeting_transcriber.engines.soniox.time.sleep")
  @patch("meeting_transcriber.engines.soniox.urllib.request.urlopen")
  def test_cost_calculation(self, mock_urlopen_fn, mock_sleep, tmp_path):
    # 1 hour = 60 minutes, cost = 60 * 0.002 = $0.12
    response = _make_soniox_response(duration_ms=3600000)
    mock_urlopen_fn.return_value = _mock_urlopen(response)

    engine = SonioxEngine()
    dummy = tmp_path / "test.wav"
    dummy.write_bytes(b"fake audio")

    with patch.dict("os.environ", {"SONIOX_API_KEY": "test-key"}):
      result = engine.transcribe_file(dummy)

    assert result.cost == pytest.approx(0.12, abs=0.001)

  @patch("meeting_transcriber.engines.soniox.time.sleep")
  @patch("meeting_transcriber.engines.soniox.urllib.request.urlopen")
  def test_cost_30_seconds(self, mock_urlopen_fn, mock_sleep, tmp_path):
    # 30s = 0.5 min, cost = 0.5 * 0.002 = $0.001
    response = _make_soniox_response(duration_ms=30000)
    mock_urlopen_fn.return_value = _mock_urlopen(response)

    engine = SonioxEngine()
    dummy = tmp_path / "test.wav"
    dummy.write_bytes(b"fake audio")

    with patch.dict("os.environ", {"SONIOX_API_KEY": "test-key"}):
      result = engine.transcribe_file(dummy)

    assert result.cost == pytest.approx(0.001, abs=0.0001)


class TestSonioxRetry:
  @patch("meeting_transcriber.engines.soniox.time.sleep")
  @patch("meeting_transcriber.engines.soniox.urllib.request.urlopen")
  def test_retry_on_failure_then_success(self, mock_urlopen_fn, mock_sleep, tmp_path):
    from urllib.error import URLError

    mock_urlopen_fn.side_effect = [
      URLError("Connection refused"),
      _mock_urlopen(_make_soniox_response()),
    ]

    engine = SonioxEngine()
    dummy = tmp_path / "test.wav"
    dummy.write_bytes(b"fake audio")

    with patch.dict("os.environ", {"SONIOX_API_KEY": "test-key"}):
      result = engine.transcribe_file(dummy)

    assert result.full_text != ""
    assert mock_urlopen_fn.call_count == 2
    mock_sleep.assert_called_once()

  @patch("meeting_transcriber.engines.soniox.time.sleep")
  @patch("meeting_transcriber.engines.soniox.urllib.request.urlopen")
  def test_raises_after_max_retries(self, mock_urlopen_fn, mock_sleep, tmp_path):
    from urllib.error import URLError

    mock_urlopen_fn.side_effect = URLError("Connection refused")

    engine = SonioxEngine()
    dummy = tmp_path / "test.wav"
    dummy.write_bytes(b"fake audio")

    with patch.dict("os.environ", {"SONIOX_API_KEY": "test-key"}):
      with pytest.raises(RuntimeError, match="failed after 3 attempts"):
        engine.transcribe_file(dummy)

    assert mock_urlopen_fn.call_count == 3

  @patch("meeting_transcriber.engines.soniox.time.sleep")
  @patch("meeting_transcriber.engines.soniox.urllib.request.urlopen")
  def test_exponential_backoff_timing(self, mock_urlopen_fn, mock_sleep, tmp_path):
    from urllib.error import URLError

    mock_urlopen_fn.side_effect = URLError("fail")

    engine = SonioxEngine()
    dummy = tmp_path / "test.wav"
    dummy.write_bytes(b"fake audio")

    with patch.dict("os.environ", {"SONIOX_API_KEY": "test-key"}):
      with pytest.raises(RuntimeError):
        engine.transcribe_file(dummy)

    # backoff: 2^0=1.0, 2^1=2.0 (sleeps before attempt 2 and 3)
    assert mock_sleep.call_count == 2
    mock_sleep.assert_any_call(1.0)
    mock_sleep.assert_any_call(2.0)


class TestSonioxLanguageHint:
  @patch("meeting_transcriber.engines.soniox.time.sleep")
  @patch("meeting_transcriber.engines.soniox.urllib.request.urlopen")
  def test_language_param_is_accepted(self, mock_urlopen_fn, mock_sleep, tmp_path):
    """Soniox auto-detects language; the language param is just a hint."""
    mock_urlopen_fn.return_value = _mock_urlopen(_make_soniox_response())

    engine = SonioxEngine()
    dummy = tmp_path / "test.wav"
    dummy.write_bytes(b"fake audio")

    with patch.dict("os.environ", {"SONIOX_API_KEY": "test-key"}):
      result = engine.transcribe_file(dummy, language="en")

    assert isinstance(result, TranscriptResult)


class TestSonioxWordGrouping:
  @patch("meeting_transcriber.engines.soniox.time.sleep")
  @patch("meeting_transcriber.engines.soniox.urllib.request.urlopen")
  def test_groups_words_by_pause(self, mock_urlopen_fn, mock_sleep, tmp_path):
    """Words separated by a long pause should be in different segments."""
    words = [
      {"text": "First", "start_ms": 0, "end_ms": 500},
      {"text": " ", "start_ms": 500, "end_ms": 500},
      {"text": "sentence.", "start_ms": 510, "end_ms": 1200},
      # Big pause (> 1000ms)
      {"text": "Second", "start_ms": 3000, "end_ms": 3500},
      {"text": " ", "start_ms": 3500, "end_ms": 3500},
      {"text": "sentence.", "start_ms": 3510, "end_ms": 4200},
    ]
    response = _make_soniox_response(words=words, duration_ms=5000)
    mock_urlopen_fn.return_value = _mock_urlopen(response)

    engine = SonioxEngine()
    dummy = tmp_path / "test.wav"
    dummy.write_bytes(b"fake audio")

    with patch.dict("os.environ", {"SONIOX_API_KEY": "test-key"}):
      result = engine.transcribe_file(dummy)

    assert len(result.segments) == 2
    assert "First" in result.segments[0].text
    assert "Second" in result.segments[1].text

  @patch("meeting_transcriber.engines.soniox.time.sleep")
  @patch("meeting_transcriber.engines.soniox.urllib.request.urlopen")
  def test_empty_words_returns_empty_result(self, mock_urlopen_fn, mock_sleep, tmp_path):
    response = _make_soniox_response(words=[], duration_ms=5000)
    mock_urlopen_fn.return_value = _mock_urlopen(response)

    engine = SonioxEngine()
    dummy = tmp_path / "test.wav"
    dummy.write_bytes(b"fake audio")

    with patch.dict("os.environ", {"SONIOX_API_KEY": "test-key"}):
      result = engine.transcribe_file(dummy)

    assert len(result.segments) == 0
    assert result.full_text == ""
    assert result.duration == pytest.approx(5.0)

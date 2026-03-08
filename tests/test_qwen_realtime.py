"""Tests for Qwen3-ASR real-time WebSocket streaming engine."""

import base64
import json
from unittest.mock import MagicMock, patch

import pytest

from meeting_transcriber.engines.qwen_realtime import (
  QwenRealtimeStreamer,
  _build_audio_append,
  _build_session_finish,
  _build_session_update,
  _to_traditional,
)


class TestMessageFormatting:
  """Test WebSocket message builders."""

  def test_session_update_has_correct_type(self):
    msg = _build_session_update("zh")
    assert msg["type"] == "session.update"

  def test_session_update_includes_language(self):
    msg = _build_session_update("en")
    assert msg["session"]["asr_options"]["language"] == "en"

  def test_session_update_includes_pcm16_format(self):
    msg = _build_session_update("zh")
    assert msg["session"]["input_audio_format"] == "pcm16"

  def test_session_update_includes_sample_rate(self):
    msg = _build_session_update("zh")
    assert msg["session"]["sample_rate"] == 16000

  def test_session_update_enables_itn(self):
    msg = _build_session_update("zh")
    assert msg["session"]["asr_options"]["enable_itn"] is True

  def test_session_update_includes_vad(self):
    msg = _build_session_update("zh")
    assert msg["session"]["turn_detection"]["type"] == "server_vad"

  def test_session_update_has_event_id(self):
    msg = _build_session_update("zh")
    assert "event_id" in msg
    assert len(msg["event_id"]) == 8

  def test_audio_append_has_correct_type(self):
    msg = _build_audio_append("AQID")
    assert msg["type"] == "input_audio_buffer.append"

  def test_audio_append_includes_audio(self):
    msg = _build_audio_append("AQID")
    assert msg["audio"] == "AQID"

  def test_audio_append_has_event_id(self):
    msg = _build_audio_append("AQID")
    assert "event_id" in msg

  def test_session_finish_has_correct_type(self):
    msg = _build_session_finish()
    assert msg["type"] == "session.finish"

  def test_session_finish_has_event_id(self):
    msg = _build_session_finish()
    assert "event_id" in msg


class TestTraditionalConversion:
  """Test simplified to traditional Chinese conversion."""

  def test_converts_simplified_to_traditional(self):
    result = _to_traditional("你好世界")
    # s2twp converts to Taiwan phrases
    assert result == "你好世界"

  def test_converts_common_words(self):
    result = _to_traditional("信息")
    assert result == "資訊"

  def test_preserves_english(self):
    result = _to_traditional("Hello world")
    assert result == "Hello world"

  def test_handles_empty_string(self):
    result = _to_traditional("")
    assert result == ""

  def test_handles_mixed_text(self):
    result = _to_traditional("Hello 信息 world")
    assert "Hello" in result
    assert "world" in result


class TestResponseParsing:
  """Test server event handling via callbacks."""

  def _make_streamer_with_callbacks(self):
    """Create a streamer with mock callbacks (no connection)."""
    on_partial = MagicMock()
    on_final = MagicMock()
    on_error = MagicMock()
    streamer = QwenRealtimeStreamer(
      api_key="test-key",
      language="zh",
      on_partial=on_partial,
      on_final=on_final,
      on_error=on_error,
    )
    return streamer, on_partial, on_final, on_error

  def test_partial_result_calls_on_partial(self):
    streamer, on_partial, _, _ = self._make_streamer_with_callbacks()
    msg = {
      "type": "conversation.item.input_audio_transcription.text",
      "stash": "你好",
    }
    streamer._handle_server_event(msg)
    on_partial.assert_called_once()

  def test_partial_result_converts_to_traditional(self):
    streamer, on_partial, _, _ = self._make_streamer_with_callbacks()
    msg = {
      "type": "conversation.item.input_audio_transcription.text",
      "stash": "信息",
    }
    streamer._handle_server_event(msg)
    on_partial.assert_called_once_with("資訊")

  def test_final_result_calls_on_final(self):
    streamer, _, on_final, _ = self._make_streamer_with_callbacks()
    msg = {
      "type": "conversation.item.input_audio_transcription.completed",
      "transcript": "你好世界",
    }
    streamer._handle_server_event(msg)
    on_final.assert_called_once()

  def test_final_result_converts_to_traditional(self):
    streamer, _, on_final, _ = self._make_streamer_with_callbacks()
    msg = {
      "type": "conversation.item.input_audio_transcription.completed",
      "transcript": "信息技术",
    }
    streamer._handle_server_event(msg)
    # s2twp: 信息技术 → 資訊科技 (Taiwan phrasing)
    args = on_final.call_args[0][0]
    assert "資訊" in args

  def test_error_event_calls_on_error(self):
    streamer, _, _, on_error = self._make_streamer_with_callbacks()
    msg = {
      "type": "error",
      "error": {"message": "Rate limit exceeded"},
    }
    streamer._handle_server_event(msg)
    on_error.assert_called_once_with("Rate limit exceeded")

  def test_error_event_default_message(self):
    streamer, _, _, on_error = self._make_streamer_with_callbacks()
    msg = {
      "type": "error",
      "error": {},
    }
    streamer._handle_server_event(msg)
    on_error.assert_called_once_with("Unknown server error")

  def test_session_finish_stops_running(self):
    streamer, _, _, _ = self._make_streamer_with_callbacks()
    streamer._running = True
    msg = {"type": "session.finish"}
    streamer._handle_server_event(msg)
    assert streamer._running is False

  def test_empty_stash_does_not_call_partial(self):
    streamer, on_partial, _, _ = self._make_streamer_with_callbacks()
    msg = {
      "type": "conversation.item.input_audio_transcription.text",
      "stash": "",
    }
    streamer._handle_server_event(msg)
    on_partial.assert_not_called()

  def test_empty_transcript_does_not_call_final(self):
    streamer, _, on_final, _ = self._make_streamer_with_callbacks()
    msg = {
      "type": "conversation.item.input_audio_transcription.completed",
      "transcript": "",
    }
    streamer._handle_server_event(msg)
    on_final.assert_not_called()

  def test_unknown_event_type_is_ignored(self):
    streamer, on_partial, on_final, on_error = self._make_streamer_with_callbacks()
    msg = {"type": "some.unknown.event"}
    streamer._handle_server_event(msg)
    on_partial.assert_not_called()
    on_final.assert_not_called()
    on_error.assert_not_called()


class TestStreamerInit:
  """Test streamer initialization and configuration."""

  def test_default_language(self):
    streamer = QwenRealtimeStreamer(api_key="test-key")
    assert streamer._language == "zh"

  def test_custom_language(self):
    streamer = QwenRealtimeStreamer(api_key="test-key", language="en")
    assert streamer._language == "en"

  def test_is_running_initially_false(self):
    streamer = QwenRealtimeStreamer(api_key="test-key")
    assert streamer.is_running is False

  def test_api_key_from_param(self):
    streamer = QwenRealtimeStreamer(api_key="my-key")
    assert streamer._api_key == "my-key"

  @patch.dict("os.environ", {"DASHSCOPE_API_KEY": "env-key"})
  def test_api_key_from_env(self):
    streamer = QwenRealtimeStreamer()
    assert streamer._api_key == "env-key"

  def test_start_raises_without_api_key(self):
    streamer = QwenRealtimeStreamer(api_key="")
    with patch.dict("os.environ", {}, clear=True):
      with pytest.raises(RuntimeError, match="DASHSCOPE_API_KEY"):
        streamer.start()


class TestSendAudio:
  """Test audio sending logic."""

  def test_send_audio_base64_encodes(self):
    streamer = QwenRealtimeStreamer(api_key="test-key")
    streamer._running = True
    mock_ws = MagicMock()
    streamer._ws = mock_ws

    pcm = b"\x00\x01\x02\x03"
    streamer.send_audio(pcm)

    mock_ws.send.assert_called_once()
    sent = json.loads(mock_ws.send.call_args[0][0])
    assert sent["type"] == "input_audio_buffer.append"
    # Verify the audio is correctly base64 encoded
    decoded = base64.b64decode(sent["audio"])
    assert decoded == pcm

  def test_send_audio_noop_when_not_running(self):
    streamer = QwenRealtimeStreamer(api_key="test-key")
    streamer._running = False
    mock_ws = MagicMock()
    streamer._ws = mock_ws

    streamer.send_audio(b"\x00\x01")
    mock_ws.send.assert_not_called()

  def test_send_audio_noop_when_no_ws(self):
    streamer = QwenRealtimeStreamer(api_key="test-key")
    streamer._running = True
    streamer._ws = None

    # Should not raise
    streamer.send_audio(b"\x00\x01")


class TestStop:
  """Test stop/cleanup logic."""

  def test_stop_sends_session_finish(self):
    streamer = QwenRealtimeStreamer(api_key="test-key")
    streamer._running = True
    mock_ws = MagicMock()
    streamer._ws = mock_ws

    streamer.stop()

    # Should have sent session.finish
    assert mock_ws.send.call_count == 1
    sent = json.loads(mock_ws.send.call_args[0][0])
    assert sent["type"] == "session.finish"
    mock_ws.close.assert_called_once()

  def test_stop_sets_running_false(self):
    streamer = QwenRealtimeStreamer(api_key="test-key")
    streamer._running = True
    mock_ws = MagicMock()
    streamer._ws = mock_ws

    streamer.stop()
    assert streamer._running is False

  def test_stop_noop_when_not_running(self):
    streamer = QwenRealtimeStreamer(api_key="test-key")
    streamer._running = False
    streamer._ws = None

    # Should not raise
    streamer.stop()

  def test_stop_handles_ws_error_gracefully(self):
    streamer = QwenRealtimeStreamer(api_key="test-key")
    streamer._running = True
    mock_ws = MagicMock()
    mock_ws.send.side_effect = Exception("connection lost")
    streamer._ws = mock_ws

    # Should not raise
    streamer.stop()
    assert streamer._running is False

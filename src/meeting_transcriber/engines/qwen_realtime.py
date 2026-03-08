"""Qwen3-ASR real-time streaming engine via DashScope WebSocket API.

Streams PCM audio over WebSocket for real-time transcription.
Model: qwen3-asr-flash-realtime (streaming, continuous audio).
Cost: ~$0.0054/min (same as sync Qwen3-ASR).
"""

import base64
import json
import logging
import os
import threading
import uuid
from typing import Callable

import websocket  # websocket-client (sync)

logger = logging.getLogger(__name__)

_WS_URL = "wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime"
_MODEL = "qwen3-asr-flash-realtime"
_COST_PER_MINUTE = 0.0054


def _to_traditional(text: str) -> str:
  """Convert simplified Chinese to traditional Chinese (Taiwan)."""
  try:
    from opencc import OpenCC

    cc = OpenCC("s2twp")
    return cc.convert(text)
  except ImportError:
    return text


def _build_session_update(language: str) -> dict:
  """Build the session.update message with language and VAD config."""
  return {
    "event_id": _event_id(),
    "type": "session.update",
    "session": {
      "input_audio_format": "pcm16",
      "sample_rate": 16000,
      "asr_options": {
        "language": language,
        "enable_itn": True,
      },
      "turn_detection": {
        "type": "server_vad",
      },
    },
  }


def _build_audio_append(audio_b64: str) -> dict:
  """Build an input_audio_buffer.append message."""
  return {
    "event_id": _event_id(),
    "type": "input_audio_buffer.append",
    "audio": audio_b64,
  }


def _build_session_finish() -> dict:
  """Build the session.finish message."""
  return {
    "event_id": _event_id(),
    "type": "session.finish",
  }


def _event_id() -> str:
  """Generate a unique event ID."""
  return str(uuid.uuid4())[:8]


class QwenRealtimeStreamer:
  """Streams PCM audio to DashScope Qwen3-ASR real-time WebSocket API.

  Uses websocket-client (sync) to avoid async-in-thread complexity.
  The receive loop runs in a separate daemon thread.
  """

  def __init__(
    self,
    api_key: str | None = None,
    language: str = "zh",
    on_partial: Callable[[str], None] | None = None,
    on_final: Callable[[str], None] | None = None,
    on_error: Callable[[str], None] | None = None,
  ) -> None:
    self._api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
    self._language = language
    self._on_partial = on_partial
    self._on_final = on_final
    self._on_error = on_error
    self._ws: websocket.WebSocket | None = None
    self._recv_thread: threading.Thread | None = None
    self._running = False
    self._connected = threading.Event()

  @property
  def is_running(self) -> bool:
    return self._running

  def start(self) -> None:
    """Connect to the WebSocket and start the receive loop."""
    if not self._api_key:
      raise RuntimeError(
        "DASHSCOPE_API_KEY environment variable is not set. "
        "Get your key at https://modelstudio.console.alibabacloud.com/"
      )

    url = f"{_WS_URL}?model={_MODEL}"
    headers = {
      "Authorization": f"Bearer {self._api_key}",
    }

    self._ws = websocket.WebSocket()
    self._ws.connect(url, header=headers)
    self._running = True

    # Wait for session.created, then send session.update
    self._wait_session_created()
    self._send_json(_build_session_update(self._language))

    # Start receive loop in background thread
    self._recv_thread = threading.Thread(
      target=self._receive_loop,
      daemon=True,
    )
    self._recv_thread.start()
    self._connected.set()

  def _wait_session_created(self) -> None:
    """Read messages until session.created is received."""
    for _ in range(10):
      raw = self._ws.recv()  # type: ignore[union-attr]
      if not raw:
        continue
      msg = json.loads(raw)
      if msg.get("type") == "session.created":
        logger.info("Session created: %s", msg.get("session", {}).get("id", ""))
        return
    raise RuntimeError("Did not receive session.created from server")

  def send_audio(self, pcm_bytes: bytes) -> None:
    """Base64 encode PCM bytes and send as audio append message."""
    if not self._running or self._ws is None:
      return
    audio_b64 = base64.b64encode(pcm_bytes).decode("ascii")
    self._send_json(_build_audio_append(audio_b64))

  def stop(self) -> None:
    """Send session.finish and close the WebSocket."""
    if not self._running:
      return

    self._running = False
    try:
      if self._ws is not None:
        self._send_json(_build_session_finish())
        self._ws.close()
    except Exception as e:
      logger.warning("Error closing WebSocket: %s", e)

    if self._recv_thread is not None:
      self._recv_thread.join(timeout=3)

  def _send_json(self, msg: dict) -> None:
    """Send a JSON message over the WebSocket."""
    if self._ws is None:
      return
    try:
      self._ws.send(json.dumps(msg))
    except Exception as e:
      logger.error("WebSocket send error: %s", e)
      if self._on_error:
        self._on_error(f"WebSocket send error: {e}")

  def _receive_loop(self) -> None:
    """Background thread: read server events and dispatch callbacks."""
    while self._running and self._ws is not None:
      try:
        raw = self._ws.recv()
        if not raw:
          continue
        msg = json.loads(raw)
        self._handle_server_event(msg)
      except websocket.WebSocketConnectionClosedException:
        logger.info("WebSocket connection closed")
        break
      except Exception as e:
        if self._running:
          logger.error("Receive loop error: %s", e)
          if self._on_error:
            self._on_error(f"Receive error: {e}")
        break

  def _handle_server_event(self, msg: dict) -> None:
    """Route a server event to the appropriate callback."""
    event_type = msg.get("type", "")

    if event_type == "conversation.item.input_audio_transcription.text":
      stash = msg.get("stash", "")
      if stash and self._on_partial:
        self._on_partial(_to_traditional(stash))

    elif event_type == "conversation.item.input_audio_transcription.completed":
      transcript = msg.get("transcript", "")
      if transcript and self._on_final:
        self._on_final(_to_traditional(transcript))

    elif event_type == "error":
      error_msg = msg.get("error", {}).get("message", "Unknown server error")
      logger.error("Server error: %s", error_msg)
      if self._on_error:
        self._on_error(error_msg)

    elif event_type == "session.finish":
      logger.info("Session finished by server")
      self._running = False

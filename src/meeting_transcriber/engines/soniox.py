"""Soniox engine.

Mid-sentence code-switching auto-detection, $0.12/hr.
No language specification needed — auto-detects Chinese/English.
Uses REST API with urllib (stdlib) to avoid extra dependencies.
"""

import io
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from meeting_transcriber.engines.base import BaseEngine
from meeting_transcriber.models import Segment, TranscriptResult

_API_URL = "https://api.soniox.com/v1/transcribe"
_MODEL = "soniox-v2"
_MAX_RETRIES = 3
_BACKOFF_BASE = 2.0
_PAUSE_THRESHOLD_MS = 1000  # pause > 1s splits segments


class SonioxEngine(BaseEngine):
  """Speech-to-text using Soniox REST API."""

  name = "soniox"
  cost_per_minute = 0.002  # ~$0.12/hr

  def __init__(self) -> None:
    self._api_key: str | None = None

  @property
  def api_key(self) -> str:
    """Lazy-load API key from environment."""
    if self._api_key is None:
      key = os.environ.get("SONIOX_API_KEY")
      if not key:
        raise RuntimeError(
          "SONIOX_API_KEY environment variable is not set. "
          "Get your key at https://soniox.com"
        )
      self._api_key = key
    return self._api_key

  def transcribe_file(
    self,
    path: Path,
    language: str = "zh",
  ) -> TranscriptResult:
    """Transcribe a single audio file via Soniox REST API."""
    # Validate API key early
    _ = self.api_key

    response_data = self._call_api(path)
    return self._parse_response(response_data)

  def _call_api(self, path: Path) -> dict:
    """Call Soniox API with retry and exponential backoff."""
    last_error: Exception | None = None

    for attempt in range(_MAX_RETRIES):
      try:
        return self._send_request(path)
      except Exception as err:
        last_error = err
        if attempt < _MAX_RETRIES - 1:
          time.sleep(_BACKOFF_BASE ** attempt)

    raise RuntimeError(
      f"Soniox transcription failed after {_MAX_RETRIES} attempts: {last_error}"
    )

  def _send_request(self, path: Path) -> dict:
    """Send multipart/form-data POST to Soniox API."""
    boundary = "----SonioxBoundary"
    body = self._build_multipart_body(path, boundary)

    request = urllib.request.Request(
      _API_URL,
      data=body,
      headers={
        "Authorization": f"Bearer {self.api_key}",
        "Content-Type": f"multipart/form-data; boundary={boundary}",
      },
      method="POST",
    )

    with urllib.request.urlopen(request) as resp:
      return json.loads(resp.read().decode("utf-8"))

  def _build_multipart_body(self, path: Path, boundary: str) -> bytes:
    """Build multipart/form-data body with audio file and model param."""
    parts = io.BytesIO()

    # Model field
    parts.write(f"--{boundary}\r\n".encode())
    parts.write(b'Content-Disposition: form-data; name="model"\r\n\r\n')
    parts.write(f"{_MODEL}\r\n".encode())

    # File field
    parts.write(f"--{boundary}\r\n".encode())
    parts.write(
      f'Content-Disposition: form-data; name="file"; '
      f'filename="{path.name}"\r\n'.encode()
    )
    parts.write(b"Content-Type: application/octet-stream\r\n\r\n")
    parts.write(path.read_bytes())
    parts.write(b"\r\n")

    # Closing boundary
    parts.write(f"--{boundary}--\r\n".encode())

    return parts.getvalue()

  def _parse_response(self, data: dict) -> TranscriptResult:
    """Parse Soniox JSON response into TranscriptResult."""
    words = data.get("words", [])
    duration_ms = data.get("duration_ms", 0)
    duration_s = duration_ms / 1000.0

    segments = self._group_words_into_segments(words)
    full_text = " ".join(seg.text for seg in segments)
    cost = (duration_s / 60.0) * self.cost_per_minute

    return TranscriptResult(
      segments=segments,
      full_text=full_text,
      duration=duration_s,
      cost=cost,
      engine=self.name,
    )

  def _group_words_into_segments(self, words: list[dict]) -> list[Segment]:
    """Group words into sentence-level segments by pauses.

    Words separated by > _PAUSE_THRESHOLD_MS are placed in
    different segments. Whitespace-only words are skipped as
    content but used for gap detection.
    """
    if not words:
      return []

    segments: list[Segment] = []
    current_texts: list[str] = []
    current_start_ms: int | None = None
    current_end_ms: int = 0

    for word in words:
      text = word.get("text", "")
      start_ms = word.get("start_ms", 0)
      end_ms = word.get("end_ms", 0)

      # Skip whitespace-only tokens for content
      is_whitespace = text.strip() == ""

      # Detect pause and split segment
      if (
        current_start_ms is not None
        and not is_whitespace
        and start_ms - current_end_ms > _PAUSE_THRESHOLD_MS
      ):
        joined = "".join(current_texts).strip()
        if joined:
          segments.append(Segment(
            start=current_start_ms / 1000.0,
            end=current_end_ms / 1000.0,
            text=joined,
          ))
        current_texts = []
        current_start_ms = None

      if not is_whitespace:
        if current_start_ms is None:
          current_start_ms = start_ms
        current_end_ms = end_ms
        current_texts.append(text)
      else:
        # Whitespace between words — append as separator
        current_texts.append(text)
        if end_ms > current_end_ms:
          current_end_ms = end_ms

    # Flush remaining
    if current_texts:
      joined = "".join(current_texts).strip()
      if joined and current_start_ms is not None:
        segments.append(Segment(
          start=current_start_ms / 1000.0,
          end=current_end_ms / 1000.0,
          text=joined,
        ))

    return segments

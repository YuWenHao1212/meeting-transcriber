"""Groq engine (LPU-accelerated Whisper).

Cheapest option at $0.04/hr, 216x real-time speed.
Best for pure English. Weak on code-switching.
Chunked HTTP (not WebSocket), max 100MB per upload.
"""

import os
import time
from pathlib import Path

import openai

from meeting_transcriber.engines.base import BaseEngine
from meeting_transcriber.models import Segment, TranscriptResult

_MAX_RETRIES = 3
_BACKOFF_BASE = 2.0
_COST_PER_MINUTE = 0.0007  # ~$0.04/hr
_API_BASE_URL = "https://api.groq.com/openai/v1"


class GroqEngine(BaseEngine):
  """Speech-to-text using Groq (whisper-large-v3-turbo)."""

  name = "groq"
  cost_per_minute = _COST_PER_MINUTE

  def __init__(self, model: str = "whisper-large-v3-turbo") -> None:
    self.model = model
    self._client = None

  @property
  def client(self):
    if self._client is None:
      api_key = os.environ.get("GROQ_API_KEY")
      if not api_key:
        raise RuntimeError(
          "GROQ_API_KEY environment variable is not set. Get your key at https://console.groq.com/"
        )
      self._client = openai.OpenAI(
        api_key=api_key,
        base_url=_API_BASE_URL,
      )
    return self._client

  def transcribe_file(
    self,
    path: Path,
    language: str = "zh",
  ) -> TranscriptResult:
    """Transcribe a single audio file via Groq API."""
    response = self._call_api(path, language)
    return self._parse_response(response)

  def _call_api(self, path: Path, language: str):
    """Call Groq API with retry and exponential backoff."""
    last_error = None
    for attempt in range(_MAX_RETRIES):
      try:
        with open(path, "rb") as audio_file:
          return self.client.audio.transcriptions.create(
            model=self.model,
            file=audio_file,
            language=language,
            response_format="verbose_json",
          )
      except Exception as err:
        last_error = err
        if attempt < _MAX_RETRIES - 1:
          time.sleep(_BACKOFF_BASE**attempt)
    raise RuntimeError(f"Groq transcription failed after {_MAX_RETRIES} attempts: {last_error}")

  def _parse_response(self, response) -> TranscriptResult:
    """Parse Groq verbose_json response into TranscriptResult."""
    segments = []
    if hasattr(response, "segments") and response.segments:
      for seg in response.segments:
        segments.append(
          Segment(
            start=seg.get("start", 0.0) if isinstance(seg, dict) else getattr(seg, "start", 0.0),
            end=seg.get("end", 0.0) if isinstance(seg, dict) else getattr(seg, "end", 0.0),
            text=(
              seg.get("text", "") if isinstance(seg, dict) else getattr(seg, "text", "")
            ).strip(),
          )
        )

    duration = getattr(response, "duration", 0.0) or 0.0
    cost = (duration / 60.0) * self.cost_per_minute

    return TranscriptResult(
      segments=segments,
      full_text=getattr(response, "text", "") or "",
      duration=duration,
      cost=cost,
      engine=self.name,
    )

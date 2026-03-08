"""OpenAI gpt-4o-transcribe engine."""

import time
from pathlib import Path

from meeting_transcriber.engines.base import BaseEngine
from meeting_transcriber.models import Segment, TranscriptResult

_MAX_RETRIES = 3
_BACKOFF_BASE = 2.0
_COST_PER_MINUTE = 0.006


class OpenAIEngine(BaseEngine):
  """Speech-to-text using OpenAI gpt-4o-transcribe."""

  name = "openai"
  cost_per_minute = _COST_PER_MINUTE

  def __init__(self, model: str = "gpt-4o-transcribe") -> None:
    self.model = model
    self._client = None

  @property
  def client(self):
    if self._client is None:
      import openai
      self._client = openai.OpenAI()
    return self._client

  def transcribe_file(
    self,
    path: Path,
    language: str = "zh",
  ) -> TranscriptResult:
    """Transcribe a single audio file via OpenAI API."""
    response = self._call_api(path, language)
    return self._parse_response(response)

  def _call_api(self, path: Path, language: str):
    """Call OpenAI API with retry and exponential backoff."""
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
          time.sleep(_BACKOFF_BASE ** attempt)
    raise RuntimeError(
      f"OpenAI transcription failed after {_MAX_RETRIES} attempts: {last_error}"
    )

  def _parse_response(self, response) -> TranscriptResult:
    """Parse OpenAI verbose_json response into TranscriptResult."""
    segments = []
    if hasattr(response, "segments") and response.segments:
      for seg in response.segments:
        segments.append(Segment(
          start=seg.get("start", 0.0) if isinstance(seg, dict) else getattr(seg, "start", 0.0),
          end=seg.get("end", 0.0) if isinstance(seg, dict) else getattr(seg, "end", 0.0),
          text=(seg.get("text", "") if isinstance(seg, dict) else getattr(seg, "text", "")).strip(),
        ))

    duration = getattr(response, "duration", 0.0) or 0.0
    cost = (duration / 60.0) * self.cost_per_minute

    return TranscriptResult(
      segments=segments,
      full_text=getattr(response, "text", "") or "",
      duration=duration,
      cost=cost,
      engine=self.name,
    )

"""Groq engine (LPU-accelerated Whisper).

Cheapest option at $0.04/hr, 216x real-time speed.
Best for pure English. Weak on code-switching.
Chunked HTTP (not WebSocket), max 100MB per upload.
"""

from pathlib import Path

from meeting_transcriber.engines.base import BaseEngine
from meeting_transcriber.models import TranscriptResult


class GroqEngine(BaseEngine):
  """Speech-to-text using Groq (whisper-large-v3-turbo)."""

  name = "groq"
  cost_per_minute = 0.0007  # ~$0.04/hr

  def transcribe_file(
    self,
    path: Path,
    language: str = "zh",
  ) -> TranscriptResult:
    raise NotImplementedError(
      "Groq engine coming in Phase 3. "
      "Use --engine openai for now."
    )

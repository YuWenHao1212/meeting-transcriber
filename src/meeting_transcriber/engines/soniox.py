"""Soniox engine.

Mid-sentence code-switching auto-detection, $0.12/hr streaming.
No language specification needed — auto-detects Chinese/English.
"""

from pathlib import Path

from meeting_transcriber.engines.base import BaseEngine
from meeting_transcriber.models import TranscriptResult


class SonioxEngine(BaseEngine):
  """Speech-to-text using Soniox API."""

  name = "soniox"
  cost_per_minute = 0.002  # ~$0.12/hr

  def transcribe_file(
    self,
    path: Path,
    language: str = "zh",
  ) -> TranscriptResult:
    raise NotImplementedError(
      "Soniox engine coming in Phase 3. "
      "Use --engine openai for now."
    )

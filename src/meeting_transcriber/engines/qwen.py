"""Qwen3-ASR engine (Alibaba Cloud Model Studio).

Best for Chinese + English code-switching.
WebSocket streaming, ~$0.40/hr, 97%+ accuracy on Chinese.
"""

from pathlib import Path

from meeting_transcriber.engines.base import BaseEngine
from meeting_transcriber.models import TranscriptResult


class QwenEngine(BaseEngine):
  """Speech-to-text using Qwen3-ASR via Alibaba Cloud."""

  name = "qwen"
  cost_per_minute = 0.0067  # ~$0.40/hr

  def transcribe_file(
    self,
    path: Path,
    language: str = "zh",
  ) -> TranscriptResult:
    raise NotImplementedError(
      "Qwen3-ASR engine coming in Phase 3. "
      "Use --engine openai for now."
    )

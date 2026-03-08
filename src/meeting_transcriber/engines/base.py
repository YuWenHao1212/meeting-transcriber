"""Base engine interface for STT providers."""

from abc import ABC, abstractmethod
from pathlib import Path

from meeting_transcriber.models import TranscriptResult


class BaseEngine(ABC):
  """Abstract base class for speech-to-text engines."""

  name: str = "base"
  cost_per_minute: float = 0.0

  @abstractmethod
  def transcribe_file(
    self,
    path: Path,
    language: str = "zh",
  ) -> TranscriptResult:
    """Transcribe a single audio file."""

  def transcribe_chunks(
    self,
    chunk_paths: list[Path],
    language: str = "zh",
  ) -> TranscriptResult:
    """Transcribe multiple chunks and merge results."""
    from meeting_transcriber.models import Segment

    all_segments: list[Segment] = []
    all_text_parts: list[str] = []
    total_duration = 0.0
    total_cost = 0.0

    for chunk_path in chunk_paths:
      result = self.transcribe_file(chunk_path, language)
      offset = total_duration
      for seg in result.segments:
        all_segments.append(Segment(
          start=seg.start + offset,
          end=seg.end + offset,
          text=seg.text,
          speaker=seg.speaker,
        ))
      all_text_parts.append(result.full_text)
      total_duration += result.duration
      total_cost += result.cost

    return TranscriptResult(
      segments=all_segments,
      full_text="\n".join(all_text_parts),
      duration=total_duration,
      cost=total_cost,
      engine=self.name,
    )

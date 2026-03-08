"""High-level transcription interface."""

from pathlib import Path

from meeting_transcriber.chunker import chunk_audio
from meeting_transcriber.engines import get_engine
from meeting_transcriber.models import TranscriptResult


def transcribe(
  audio_path: Path,
  language: str = "zh",
  engine_name: str = "openai",
  chunk_duration: int = 30,
  overlap: int = 2,
) -> TranscriptResult:
  """Transcribe an audio file using the specified engine.

  Chunks the audio and transcribes each chunk, then merges results.
  """
  engine = get_engine(engine_name)
  chunks = chunk_audio(audio_path, chunk_duration=chunk_duration, overlap=overlap)
  return engine.transcribe_chunks(chunks, language=language)


def transcribe_file(
  audio_path: Path,
  language: str = "zh",
  engine_name: str = "openai",
) -> TranscriptResult:
  """Transcribe a single audio file without chunking."""
  engine = get_engine(engine_name)
  return engine.transcribe_file(audio_path, language=language)

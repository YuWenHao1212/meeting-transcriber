"""Audio chunking module for splitting WAV files into overlapping segments."""

import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf


def chunk_audio(
  input_path: Path,
  chunk_duration: int = 30,
  overlap: int = 2,
) -> list[Path]:
  """Split a WAV file into overlapping chunks.

  Args:
    input_path: Path to the input WAV file.
    chunk_duration: Duration of each chunk in seconds.
    overlap: Overlap between consecutive chunks in seconds.

  Returns:
    List of paths to the generated chunk WAV files.

  Raises:
    FileNotFoundError: If input_path does not exist.
    RuntimeError: If the file cannot be read as audio.
  """
  if not input_path.exists():
    raise FileNotFoundError(f"Audio file not found: {input_path}")

  data, sample_rate = sf.read(str(input_path), dtype="float32")
  # Ensure mono
  if data.ndim > 1:
    data = data[:, 0]

  total_samples = len(data)
  chunk_samples = chunk_duration * sample_rate
  overlap_samples = overlap * sample_rate

  boundaries = _compute_boundaries(total_samples, chunk_samples, overlap_samples)
  return _write_chunks(data, sample_rate, boundaries)


def _compute_boundaries(
  total_samples: int,
  chunk_samples: int,
  overlap_samples: int,
) -> list[tuple[int, int]]:
  """Compute (start, end) sample boundaries for each chunk.

  First chunk: [0, chunk_samples).
  Each subsequent chunk starts `overlap_samples` before the previous
  chunk's end, giving it chunk_samples + overlap_samples length
  (or less for the final chunk).
  """
  if total_samples <= chunk_samples:
    return [(0, total_samples)]

  boundaries: list[tuple[int, int]] = []
  pos = 0
  while pos < total_samples:
    start = max(pos - overlap_samples, 0) if boundaries else 0
    end = min(pos + chunk_samples, total_samples)
    boundaries.append((start, end))
    pos += chunk_samples
  return boundaries


def _write_chunks(
  data: np.ndarray,
  sample_rate: int,
  boundaries: list[tuple[int, int]],
) -> list[Path]:
  """Write each chunk segment to a temporary WAV file."""
  tmp_dir = Path(tempfile.mkdtemp(prefix="mt_chunks_"))
  paths: list[Path] = []

  for i, (start, end) in enumerate(boundaries):
    chunk_path = tmp_dir / f"chunk_{i:03d}.wav"
    sf.write(str(chunk_path), data[start:end], sample_rate)
    paths.append(chunk_path)

  return paths

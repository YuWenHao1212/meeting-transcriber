"""Tests for audio chunking module."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from meeting_transcriber.chunker import chunk_audio

SAMPLE_RATE = 16000


def _make_sine_wav(path: Path, duration_sec: float, sr: int = SAMPLE_RATE) -> Path:
  """Create a mono sine-wave WAV file for testing."""
  t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
  data = np.sin(2 * np.pi * 440 * t).astype(np.float32)
  sf.write(str(path), data, sr)
  return path


class TestChunkAudioSplitting:
  """Test that chunk_audio splits a 90s file into 3 chunks of ~30s."""

  def test_splits_into_expected_number_of_chunks(self, tmp_path: Path) -> None:
    wav = _make_sine_wav(tmp_path / "90s.wav", 90.0)
    chunks = chunk_audio(wav, chunk_duration=30, overlap=2)
    assert len(chunks) == 3

  def test_chunk_durations_are_correct(self, tmp_path: Path) -> None:
    wav = _make_sine_wav(tmp_path / "90s.wav", 90.0)
    chunks = chunk_audio(wav, chunk_duration=30, overlap=2)

    for i, chunk_path in enumerate(chunks):
      data, sr = sf.read(str(chunk_path))
      duration = len(data) / sr
      if i == 0:
        # First chunk: exactly chunk_duration
        assert abs(duration - 30.0) < 0.01
      else:
        # Subsequent chunks include overlap from previous
        assert abs(duration - 32.0) < 0.01


class TestChunkValidity:
  """Test each chunk is a valid WAV file readable by soundfile."""

  def test_chunks_are_valid_wav_files(self, tmp_path: Path) -> None:
    wav = _make_sine_wav(tmp_path / "90s.wav", 90.0)
    chunks = chunk_audio(wav, chunk_duration=30, overlap=2)

    for chunk_path in chunks:
      assert chunk_path.exists()
      assert chunk_path.suffix == ".wav"
      data, sr = sf.read(str(chunk_path))
      assert sr == SAMPLE_RATE
      assert len(data) > 0

  def test_chunks_are_mono(self, tmp_path: Path) -> None:
    wav = _make_sine_wav(tmp_path / "90s.wav", 90.0)
    chunks = chunk_audio(wav, chunk_duration=30, overlap=2)

    for chunk_path in chunks:
      data, sr = sf.read(str(chunk_path))
      # Mono data is 1D array
      assert data.ndim == 1


class TestChunkOverlap:
  """Test overlap: second chunk starts 2 seconds before first chunk ends."""

  def test_overlap_content_matches(self, tmp_path: Path) -> None:
    wav = _make_sine_wav(tmp_path / "90s.wav", 90.0)
    chunks = chunk_audio(wav, chunk_duration=30, overlap=2)

    # Read the tail of chunk 0 and head of chunk 1
    data0, _ = sf.read(str(chunks[0]))
    data1, _ = sf.read(str(chunks[1]))

    overlap_samples = 2 * SAMPLE_RATE
    tail_of_first = data0[-overlap_samples:]
    head_of_second = data1[:overlap_samples]

    np.testing.assert_array_almost_equal(tail_of_first, head_of_second, decimal=5)


class TestShortAudio:
  """Test short audio (< chunk_duration) returns a single chunk."""

  def test_short_audio_returns_single_chunk(self, tmp_path: Path) -> None:
    wav = _make_sine_wav(tmp_path / "10s.wav", 10.0)
    chunks = chunk_audio(wav, chunk_duration=30, overlap=2)
    assert len(chunks) == 1

  def test_short_audio_preserves_content(self, tmp_path: Path) -> None:
    wav = _make_sine_wav(tmp_path / "10s.wav", 10.0)
    chunks = chunk_audio(wav, chunk_duration=30, overlap=2)

    original, _ = sf.read(str(wav))
    chunk_data, _ = sf.read(str(chunks[0]))
    np.testing.assert_array_almost_equal(original, chunk_data, decimal=5)


class TestErrorHandling:
  """Test empty/invalid file raises appropriate error."""

  def test_nonexistent_file_raises_error(self) -> None:
    with pytest.raises(FileNotFoundError):
      chunk_audio(Path("/nonexistent/file.wav"))

  def test_empty_file_raises_error(self, tmp_path: Path) -> None:
    empty = tmp_path / "empty.wav"
    empty.write_bytes(b"")
    with pytest.raises((sf.LibsndfileError, RuntimeError, ValueError)):
      chunk_audio(empty)

  def test_invalid_file_raises_error(self, tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.wav"
    invalid.write_text("this is not audio")
    with pytest.raises((sf.LibsndfileError, RuntimeError, ValueError)):
      chunk_audio(invalid)

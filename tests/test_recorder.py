"""Tests for the audio recorder module."""

from dataclasses import fields
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from meeting_transcriber.recorder import Recorder, Recording, list_devices


class TestRecordingDataclass:
  """Recording dataclass structure and defaults."""

  def test_recording_has_required_fields(self):
    rec = Recording(path=Path("/tmp/test.wav"), duration=10.5, sample_rate=16000)
    assert rec.path == Path("/tmp/test.wav")
    assert rec.duration == 10.5
    assert rec.sample_rate == 16000

  def test_recording_is_dataclass(self):
    field_names = {f.name for f in fields(Recording)}
    assert field_names == {"path", "duration", "sample_rate"}


class TestRecorderInit:
  """Recorder initialisation and defaults."""

  def test_default_values(self):
    rec = Recorder()
    assert rec.device_id is None
    assert rec.sample_rate == 16000
    assert rec.channels == 1

  def test_custom_values(self):
    rec = Recorder(device_id=2, sample_rate=44100, channels=2)
    assert rec.device_id == 2
    assert rec.sample_rate == 44100
    assert rec.channels == 2


class TestRecorderStart:
  """Recorder.start() opens a WAV file and starts an InputStream."""

  @patch("meeting_transcriber.recorder.sd")
  @patch("meeting_transcriber.recorder.sf")
  def test_start_creates_wav_and_stream(self, mock_sf, mock_sd):
    mock_file = MagicMock()
    mock_sf.SoundFile.return_value = mock_file

    mock_stream = MagicMock()
    mock_sd.InputStream.return_value = mock_stream

    recorder = Recorder(device_id=3, sample_rate=16000, channels=1)
    recorder.start(Path("/tmp/out.wav"))

    # SoundFile opened with correct params
    mock_sf.SoundFile.assert_called_once_with(
      Path("/tmp/out.wav"),
      mode="w",
      samplerate=16000,
      channels=1,
      format="WAV",
      subtype="FLOAT",
    )

    # InputStream created with correct params
    mock_sd.InputStream.assert_called_once()
    kwargs = mock_sd.InputStream.call_args[1]
    assert kwargs["device"] == 3
    assert kwargs["samplerate"] == 16000
    assert kwargs["channels"] == 1
    assert kwargs["dtype"] == "float32"
    assert kwargs["callback"] is not None

    # Stream started
    mock_stream.start.assert_called_once()

  @patch("meeting_transcriber.recorder.sd")
  @patch("meeting_transcriber.recorder.sf")
  def test_start_stores_output_path(self, mock_sf, mock_sd):
    mock_sf.SoundFile.return_value = MagicMock()
    mock_sd.InputStream.return_value = MagicMock()

    recorder = Recorder()
    recorder.start(Path("/tmp/out.wav"))
    assert recorder._output_path == Path("/tmp/out.wav")


class TestRecorderStop:
  """Recorder.stop() closes resources and returns Recording."""

  @patch("meeting_transcriber.recorder.sd")
  @patch("meeting_transcriber.recorder.sf")
  def test_stop_returns_recording(self, mock_sf, mock_sd):
    mock_file = MagicMock()
    mock_file.frames = 48000  # 3 seconds at 16000 Hz
    mock_sf.SoundFile.return_value = mock_file

    mock_stream = MagicMock()
    mock_sd.InputStream.return_value = mock_stream

    recorder = Recorder(sample_rate=16000)
    recorder.start(Path("/tmp/out.wav"))
    result = recorder.stop()

    assert isinstance(result, Recording)
    assert result.path == Path("/tmp/out.wav")
    assert result.duration == pytest.approx(3.0)
    assert result.sample_rate == 16000

  @patch("meeting_transcriber.recorder.sd")
  @patch("meeting_transcriber.recorder.sf")
  def test_stop_closes_stream_and_file(self, mock_sf, mock_sd):
    mock_file = MagicMock()
    mock_file.frames = 16000
    mock_sf.SoundFile.return_value = mock_file

    mock_stream = MagicMock()
    mock_sd.InputStream.return_value = mock_stream

    recorder = Recorder()
    recorder.start(Path("/tmp/out.wav"))
    recorder.stop()

    mock_stream.stop.assert_called_once()
    mock_stream.close.assert_called_once()
    mock_file.close.assert_called_once()


class TestRecorderCallback:
  """The InputStream callback writes data to the WAV file."""

  @patch("meeting_transcriber.recorder.sd")
  @patch("meeting_transcriber.recorder.sf")
  def test_callback_writes_data_to_file(self, mock_sf, mock_sd):
    import numpy as np

    mock_file = MagicMock()
    mock_sf.SoundFile.return_value = mock_file
    mock_sd.InputStream.return_value = MagicMock()

    recorder = Recorder()
    recorder.start(Path("/tmp/out.wav"))

    # Extract the callback passed to InputStream
    callback = mock_sd.InputStream.call_args[1]["callback"]

    # Simulate audio data
    indata = np.zeros((1024, 1), dtype="float32")
    callback(indata, 1024, None, None)

    # Verify data was written
    mock_file.write.assert_called_once_with(indata)


class TestListDevices:
  """list_devices() returns formatted device info."""

  @patch("meeting_transcriber.recorder.sd")
  def test_list_devices_returns_list(self, mock_sd):
    mock_sd.query_devices.return_value = [
      {"name": "MacBook Pro Microphone", "max_input_channels": 1, "index": 0},
      {"name": "External USB Mic", "max_input_channels": 2, "index": 1},
    ]

    result = list_devices()

    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0]["name"] == "MacBook Pro Microphone"
    assert result[1]["name"] == "External USB Mic"

  @patch("meeting_transcriber.recorder.sd")
  def test_list_devices_empty(self, mock_sd):
    mock_sd.query_devices.return_value = []
    result = list_devices()
    assert result == []

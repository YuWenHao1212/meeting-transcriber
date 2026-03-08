"""Audio recorder — streams microphone input to WAV file."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf


@dataclass
class Recording:
  """Result of a completed recording session."""

  path: Path
  duration: float
  sample_rate: int


class Recorder:
  """Streams microphone audio to a WAV file via sounddevice + soundfile."""

  def __init__(
    self,
    device_id: Optional[int] = None,
    sample_rate: int = 16000,
    channels: int = 1,
  ) -> None:
    self.device_id = device_id
    self.sample_rate = sample_rate
    self.channels = channels
    self._stream: Optional[sd.InputStream] = None
    self._file: Optional[sf.SoundFile] = None
    self._output_path: Optional[Path] = None

  def start(self, output_path: Path) -> None:
    """Open a WAV file and start recording from the microphone."""
    self._output_path = output_path
    self._file = sf.SoundFile(
      output_path,
      mode="w",
      samplerate=self.sample_rate,
      channels=self.channels,
      format="WAV",
      subtype="FLOAT",
    )
    self._stream = sd.InputStream(
      device=self.device_id,
      samplerate=self.sample_rate,
      channels=self.channels,
      dtype="float32",
      callback=self._callback,
    )
    self._stream.start()

  def stop(self) -> Recording:
    """Stop recording, close resources, and return a Recording."""
    if self._stream is not None:
      self._stream.stop()
      self._stream.close()

    frames = self._file.frames if self._file is not None else 0
    duration = frames / self.sample_rate

    if self._file is not None:
      self._file.close()

    return Recording(
      path=self._output_path,  # type: ignore[arg-type]
      duration=duration,
      sample_rate=self.sample_rate,
    )

  def _callback(
    self,
    indata: np.ndarray,
    frame_count: int,
    time_info: object,
    status: object,
  ) -> None:
    """Write incoming audio data directly to the WAV file."""
    if self._file is not None:
      self._file.write(indata)


def list_devices() -> list[dict]:
  """Return available audio devices as a list of dicts."""
  devices = sd.query_devices()
  if isinstance(devices, dict):
    return [devices]
  return [dict(d) if not isinstance(d, dict) else d for d in devices]

"""Qwen3-ASR engine via Alibaba Cloud DashScope International.

Best for Chinese + English code-switching.
Uses OpenAI-compatible chat/completions with base64 audio input.
Model: qwen3-asr-flash (sync, up to 5 min per chunk).
Cost: ~$0.0054/min (25 tokens/sec audio, $0.0036/1K tokens).
"""

import base64
import os
import time
from pathlib import Path

import openai
import soundfile as sf

from meeting_transcriber.engines.base import BaseEngine
from meeting_transcriber.models import Segment, TranscriptResult

_MAX_RETRIES = 3
_BACKOFF_BASE = 2.0
_COST_PER_MINUTE = 0.0054  # ~$0.32/hr
_DASHSCOPE_INTL_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"


class QwenEngine(BaseEngine):
  """Speech-to-text using Qwen3-ASR via DashScope International."""

  name = "qwen"
  cost_per_minute = _COST_PER_MINUTE

  def __init__(self, model: str = "qwen3-asr-flash") -> None:
    self.model = model
    self._client = None

  @property
  def client(self):
    if self._client is None:
      api_key = os.environ.get("DASHSCOPE_API_KEY", "")
      if not api_key:
        raise RuntimeError(
          "DASHSCOPE_API_KEY environment variable is not set. "
          "Get your key at https://modelstudio.console.alibabacloud.com/"
        )
      self._client = openai.OpenAI(
        api_key=api_key,
        base_url=_DASHSCOPE_INTL_BASE_URL,
      )
    return self._client

  def transcribe_file(
    self,
    path: Path,
    language: str = "zh",
  ) -> TranscriptResult:
    """Transcribe a single audio file via Qwen3-ASR."""
    audio_b64 = self._encode_audio(path)
    duration = self._get_duration(path)
    response = self._call_api(audio_b64, language)
    return self._parse_response(response, duration)

  def _encode_audio(self, path: Path) -> str:
    """Read audio file and return base64-encoded data URI."""
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    suffix = path.suffix.lstrip(".").lower()
    mime = {
      "wav": "audio/wav",
      "mp3": "audio/mp3",
      "flac": "audio/flac",
      "ogg": "audio/ogg",
      "m4a": "audio/mp4",
    }.get(suffix, "audio/wav")
    return f"data:{mime};base64,{b64}"

  def _get_duration(self, path: Path) -> float:
    """Get audio duration in seconds."""
    try:
      info = sf.info(path)
      return info.duration
    except Exception:
      return 0.0

  def _call_api(self, audio_data_uri: str, language: str):
    """Call Qwen3-ASR chat/completions with retry."""
    last_error = None
    for attempt in range(_MAX_RETRIES):
      try:
        return self.client.chat.completions.create(
          model=self.model,
          messages=[
            {
              "role": "user",
              "content": [
                {
                  "type": "input_audio",
                  "input_audio": {
                    "data": audio_data_uri,
                  },
                },
              ],
            },
          ],
          extra_body={
            "asr_options": {
              "language": language,
              "enable_itn": True,
            },
          },
          stream=False,
        )
      except Exception as err:
        last_error = err
        if attempt < _MAX_RETRIES - 1:
          time.sleep(_BACKOFF_BASE**attempt)
    raise RuntimeError(f"Qwen transcription failed after {_MAX_RETRIES} attempts: {last_error}")

  def _parse_response(self, response, duration: float) -> TranscriptResult:
    """Parse chat completion response into TranscriptResult."""
    text = ""
    if response.choices and response.choices[0].message:
      text = response.choices[0].message.content or ""

    # Estimate cost from duration (25 tokens/sec audio)
    cost = (duration / 60.0) * self.cost_per_minute

    # Qwen3-ASR returns plain text, no segment-level timestamps in sync mode
    segments = []
    if text.strip():
      segments = [
        Segment(
          start=0.0,
          end=duration,
          text=text.strip(),
        )
      ]

    return TranscriptResult(
      segments=segments,
      full_text=text.strip(),
      duration=duration,
      cost=cost,
      engine=self.name,
    )

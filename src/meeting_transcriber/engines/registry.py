"""Engine registry — factory for STT engines."""

from meeting_transcriber.engines.base import BaseEngine

_ENGINE_MAP: dict[str, type[BaseEngine]] = {}


def _register_defaults() -> None:
  """Lazily register all built-in engines."""
  if _ENGINE_MAP:
    return

  from meeting_transcriber.engines.openai_engine import OpenAIEngine
  from meeting_transcriber.engines.qwen import QwenEngine
  from meeting_transcriber.engines.soniox import SonioxEngine
  from meeting_transcriber.engines.groq import GroqEngine

  _ENGINE_MAP["openai"] = OpenAIEngine
  _ENGINE_MAP["qwen"] = QwenEngine
  _ENGINE_MAP["soniox"] = SonioxEngine
  _ENGINE_MAP["groq"] = GroqEngine


def get_engine(name: str) -> BaseEngine:
  """Get an engine instance by name."""
  _register_defaults()
  engine_cls = _ENGINE_MAP.get(name)
  if engine_cls is None:
    available = ", ".join(sorted(_ENGINE_MAP.keys()))
    raise ValueError(
      f"Unknown engine '{name}'. Available engines: {available}"
    )
  return engine_cls()


def list_engines() -> list[dict[str, str | float]]:
  """List all available engines with metadata."""
  _register_defaults()
  return [
    {
      "name": cls.name,
      "cost_per_minute": cls.cost_per_minute,
      "class": cls.__name__,
    }
    for cls in _ENGINE_MAP.values()
  ]

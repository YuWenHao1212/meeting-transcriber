from dataclasses import dataclass, field


@dataclass
class Segment:
  start: float
  end: float
  text: str
  speaker: str | None = None


@dataclass
class TranscriptResult:
  segments: list[Segment] = field(default_factory=list)
  full_text: str = ""
  duration: float = 0.0
  cost: float = 0.0
  engine: str = "openai"

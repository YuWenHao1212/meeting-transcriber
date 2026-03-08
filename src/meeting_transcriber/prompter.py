"""Real-time coaching prompt engine for meeting assistance."""

from dataclasses import dataclass, field
from pathlib import Path
import json
import logging
import re

import anthropic

logger = logging.getLogger(__name__)


@dataclass
class ContextChunk:
  text: str
  source: str
  keywords: list[str] = field(default_factory=list)


@dataclass
class DetectedQuestion:
  question: str
  keywords: list[str] = field(default_factory=list)


@dataclass
class ContextMatch:
  chunk: ContextChunk
  score: float
  matched_keywords: list[str] = field(default_factory=list)


@dataclass
class ActionItem:
  text: str
  owner: str | None = None
  deadline: str | None = None


_HAIKU_MODEL = "claude-haiku-4-5-20251001"


def load_context(paths: list[str]) -> list[ContextChunk]:
  """Load and parse context files into searchable chunks.

  Splits each file into paragraph-level chunks (split by double newline).
  Extracts keywords from each chunk using simple regex.
  """
  chunks: list[ContextChunk] = []
  for p in paths:
    path = Path(p)
    if not path.exists() or not path.is_file():
      continue
    text = path.read_text(encoding="utf-8")
    source = path.name
    paragraphs = [para.strip() for para in text.split("\n\n") if para.strip()]
    for para in paragraphs:
      keywords = _extract_keywords(para)
      chunks.append(ContextChunk(text=para, source=source, keywords=keywords))
  return chunks


def _extract_keywords(text: str) -> list[str]:
  """Extract keywords from text using simple heuristics.

  - CJK sequences (Chinese keywords, 2+ chars)
  - Capitalized words (proper nouns)
  - Content after bullet points (first few words)
  """
  keywords: list[str] = []
  # Extract CJK sequences (Chinese keywords)
  cjk = re.findall(r'[\u4e00-\u9fff\u3400-\u4dbf]{2,}', text)
  keywords.extend(cjk)
  # Extract capitalized words
  words = re.findall(r'\b[A-Z][a-zA-Z]+\b', text)
  keywords.extend(words)
  # Extract content after bullet points
  bullets = re.findall(r'[-*]\s+(.+?)(?:\n|$)', text)
  for bullet in bullets:
    parts = bullet.strip().split()[:4]
    keywords.append(' '.join(parts))
  return list(set(keywords))


def detect_questions(transcript_chunk: str) -> list[DetectedQuestion]:
  """Use Haiku to extract questions and key topics from transcript.

  Fast + cheap (~0.5s latency, <$0.001 per call).
  Never raises — returns empty list on any error.
  """
  if not transcript_chunk.strip():
    return []

  try:
    client = anthropic.Anthropic()
    response = client.messages.create(
      model=_HAIKU_MODEL,
      max_tokens=512,
      system=(
        "Extract questions and key discussion topics from meeting transcript. "
        "Return JSON array: [{\"question\": \"...\", \"keywords\": [\"...\", ...]}]. "
        "Focus on questions from the OTHER party (not the user). "
        "Include both explicit questions and implicit topic shifts. "
        "Keywords should be specific nouns/terms, both English and Chinese."
      ),
      messages=[{"role": "user", "content": transcript_chunk}],
    )
    text = _strip_code_block(response.content[0].text)
    items = json.loads(text)
    return [
      DetectedQuestion(
        question=item["question"],
        keywords=item.get("keywords", []),
      )
      for item in items
    ]
  except Exception:
    logger.warning("detect_questions failed", exc_info=True)
    return []


def match_context(
  keywords: list[str],
  context_chunks: list[ContextChunk],
) -> list[ContextMatch]:
  """Match keywords against context chunks using keyword overlap.

  Simple, fast, local matching — no API calls needed.
  Case-insensitive, supports partial matching for Chinese.
  """
  if not keywords or not context_chunks:
    return []

  matches: list[ContextMatch] = []
  lower_keywords = [k.lower() for k in keywords]

  for chunk in context_chunks:
    chunk_text_lower = chunk.text.lower()
    chunk_keywords_lower = [k.lower() for k in chunk.keywords]

    matched: list[str] = []
    for kw in lower_keywords:
      if kw in chunk_text_lower:
        matched.append(kw)
      elif any(kw in ck for ck in chunk_keywords_lower):
        matched.append(kw)

    if matched:
      score = len(matched) / len(lower_keywords)
      matches.append(ContextMatch(
        chunk=chunk,
        score=score,
        matched_keywords=matched,
      ))

  matches.sort(key=lambda m: m.score, reverse=True)
  return matches


def generate_prompt_card(
  question: DetectedQuestion,
  matches: list[ContextMatch],
  max_matches: int = 3,
) -> str:
  """Format a coaching prompt card from question + matched context."""
  lines = [f"**{question.question}**", ""]

  for match in matches[:max_matches]:
    excerpt = match.chunk.text[:200]
    if len(match.chunk.text) > 200:
      excerpt += "..."
    lines.append(f"*{match.chunk.source}*: {excerpt}")
    lines.append("")

  if not matches:
    lines.append("(No matching context found)")

  return "\n".join(lines).strip()


def detect_action_items(transcript_chunk: str) -> list[ActionItem]:
  """Use Haiku to extract action items from transcript chunk.

  Never raises — returns empty list on any error.
  """
  if not transcript_chunk.strip():
    return []

  try:
    client = anthropic.Anthropic()
    response = client.messages.create(
      model=_HAIKU_MODEL,
      max_tokens=512,
      system=(
        "Extract action items from meeting transcript. "
        "Return JSON array: "
        "[{\"text\": \"...\", \"owner\": \"name or null\", "
        "\"deadline\": \"date or null\"}]. "
        "Only include clear commitments, not general discussion."
      ),
      messages=[{"role": "user", "content": transcript_chunk}],
    )
    text = _strip_code_block(response.content[0].text)
    items = json.loads(text)
    return [
      ActionItem(
        text=item["text"],
        owner=item.get("owner"),
        deadline=item.get("deadline"),
      )
      for item in items
    ]
  except Exception:
    logger.warning("detect_action_items failed", exc_info=True)
    return []


def _strip_code_block(text: str) -> str:
  """Strip markdown code block wrapper if present."""
  text = text.strip()
  if text.startswith("```"):
    text = text.split("\n", 1)[1].rsplit("```", 1)[0]
  return text

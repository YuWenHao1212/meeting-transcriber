"""Real-time coaching prompt engine using Azure OpenAI GPT-4.1 Nano.

Detects questions and action items from live transcript segments,
matches them against uploaded playbook/context, and generates
coaching prompt cards for the meeting facilitator.

Uses GPT-4.1 Nano on Azure OpenAI (same endpoint as neatoolkit)
for low-latency, low-cost inference (~$0.10/M input tokens).
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import openai

logger = logging.getLogger(__name__)

# Azure OpenAI config — same endpoint as neatoolkit
_AZURE_ENDPOINT = os.environ.get(
  "AZURE_OPENAI_ENDPOINT", "https://neatoolkit-openai.openai.azure.com/"
)
_AZURE_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY", "")
_AZURE_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
_MODEL = os.environ.get("COACHING_MODEL", "gpt-4.1-nano")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ContextChunk:
  """A chunk of context from an uploaded playbook/document."""

  text: str
  source: str
  keywords: list[str] = field(default_factory=list)


@dataclass
class DetectedQuestion:
  """A question detected from the transcript."""

  question: str
  keywords: list[str] = field(default_factory=list)


@dataclass
class ContextMatch:
  """A context chunk matched against a question's keywords."""

  chunk: ContextChunk
  score: float
  matched_keywords: list[str] = field(default_factory=list)


@dataclass
class ActionItem:
  """An action item extracted from the transcript."""

  text: str
  owner: str | None = None
  deadline: str | None = None


# ---------------------------------------------------------------------------
# Client singleton
# ---------------------------------------------------------------------------

_client: openai.AzureOpenAI | None = None


def _get_client() -> openai.AzureOpenAI:
  """Get or create Azure OpenAI client singleton."""
  global _client
  if _client is None:
    _client = openai.AzureOpenAI(
      api_key=_AZURE_API_KEY,
      azure_endpoint=_AZURE_ENDPOINT,
      api_version=_AZURE_API_VERSION,
    )
  return _client


def _call_llm(system_prompt: str, user_text: str) -> str:
  """Call GPT-4.1 Nano and return the response text."""
  client = _get_client()
  response = client.chat.completions.create(
    model=_MODEL,
    messages=[
      {"role": "system", "content": system_prompt},
      {"role": "user", "content": user_text},
    ],
    temperature=0.3,
  )
  return response.choices[0].message.content or ""


def _parse_json_response(text: str) -> list[dict]:
  """Parse JSON from LLM response, handling markdown code blocks."""
  text = text.strip()
  # Strip markdown code block wrapper
  match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
  if match:
    text = match.group(1).strip()
  try:
    result = json.loads(text)
    if isinstance(result, list):
      return result
    return []
  except (json.JSONDecodeError, ValueError):
    return []


# ---------------------------------------------------------------------------
# Context loading
# ---------------------------------------------------------------------------

# Chinese keyword extraction pattern: CJK sequences of 2+ characters
_CJK_PATTERN = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]{2,}")
# English keyword pattern: capitalized words
_EN_KEYWORD_PATTERN = re.compile(r"\b[A-Z][a-zA-Z]+\b")


def _extract_keywords(text: str) -> list[str]:
  """Extract keywords from text (Chinese + English)."""
  keywords: list[str] = []
  # Chinese multi-char terms
  keywords.extend(_CJK_PATTERN.findall(text))
  # Capitalized English words
  keywords.extend(_EN_KEYWORD_PATTERN.findall(text))
  # Content after bullet points (first few words)
  bullets = re.findall(r"[-*]\s+(.+?)(?:\n|$)", text)
  for bullet in bullets:
    parts = bullet.strip().split()[:4]
    keywords.append(" ".join(parts))
  return list(set(keywords))


def load_context(paths: list[str]) -> list[ContextChunk]:
  """Load and chunk context files into searchable pieces."""
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


# ---------------------------------------------------------------------------
# Question detection
# ---------------------------------------------------------------------------

_QUESTION_SYSTEM_PROMPT = (
  "Extract questions and key discussion topics from meeting transcript. "
  'Return JSON array: [{"question": "...", "keywords": ["...", ...]}]. '
  "Focus on questions from the OTHER party (not the user). "
  "Include both explicit questions and implicit topic shifts. "
  "Keywords should be specific nouns/terms, both English and Chinese. "
  "Return ONLY valid JSON, no markdown code blocks."
)


def detect_questions(transcript_chunk: str) -> list[DetectedQuestion]:
  """Detect questions from a transcript segment using GPT-4.1 Nano.

  Fast + cheap (~0.3s latency, <$0.001 per call).
  Never raises — returns empty list on any error.
  """
  if not transcript_chunk.strip():
    return []

  try:
    response_text = _call_llm(_QUESTION_SYSTEM_PROMPT, transcript_chunk)
    items = _parse_json_response(response_text)
    return [
      DetectedQuestion(
        question=item["question"],
        keywords=item.get("keywords", []),
      )
      for item in items
      if item.get("question")
    ]
  except Exception:
    logger.warning("detect_questions failed", exc_info=True)
    return []


# ---------------------------------------------------------------------------
# Context matching (pure logic, no LLM)
# ---------------------------------------------------------------------------


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
      matches.append(
        ContextMatch(
          chunk=chunk,
          score=score,
          matched_keywords=matched,
        )
      )

  matches.sort(key=lambda m: m.score, reverse=True)
  return matches


# ---------------------------------------------------------------------------
# Prompt card generation (pure formatting, no LLM)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Action item detection
# ---------------------------------------------------------------------------

_ACTION_ITEM_SYSTEM_PROMPT = (
  "Extract action items from meeting transcript. "
  "Return JSON array: "
  '[{"text": "...", "owner": "name or null", '
  '"deadline": "date or null"}]. '
  "Only include clear commitments, not general discussion. "
  "Return ONLY valid JSON, no markdown code blocks."
)


def detect_action_items(transcript_chunk: str) -> list[ActionItem]:
  """Detect action items from a transcript segment using GPT-4.1 Nano.

  Never raises — returns empty list on any error.
  """
  if not transcript_chunk.strip():
    return []

  try:
    response_text = _call_llm(_ACTION_ITEM_SYSTEM_PROMPT, transcript_chunk)
    items = _parse_json_response(response_text)
    return [
      ActionItem(
        text=item["text"],
        owner=item.get("owner"),
        deadline=item.get("deadline"),
      )
      for item in items
      if item.get("text")
    ]
  except Exception:
    logger.warning("detect_action_items failed", exc_info=True)
    return []


# ---------------------------------------------------------------------------
# On-demand coaching strategy
# ---------------------------------------------------------------------------

_COACHING_SYSTEM_PROMPT = """\
你是一位即時會議教練。根據最近的對話內容和 Playbook，提供簡短的回應策略。

規則：
1. 用繁體中文回答
2. 分析對方剛才說了什麼、問了什麼
3. 從 Playbook 中找到相關的準備內容
4. 給出 2-3 句具體的回應建議
5. 如果有數字、報價、時程等具體資訊，直接引用
6. 保持簡短，不超過 100 字

格式：
📌 對方重點：（一句話）
💡 建議回應：（2-3 句策略）
"""


def generate_coaching_strategy(
  recent_transcript: str,
  playbook_text: str,
) -> str:
  """Generate on-demand coaching strategy from recent transcript + playbook.

  Called manually by the user (not auto-triggered).
  Returns a coaching strategy string.
  """
  if not recent_transcript.strip():
    return "沒有最近的對話內容可分析。"

  user_msg = f"## 最近對話\n{recent_transcript}"
  if playbook_text.strip():
    user_msg += f"\n\n## Playbook\n{playbook_text[:3000]}"

  try:
    return _call_llm(_COACHING_SYSTEM_PROMPT, user_msg)
  except Exception as e:
    logger.warning("generate_coaching_strategy failed: %s", e)
    return f"教練分析失敗：{e}"

"""Meeting coaching via Anthropic Claude API.

Two tiers:
- Quick Coach: Sonnet (~3-5s, ~$0.014/call)
- Deep Coach: Opus (~10-15s, ~$0.05/call)

Both use the Anthropic Python SDK with a shared client singleton.
"""

import logging
import os
import threading

import anthropic

logger = logging.getLogger(__name__)

SONNET_MODEL = os.environ.get("COACH_SONNET_MODEL", "claude-sonnet-4-20250514")
OPUS_MODEL = os.environ.get("COACH_OPUS_MODEL", "claude-opus-4-20250514")
MAX_TOKENS = 1024

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
  """Return a shared Anthropic client (lazy init, reuses connection)."""
  global _client
  if _client is None:
    _client = anthropic.Anthropic()
  return _client


_QUICK_SYSTEM = """\
你是一位即時會議教練。根據最近的對話內容和 Playbook，提供簡短的回應策略。
規則：
1. 用繁體中文回答
2. 分析對方剛才說了什麼、問了什麼
3. 從 Playbook 中找到相關的準備內容
4. 給出 2-3 句具體的回應建議
5. 如果有數字、報價、時程等具體資訊，直接引用
6. 保持簡短，不超過 150 字
格式：
📌 對方重點：（一句話）
💡 建議回應：（2-3 句策略）
"""

_DEEP_SYSTEM = """\
你是一位資深會議教練。深度分析即時會議對話，提供完整策略建議。
規則：
1. 用繁體中文回答，使用 Markdown 格式（標題、列表、粗體）
2. 識別對方的核心關注點和潛在需求
3. 從 Playbook 中找出最相關的準備內容
4. 提供 3-5 個具體的回應策略，包含：
   - 可以直接使用的話術
   - 數字/報價/時程等具體資訊引用
   - 需要注意的風險或紅旗
5. 建議下一步行動
"""


def _build_user_message(recent_transcript: str, playbook_text: str) -> str:
  """Build user message with transcript and optional playbook."""
  parts = [f"## 最近對話\n{recent_transcript}"]
  if playbook_text.strip():
    parts.append(f"## Playbook\n{playbook_text[:4000]}")
  return "\n\n".join(parts)


def _call_anthropic(system: str, user_msg: str, model: str) -> str:
  """Call Anthropic API and return response text."""
  client = _get_client()
  response = client.messages.create(
    model=model,
    max_tokens=MAX_TOKENS,
    system=system,
    messages=[{"role": "user", "content": user_msg}],
  )
  return response.content[0].text


def run_quick_coaching(
  recent_transcript: str,
  playbook_text: str,
  callback: callable,
) -> None:
  """Quick coaching via Sonnet. Runs in background thread."""
  thread = threading.Thread(
    target=_run_coaching,
    args=(recent_transcript, playbook_text, callback, SONNET_MODEL, _QUICK_SYSTEM),
    daemon=True,
  )
  thread.start()


def run_deep_coaching(
  recent_transcript: str,
  playbook_text: str,
  callback: callable,
) -> None:
  """Deep coaching via Opus. Runs in background thread."""
  thread = threading.Thread(
    target=_run_coaching,
    args=(recent_transcript, playbook_text, callback, OPUS_MODEL, _DEEP_SYSTEM),
    daemon=True,
  )
  thread.start()


def _run_coaching(
  recent_transcript: str,
  playbook_text: str,
  callback: callable,
  model: str,
  system: str,
) -> None:
  """Execute coaching call and invoke callback with result."""
  user_msg = _build_user_message(recent_transcript, playbook_text)
  try:
    result = _call_anthropic(system, user_msg, model)
    if result.strip():
      callback(result)
    else:
      callback("No analysis generated.")
  except anthropic.AuthenticationError:
    logger.error("Anthropic API key not set or invalid")
    callback("Coaching error: ANTHROPIC_API_KEY not set or invalid.")
  except anthropic.RateLimitError:
    logger.warning("Anthropic rate limit hit")
    callback("Coaching error: rate limit exceeded. Try again in a moment.")
  except Exception as e:
    logger.warning("Coaching error (%s): %s", model, e)
    callback(f"Coaching error: {e}")

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
QUICK_MAX_TOKENS = 300
DEEP_MAX_TOKENS = 600

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
  """Return a shared Anthropic client (lazy init, reuses connection)."""
  global _client
  if _client is None:
    _client = anthropic.Anthropic()
  return _client


_SHARED_SYSTEM = """\
你是一位即時會議戰術顧問，輔助「我方」進行會議。

重要：逐字稿可能有 [我方] / [對方] 標記（雙聲道模式），也可能沒有標記（單聲道模式）。\
若有標記，直接使用；若無標記，根據 Playbook 中的「我方/對方」角色資訊，\
從語境判斷每句話是誰說的。

嚴格規則：
1. 用繁體中文回答
2. 先釐清「誰說了什麼」— 區分我方 vs 對方的發言
3. 不要重述 Playbook 內容 — 使用者已經讀過，只引用關鍵數字或事實
4. 聚焦「現在該怎麼回應」，不是「背景分析」
5. 總長度不超過 300 字
6. 使用 Markdown 格式

格式：
### 🎯 對方意圖
（一句話：對方說了什麼、想要什麼）

### 💬 建議話術
（2-3 句我方可以直接說的話，引用具體數字/事實）

### ⚠️ 注意
（一句話：這個時刻要避免什麼陷阱）

### 👉 下一步
（一句話：接下來引導對話往哪走）
"""

_DEEP_SYSTEM = _SHARED_SYSTEM


def _build_user_message(recent_transcript: str, playbook_text: str) -> str:
  """Build user message with transcript and optional playbook.

  Playbook is placed BEFORE transcript so the coach reads role info first.
  """
  parts = []
  if playbook_text.strip():
    parts.append(f"## Playbook（角色與背景）\n{playbook_text[:4000]}")
  # Check if transcript has speaker labels from stereo mode
  has_labels = "[我方]" in recent_transcript or "[對方]" in recent_transcript
  label_note = "（已標記說話者）" if has_labels else "（未標記說話者）"
  parts.append(f"## 最近對話{label_note}\n{recent_transcript}")
  return "\n\n".join(parts)


def _call_anthropic(system: str, user_msg: str, model: str, max_tokens: int = 600) -> str:
  """Call Anthropic API and return response text."""
  client = _get_client()
  response = client.messages.create(
    model=model,
    max_tokens=max_tokens,
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
    args=(
      recent_transcript,
      playbook_text,
      callback,
      SONNET_MODEL,
      _SHARED_SYSTEM,
      QUICK_MAX_TOKENS,
    ),
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
    args=(recent_transcript, playbook_text, callback, OPUS_MODEL, _DEEP_SYSTEM, DEEP_MAX_TOKENS),
    daemon=True,
  )
  thread.start()


def _run_coaching(
  recent_transcript: str,
  playbook_text: str,
  callback: callable,
  model: str,
  system: str,
  max_tokens: int = 600,
) -> None:
  """Execute coaching call and invoke callback with result."""
  user_msg = _build_user_message(recent_transcript, playbook_text)
  try:
    result = _call_anthropic(system, user_msg, model, max_tokens)
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

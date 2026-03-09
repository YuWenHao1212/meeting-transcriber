"""Meeting coaching via Claude Code CLI (subscription-based, no API billing).

Two tiers:
- Quick Coach: fast tactical advice
- Deep Coach: in-depth strategic analysis
"""

import logging
import threading

from meeting_transcriber.claude_cli import call_claude_cli

logger = logging.getLogger(__name__)

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


def _build_prompt(
  system: str,
  recent_transcript: str,
  playbook_text: str,
) -> str:
  """Build the full prompt for Claude CLI."""
  parts = [system, ""]
  if playbook_text.strip():
    parts.append(f"## Playbook（角色與背景）\n{playbook_text[:4000]}")
    parts.append("")
  has_labels = "[我方]" in recent_transcript or "[對方]" in recent_transcript
  label_note = "（已標記說話者）" if has_labels else "（未標記說話者）"
  parts.append(f"## 最近對話{label_note}\n{recent_transcript}")
  return "\n".join(parts)


def run_quick_coaching(
  recent_transcript: str,
  playbook_text: str,
  callback: callable,
) -> None:
  """Quick coaching. Runs in background thread."""
  thread = threading.Thread(
    target=_run_coaching,
    args=(recent_transcript, playbook_text, callback, _SHARED_SYSTEM),
    daemon=True,
  )
  thread.start()


def run_deep_coaching(
  recent_transcript: str,
  playbook_text: str,
  callback: callable,
) -> None:
  """Deep coaching. Runs in background thread."""
  thread = threading.Thread(
    target=_run_coaching,
    args=(recent_transcript, playbook_text, callback, _DEEP_SYSTEM),
    daemon=True,
  )
  thread.start()


def _run_coaching(
  recent_transcript: str,
  playbook_text: str,
  callback: callable,
  system: str,
) -> None:
  """Execute coaching call and invoke callback with result."""
  prompt = _build_prompt(system, recent_transcript, playbook_text)
  try:
    result = call_claude_cli(prompt)
    callback(result)
  except Exception as e:
    logger.warning("Coaching error: %s", e)
    callback(f"Coaching error: {e}")

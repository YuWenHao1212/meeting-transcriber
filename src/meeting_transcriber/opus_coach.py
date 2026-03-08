"""Deep coaching via Claude Code CLI (Opus subscription)."""

import logging
import subprocess
import threading

logger = logging.getLogger(__name__)


def run_opus_coaching(
  recent_transcript: str,
  playbook_text: str,
  callback: callable,
) -> None:
  """Run Claude Code CLI for deep coaching analysis.

  Spawns `claude` CLI as subprocess with a coaching prompt.
  Calls callback(text) with the result when done.
  Runs in a background thread to avoid blocking.
  """
  thread = threading.Thread(
    target=_run_claude,
    args=(recent_transcript, playbook_text, callback),
    daemon=True,
  )
  thread.start()


def _run_claude(
  recent_transcript: str,
  playbook_text: str,
  callback: callable,
) -> None:
  """Actually invoke claude CLI and collect output."""
  prompt = _build_prompt(recent_transcript, playbook_text)

  try:
    result = subprocess.run(
      ["claude", "-p", prompt, "--no-input"],
      capture_output=True,
      text=True,
      timeout=30,
    )
    output = result.stdout.strip()
    if result.returncode != 0:
      error = result.stderr.strip() or "Claude CLI returned error"
      logger.warning("Claude CLI error: %s", error)
      callback(f"Deep coaching error: {error}")
      return

    if output:
      callback(output)
    else:
      callback("No analysis generated.")

  except FileNotFoundError:
    logger.error("claude CLI not found in PATH")
    callback("Claude Code CLI not found. Install: npm install -g @anthropic-ai/claude-code")
  except subprocess.TimeoutExpired:
    logger.warning("Claude CLI timed out after 30s")
    callback("Deep coaching timed out (30s limit).")
  except Exception as e:
    logger.warning("Claude CLI error: %s", e)
    callback(f"Deep coaching error: {e}")


def _build_prompt(recent_transcript: str, playbook_text: str) -> str:
  """Build the coaching prompt for Claude CLI."""
  parts = [
    "你是一位資深會議教練。分析以下即時會議對話，提供深度策略建議。",
    "",
    "## 分析要求",
    "1. 識別對方的核心關注點和潛在需求",
    "2. 從 Playbook 中找出最相關的準備內容",
    "3. 提供 3-5 個具體的回應策略，包含：",
    "   - 可以直接使用的話術",
    "   - 數字/報價/時程等具體資訊引用",
    "   - 需要注意的風險或紅旗",
    "4. 建議下一步行動",
    "",
    "## 格式",
    "用繁體中文回答。使用 Markdown 格式（標題、列表、粗體）。",
    "",
    "## 最近對話",
    recent_transcript,
  ]
  if playbook_text.strip():
    parts.extend([
      "",
      "## Playbook",
      playbook_text[:4000],
    ])
  return "\n".join(parts)

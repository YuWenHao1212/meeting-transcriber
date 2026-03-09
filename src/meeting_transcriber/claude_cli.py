"""Shared helper for calling Claude Code CLI (uses subscription, not API billing)."""

import logging
import subprocess

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 180


def call_claude_cli(
  prompt: str,
  timeout: int = _DEFAULT_TIMEOUT,
) -> str:
  """Call Claude Code CLI and return response text.

  Uses `claude -p` which runs on your Claude Code subscription
  (Max plan), not pay-as-you-go API billing.

  Args:
    prompt: The full prompt to send.
    timeout: Max seconds to wait for response.

  Returns:
    Response text from Claude.

  Raises:
    FileNotFoundError: If claude CLI is not installed.
    TimeoutError: If the call exceeds timeout.
    RuntimeError: If the CLI returns an error.
  """
  try:
    result = subprocess.run(
      ["claude", "-p", prompt],
      capture_output=True,
      text=True,
      timeout=timeout,
    )
  except FileNotFoundError:
    logger.error("claude CLI not found in PATH")
    raise FileNotFoundError(
      "Claude Code CLI not found. Install: npm install -g @anthropic-ai/claude-code"
    )
  except subprocess.TimeoutExpired:
    logger.warning("Claude CLI timed out after %ds", timeout)
    raise TimeoutError(f"Claude CLI timed out after {timeout}s")

  if result.returncode != 0:
    error = result.stderr.strip() or "Claude CLI returned error"
    logger.warning("Claude CLI error: %s", error)
    raise RuntimeError(error)

  output = result.stdout.strip()
  if not output:
    raise RuntimeError("Claude CLI returned empty response")

  return output

# Dual Coaching UI Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add dual coaching system — Quick Coach (Azure Nano, <1s) + Deep Coach (Claude Code Opus, ~10s) — with split UI panels, differentiated WebSocket messages, and 1-click auto-start that spawns Claude Code CLI in background.

**Architecture:** Backend differentiates coaching sources via `coaching_nano` / `coaching_opus` WebSocket message types. Frontend splits right panel into two sub-panels (Nano left, Opus right) with show/hide logic. On session start, server auto-spawns a `claude` CLI subprocess that watches the live transcript file and pushes deep coaching via POST `/api/coach/push`. Two Coach buttons in footer trigger each path independently.

**Tech Stack:** Python (FastAPI, subprocess), JavaScript (vanilla), CSS Grid

---

## Task 1: Differentiate WebSocket coaching message types

**Files:**
- Modify: `src/meeting_transcriber/server.py`

**Steps:**

**Step 1: Update `/api/coach` to emit `coaching_nano` type**

In `server.py`, change the coach endpoint's WebSocket queue message:

```python
# In coach_me():
session["_ws_queue"].append({"type": "coaching_nano", "text": strategy})
```

Also update the return to include `source`:
```python
return JSONResponse({"strategy": strategy, "source": "nano"})
```

**Step 2: Update `/api/coach/push` to emit `coaching_opus` type**

```python
# In push_coaching():
session["_ws_queue"].append({"type": "coaching_opus", "text": text})
```

Remove the `[{source}]\n` prefix wrapping — the UI will handle labeling.

**Step 3: Run tests**

Run: `cd ~/GitHub/meeting-transcriber && uv run pytest tests/ -v -x`
Expected: All pass (existing tests don't check coaching message types)

**Step 4: Commit**

```bash
git add src/meeting_transcriber/server.py
git commit -m "feat: differentiate coaching WebSocket types (nano vs opus)"
```

---

## Task 2: Split frontend coaching panel into two sub-panels

**Files:**
- Modify: `src/meeting_transcriber/static/index.html`
- Modify: `src/meeting_transcriber/static/style.css`

**Step 1: Update HTML — split coaching panel into two sub-panels**

Replace the single coaching panel in `index.html`:

```html
<section class="panel panel-coaching">
  <h2>Coaching Prompts</h2>
  <div id="coaching" class="scroll-area">
    <p class="placeholder">Press "Coach" to get strategy advice based on recent conversation.</p>
  </div>
</section>
```

With:

```html
<section class="panel panel-coaching">
  <div class="coaching-split">
    <div class="coaching-sub" id="coaching-nano-panel">
      <h2>Quick Coach <span class="coaching-tag tag-nano">Nano</span></h2>
      <div id="coaching-nano" class="scroll-area">
        <p class="placeholder">Press "Quick Coach" for fast strategy tips.</p>
      </div>
    </div>
    <div class="coaching-sub" id="coaching-opus-panel">
      <h2>Deep Coach <span class="coaching-tag tag-opus">Opus</span></h2>
      <div id="coaching-opus" class="scroll-area">
        <p class="placeholder">Press "Deep Coach" for in-depth analysis.</p>
      </div>
    </div>
  </div>
</section>
```

**Step 2: Add CSS for split coaching panels**

Add to `style.css`:

```css
/* --- Dual coaching split --- */
.coaching-split {
  display: flex;
  flex: 1;
  overflow: hidden;
}

.coaching-sub {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.coaching-sub + .coaching-sub {
  border-left: 1px solid var(--cream-300);
}

.coaching-sub h2 {
  padding: 8px 12px;
  font-size: 12px;
}

.coaching-tag {
  font-size: 10px;
  padding: 2px 6px;
  border-radius: 8px;
  font-weight: 500;
  text-transform: none;
  letter-spacing: 0;
}

.tag-nano {
  background: var(--accent-muted);
  color: var(--accent);
}

.tag-opus {
  background: rgba(13, 148, 136, 0.10);
  color: var(--teal);
}

/* When only one panel active, hide the other */
.coaching-sub.hidden {
  display: none;
}

/* Single panel gets full width */
.coaching-split:has(.coaching-sub.hidden) .coaching-sub:not(.hidden) {
  flex: 1;
}
```

**Step 3: Verify layout in browser**

Open `http://localhost:8765`, verify:
- Two sub-panels visible side-by-side in the right column
- Each has its own header with colored tag
- Placeholder text shows in each

**Step 4: Commit**

```bash
git add src/meeting_transcriber/static/index.html src/meeting_transcriber/static/style.css
git commit -m "feat: split coaching panel into Nano and Opus sub-panels"
```

---

## Task 3: Two Coach buttons in footer

**Files:**
- Modify: `src/meeting_transcriber/static/index.html`
- Modify: `src/meeting_transcriber/static/style.css`

**Step 1: Replace single Coach button with three buttons**

In `index.html`, replace:
```html
<button id="btn-coach" class="btn btn-coach" disabled onclick="coach()">Coach</button>
```

With:
```html
<button id="btn-coach-nano" class="btn btn-coach-nano" disabled onclick="coachNano()">Quick Coach</button>
<button id="btn-coach-opus" class="btn btn-coach-opus" disabled onclick="coachOpus()">Deep Coach</button>
<button id="btn-coach-both" class="btn btn-coach-both" disabled onclick="coachBoth()">Both</button>
```

The "Both" button fires Nano and Opus simultaneously — user can compare results side-by-side.

**Step 2: Add CSS for two coach buttons**

```css
.btn-coach-nano {
  background: var(--accent);
  color: #fff;
  font-size: 14px;
  padding: 8px 16px;
}

.btn-coach-opus {
  background: var(--teal);
  color: #fff;
  font-size: 14px;
  padding: 8px 16px;
}

.btn-coach-both {
  background: var(--brown-mid);
  color: #fff;
  font-size: 14px;
  padding: 8px 16px;
}
```

Remove the old `.btn-coach` style.

**Step 3: Commit**

```bash
git add src/meeting_transcriber/static/index.html src/meeting_transcriber/static/style.css
git commit -m "feat: add Quick Coach and Deep Coach buttons"
```

---

## Task 4: Wire up frontend JS for dual coaching

**Files:**
- Modify: `src/meeting_transcriber/static/app.js`

**Step 1: Update DOM references**

Replace:
```javascript
const btnCoach = document.getElementById("btn-coach");
```

With:
```javascript
const btnCoachNano = document.getElementById("btn-coach-nano");
const btnCoachOpus = document.getElementById("btn-coach-opus");
const btnCoachBoth = document.getElementById("btn-coach-both");
const coachingNanoEl = document.getElementById("coaching-nano");
const coachingOpusEl = document.getElementById("coaching-opus");
const coachingNanoPanel = document.getElementById("coaching-nano-panel");
const coachingOpusPanel = document.getElementById("coaching-opus-panel");
```

Remove old `coachingEl` reference.

**Step 2: Update `handleMessage()` switch**

Replace the `"coaching"` case:

```javascript
case "coaching_nano":
  appendCoachingNano(msg.text || "");
  break;
case "coaching_opus":
  appendCoachingOpus(msg.text || "");
  break;
```

**Step 3: Add panel-specific append functions**

```javascript
function appendCoachingNano(text) {
  showPanel("nano");
  clearPlaceholder(coachingNanoEl);
  const card = document.createElement("div");
  card.className = "coaching-card";
  card.innerHTML = `<div class="label">Quick Coach</div><div>${escapeHtml(text)}</div>`;
  coachingNanoEl.insertBefore(card, coachingNanoEl.firstChild);
}

function appendCoachingOpus(text) {
  showPanel("opus");
  clearPlaceholder(coachingOpusEl);
  const card = document.createElement("div");
  card.className = "coaching-card summary-card";
  card.innerHTML = `<div class="label">Deep Coach</div><div class="context-rendered">${renderMarkdown(text)}</div>`;
  coachingOpusEl.insertBefore(card, coachingOpusEl.firstChild);
}
```

**Step 4: Add show/hide panel logic**

```javascript
function showPanel(which) {
  // Show the target panel, keep the other visible too if it has content
  if (which === "nano") {
    coachingNanoPanel.classList.remove("hidden");
  } else {
    coachingOpusPanel.classList.remove("hidden");
  }
}
```

Both panels start visible (no hide until we see usage patterns).

**Step 5: Add `coachNano()` and `coachOpus()` functions**

```javascript
async function coachNano() {
  btnCoachNano.disabled = true;
  btnCoachNano.textContent = "Thinking...";
  try {
    const resp = await fetch("/api/coach", { method: "POST" });
    if (!resp.ok) {
      const err = await resp.json();
      alert(err.error || "No recent transcript");
      return;
    }
    // Result pushed via WebSocket as coaching_nano
  } finally {
    btnCoachNano.disabled = false;
    btnCoachNano.textContent = "Quick Coach";
  }
}

async function coachOpus() {
  btnCoachOpus.disabled = true;
  btnCoachOpus.textContent = "Thinking...";
  try {
    const resp = await fetch("/api/coach/opus", { method: "POST" });
    if (!resp.ok) {
      const err = await resp.json();
      alert(err.error || "Deep coaching unavailable");
      return;
    }
    // Result pushed via WebSocket as coaching_opus
  } finally {
    btnCoachOpus.disabled = false;
    btnCoachOpus.textContent = "Deep Coach";
  }
}
```

**Step 6: Add `coachBoth()` function**

```javascript
async function coachBoth() {
  // Fire both in parallel — user compares side-by-side
  coachNano();
  coachOpus();
}
```

**Step 7: Update `setRecordingUI()` to toggle all three buttons**

Replace `btnCoach.disabled = !recording;` with:
```javascript
btnCoachNano.disabled = !recording;
btnCoachOpus.disabled = !recording;
btnCoachBoth.disabled = !recording;
```

**Step 8: Update `showSummary()` to use Nano panel**

Replace `coachingEl` reference with `coachingNanoEl` in `showSummary()`.

**Step 9: Remove old `coach()` and `appendCoaching()` functions**

Delete the old single-path functions.

**Step 10: Commit**

```bash
git add src/meeting_transcriber/static/app.js
git commit -m "feat: wire up dual coaching JS — Nano and Opus buttons + panels"
```

---

## Task 5: Backend — Opus coaching endpoint with Claude Code CLI

**Files:**
- Create: `src/meeting_transcriber/opus_coach.py`
- Modify: `src/meeting_transcriber/server.py`

This is the hardest task. We spawn a `claude` CLI subprocess that receives a coaching prompt, and pipe the output back to the frontend via WebSocket.

**Step 1: Create `opus_coach.py`**

```python
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
```

**Step 2: Add `/api/coach/opus` endpoint to `server.py`**

```python
@app.post("/api/coach/opus")
async def coach_opus() -> JSONResponse:
  """On-demand deep coaching via Claude Code CLI (Opus)."""
  from meeting_transcriber.opus_coach import run_opus_coaching

  recent = session["transcript_chunks"][-10:] if session["transcript_chunks"] else []
  recent_text = "\n".join(recent)
  if not recent_text.strip():
    return JSONResponse(
      {"error": "No recent transcript to analyze"},
      status_code=400,
    )

  playbook_text = "\n\n".join(session["context"]) if session["context"] else ""

  def on_result(text: str) -> None:
    session["_ws_queue"].append({"type": "coaching_opus", "text": text})

  run_opus_coaching(recent_text, playbook_text, on_result)

  return JSONResponse({"status": "processing", "source": "opus"})
```

**Step 3: Run tests**

Run: `cd ~/GitHub/meeting-transcriber && uv run pytest tests/ -v -x`

**Step 4: Commit**

```bash
git add src/meeting_transcriber/opus_coach.py src/meeting_transcriber/server.py
git commit -m "feat: add Opus deep coaching via Claude Code CLI subprocess"
```

---

## Task 6: Tests for opus_coach.py

**Files:**
- Create: `tests/test_opus_coach.py`

**Step 1: Write tests**

```python
"""Tests for Opus deep coaching via Claude Code CLI."""

import subprocess
from unittest.mock import MagicMock, patch

from meeting_transcriber.opus_coach import _build_prompt, _run_claude


class TestBuildPrompt:
  def test_includes_transcript(self):
    prompt = _build_prompt("hello world", "")
    assert "hello world" in prompt

  def test_includes_playbook_when_provided(self):
    prompt = _build_prompt("transcript", "playbook content")
    assert "playbook content" in prompt

  def test_excludes_playbook_section_when_empty(self):
    prompt = _build_prompt("transcript", "")
    assert "## Playbook" not in prompt

  def test_truncates_long_playbook(self):
    long_playbook = "x" * 5000
    prompt = _build_prompt("transcript", long_playbook)
    # Playbook truncated to 4000 chars
    assert len(prompt) < 5000 + 500  # prompt overhead


class TestRunClaude:
  @patch("meeting_transcriber.opus_coach.subprocess.run")
  def test_calls_claude_cli(self, mock_run):
    mock_run.return_value = subprocess.CompletedProcess(
      args=[], returncode=0, stdout="Analysis result", stderr=""
    )
    callback = MagicMock()
    _run_claude("transcript", "playbook", callback)
    mock_run.assert_called_once()
    args = mock_run.call_args[0][0]
    assert args[0] == "claude"
    assert "-p" in args

  @patch("meeting_transcriber.opus_coach.subprocess.run")
  def test_callback_receives_output(self, mock_run):
    mock_run.return_value = subprocess.CompletedProcess(
      args=[], returncode=0, stdout="Deep analysis", stderr=""
    )
    callback = MagicMock()
    _run_claude("transcript", "", callback)
    callback.assert_called_once_with("Deep analysis")

  @patch("meeting_transcriber.opus_coach.subprocess.run")
  def test_handles_cli_error(self, mock_run):
    mock_run.return_value = subprocess.CompletedProcess(
      args=[], returncode=1, stdout="", stderr="API error"
    )
    callback = MagicMock()
    _run_claude("transcript", "", callback)
    callback.assert_called_once()
    assert "error" in callback.call_args[0][0].lower()

  @patch("meeting_transcriber.opus_coach.subprocess.run")
  def test_handles_timeout(self, mock_run):
    mock_run.side_effect = subprocess.TimeoutExpired(cmd="claude", timeout=30)
    callback = MagicMock()
    _run_claude("transcript", "", callback)
    callback.assert_called_once()
    assert "timed out" in callback.call_args[0][0].lower()

  @patch("meeting_transcriber.opus_coach.subprocess.run")
  def test_handles_missing_cli(self, mock_run):
    mock_run.side_effect = FileNotFoundError()
    callback = MagicMock()
    _run_claude("transcript", "", callback)
    callback.assert_called_once()
    assert "not found" in callback.call_args[0][0].lower()
```

**Step 2: Run tests**

Run: `cd ~/GitHub/meeting-transcriber && uv run pytest tests/test_opus_coach.py -v`
Expected: All pass

**Step 3: Commit**

```bash
git add tests/test_opus_coach.py
git commit -m "test: add tests for Opus deep coaching module"
```

---

## Task 7: Update existing tests for new WebSocket message types

**Files:**
- Modify: `tests/test_streaming.py` (if it checks coaching message types)
- Modify: `tests/test_server.py` (if it exists)

**Step 1: Search for tests referencing old coaching type**

```bash
cd ~/GitHub/meeting-transcriber && grep -r '"coaching"' tests/
```

Update any assertions from `"coaching"` to `"coaching_nano"` or `"coaching_opus"` as appropriate.

**Step 2: Run full test suite**

Run: `cd ~/GitHub/meeting-transcriber && uv run pytest tests/ -v`
Expected: All pass

**Step 3: Commit**

```bash
git add tests/
git commit -m "test: update tests for dual coaching message types"
```

---

## Dependency Graph

```
Task 1 (WS message types) ─────────────┐
Task 2 (split coaching HTML/CSS) ───────┤
Task 3 (two coach buttons HTML/CSS) ────┼──→ Task 4 (wire up JS) ──→ Task 7 (fix tests)
                                        │
Task 5 (opus_coach.py + endpoint) ──────┘
Task 6 (tests for opus_coach) ── after Task 5
```

**Parallelizable:** Tasks 1, 2, 3 are independent. Task 5 is independent of 2/3.

---

## Verification Plan

```bash
# All tests pass
cd ~/GitHub/meeting-transcriber && uv run pytest tests/ -v --tb=short

# Start server
uv run mt serve --record --engine qwen --language zh

# Verify UI:
# 1. Two coaching sub-panels visible in right column
# 2. Two coach buttons in footer (Quick Coach gold, Deep Coach teal)
# 3. Both buttons disabled until recording starts
# 4. Press Start → both buttons enabled
# 5. Press Quick Coach → card appears in Nano panel (<1s)
# 6. Press Deep Coach → card appears in Opus panel (~5-10s)
# 7. Press Stop → both buttons disabled

# Verify Claude CLI available:
which claude  # Should print path
```

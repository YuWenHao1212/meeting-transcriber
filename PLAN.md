# Meeting Transcriber — Implementation Plan

> **Goal:** Build an open-source meeting assistant — CLI + Web UI for recording, transcribing (Chinese+English), real-time coaching prompts, and structured meeting notes, with a FLUX Vault skill layer for playbook generation and knowledge management integration.

**Architecture:** Three-layer design — Layer 1 is a standalone CLI (`mt`) installable via pip. Layer 2 is a local Web UI (`mt serve`) for real-time meeting experience. Layer 3 is a Claude Code skill (`/meeting`) that integrates with calendar, email, and vault knowledge. Layers are independent: CLI works alone, Web UI wraps CLI, Vault skill wraps both.

**Tech Stack:** Python 3.11+, Typer (CLI), FastAPI + WebSocket (Web UI), sounddevice (audio), multi-engine STT, Anthropic Claude (summarization + real-time prompts), uv (packaging)

---

## Context

Yu Wen-Hao 經常有線上（Google Meet）和線下會議，中英文混雜。目前沒有自動化的會議記錄流程。需求：

1. 會前：從行事曆 + email 自動產出 playbook
2. 會中：即時錄音 + 轉文字（30 秒 chunk，~33 秒延遲）
3. 會後：playbook + 逐字稿 → 完整會議記錄 + action items

週三 3/11 和 Vista 的會議是首次實測目標。

---

## STT Engine Selection — 中英混雜場景

| 排名 | 服務 | 串流成本/hr | 中英 code-switching | 串流方式 | 備註 |
|------|------|-----------|-------------------|---------|------|
| **1** | **Qwen3-ASR** | ~$0.40 | 原生支援，最強 | WebSocket | 22 種中文方言也行，阿里雲 Model Studio API |
| **2** | **Soniox** | $0.12 | mid-sentence 自動偵測 | WebSocket | 單一引擎搞定，最省事 |
| 3 | Gladia | $0.25 | 支援 | WebSocket | 功能全包（diarization 等） |
| 4 | OpenAI gpt-4o-transcribe | $0.36 | 有但非強項 | REST / Realtime API | 沒有更新的模型，Realtime API 很貴 |
| 5 | Groq whisper-v3-turbo | $0.04 | 弱 | chunked HTTP | 便宜但 code-switching 差，僅適合純英文 |

**決策：**
- MVP 預設引擎：**Qwen3-ASR**（中英混雜最強）
- 替代方案：**Soniox**（$0.12/hr，單引擎免路由）
- Fallback：**OpenAI gpt-4o-transcribe**（大家都有 key，降低上手門檻）
- 設計成 `--engine` flag：`qwen`（default）、`soniox`、`openai`、`groq`

**參考來源：**
- [Qwen3-ASR GitHub](https://github.com/QwenLM/Qwen3-ASR) — 1.7B 模型，30 語言，Apache 2.0
- [Soniox Chinese STT](https://soniox.com/speech-to-text/chinese)
- [Deepgram Nova-3](https://deepgram.com/learn/introducing-nova-3-speech-to-text-api) — 中文 code-switching 尚未確認支援
- [STT APIs 2026 Benchmark Guide](https://futureagi.substack.com/p/speech-to-text-apis-in-2026-benchmarks)

---

## Architecture

```
Layer 1: Standalone CLI (GitHub: meeting-transcriber)
┌──────────────────────────────────────────────────────┐
│  mt record   │ mt transcribe │ mt summarize │ mt live │
│  (audio)     │ (multi-engine)│ (claude API) │(combo)  │
├──────────────┴───────────────┴──────────────┴─────────┤
│  recorder.py │ transcriber.py │ summarizer.py         │
│  chunker.py  │ engines/       │ config.py             │
│              │  ├ base.py     │ formats.py            │
│              │  ├ qwen.py     │ models.py             │
│              │  ├ openai.py   │                       │
│              │  ├ soniox.py   │                       │
│              │  └ groq.py     │                       │
└──────────────────────────────────────────────────────┘

Layer 2: Web UI (mt serve)
┌──────────────────────────────────────────────────────┐
│  FastAPI + WebSocket                                 │
│  ┌─────────────┬──────────────┬───────────────────┐  │
│  │ Playbook    │ Real-time    │ Coaching Prompts  │  │
│  │ Sidebar     │ Transcript   │ (Haiku detection) │  │
│  │             │              │                   │  │
│  │ 載入 context│ 每 30s chunk │ 偵測問題/關鍵字   │  │
│  │ (playbook,  │ → STT → 顯示│ → 比對 context    │  │
│  │  resume,    │              │ → 顯示提示卡片    │  │
│  │  cheat sheet│ Action Items │                   │  │
│  │  )          │ 即時擷取     │ Post-meeting:     │  │
│  │             │              │ 一鍵 summarize    │  │
│  └─────────────┴──────────────┴───────────────────┘  │
│  server.py │ prompter.py │ static/ (HTML/JS/CSS)    │
└──────────────────────────────────────────────────────┘

Layer 3: FLUX Vault Skill (.claude/skills/meeting/)
┌──────────────────────────────────────────────────────┐
│  meeting prep          │  meeting notes              │
│  cal_ops + email_ops   │  transcript + playbook      │
│  → playbook.md         │  → meeting_notes.md         │
│                        │  → FLUX Vault auto-save     │
└──────────────────────────────────────────────────────┘

Data Flow:
  Pre:  Calendar + Email → [meeting prep] → playbook.md
  Live: Mic/System Audio → [mt serve / mt live] → real-time transcript
        Each chunk → Haiku question detection → context match → coaching prompt
  Post: transcript + playbook → [mt summarize] → meeting_notes.md
        Web UI: one-click save to FLUX Vault
```

### Web UI Layout（`mt serve`）

```
┌─────────────────────────────────────────────────────────────┐
│  Meeting Transcriber                    [■ Recording] 01:23 │
├──────────────┬──────────────────────────┬───────────────────┤
│              │                          │                   │
│  PLAYBOOK    │  TRANSCRIPT              │  COACHING         │
│              │                          │                   │
│  ## 會議目標 │  [00:00] 好，我們開始吧  │  💡 報價建議      │
│  - 確認報價  │  [00:32] 先聊一下需求    │                   │
│  - 了解時程  │  [01:05] 這個案子預算    │  對方問到預算，    │
│              │    大概多少？            │  playbook 建議：   │
│  ## 提問清單 │  [01:15] ...             │  「先了解範圍再    │
│  - 預算範圍？│                          │    報價，區間      │
│  - 決策者？  │  ── Action Items ──      │    $50-80K」       │
│  - Timeline? │  □ 下週一前寄報價單      │                   │
│              │  □ 確認技術需求規格      │  ── 歷史提示 ──   │
│              │                          │  (可收合)          │
├──────────────┴──────────────────────────┴───────────────────┤
│  [⏹ Stop] [📝 Summarize] [💾 Save to Vault]               │
└─────────────────────────────────────────────────────────────┘
```

---

## Phase 1: MVP CLI (GitHub Repo)

### Task 1.1: Scaffold repo

**Files:**
- Create: `~/GitHub/meeting-transcriber/pyproject.toml`
- Create: `~/GitHub/meeting-transcriber/src/meeting_transcriber/__init__.py`
- Create: `~/GitHub/meeting-transcriber/src/meeting_transcriber/cli.py` (skeleton)
- Create: `~/GitHub/meeting-transcriber/LICENSE` (MIT)
- Create: `~/GitHub/meeting-transcriber/.gitignore`
- Create: `~/GitHub/meeting-transcriber/.env.example`
- Create: `~/GitHub/meeting-transcriber/README.md` (minimal)

**Steps:**
1. `mkdir -p ~/GitHub/meeting-transcriber/src/meeting_transcriber`
2. `mkdir -p ~/GitHub/meeting-transcriber/tests/fixtures`
3. Write `pyproject.toml` with uv + hatchling, dependencies: typer, openai, anthropic, sounddevice, python-dotenv, platformdirs, rich, soundfile
4. Write `cli.py` skeleton with 4 empty Typer commands
5. Write MIT LICENSE, .gitignore (Python template), .env.example
6. `cd ~/GitHub/meeting-transcriber && git init && git add -A && git commit -m "chore: scaffold project"`
7. Create GitHub repo: `gh repo create meeting-transcriber --public --source=. --push`

**Verify:** `cd ~/GitHub/meeting-transcriber && uv sync && uv run mt --help` shows 4 commands

---

### Task 1.2: config.py

**Files:**
- Create: `src/meeting_transcriber/config.py`
- Create: `tests/test_config.py`

**Steps:**
1. Write test: load config from temp dir, env var override, missing keys raise helpful error
2. Run test → FAIL
3. Implement config.py:
   - `get_config_dir()` → `~/.config/meeting-transcriber/`
   - `load_config()` → load .env + config.toml, env vars override
   - `init_config()` → create config dir + .env template (for `mt setup`)
   - Config keys: `openai_api_key`, `anthropic_api_key`, `audio.device`, `transcription.language`, `transcription.engine` (default: `qwen`), `transcription.model`, `summary.model`
4. Run test → PASS
5. Commit

---

### Task 1.3: recorder.py

**Files:**
- Create: `src/meeting_transcriber/recorder.py`
- Create: `tests/test_recorder.py`

**Steps:**
1. Write test: mock sounddevice, verify WAV output (16kHz mono), start/stop lifecycle, file size > 0
2. Run test → FAIL
3. Implement recorder.py:
   - `class Recorder`: init with device_id, sample_rate=16000, channels=1
   - `start(output_path)`: open WAV via soundfile, start sounddevice.InputStream with callback
   - `stop()`: close stream + file, return Recording(path, duration, sample_rate)
   - Callback writes PCM data to WAV file (streaming, no memory accumulation)
   - `list_devices()`: return sounddevice.query_devices() formatted list
4. Run test → PASS
5. Commit

---

### Task 1.4: chunker.py

**Files:**
- Create: `src/meeting_transcriber/chunker.py`
- Create: `tests/test_chunker.py`

**Steps:**
1. Write test: create 90-second synthetic WAV, chunk into 30s segments, verify 3 chunks, each valid WAV
2. Run test → FAIL
3. Implement chunker.py:
   - `chunk_audio(input_path, chunk_duration=30, overlap=2)` → list of chunk file paths
   - Read WAV with soundfile, split by sample count
   - Each chunk includes 2-second overlap from previous chunk (for sentence boundary safety)
   - Write chunks to temp dir as WAV files
   - Return list of paths
4. Run test → PASS
5. Commit

---

### Task 1.5: transcriber.py — Multi-engine architecture (depends on 1.4)

**Files:**
- Create: `src/meeting_transcriber/engines/__init__.py`
- Create: `src/meeting_transcriber/engines/base.py`
- Create: `src/meeting_transcriber/engines/openai_engine.py`
- Create: `src/meeting_transcriber/engines/qwen.py` (Phase 3 placeholder)
- Create: `src/meeting_transcriber/engines/soniox.py` (Phase 3 placeholder)
- Create: `src/meeting_transcriber/engines/groq.py` (Phase 3 placeholder)
- Create: `src/meeting_transcriber/transcriber.py`
- Create: `tests/test_transcriber.py`

**Steps:**
1. Write test: mock openai client, verify API call format, language param, response parsing, cost calculation, engine selection
2. Run test → FAIL
3. Implement:
   - `engines/base.py`: `BaseEngine` ABC with `transcribe_file()` and `transcribe_chunks()` methods
   - `engines/openai_engine.py`: OpenAI implementation (gpt-4o-transcribe)
   - `engines/qwen.py`: placeholder `raise NotImplementedError("Qwen3-ASR engine coming in Phase 3")`
   - `engines/soniox.py`: placeholder
   - `engines/groq.py`: placeholder
   - `transcriber.py`: `get_engine(name)` factory, `TranscriptResult`, `Segment` dataclasses
   - `TranscriptResult`: `segments`, `full_text`, `duration`, `cost`, `engine`
   - Cost tracking per engine
   - Retry with exponential backoff (3 attempts)
4. Run test → PASS
5. Commit

---

### Task 1.6: summarizer.py

**Files:**
- Create: `src/meeting_transcriber/summarizer.py`
- Create: `tests/test_summarizer.py`

**Steps:**
1. Write test: mock anthropic client, verify prompt includes transcript, optional playbook injection, output is markdown
2. Run test → FAIL
3. Implement summarizer.py:
   - `summarize(transcript: str, playbook: str | None = None, template: str | None = None)` → str
   - System prompt: structured meeting notes (decisions, action items, key discussions, follow-ups)
   - If playbook provided: add "Cross-reference with pre-meeting objectives" instruction
   - Model: `claude-sonnet-4-20250514` (configurable)
   - Output: markdown string
4. Run test → PASS
5. Commit

---

### Task 1.7: formats.py

**Files:**
- Create: `src/meeting_transcriber/formats.py`
- Create: `tests/test_formats.py`

**Steps:**
1. Write test: TranscriptResult → markdown, with timestamps, with/without speakers
2. Run test → FAIL
3. Implement formats.py:
   - `transcript_to_markdown(result: TranscriptResult)` → str
   - `meeting_notes_header(title, date, duration, attendees)` → str
   - Format: timestamp + speaker (if available) + text, grouped by minute
4. Run test → PASS
5. Commit

---

### Task 1.8: cli.py — Wire up commands (depends on 1.2-1.7)

**Files:**
- Modify: `src/meeting_transcriber/cli.py`
- Create: `tests/test_cli.py`

**Steps:**
1. Write integration test: Typer CliRunner, mock all external deps, verify each command produces expected output
2. Implement 4 commands:

```
mt record [--device ID] [--output PATH]
  → Recorder.start() → Ctrl+C → Recorder.stop() → print path + duration

mt transcribe <file> [--language zh] [--engine qwen] [--output PATH]
  → chunker.chunk_audio() → engine.transcribe_chunks() → formats.transcript_to_markdown() → write file

mt summarize <transcript> [--playbook PATH] [--output PATH]
  → read transcript → summarizer.summarize() → write file

mt live [--device ID] [--output-dir PATH] [--language zh] [--engine qwen]
  → Start recording
  → Every 30s: chunk latest → transcribe → append to live transcript file
  → Show real-time cost in Rich console
  → Ctrl+C: stop, finalize, offer to summarize
```

3. Run tests → PASS
4. Commit

**Verify:** `uv run mt record --help`, `uv run mt live --help` all work

---

### Task 1.9: README + polish

**Files:**
- Modify: `~/GitHub/meeting-transcriber/README.md`

**Steps:**
1. Write comprehensive README:
   - Features, Quick Start, Installation (`pip install meeting-transcriber`)
   - Setup: API keys, BlackHole for system audio (with macOS setup instructions)
   - Engine comparison table (from STT Engine Selection section above)
   - Usage examples for all 4 commands
   - Cost estimate table per engine
   - Development section (uv sync, pytest)
2. Commit + push

---

## Phase 2: FLUX Vault Skill

### Task 2.1: Playbook + notes templates

**Files:**
- Create: `FLUX Vault/.claude/skills/meeting/references/playbook_template.md`
- Create: `FLUX Vault/.claude/skills/meeting/references/meeting_notes_template.md`

**Steps:**
1. Extract common structure from existing playbooks (arsh-discovery-call, studio-a)
2. Write playbook template with placeholders: `{{TITLE}}`, `{{DATE}}`, `{{ATTENDEES}}`, `{{PURPOSE}}`, `{{KNOWN_INFO}}`, `{{OBJECTIVES}}`, `{{QUESTIONS}}`
3. Write meeting notes template: Summary, Decisions, Action Items, Key Discussions, Follow-ups, Raw Transcript link
4. Commit

**Reference files:**
- `efforts/projects/active/openclaw/arsh-discovery-call-playbook.md`
- `efforts/areas/knowledge-monetization/courses/studio-a/meeting-playbook-2026-03-06.md`

---

### Task 2.2: meeting_ops.py

**Files:**
- Create: `FLUX Vault/.claude/skills/meeting/scripts/meeting_ops.py`

**Steps:**
1. Implement two commands:
   - `meeting_ops.py prep <date> [--meeting-title HINT]` → queries cal_ops.py + email_ops.py, outputs JSON context
   - `meeting_ops.py list-today` → list today's meetings from all calendar sources
2. This script gathers context; the actual playbook writing is done by Claude via SKILL.md workflow
3. Commit

---

### Task 2.3: SKILL.md

**Files:**
- Create: `FLUX Vault/.claude/skills/meeting/SKILL.md`

**Steps:**
1. Write YAML frontmatter: name, description (triggers: "meeting prep", "meeting playbook", "meeting notes", "會議準備", "會議筆記"), allowed-tools
2. Write two workflows:

**Workflow A: meeting prep**
1. Run `meeting_ops.py list-today` or query specific date
2. User selects meeting
3. Run `meeting_ops.py prep <date> --meeting-title <hint>` → get calendar + email context JSON
4. Search vault for related effort files (Grep attendee names in `efforts/`)
5. Read playbook template
6. Generate playbook with Claude (fill template + add AI-generated objectives/questions)
7. Save to `Calendar/Records/Meetings/YYYY-MM-DD-<slug>-playbook.md`

**Workflow B: meeting notes**
1. User provides transcript path (or auto-detect from today's recordings)
2. Auto-detect matching playbook from `Calendar/Records/Meetings/`
3. Run `mt summarize <transcript> --playbook <playbook>`
4. Enhance output: add `[[wikilinks]]` to related efforts, format action items
5. Save to `Calendar/Records/Meetings/YYYY-MM-DD-<slug>-notes.md`
6. Offer to extract action items → daily note or BACKLOG.md

3. Commit

---

### Task 2.4: Create Meetings directory + update CLAUDE.md

**Files:**
- Create: `FLUX Vault/Calendar/Records/Meetings/.gitkeep`
- Modify: `FLUX Vault/CLAUDE.md` — add `/meeting` to skills table

**Steps:**
1. Create directory
2. Add skill to CLAUDE.md skills table
3. Commit

---

## Phase 3A: STT Engine Implementation

### Task 3A.1: Qwen3-ASR engine

**Files:**
- Modify: `src/meeting_transcriber/engines/qwen.py`
- Create: `tests/test_engine_qwen.py`

**Steps:**
1. Implement `QwenEngine(BaseEngine)` with Alibaba Cloud Model Studio API
2. WebSocket streaming: send audio chunks, receive partial transcripts
3. Native Chinese+English code-switching support
4. Cost tracking: ~$0.40/hr
5. Config keys: `DASHSCOPE_API_KEY` (Alibaba Cloud)
6. Write tests with mocked WebSocket
7. Commit

---

### Task 3A.2: Soniox engine

**Files:**
- Modify: `src/meeting_transcriber/engines/soniox.py`
- Create: `tests/test_engine_soniox.py`

**Steps:**
1. Implement `SonioxEngine(BaseEngine)` with Soniox API
2. WebSocket streaming with mid-sentence language auto-detect
3. Cost tracking: $0.12/hr
4. Config keys: `SONIOX_API_KEY`
5. Write tests with mocked WebSocket
6. Commit

---

### Task 3A.3: Groq engine

**Files:**
- Modify: `src/meeting_transcriber/engines/groq.py`
- Create: `tests/test_engine_groq.py`

**Steps:**
1. Implement `GroqEngine(BaseEngine)` with chunked HTTP upload
2. Uses whisper-v3-turbo model, best for pure English
3. Cost tracking: $0.04/hr
4. Config keys: `GROQ_API_KEY`
5. Write tests with mocked HTTP client
6. Commit

---

## Phase 3B: Web UI（`mt serve`）

### Real-time Coaching Prompt Architecture

```
每 30 秒 chunk:
  Audio chunk → STT engine → transcript text
                                  ↓
                    Question/Keyword Detection (Haiku, ~0.5s)
                    「這個案子預算大概多少？」→ 偵測到「預算」
                                  ↓
                    Context Matching (本地比對)
                    在 playbook/resume/cheat sheet 中搜尋「預算」
                                  ↓
                    Prompt Card (WebSocket push)
                    「報價建議：先了解範圍再報價，區間 $50-80K」
```

**兩種使用情境，同一架構：**

| 情境 | Context 來源 | Prompt 類型 |
|------|-------------|------------|
| 會議 | playbook.md（會議目標、提問清單、注意事項） | 回應建議、數據提醒、注意事項 |
| 面試 | resume.md + Q&A cheat sheet | 小抄提示、STAR 故事、數據佐證 |

### Task 3B.1: FastAPI server skeleton

**Files:**
- Create: `src/meeting_transcriber/server.py`
- Create: `src/meeting_transcriber/static/index.html`
- Create: `src/meeting_transcriber/static/app.js`
- Create: `src/meeting_transcriber/static/style.css`
- Create: `tests/test_server.py`

**Steps:**
1. FastAPI app with:
   - `GET /` → serve static HTML
   - `POST /api/start` → start recording session (accepts context files: playbook/resume/cheat sheet)
   - `POST /api/stop` → stop recording
   - `WebSocket /ws` → real-time transcript + coaching prompts push
2. Session state management (one active session at a time)
3. CLI integration: `mt serve [--port 8765] [--context playbook.md]`
4. Minimal frontend: three-column layout (playbook | transcript | coaching)
5. Write tests for API endpoints
6. Commit

**Dependencies:** `fastapi`, `uvicorn`, `websockets` (add to pyproject.toml)

---

### Task 3B.2: Real-time transcript streaming

**Files:**
- Modify: `src/meeting_transcriber/server.py`
- Modify: `src/meeting_transcriber/static/app.js`

**Steps:**
1. Backend: recording thread → every 30s chunk → STT engine → WebSocket push `{type: "transcript", text: "...", timestamp: "01:23"}`
2. Frontend: append transcript lines with timestamps, auto-scroll
3. Live cost tracking: WebSocket push `{type: "cost", total: 0.12, engine: "qwen"}`
4. Recording duration timer in header
5. Commit

---

### Task 3B.3: Coaching prompt engine（prompter.py）

**Files:**
- Create: `src/meeting_transcriber/prompter.py`
- Create: `tests/test_prompter.py`

**Steps:**
1. `load_context(paths: list[str])` → parse playbook/resume/cheat sheet into searchable chunks
2. `detect_question(transcript_chunk: str)` → call Haiku to extract questions/keywords
   - Model: `claude-haiku-4-5-20251001`（fast + cheap, ~0.5s latency）
   - Prompt: 「從這段逐字稿中提取對方提出的問題或關鍵議題，JSON format」
   - Return: `list[{question: str, keywords: list[str]}]`
3. `match_context(keywords: list[str], context_chunks)` → find relevant context passages
   - Simple keyword + fuzzy matching（不需要 embedding，本地即可）
4. `generate_prompt_card(question, matched_context)` → formatted coaching prompt
5. Wire into server: each transcript chunk → detect → match → push coaching card via WebSocket
6. Frontend: coaching cards with fade-in animation, collapsible history
7. Write tests for detection + matching logic
8. Commit

---

### Task 3B.4: Action item detection

**Files:**
- Modify: `src/meeting_transcriber/prompter.py`
- Modify: `src/meeting_transcriber/static/app.js`

**Steps:**
1. Add `detect_action_items(transcript_chunk: str)` → Haiku call to extract action items
   - 「從這段逐字稿中提取行動項目（誰、做什麼、什麼時候）」
2. WebSocket push `{type: "action_item", text: "...", owner: "...", deadline: "..."}`
3. Frontend: action items panel below transcript, checkbox format
4. Commit

---

### Task 3B.5: Post-meeting summarize + save

**Files:**
- Modify: `src/meeting_transcriber/server.py`
- Modify: `src/meeting_transcriber/static/app.js`

**Steps:**
1. `POST /api/summarize` → call summarizer.summarize() with full transcript + loaded context
2. Enhanced summarizer: cross-reference playbook objectives (✅ 已討論 / ❌ 未提及)
3. `POST /api/save` → save meeting notes to specified path
   - Default: `Calendar/Records/Meetings/YYYY-MM-DD-<slug>-notes.md`
   - Include `[[wikilinks]]` to related efforts
4. Frontend: "Summarize" button → show preview → "Save to Vault" button
5. Commit

---

## Phase 3C: Enhanced Summarizer

### Task 3C.1: Playbook cross-reference

**Files:**
- Modify: `src/meeting_transcriber/summarizer.py`
- Modify: `tests/test_summarizer.py`

**Steps:**
1. When playbook provided, add cross-reference section to output:
   ```
   ## Playbook 覆蓋率
   ✅ 確認報價範圍 — 在 [01:05] 討論
   ✅ 了解決策者 — 在 [03:22] 提到 CEO 拍板
   ❌ Timeline 確認 — 未討論到
   ```
2. Prompt instructs Claude to map each playbook objective to transcript evidence
3. Update tests
4. Commit

---

## Phase 3D: Polish

| Task | Description |
|------|-------------|
| 3D.1 | BlackHole system audio: detection in `mt setup`, Aggregate Device guide in README |
| 3D.2 | Speaker diarization: `--diarize` flag |
| 3D.3 | `mt setup` command: check dependencies, list audio devices, test API keys |
| 3D.4 | GitHub Actions CI: lint (ruff) + type check (mypy) + test (pytest) |
| 3D.5 | PyPI publish workflow |
| 3D.6 | Daily skill integration: `/daily` auto-detect today's meetings, suggest prep |
| 3D.7 | Glossary / terminology table: custom term mapping for consistent transcription |

---

## Dependency Graph

```
Phase 1 (Standalone CLI) — ✅ COMPLETE
  1.1 scaffold ─────┐
  1.2 config ───────┤
  1.3 recorder ─────┤
  1.4 chunker ──→ 1.5 transcriber (multi-engine) ─┐
  1.6 summarizer ───┤                              ├──→ 1.8 cli ──→ 1.9 README
  1.7 formats ──────┘                              │
                                                   └──→ 91 tests, 82% coverage

Phase 2 (FLUX Vault Skill) — ✅ COMPLETE
  2.1 templates ────┐
  Phase 1 complete ─┼──→ 2.2 meeting_ops ──→ 2.3 SKILL.md ──→ 2.4 vault setup

Phase 3A (STT Engines):
  3A.1 Qwen ──┐
  3A.2 Soniox ├── all independent, any order after Phase 1
  3A.3 Groq ──┘

Phase 3B (Web UI) — depends on Phase 1:
  3B.1 server skeleton ──→ 3B.2 transcript streaming ──→ 3B.3 coaching prompts
                                                          ↓
                                              3B.4 action items ──→ 3B.5 post-meeting save

Phase 3C (Enhanced Summarizer):
  3C.1 playbook cross-reference — depends on Phase 1

Phase 3D (Polish):
  All independent, any order
```

## Parallelization Strategy

**Phase 3A 全部可平行**：Qwen、Soniox、Groq 引擎互不依賴

**Phase 3B 部分可平行**：
- 3B.1 server skeleton 先做
- 3B.2 (transcript) 和 3B.3 (coaching) 可在 server 骨架上平行開發
- 3B.4 (action items) 依賴 3B.3 的 Haiku detection 架構
- 3B.5 (post-meeting) 最後整合

**Phase 3A + 3B + 3C 可跨 phase 平行**：三者互不依賴

---

## Verification Plan

### Phase 1 驗證 ✅
```bash
cd ~/GitHub/meeting-transcriber
uv sync
uv run pytest --cov                    # 91 tests pass, 82% coverage
uv run mt record --output test.wav     # Record 10 seconds, Ctrl+C
uv run mt transcribe test.wav          # Produces transcript.md
uv run mt summarize transcript.md      # Produces meeting_notes.md
uv run mt live                         # Record + transcribe in real-time
```

### Phase 2 驗證 ✅
- 在 Claude Code 中說 "meeting prep" → 產出 playbook
- 提供 transcript 路徑 → 說 "meeting notes" → 產出完整會議記錄
- 確認檔案正確存到 `Calendar/Records/Meetings/`

### Phase 3A 驗證
```bash
uv run mt transcribe test.wav --engine qwen    # Qwen3-ASR 引擎
uv run mt transcribe test.wav --engine soniox  # Soniox 引擎
uv run mt transcribe test.wav --engine groq    # Groq 引擎
```

### Phase 3B 驗證
```bash
uv run mt serve --context playbook.md          # 開啟 Web UI
# 瀏覽器開 http://localhost:8765
# 1. 左側顯示 playbook 內容
# 2. 按 Record → 即時逐字稿出現在中間
# 3. 對方問問題 → 右側出現 coaching prompt card
# 4. 按 Stop → Summarize → Save to Vault
```

### Phase 3C 驗證
```bash
uv run mt summarize transcript.md --playbook playbook.md
# 輸出包含「Playbook 覆蓋率」section，✅/❌ 標記每個目標
```

### 實戰測試
- 3/11 週三 Vista 會議：完整 prep → record → transcribe → notes 流程
- Web UI 實測：`mt serve --context playbook.md` → 即時 coaching 體驗

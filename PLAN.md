# Meeting Transcriber — Implementation Plan

> **Goal:** Build an open-source Python CLI tool for recording, transcribing (Chinese+English), and summarizing meetings, with a FLUX Vault skill layer for playbook generation and knowledge management integration.

**Architecture:** Two-layer design — Layer 1 is a standalone CLI (`mt`) installable via pip, zero FLUX Vault dependency. Layer 2 is a Claude Code skill (`/meeting`) that wraps the CLI and integrates with calendar, email, and vault knowledge. This separation makes the tool shareable while keeping personal workflow integration private.

**Tech Stack:** Python 3.11+, Typer (CLI), sounddevice (audio), multi-engine STT, Anthropic Claude (summarization), uv (packaging)

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
│              │  ├ qwen.py     │                       │
│              │  ├ openai.py   │                       │
│              │  ├ soniox.py   │                       │
│              │  └ groq.py     │                       │
└──────────────────────────────────────────────────────┘

Layer 2: FLUX Vault Skill (.claude/skills/meeting/)
┌──────────────────────────────────────────────────────┐
│  meeting prep          │  meeting notes              │
│  cal_ops + email_ops   │  transcript + playbook      │
│  → playbook.md         │  → meeting_notes.md         │
└──────────────────────────────────────────────────────┘

Data Flow:
  Pre:  Calendar + Email → [meeting prep] → playbook.md
  Live: Mic/System Audio → [mt live] → audio.wav + transcript.md
  Post: transcript + playbook → [mt summarize] → meeting_notes.md
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

## Phase 3: Polish + Advanced (Post-MVP)

| Task | Description |
|------|-------------|
| 3.1 | BlackHole system audio: detection in `mt setup`, Aggregate Device guide in README |
| 3.2 | Speaker diarization: `--diarize` flag using `gpt-4o-transcribe-diarize` |
| 3.3 | **Qwen3-ASR engine**: implement `engines/qwen.py` with WebSocket streaming via Alibaba Cloud API |
| 3.4 | **Soniox engine**: implement `engines/soniox.py` with WebSocket streaming |
| 3.5 | **Groq engine**: implement `engines/groq.py` with chunked HTTP |
| 3.6 | `mt setup` command: check dependencies, list audio devices, generate config, test API keys |
| 3.7 | GitHub Actions CI: lint (ruff) + type check (mypy) + test (pytest) |
| 3.8 | PyPI publish workflow |
| 3.9 | Daily skill integration: `/daily` Phase 2 auto-detect today's meetings, suggest prep |
| 3.10 | Glossary / terminology table: custom term mapping for consistent transcription |
| 3.11 | Real-time cost display in `mt live` Rich console |
| 3.12 | Multi-engine auto-routing: detect language per chunk and route to optimal engine |

---

## Dependency Graph

```
Phase 1 (Standalone CLI):
  1.1 scaffold ─────┐
  1.2 config ───────┤
  1.3 recorder ─────┤
  1.4 chunker ──→ 1.5 transcriber (multi-engine) ─┐
  1.6 summarizer ───┤                              ├──→ 1.8 cli ──→ 1.9 README
  1.7 formats ──────┘                              │
                                                   └──→ tests (alongside each)

Phase 2 (FLUX Vault Skill):
  2.1 templates ────┐
  Phase 1 complete ─┼──→ 2.2 meeting_ops ──→ 2.3 SKILL.md ──→ 2.4 vault setup

Phase 3 (Polish):
  All independent, any order after Phase 1
```

## Parallelization Strategy

**Phase 1 可平行的 tasks:**
- 1.1, 1.2, 1.3, 1.4, 1.6, 1.7 全部可同時開工
- 1.5 等 1.4 完成
- 1.8 等 1.2-1.7 全部完成

**Phase 2 可平行的 tasks:**
- 2.1 templates 可和 Phase 1 平行
- 2.2, 2.3 必須等 Phase 1

---

## Verification Plan

### Phase 1 驗證
```bash
cd ~/GitHub/meeting-transcriber
uv sync
uv run pytest --cov                    # All tests pass, coverage > 80%
uv run mt record --output test.wav     # Record 10 seconds, Ctrl+C
uv run mt transcribe test.wav          # Produces transcript.md
uv run mt summarize transcript.md      # Produces meeting_notes.md
uv run mt live                         # Record + transcribe in real-time
```

### Phase 2 驗證
- 在 Claude Code 中說 "meeting prep" → 產出 playbook
- 提供 transcript 路徑 → 說 "meeting notes" → 產出完整會議記錄
- 確認檔案正確存到 `Calendar/Records/Meetings/`

### 實戰測試
- 3/11 週三 Vista 會議：完整 prep → record → transcribe → notes 流程

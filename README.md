# Meeting Transcriber

Record, transcribe (Chinese+English code-switching), and summarize meetings from the command line.

<!-- Badges placeholder -->
<!-- [![PyPI](https://img.shields.io/pypi/v/meeting-transcriber)](https://pypi.org/project/meeting-transcriber/) -->
<!-- [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE) -->
<!-- [![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org) -->

## Features

- **`mt record`** — Record audio from microphone or system audio (via BlackHole)
- **`mt transcribe`** — Transcribe audio files with multi-engine STT support
- **`mt summarize`** — Generate structured meeting notes with Claude (decisions, action items, follow-ups)
- **`mt live`** — Record + transcribe in real-time with 30-second chunks and live cost tracking
- **Multi-engine architecture** — Switch between OpenAI, Qwen, Soniox, and Groq with `--engine`
- **Chinese+English code-switching** — Built for bilingual meetings where speakers mix languages mid-sentence

## Quick Start

```bash
pip install meeting-transcriber

# Create config template with API keys
mt init

# Record a meeting
mt record --output meeting.wav

# Transcribe it
mt transcribe meeting.wav --engine openai

# Generate meeting notes
mt summarize transcript.md
```

## Installation

**pip:**

```bash
pip install meeting-transcriber
```

**uv (recommended):**

```bash
uv pip install meeting-transcriber
```

**From source:**

```bash
git clone https://github.com/yuwenhao/meeting-transcriber.git
cd meeting-transcriber
uv sync
```

## Setup

### API Keys

Set the following environment variables or add them to `~/.config/meeting-transcriber/.env`:

```bash
OPENAI_API_KEY=sk-...       # Required for OpenAI transcription engine
ANTHROPIC_API_KEY=sk-ant-...  # Required for meeting summarization (Claude)
```

Run `mt init` to generate the config template at `~/.config/meeting-transcriber/`.

### macOS System Audio Capture (BlackHole)

To capture system audio (e.g., Google Meet, Zoom), install [BlackHole](https://existential.audio/blackhole/):

```bash
brew install blackhole-2ch
```

Then create an Aggregate Device in **Audio MIDI Setup** (Applications > Utilities):

1. Open Audio MIDI Setup
2. Click **+** > **Create Aggregate Device**
3. Check both your built-in output and **BlackHole 2ch**
4. Set this Aggregate Device as your system output
5. Use BlackHole as the input device for `mt record`:

```bash
mt record --device "BlackHole 2ch"
```

## STT Engine Comparison

| Engine | Cost/hr | Code-switching | Status |
|--------|---------|----------------|--------|
| **Qwen3-ASR** | ~$0.40 | Best (native support) | Coming Phase 3 |
| **Soniox** | $0.12 | Auto-detect mid-sentence | Coming Phase 3 |
| **OpenAI** (gpt-4o-transcribe) | $0.36 | OK | Available |
| **Groq** (whisper-v3-turbo) | $0.04 | Weak (English-only recommended) | Coming Phase 3 |

Select an engine with the `--engine` flag:

```bash
mt transcribe meeting.wav --engine openai   # Available now
mt transcribe meeting.wav --engine qwen     # Phase 3
mt transcribe meeting.wav --engine soniox   # Phase 3
mt transcribe meeting.wav --engine groq     # Phase 3
```

## Usage

### Record

```bash
# Record from default microphone
mt record

# Specify output path and audio device
mt record --output ./recordings/meeting.wav --device "BlackHole 2ch"
```

Press `Ctrl+C` to stop recording.

### Transcribe

```bash
# Transcribe with default engine
mt transcribe meeting.wav

# Specify language and engine
mt transcribe meeting.wav --language zh --engine openai

# Custom output path
mt transcribe meeting.wav --output transcript.md
```

### Summarize

```bash
# Generate meeting notes from transcript
mt summarize transcript.md

# Include a pre-meeting playbook for cross-referencing
mt summarize transcript.md --playbook playbook.md

# Custom output path
mt summarize transcript.md --output meeting-notes.md
```

### Live (Record + Transcribe)

```bash
# Real-time recording and transcription
mt live

# With options
mt live --device "BlackHole 2ch" --engine openai --language zh --output-dir ./meetings/
```

Press `Ctrl+C` to stop. The tool will finalize the transcript and offer to generate a summary.

## Cost Estimate

| Scenario | Engine | Cost |
|----------|--------|------|
| 1-hour meeting | OpenAI (gpt-4o-transcribe) | ~$0.36 |
| 1-hour meeting | Soniox | ~$0.12 |
| 1-hour meeting | Groq | ~$0.04 |
| Summarization | Claude Sonnet | ~$0.02 |

A typical 1-hour meeting costs **under $0.40** end-to-end with the OpenAI engine. Future Soniox and Groq engines will reduce this significantly.

## Development

```bash
# Clone and install with dev dependencies
git clone https://github.com/yuwenhao/meeting-transcriber.git
cd meeting-transcriber
uv sync --extra dev

# Run tests
pytest

# Run tests with coverage
pytest --cov

# Lint
ruff check src/ tests/

# Type check
mypy src/
```

## License

[MIT](LICENSE)

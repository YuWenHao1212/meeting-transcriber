# Meeting Transcriber

Record, transcribe (Chinese+English), and summarize meetings from CLI.

## Quick Start

```bash
pip install meeting-transcriber
mt --help
```

## Setup

```bash
cp .env.example .env
# Fill in your API keys
```

## Commands

- `mt record` — Record audio from microphone
- `mt transcribe` — Transcribe audio file
- `mt summarize` — Summarize transcript
- `mt live` — Record + transcribe in real-time

## Development

```bash
uv sync --extra dev
uv run pytest
```

## License

MIT

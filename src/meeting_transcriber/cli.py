"""CLI entry point for meeting-transcriber."""

import signal
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

load_dotenv()

app = typer.Typer(
  name="mt",
  help="Record, transcribe, and summarize meetings.",
  no_args_is_help=True,
)

console = Console()


@app.command()
def record(
  device: Annotated[Optional[int], typer.Option("--device", "-d", help="Audio device ID")] = None,
  output: Annotated[Optional[Path], typer.Option("--output", "-o", help="Output WAV path")] = None,
  list_devices: Annotated[bool, typer.Option("--list-devices", help="List audio devices")] = False,
) -> None:
  """Record audio from microphone or system audio."""
  if list_devices:
    _show_devices()
    return

  from meeting_transcriber.recorder import Recorder

  if output is None:
    from datetime import datetime

    output = Path(f"recording-{datetime.now().strftime('%Y%m%d-%H%M%S')}.wav")

  recorder = Recorder(device_id=device)
  recorder.start(output)
  console.print(f"[bold green]Recording...[/] Press Ctrl+C to stop. Output: {output}")

  def _stop(sig, frame):
    result = recorder.stop()
    console.print(f"\n[bold]Done.[/] Duration: {result.duration:.1f}s → {result.path}")
    sys.exit(0)

  signal.signal(signal.SIGINT, _stop)
  signal.pause()


@app.command()
def transcribe(
  file: Annotated[Path, typer.Argument(help="Audio file to transcribe")],
  language: Annotated[str, typer.Option("--language", "-l", help="Language code")] = "zh",
  engine: Annotated[str, typer.Option("--engine", "-e", help="STT engine")] = "openai",
  output: Annotated[
    Optional[Path],
    typer.Option("--output", "-o", help="Output markdown path"),
  ] = None,
) -> None:
  """Transcribe an audio file to markdown."""
  from meeting_transcriber.formats import transcript_to_markdown
  from meeting_transcriber.transcriber import transcribe as do_transcribe

  if not file.exists():
    console.print(f"[red]File not found:[/] {file}")
    raise typer.Exit(1)

  if output is None:
    output = file.with_suffix(".md")

  console.print(f"[bold]Transcribing[/] {file} with engine={engine}, language={language}...")
  result = do_transcribe(file, language=language, engine_name=engine)

  md = transcript_to_markdown(result)
  output.write_text(md, encoding="utf-8")

  console.print(f"[bold green]Done.[/] Duration: {result.duration:.1f}s, Cost: ${result.cost:.4f}")
  console.print(f"Output: {output}")


@app.command()
def summarize(
  transcript: Annotated[Path, typer.Argument(help="Transcript markdown file")],
  playbook: Annotated[
    Optional[Path],
    typer.Option("--playbook", "-p", help="Pre-meeting playbook"),
  ] = None,
  output: Annotated[Optional[Path], typer.Option("--output", "-o", help="Output path")] = None,
) -> None:
  """Summarize a transcript into structured meeting notes."""
  from meeting_transcriber.summarizer import summarize as do_summarize

  if not transcript.exists():
    console.print(f"[red]File not found:[/] {transcript}")
    raise typer.Exit(1)

  transcript_text = transcript.read_text(encoding="utf-8")
  playbook_text = None
  if playbook and playbook.exists():
    playbook_text = playbook.read_text(encoding="utf-8")

  if output is None:
    output = transcript.with_name(transcript.stem + "-notes.md")

  console.print(f"[bold]Summarizing[/] {transcript}...")
  result = do_summarize(transcript_text, playbook=playbook_text)

  output.write_text(result, encoding="utf-8")
  console.print(f"[bold green]Done.[/] Output: {output}")


@app.command()
def live(
  device: Annotated[Optional[int], typer.Option("--device", "-d", help="Audio device ID")] = None,
  output_dir: Annotated[
    Optional[Path],
    typer.Option("--output-dir", help="Output directory"),
  ] = None,
  language: Annotated[str, typer.Option("--language", "-l", help="Language code")] = "zh",
  engine: Annotated[str, typer.Option("--engine", "-e", help="STT engine")] = "openai",
  chunk_duration: Annotated[int, typer.Option("--chunk-duration", help="Seconds per chunk")] = 30,
) -> None:
  """Record and transcribe in real-time."""
  import threading
  from datetime import datetime

  from meeting_transcriber.chunker import chunk_audio
  from meeting_transcriber.engines import get_engine
  from meeting_transcriber.formats import transcript_to_markdown
  from meeting_transcriber.models import Segment, TranscriptResult
  from meeting_transcriber.recorder import Recorder

  if output_dir is None:
    output_dir = Path(".")
  output_dir.mkdir(parents=True, exist_ok=True)

  timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
  audio_path = output_dir / f"live-{timestamp}.wav"
  transcript_path = output_dir / f"live-{timestamp}.md"

  stt_engine = get_engine(engine)
  recorder = Recorder(device_id=device)
  recorder.start(audio_path)

  all_segments: list[Segment] = []
  total_cost = 0.0
  stop_event = threading.Event()

  console.print("[bold green]Live recording + transcription[/]")
  console.print(f"Engine: {engine}, Language: {language}, Chunk: {chunk_duration}s")
  console.print(f"Audio: {audio_path}")
  console.print(f"Transcript: {transcript_path}")
  console.print("Press Ctrl+C to stop.\n")

  def _transcribe_loop():
    nonlocal total_cost
    chunk_index = 0
    while not stop_event.is_set():
      stop_event.wait(timeout=chunk_duration)
      if stop_event.is_set():
        break
      try:
        chunks = chunk_audio(audio_path, chunk_duration=chunk_duration, overlap=2)
        if chunk_index < len(chunks):
          for i in range(chunk_index, len(chunks)):
            result = stt_engine.transcribe_file(chunks[i], language=language)
            offset = i * chunk_duration
            for seg in result.segments:
              all_segments.append(
                Segment(
                  start=seg.start + offset,
                  end=seg.end + offset,
                  text=seg.text,
                  speaker=seg.speaker,
                )
              )
            total_cost += result.cost
            console.print(f"[dim][chunk {i + 1}][/] {result.full_text[:80]}")
          chunk_index = len(chunks)
      except Exception as err:
        console.print(f"[yellow]Transcription error:[/] {err}")

  worker = threading.Thread(target=_transcribe_loop, daemon=True)
  worker.start()

  def _stop(sig, frame):
    stop_event.set()
    recording = recorder.stop()
    worker.join(timeout=5)

    # Final transcription pass
    final_result = TranscriptResult(
      segments=all_segments,
      full_text="\n".join(s.text for s in all_segments),
      duration=recording.duration,
      cost=total_cost,
      engine=engine,
    )

    md = transcript_to_markdown(final_result)
    transcript_path.write_text(md, encoding="utf-8")

    console.print("\n[bold green]Done.[/]")
    console.print(f"Duration: {recording.duration:.1f}s, Cost: ${total_cost:.4f}")
    console.print(f"Audio: {recording.path}")
    console.print(f"Transcript: {transcript_path}")
    sys.exit(0)

  signal.signal(signal.SIGINT, _stop)
  signal.pause()


@app.command(name="start")
def serve(
  port: Annotated[int, typer.Option("--port", "-p", help="Server port")] = 8765,
  context: Annotated[
    Optional[list[Path]],
    typer.Option("--context", "-c", help="Context files"),
  ] = None,
  engine: Annotated[str, typer.Option("--engine", "-e", help="STT engine")] = "qwen",
  language: Annotated[str, typer.Option("--language", "-l", help="Language code")] = "zh",
  record: Annotated[
    bool,
    typer.Option("--record/--no-record", "-r", help="Enable live recording and transcription"),
  ] = True,
  chunk_duration: Annotated[
    int,
    typer.Option("--chunk-duration", help="Seconds per transcription chunk"),
  ] = 5,
  stereo: Annotated[
    bool,
    typer.Option("--stereo", help="Stereo: split mic (ours) and system audio (theirs)"),
  ] = False,
) -> None:
  """Start the Web UI for real-time meeting transcription."""
  import uvicorn

  from meeting_transcriber.server import create_app

  web_app = create_app(
    context_paths=[str(p) for p in context] if context else [],
    engine_name=engine,
    language=language,
    record=record,
    chunk_duration=chunk_duration,
    stereo=stereo,
  )
  import webbrowser

  console.print("[bold green]Meeting Transcriber Web UI[/]")
  console.print(f"Open [link=http://localhost:{port}]http://localhost:{port}[/link]")
  webbrowser.open(f"http://localhost:{port}")
  uvicorn.run(web_app, host="0.0.0.0", port=port)


@app.command()
def setup() -> None:
  """Check dependencies, configure API keys, and list audio devices."""
  import os

  from meeting_transcriber.config import init_config

  console.print("[bold]Meeting Transcriber Setup[/]\n")

  # Step 1: Create config directory and .env template
  config_dir = init_config()
  env_file = config_dir / ".env"
  console.print(f"[green]\u2713[/] Config directory: {config_dir}")
  console.print(f"  .env file: {env_file}\n")

  # Step 2: Check API keys
  console.print("[bold]API Keys[/]")
  api_keys = [
    ("OPENAI_API_KEY", "OpenAI (transcription)"),
    ("ANTHROPIC_API_KEY", "Anthropic (summarization)"),
  ]
  all_keys_set = True
  for env_var, label in api_keys:
    value = os.environ.get(env_var, "")
    if value:
      masked = value[:4] + "..." + value[-4:]
      console.print(f"  [green]\u2713[/] {label}: {masked}")
    else:
      console.print(f"  [red]\u2717[/] {label}: not set")
      all_keys_set = False

  if not all_keys_set:
    console.print(f"\n  [dim]Set keys in {env_file} or as environment variables.[/]")
  console.print()

  # Step 3: List audio devices and check BlackHole
  console.print("[bold]Audio Devices[/]")
  try:
    from meeting_transcriber.recorder import list_devices

    devices = list_devices()
    _show_devices()

    blackhole_found = any("blackhole" in str(dev.get("name", "")).lower() for dev in devices)
    if blackhole_found:
      console.print("  [green]\u2713[/] BlackHole detected — system audio capture available.")
    else:
      console.print("  [yellow]\u2717[/] BlackHole not found.")
      console.print(
        "  [dim]Install BlackHole to capture system audio: https://existential.audio/blackhole/[/]"
      )
  except Exception as err:
    console.print(f"  [red]Could not list devices:[/] {err}")

  console.print()

  # Step 4: Summary
  console.print("[bold]Summary[/]")
  if all_keys_set:
    console.print("  [green]\u2713[/] All API keys configured.")
  else:
    console.print(
      "  [yellow]\u2717[/] Some API keys missing — transcription/summarization won't work.",
    )
  console.print(f"  Config: {env_file}")
  console.print("  [dim]Run [bold]mt setup[/bold] again after making changes.[/]")


def _show_devices() -> None:
  """Print a table of available audio devices."""
  from meeting_transcriber.recorder import list_devices

  devices = list_devices()
  table = Table(title="Audio Devices")
  table.add_column("ID", style="cyan")
  table.add_column("Name")
  table.add_column("Inputs", style="green")
  table.add_column("Outputs", style="blue")

  for i, dev in enumerate(devices):
    table.add_row(
      str(i),
      str(dev.get("name", "")),
      str(dev.get("max_input_channels", 0)),
      str(dev.get("max_output_channels", 0)),
    )
  console.print(table)


if __name__ == "__main__":
  app()

"""CLI entry point for meeting-transcriber."""

import typer

app = typer.Typer(
  name="mt",
  help="Record, transcribe, and summarize meetings.",
  no_args_is_help=True,
)


@app.command()
def record() -> None:
  """Record audio from microphone or system audio."""
  typer.echo("record: not yet implemented")
  raise typer.Exit(1)


@app.command()
def transcribe() -> None:
  """Transcribe an audio file using OpenAI Whisper."""
  typer.echo("transcribe: not yet implemented")
  raise typer.Exit(1)


@app.command()
def summarize() -> None:
  """Summarize a transcript into structured meeting notes."""
  typer.echo("summarize: not yet implemented")
  raise typer.Exit(1)


@app.command()
def live() -> None:
  """Record and transcribe in real-time."""
  typer.echo("live: not yet implemented")
  raise typer.Exit(1)


if __name__ == "__main__":
  app()

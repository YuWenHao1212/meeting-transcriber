"""Tests for CLI commands."""

from unittest.mock import patch

from typer.testing import CliRunner

from meeting_transcriber.cli import app
from meeting_transcriber.models import TranscriptResult

runner = CliRunner()


class TestCliHelp:
  def test_help_shows_commands(self):
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "record" in result.output
    assert "transcribe" in result.output
    assert "summarize" in result.output
    assert "live" in result.output
    assert "setup" in result.output

  def test_record_help(self):
    result = runner.invoke(app, ["record", "--help"])
    assert result.exit_code == 0
    assert "--device" in result.output
    assert "--output" in result.output

  def test_transcribe_help(self):
    result = runner.invoke(app, ["transcribe", "--help"])
    assert result.exit_code == 0
    assert "--language" in result.output
    assert "--engine" in result.output

  def test_summarize_help(self):
    result = runner.invoke(app, ["summarize", "--help"])
    assert result.exit_code == 0
    assert "--playbook" in result.output

  def test_live_help(self):
    result = runner.invoke(app, ["live", "--help"])
    assert result.exit_code == 0
    assert "--engine" in result.output
    assert "--chunk-duration" in result.output


class TestTranscribeCommand:
  def test_file_not_found(self):
    result = runner.invoke(app, ["transcribe", "/nonexistent/file.wav"])
    assert result.exit_code == 1
    assert "not found" in result.output

  @patch("meeting_transcriber.transcriber.transcribe")
  @patch("meeting_transcriber.formats.transcript_to_markdown")
  def test_transcribe_produces_output(self, mock_fmt, mock_transcribe, tmp_path):
    audio = tmp_path / "test.wav"
    audio.write_bytes(b"fake audio")

    mock_transcribe.return_value = TranscriptResult(
      full_text="Hello world",
      duration=10.0,
      cost=0.001,
      engine="openai",
    )
    mock_fmt.return_value = "# Transcript\n\nHello world"

    result = runner.invoke(
      app,
      [
        "transcribe",
        str(audio),
        "--language",
        "zh",
        "--engine",
        "openai",
      ],
    )

    assert result.exit_code == 0
    assert "Done" in result.output
    assert "10.0s" in result.output

    output_md = audio.with_suffix(".md")
    assert output_md.exists()

  @patch("meeting_transcriber.transcriber.transcribe")
  @patch("meeting_transcriber.formats.transcript_to_markdown")
  def test_transcribe_custom_output(self, mock_fmt, mock_transcribe, tmp_path):
    audio = tmp_path / "test.wav"
    audio.write_bytes(b"fake audio")
    out = tmp_path / "custom.md"

    mock_transcribe.return_value = TranscriptResult(
      full_text="test",
      duration=5.0,
      cost=0.0,
      engine="openai",
    )
    mock_fmt.return_value = "# Transcript"

    result = runner.invoke(
      app,
      [
        "transcribe",
        str(audio),
        "--output",
        str(out),
      ],
    )

    assert result.exit_code == 0
    assert out.exists()


class TestSummarizeCommand:
  def test_file_not_found(self):
    result = runner.invoke(app, ["summarize", "/nonexistent/transcript.md"])
    assert result.exit_code == 1
    assert "not found" in result.output

  @patch("meeting_transcriber.summarizer.summarize")
  def test_summarize_produces_output(self, mock_summarize, tmp_path):
    transcript = tmp_path / "transcript.md"
    transcript.write_text("Some meeting transcript", encoding="utf-8")

    mock_summarize.return_value = "# Meeting Notes\n\nSummary here."

    result = runner.invoke(app, ["summarize", str(transcript)])

    assert result.exit_code == 0
    assert "Done" in result.output

    notes = tmp_path / "transcript-notes.md"
    assert notes.exists()
    assert "Meeting Notes" in notes.read_text(encoding="utf-8")

  @patch("meeting_transcriber.summarizer.summarize")
  def test_summarize_with_playbook(self, mock_summarize, tmp_path):
    transcript = tmp_path / "transcript.md"
    transcript.write_text("transcript text", encoding="utf-8")
    playbook = tmp_path / "playbook.md"
    playbook.write_text("playbook text", encoding="utf-8")

    mock_summarize.return_value = "notes"

    result = runner.invoke(
      app,
      [
        "summarize",
        str(transcript),
        "--playbook",
        str(playbook),
      ],
    )

    assert result.exit_code == 0
    mock_summarize.assert_called_once_with("transcript text", playbook="playbook text")


class TestSetupCommand:
  @patch("meeting_transcriber.recorder.list_devices")
  @patch("meeting_transcriber.config.init_config")
  def test_setup_runs_without_error(self, mock_init, mock_list, tmp_path):
    mock_init.return_value = tmp_path
    (tmp_path / ".env").write_text("# empty")
    mock_list.return_value = [
      {"name": "MacBook Mic", "max_input_channels": 1, "max_output_channels": 0},
    ]
    result = runner.invoke(app, ["setup"])
    assert result.exit_code == 0
    assert "Meeting Transcriber Setup" in result.output
    mock_init.assert_called_once()

  @patch("meeting_transcriber.recorder.list_devices")
  @patch("meeting_transcriber.config.init_config")
  def test_setup_creates_config_directory(self, mock_init, mock_list, tmp_path):
    config_dir = tmp_path / "mt-config"
    config_dir.mkdir()
    (config_dir / ".env").write_text("# empty")
    mock_init.return_value = config_dir
    mock_list.return_value = []
    result = runner.invoke(app, ["setup"])
    assert result.exit_code == 0
    assert "mt-config" in result.output

  @patch("meeting_transcriber.recorder.list_devices")
  @patch("meeting_transcriber.config.init_config")
  def test_setup_shows_device_listing(self, mock_init, mock_list, tmp_path):
    mock_init.return_value = tmp_path
    (tmp_path / ".env").write_text("# empty")
    mock_list.return_value = [
      {"name": "MacBook Mic", "max_input_channels": 1, "max_output_channels": 0},
      {"name": "BlackHole 2ch", "max_input_channels": 2, "max_output_channels": 2},
    ]
    result = runner.invoke(app, ["setup"])
    assert result.exit_code == 0
    assert "MacBook Mic" in result.output
    assert "BlackHole" in result.output
    assert "BlackHole detected" in result.output

  @patch("meeting_transcriber.recorder.list_devices")
  @patch("meeting_transcriber.config.init_config")
  def test_setup_warns_missing_blackhole(self, mock_init, mock_list, tmp_path):
    mock_init.return_value = tmp_path
    (tmp_path / ".env").write_text("# empty")
    mock_list.return_value = [
      {"name": "MacBook Mic", "max_input_channels": 1, "max_output_channels": 0},
    ]
    result = runner.invoke(app, ["setup"])
    assert result.exit_code == 0
    assert "BlackHole not found" in result.output

  @patch.dict(
    "os.environ",
    {"OPENAI_API_KEY": "sk-test1234abcd", "ANTHROPIC_API_KEY": ""},
    clear=False,
  )
  @patch("meeting_transcriber.recorder.list_devices")
  @patch("meeting_transcriber.config.init_config")
  def test_setup_shows_api_key_status(self, mock_init, mock_list, tmp_path):
    mock_init.return_value = tmp_path
    (tmp_path / ".env").write_text("# empty")
    mock_list.return_value = []
    result = runner.invoke(app, ["setup"])
    assert result.exit_code == 0
    assert "sk-t...abcd" in result.output
    assert "not set" in result.output
    assert "Some API keys missing" in result.output


class TestRecordListDevices:
  @patch("meeting_transcriber.recorder.list_devices")
  def test_list_devices_flag(self, mock_list):
    mock_list.return_value = [
      {"name": "MacBook Mic", "max_input_channels": 1, "max_output_channels": 0},
      {"name": "BlackHole 2ch", "max_input_channels": 2, "max_output_channels": 2},
    ]
    result = runner.invoke(app, ["record", "--list-devices"])
    assert result.exit_code == 0
    assert "MacBook Mic" in result.output
    assert "BlackHole" in result.output

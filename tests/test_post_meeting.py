"""Tests for post-meeting summarize, save, and export endpoints."""

from unittest.mock import patch

from starlette.testclient import TestClient

from meeting_transcriber.server import create_app

SAMPLE_CHUNKS = ["Alice: Let's discuss Q2.", "Bob: Sounds good."]
SAMPLE_CONTEXT = ["## Playbook\n1. Discuss Q2\n2. Assign owners"]
MOCK_SUMMARY = "## Meeting Summary\n\nDiscussed Q2 roadmap."


def _app_with_transcript(
  chunks: list[str] | None = None,
  context: list[str] | None = None,
  action_items: list[str] | None = None,
) -> tuple:
  """Create an app with pre-populated session data."""
  app = create_app()
  session = app.state.session
  session["transcript_chunks"] = list(SAMPLE_CHUNKS) if chunks is None else chunks
  if context is not None:
    session["context"] = context
  if action_items is not None:
    session["action_items"] = action_items
  return app, TestClient(app)


class TestSummarizeEndpoint:
  """Test /api/summarize calls the summarizer correctly."""

  @patch("meeting_transcriber.summarizer.summarize")
  def test_summarize_calls_summarizer(self, mock_summarize):
    mock_summarize.return_value = MOCK_SUMMARY
    _, client = _app_with_transcript()

    resp = client.post("/api/summarize")

    assert resp.status_code == 200
    mock_summarize.assert_called_once()
    call_args = mock_summarize.call_args
    # First positional arg is the joined transcript
    assert "Alice: Let's discuss Q2." in call_args[0][0]
    assert "Bob: Sounds good." in call_args[0][0]
    # No playbook when no context
    assert call_args[1]["playbook"] is None

  @patch("meeting_transcriber.summarizer.summarize")
  def test_summarize_returns_actual_summary(self, mock_summarize):
    mock_summarize.return_value = MOCK_SUMMARY
    _, client = _app_with_transcript()

    resp = client.post("/api/summarize")
    data = resp.json()

    assert data["summary"] == MOCK_SUMMARY

  @patch("meeting_transcriber.summarizer.summarize")
  def test_summarize_with_playbook_context(self, mock_summarize):
    mock_summarize.return_value = MOCK_SUMMARY
    _, client = _app_with_transcript(context=SAMPLE_CONTEXT)

    resp = client.post("/api/summarize")

    assert resp.status_code == 200
    call_args = mock_summarize.call_args
    assert call_args[1]["playbook"] is not None
    assert "Playbook" in call_args[1]["playbook"]

  @patch("meeting_transcriber.summarizer.summarize")
  def test_summarize_stores_in_session(self, mock_summarize):
    mock_summarize.return_value = MOCK_SUMMARY
    app, client = _app_with_transcript()

    client.post("/api/summarize")

    assert app.state.session["summary"] == MOCK_SUMMARY

  @patch("meeting_transcriber.summarizer.summarize")
  def test_summarize_queues_ws_message(self, mock_summarize):
    mock_summarize.return_value = MOCK_SUMMARY
    app, client = _app_with_transcript()

    client.post("/api/summarize")

    queue = app.state.session["_ws_queue"]
    summary_msgs = [m for m in queue if m["type"] == "summary"]
    assert len(summary_msgs) == 1
    assert summary_msgs[0]["text"] == MOCK_SUMMARY

  @patch("meeting_transcriber.summarizer.summarize")
  def test_summarize_returns_action_items(self, mock_summarize):
    mock_summarize.return_value = MOCK_SUMMARY
    items = ["Draft proposal by Friday", "Schedule follow-up"]
    _, client = _app_with_transcript(action_items=items)

    resp = client.post("/api/summarize")
    data = resp.json()

    assert data["action_items"] == items

  def test_summarize_no_transcript_returns_400(self):
    _, client = _app_with_transcript(chunks=[])
    resp = client.post("/api/summarize")
    assert resp.status_code == 400

  @patch("meeting_transcriber.summarizer.summarize")
  def test_summarize_with_multiple_context_files(self, mock_summarize):
    mock_summarize.return_value = MOCK_SUMMARY
    contexts = ["## Playbook A\n1. Item A", "## Playbook B\n1. Item B"]
    _, client = _app_with_transcript(context=contexts)

    client.post("/api/summarize")

    call_args = mock_summarize.call_args
    playbook = call_args[1]["playbook"]
    assert "Playbook A" in playbook
    assert "Playbook B" in playbook


class TestSaveEndpoint:
  """Test /api/save writes correct markdown format."""

  def test_save_writes_metadata_header(self, tmp_path):
    _, client = _app_with_transcript()
    out = tmp_path / "notes.md"

    resp = client.post("/api/save", json={"output_path": str(out)})

    assert resp.status_code == 200
    content = out.read_text(encoding="utf-8")
    assert "# Meeting Notes" in content
    assert "## Metadata" in content
    assert "**Engine**" in content
    assert "**Cost**" in content

  def test_save_includes_transcript(self, tmp_path):
    _, client = _app_with_transcript()
    out = tmp_path / "notes.md"

    client.post("/api/save", json={"output_path": str(out)})

    content = out.read_text(encoding="utf-8")
    assert "## Transcript" in content
    assert "Alice: Let's discuss Q2." in content
    assert "Bob: Sounds good." in content

  @patch("meeting_transcriber.summarizer.summarize")
  def test_save_includes_summary_when_available(self, mock_summarize, tmp_path):
    mock_summarize.return_value = MOCK_SUMMARY
    app, client = _app_with_transcript()

    # First summarize, then save
    client.post("/api/summarize")
    out = tmp_path / "notes.md"
    client.post("/api/save", json={"output_path": str(out)})

    content = out.read_text(encoding="utf-8")
    assert MOCK_SUMMARY in content

  def test_save_includes_action_items(self, tmp_path):
    items = ["Draft proposal", "Book room"]
    _, client = _app_with_transcript(action_items=items)
    out = tmp_path / "notes.md"

    client.post("/api/save", json={"output_path": str(out)})

    content = out.read_text(encoding="utf-8")
    assert "## Action Items" in content
    assert "Draft proposal" in content
    assert "Book room" in content

  def test_save_includes_context(self, tmp_path):
    _, client = _app_with_transcript(context=SAMPLE_CONTEXT)
    out = tmp_path / "notes.md"

    client.post("/api/save", json={"output_path": str(out)})

    content = out.read_text(encoding="utf-8")
    assert "## Playbook" in content

  def test_save_no_transcript_returns_400(self):
    _, client = _app_with_transcript(chunks=[])
    resp = client.post("/api/save", json={"output_path": "/tmp/out.md"})
    assert resp.status_code == 400


class TestExportEndpoint:
  """Test /api/export returns complete markdown."""

  def test_export_returns_markdown_and_filename(self):
    _, client = _app_with_transcript()

    resp = client.post("/api/export")

    assert resp.status_code == 200
    data = resp.json()
    assert "markdown" in data
    assert "suggested_filename" in data
    assert data["suggested_filename"].startswith("meeting-")
    assert data["suggested_filename"].endswith(".md")

  def test_export_markdown_contains_all_sections(self):
    _, client = _app_with_transcript(
      context=SAMPLE_CONTEXT,
      action_items=["Follow up"],
    )

    resp = client.post("/api/export")
    md = resp.json()["markdown"]

    assert "# Meeting Notes" in md
    assert "## Metadata" in md
    assert "## Playbook" in md
    assert "## Action Items" in md
    assert "## Transcript" in md

  @patch("meeting_transcriber.summarizer.summarize")
  def test_export_includes_summary(self, mock_summarize):
    mock_summarize.return_value = MOCK_SUMMARY
    _, client = _app_with_transcript()

    client.post("/api/summarize")
    resp = client.post("/api/export")
    md = resp.json()["markdown"]

    assert MOCK_SUMMARY in md

  def test_export_no_transcript_returns_400(self):
    _, client = _app_with_transcript(chunks=[])
    resp = client.post("/api/export")
    assert resp.status_code == 400

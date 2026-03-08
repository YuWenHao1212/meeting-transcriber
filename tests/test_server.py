"""Tests for the FastAPI web server."""

from pathlib import Path
from unittest.mock import patch

from starlette.testclient import TestClient

from meeting_transcriber.server import create_app


def _make_client() -> TestClient:
  """Create a fresh test client with isolated app."""
  app = create_app()
  return TestClient(app)


class TestIndexRoute:
  def test_index_returns_html(self):
    client = _make_client()
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


class TestStartStop:
  def test_start_returns_session_id(self):
    client = _make_client()
    resp = client.post("/api/start")
    assert resp.status_code == 200
    data = resp.json()
    assert "session_id" in data

  def test_start_when_already_recording_returns_409(self):
    client = _make_client()
    client.post("/api/start")
    resp = client.post("/api/start")
    assert resp.status_code == 409
    assert "Already recording" in resp.json()["error"]

  def test_stop_returns_duration(self):
    client = _make_client()
    client.post("/api/start")
    resp = client.post("/api/stop")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "stopped"
    assert "duration" in data

  def test_stop_when_not_recording_returns_404(self):
    client = _make_client()
    resp = client.post("/api/stop")
    assert resp.status_code == 404
    assert "No active session" in resp.json()["error"]


class TestStatus:
  def test_status_idle(self):
    client = _make_client()
    resp = client.get("/api/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["recording"] is False
    assert data["duration"] == 0.0
    assert data["cost"] == 0.0

  def test_status_while_recording(self):
    client = _make_client()
    client.post("/api/start")
    resp = client.get("/api/status")
    data = resp.json()
    assert data["recording"] is True
    assert data["duration"] >= 0.0


class TestContextLoading:
  def test_start_with_context_paths(self, tmp_path):
    ctx_file = tmp_path / "playbook.md"
    ctx_file.write_text("# Meeting Playbook\nAgenda items", encoding="utf-8")

    client = _make_client()
    resp = client.post(
      "/api/start",
      json={
        "context_paths": [str(ctx_file)],
      },
    )
    assert resp.status_code == 200

  def test_start_with_nonexistent_context_ignored(self):
    client = _make_client()
    resp = client.post(
      "/api/start",
      json={
        "context_paths": ["/nonexistent/file.md"],
      },
    )
    assert resp.status_code == 200


class TestSummarize:
  def test_summarize_no_transcript(self):
    client = _make_client()
    resp = client.post("/api/summarize")
    assert resp.status_code == 400

  @patch("meeting_transcriber.summarizer.summarize", return_value="Mock summary")
  def test_summarize_with_transcript(self, mock_summarize):
    app = create_app()
    app.state.session["transcript_chunks"] = ["chunk one", "chunk two"]
    client = TestClient(app)
    resp = client.post("/api/summarize")
    assert resp.status_code == 200
    assert "summary" in resp.json()
    assert resp.json()["summary"] == "Mock summary"


class TestSave:
  def test_save_no_transcript(self):
    client = _make_client()
    resp = client.post("/api/save", json={"output_path": "/tmp/out.md"})
    assert resp.status_code == 400

  def test_save_writes_file(self, tmp_path, monkeypatch):
    app = create_app()
    app.state.session["transcript_chunks"] = ["line 1", "line 2"]
    # Patch _get_save_directory to use tmp_path
    monkeypatch.setattr(
      "meeting_transcriber.server._get_save_directory",
      lambda session: tmp_path,
    )
    client = TestClient(app)
    resp = client.post("/api/save")
    assert resp.status_code == 200
    saved = Path(resp.json()["path"])
    assert saved.exists()
    assert "line 1" in saved.read_text(encoding="utf-8")


class TestWebSocket:
  def test_websocket_connect(self):
    client = _make_client()
    with client.websocket_connect("/ws") as ws:
      ws.send_text("ping")
      data = ws.receive_json()
      assert data["type"] == "status"
      assert "recording" in data


class TestFactoryIsolation:
  def test_separate_apps_have_independent_sessions(self):
    client_a = _make_client()
    client_b = _make_client()

    client_a.post("/api/start")
    resp_b = client_b.get("/api/status")
    assert resp_b.json()["recording"] is False

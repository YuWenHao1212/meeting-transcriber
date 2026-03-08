"""Tests for the real-time streaming pipeline (recording -> chunking -> WS)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from meeting_transcriber.models import TranscriptResult
from meeting_transcriber.server import (
  _new_session,
  _queue_context,
  _recording_loop,
  _run_prompter,
  _transcribe_new_chunks,
)


@pytest.fixture()
def session():
  """Return a fresh session dict."""
  return _new_session()


class TestNewSessionDefaults:
  def test_has_ws_queue(self, session):
    assert "_ws_queue" in session
    assert session["_ws_queue"] == []

  def test_has_thread_field(self, session):
    assert session["_thread"] is None

  def test_has_recorder_field(self, session):
    assert session["recorder"] is None


class TestQueueContext:
  def test_queues_context_messages(self, session):
    session["context"] = ["First doc", "Second doc"]
    _queue_context(session)
    assert len(session["_ws_queue"]) == 2
    assert session["_ws_queue"][0] == {"type": "context", "text": "First doc"}
    assert session["_ws_queue"][1] == {"type": "context", "text": "Second doc"}

  def test_empty_context_no_messages(self, session):
    _queue_context(session)
    assert session["_ws_queue"] == []


class TestTranscribeNewChunks:
  @patch("meeting_transcriber.server._run_prompter")
  def test_transcribes_and_queues(self, mock_prompter, session):
    mock_engine = MagicMock()
    mock_engine.transcribe_file.return_value = TranscriptResult(
      full_text="Hello world",
      duration=30.0,
      cost=0.05,
      engine="mock",
    )

    chunks = [Path("/tmp/c0.wav"), Path("/tmp/c1.wav")]
    _transcribe_new_chunks(session, mock_engine, chunks, 0, 30, "zh")

    assert len(session["transcript_chunks"]) == 2
    assert session["transcript_chunks"][0] == "Hello world"
    assert session["total_cost"] == pytest.approx(0.10)
    # 4 messages: 2 transcript + 2 cost
    assert len(session["_ws_queue"]) == 4
    assert session["_ws_queue"][0]["type"] == "transcript"
    assert session["_ws_queue"][0]["timestamp"] == "00:00"
    assert session["_ws_queue"][1]["type"] == "cost"
    assert session["_ws_queue"][2]["timestamp"] == "00:30"

  @patch("meeting_transcriber.server._run_prompter")
  def test_skips_already_processed_chunks(self, mock_prompter, session):
    mock_engine = MagicMock()
    mock_engine.transcribe_file.return_value = TranscriptResult(
      full_text="new chunk",
      duration=30.0,
      cost=0.01,
      engine="mock",
    )

    chunks = [Path("/tmp/c0.wav"), Path("/tmp/c1.wav"), Path("/tmp/c2.wav")]
    # Start from index 2 — only 1 new chunk
    _transcribe_new_chunks(session, mock_engine, chunks, 2, 30, "en")

    assert mock_engine.transcribe_file.call_count == 1
    assert len(session["transcript_chunks"]) == 1
    assert session["_ws_queue"][0]["timestamp"] == "01:00"

  @patch("meeting_transcriber.server._run_prompter")
  def test_timestamp_format(self, mock_prompter, session):
    mock_engine = MagicMock()
    mock_engine.transcribe_file.return_value = TranscriptResult(
      full_text="text",
      duration=30.0,
      cost=0.0,
      engine="mock",
    )
    chunks = [Path(f"/tmp/c{i}.wav") for i in range(5)]
    _transcribe_new_chunks(session, mock_engine, chunks, 0, 30, "zh")

    timestamps = [m["timestamp"] for m in session["_ws_queue"] if m["type"] == "transcript"]
    assert timestamps == ["00:00", "00:30", "01:00", "01:30", "02:00"]


class TestRecordingLoop:
  @patch("meeting_transcriber.chunker.chunk_audio")
  @patch("meeting_transcriber.engines.get_engine")
  @patch("meeting_transcriber.recorder.Recorder")
  def test_starts_recorder(self, mock_recorder_cls, mock_get_engine, mock_chunk, session):
    """Recording loop should call recorder.start() with a valid path."""
    mock_recorder = MagicMock()
    mock_recorder_cls.return_value = mock_recorder

    mock_engine = MagicMock()
    mock_get_engine.return_value = mock_engine

    session["active"] = True

    def stop_after_sleep(duration):
      session["active"] = False

    with patch("meeting_transcriber.server.time.sleep", side_effect=stop_after_sleep):
      _recording_loop(session, "openai", "zh", chunk_duration=30)

    mock_recorder.start.assert_called_once()
    call_path = mock_recorder.start.call_args[0][0]
    assert str(call_path).endswith(".wav")

  @patch("meeting_transcriber.chunker.chunk_audio")
  @patch("meeting_transcriber.engines.get_engine")
  @patch("meeting_transcriber.recorder.Recorder")
  def test_transcribes_chunks(self, mock_recorder_cls, mock_get_engine, mock_chunk, session):
    """Loop should transcribe chunks and populate ws_queue."""
    mock_recorder_cls.return_value = MagicMock()

    mock_engine = MagicMock()
    mock_engine.transcribe_file.return_value = TranscriptResult(
      full_text="transcribed text",
      duration=30.0,
      cost=0.02,
      engine="mock",
    )
    mock_get_engine.return_value = mock_engine
    mock_chunk.return_value = [Path("/tmp/chunk_000.wav")]

    session["active"] = True
    call_count = 0

    def sleep_then_stop(duration):
      nonlocal call_count
      call_count += 1
      if call_count >= 2:
        session["active"] = False

    with patch("meeting_transcriber.server.time.sleep", side_effect=sleep_then_stop):
      _recording_loop(session, "openai", "zh", chunk_duration=30)

    assert len(session["transcript_chunks"]) == 1
    assert session["total_cost"] == pytest.approx(0.02)
    transcript_msgs = [m for m in session["_ws_queue"] if m["type"] == "transcript"]
    assert len(transcript_msgs) == 1
    assert transcript_msgs[0]["text"] == "transcribed text"

  @patch("meeting_transcriber.chunker.chunk_audio")
  @patch("meeting_transcriber.engines.get_engine")
  @patch("meeting_transcriber.recorder.Recorder")
  def test_handles_chunk_error(self, mock_recorder_cls, mock_get_engine, mock_chunk, session):
    """Errors during chunking should be queued, not crash the loop."""
    mock_recorder_cls.return_value = MagicMock()
    mock_get_engine.return_value = MagicMock()
    mock_chunk.side_effect = RuntimeError("Cannot read file")

    session["active"] = True
    call_count = 0

    def sleep_then_stop(duration):
      nonlocal call_count
      call_count += 1
      if call_count >= 2:
        session["active"] = False

    with patch("meeting_transcriber.server.time.sleep", side_effect=sleep_then_stop):
      _recording_loop(session, "openai", "zh", chunk_duration=30)

    error_msgs = [m for m in session["_ws_queue"] if m["type"] == "error"]
    assert len(error_msgs) == 1
    assert "Cannot read file" in error_msgs[0]["text"]

  @patch("meeting_transcriber.chunker.chunk_audio")
  @patch("meeting_transcriber.engines.get_engine")
  @patch("meeting_transcriber.recorder.Recorder")
  def test_stores_audio_path(self, mock_recorder_cls, mock_get_engine, mock_chunk, session):
    """Session should have audio_path set after loop starts."""
    mock_recorder_cls.return_value = MagicMock()
    mock_get_engine.return_value = MagicMock()

    session["active"] = True

    def stop_immediately(duration):
      session["active"] = False

    with patch("meeting_transcriber.server.time.sleep", side_effect=stop_immediately):
      _recording_loop(session, "openai", "zh")

    assert session["audio_path"] is not None
    assert str(session["audio_path"]).endswith(".wav")


class TestStartStopWithRecording:
  """Integration tests for start/stop with mocked recording thread."""

  def test_start_spawns_thread_and_stop_joins(self):
    from starlette.testclient import TestClient

    with patch("meeting_transcriber.server._recording_loop"):
      app = __import__("meeting_transcriber.server", fromlist=["create_app"]).create_app(
        record=True
      )
      client = TestClient(app)

      resp = client.post("/api/start")
      assert resp.status_code == 200
      session = app.state.session
      assert session["_thread"] is not None

      resp = client.post("/api/stop")
      assert resp.status_code == 200
      assert session["active"] is False

  def test_stop_calls_recorder_stop(self):
    from starlette.testclient import TestClient

    mock_recorder = MagicMock()

    def fake_loop(session, engine_name, language, chunk_duration=30):
      session["recorder"] = mock_recorder

    with patch("meeting_transcriber.server._recording_loop", side_effect=fake_loop):
      app = __import__("meeting_transcriber.server", fromlist=["create_app"]).create_app(
        record=True
      )
      client = TestClient(app)

      client.post("/api/start")
      # Wait for thread to finish (fake_loop is synchronous)
      app.state.session["_thread"].join(timeout=2)
      client.post("/api/stop")

      mock_recorder.stop.assert_called_once()


class TestWebSocketBroadcast:
  def test_ws_receives_queued_messages(self):
    from starlette.testclient import TestClient

    with patch("meeting_transcriber.server._recording_loop"):
      app = __import__("meeting_transcriber.server", fromlist=["create_app"]).create_app()
      session = app.state.session
      # Pre-populate queue
      session["_ws_queue"].append({"type": "transcript", "timestamp": "00:00", "text": "hi"})
      session["_ws_queue"].append({"type": "cost", "value": 0.01})

      client = TestClient(app)
      with client.websocket_connect("/ws") as ws:
        # The WS loop will drain the queue, then wait for receive
        # Send a ping to trigger a loop iteration
        ws.send_text("ping")
        msg = ws.receive_json()
        # Could be either a queued message or the status echo
        assert msg["type"] in ("transcript", "cost", "status")

  def test_context_pushed_on_start(self, tmp_path):
    from starlette.testclient import TestClient

    ctx = tmp_path / "playbook.md"
    ctx.write_text("# Agenda\n1. Intro", encoding="utf-8")

    with patch("meeting_transcriber.server._recording_loop"):
      app = __import__("meeting_transcriber.server", fromlist=["create_app"]).create_app(
        context_paths=[str(ctx)]
      )
      session = app.state.session

      client = TestClient(app)
      client.post("/api/start")

      # Context is displayed client-side on upload, not pushed via WS on start
      context_msgs = [m for m in session["_ws_queue"] if m["type"] == "context"]
      assert len(context_msgs) == 0


class TestCoachingMessages:
  """Verify coaching messages are queued after transcription."""

  @patch("meeting_transcriber.prompter.detect_action_items", return_value=[])
  @patch("meeting_transcriber.prompter.detect_questions")
  def test_coaching_messages_queued(self, mock_detect_q, mock_detect_ai, session):
    from meeting_transcriber.prompter import ContextChunk, DetectedQuestion

    mock_detect_q.return_value = [
      DetectedQuestion(question="What is the timeline?", keywords=["timeline"]),
    ]
    context_chunks = [
      ContextChunk(text="Project timeline is Q2", source="plan.md", keywords=["timeline"]),
    ]

    _run_prompter(session, "What is the timeline?", context_chunks)

    coaching_msgs = [m for m in session["_ws_queue"] if m["type"] == "coaching_nano"]
    assert len(coaching_msgs) == 1
    assert "What is the timeline?" in coaching_msgs[0]["text"]
    assert "plan.md" in coaching_msgs[0]["text"]

  @patch("meeting_transcriber.prompter.detect_action_items", return_value=[])
  @patch("meeting_transcriber.prompter.detect_questions")
  def test_multiple_questions_produce_multiple_cards(
    self,
    mock_detect_q,
    mock_detect_ai,
    session,
  ):
    from meeting_transcriber.prompter import DetectedQuestion

    mock_detect_q.return_value = [
      DetectedQuestion(question="Q1?", keywords=["a"]),
      DetectedQuestion(question="Q2?", keywords=["b"]),
    ]

    _run_prompter(session, "some transcript", [])

    coaching_msgs = [m for m in session["_ws_queue"] if m["type"] == "coaching_nano"]
    assert len(coaching_msgs) == 2


class TestActionItemMessages:
  """Verify action items are queued after transcription."""

  @patch("meeting_transcriber.prompter.detect_questions", return_value=[])
  @patch("meeting_transcriber.prompter.detect_action_items")
  def test_action_items_queued(self, mock_detect_ai, mock_detect_q, session):
    from meeting_transcriber.prompter import ActionItem

    mock_detect_ai.return_value = [
      ActionItem(text="Send report", owner="Alice", deadline="Friday"),
    ]

    _run_prompter(session, "Alice will send the report by Friday", [])

    ai_msgs = [m for m in session["_ws_queue"] if m["type"] == "action_item"]
    assert len(ai_msgs) == 1
    assert ai_msgs[0]["text"] == "Send report"
    assert ai_msgs[0]["owner"] == "Alice"
    assert ai_msgs[0]["deadline"] == "Friday"

  @patch("meeting_transcriber.prompter.detect_questions", return_value=[])
  @patch("meeting_transcriber.prompter.detect_action_items")
  def test_action_items_stored_in_session(self, mock_detect_ai, mock_detect_q, session):
    from meeting_transcriber.prompter import ActionItem

    mock_detect_ai.return_value = [
      ActionItem(text="Review PR", owner=None, deadline=None),
    ]

    _run_prompter(session, "We need to review the PR", [])

    assert len(session["action_items"]) == 1
    assert session["action_items"][0]["text"] == "Review PR"
    assert session["action_items"][0]["owner"] is None

  @patch("meeting_transcriber.prompter.detect_questions", return_value=[])
  @patch("meeting_transcriber.prompter.detect_action_items")
  def test_multiple_action_items(self, mock_detect_ai, mock_detect_q, session):
    from meeting_transcriber.prompter import ActionItem

    mock_detect_ai.return_value = [
      ActionItem(text="Task 1", owner="A", deadline=None),
      ActionItem(text="Task 2", owner="B", deadline="Monday"),
    ]

    _run_prompter(session, "transcript", [])

    ai_msgs = [m for m in session["_ws_queue"] if m["type"] == "action_item"]
    assert len(ai_msgs) == 2
    assert len(session["action_items"]) == 2


class TestCoachingWithoutContext:
  """Coaching still works when no context is loaded."""

  @patch("meeting_transcriber.prompter.detect_action_items", return_value=[])
  @patch("meeting_transcriber.prompter.detect_questions")
  def test_no_context_still_produces_cards(self, mock_detect_q, mock_detect_ai, session):
    from meeting_transcriber.prompter import DetectedQuestion

    mock_detect_q.return_value = [
      DetectedQuestion(question="What about pricing?", keywords=["pricing"]),
    ]

    _run_prompter(session, "What about pricing?", [])

    coaching_msgs = [m for m in session["_ws_queue"] if m["type"] == "coaching_nano"]
    assert len(coaching_msgs) == 1
    # Card should indicate no matching context
    assert "No matching context found" in coaching_msgs[0]["text"]

  @patch("meeting_transcriber.prompter.detect_action_items", return_value=[])
  @patch("meeting_transcriber.prompter.detect_questions", return_value=[])
  def test_no_questions_no_coaching(self, mock_detect_q, mock_detect_ai, session):
    _run_prompter(session, "Just chatting", [])

    coaching_msgs = [m for m in session["_ws_queue"] if m["type"] == "coaching_nano"]
    assert len(coaching_msgs) == 0


class TestPrompterErrorResilience:
  """Prompter errors don't crash the streaming loop."""

  @patch("meeting_transcriber.prompter.detect_action_items", return_value=[])
  @patch(
    "meeting_transcriber.prompter.detect_questions",
    side_effect=RuntimeError("API down"),
  )
  def test_detect_questions_error_does_not_crash(
    self,
    mock_detect_q,
    mock_detect_ai,
    session,
  ):
    # Should not raise
    _run_prompter(session, "some text", [])

    # No coaching messages, but no crash
    coaching_msgs = [m for m in session["_ws_queue"] if m["type"] == "coaching_nano"]
    assert len(coaching_msgs) == 0

  @patch(
    "meeting_transcriber.prompter.detect_action_items",
    side_effect=RuntimeError("API down"),
  )
  @patch("meeting_transcriber.prompter.detect_questions", return_value=[])
  def test_detect_action_items_error_does_not_crash(
    self,
    mock_detect_q,
    mock_detect_ai,
    session,
  ):
    # Should not raise
    _run_prompter(session, "some text", [])

    ai_msgs = [m for m in session["_ws_queue"] if m["type"] == "action_item"]
    assert len(ai_msgs) == 0

  @patch(
    "meeting_transcriber.prompter.detect_action_items",
    side_effect=RuntimeError("API down"),
  )
  @patch(
    "meeting_transcriber.prompter.detect_questions",
    side_effect=RuntimeError("API down"),
  )
  def test_both_fail_does_not_crash(self, mock_detect_q, mock_detect_ai, session):
    # Should not raise even when both fail
    _run_prompter(session, "some text", [])
    assert session["_ws_queue"] == []

  def test_transcribe_with_prompter_error_still_queues_transcript(self, session):
    """Even if prompter fails, transcript + cost messages still appear."""
    mock_engine = MagicMock()
    mock_engine.transcribe_file.return_value = TranscriptResult(
      full_text="Hello",
      duration=30.0,
      cost=0.01,
      engine="mock",
    )

    with (
      patch(
        "meeting_transcriber.prompter.detect_questions",
        side_effect=RuntimeError("fail"),
      ),
      patch(
        "meeting_transcriber.prompter.detect_action_items",
        side_effect=RuntimeError("fail"),
      ),
    ):
      chunks = [Path("/tmp/c0.wav")]
      _transcribe_new_chunks(session, mock_engine, chunks, 0, 30, "zh")

    assert len(session["transcript_chunks"]) == 1
    transcript_msgs = [m for m in session["_ws_queue"] if m["type"] == "transcript"]
    assert len(transcript_msgs) == 1
    assert transcript_msgs[0]["text"] == "Hello"

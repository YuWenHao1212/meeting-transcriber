"""FastAPI server for Meeting Transcriber Web UI."""

import asyncio
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

STATIC_DIR = Path(__file__).parent / "static"

# Audio send interval for realtime streaming (seconds)
_REALTIME_SEND_INTERVAL = 0.1
# PCM16 16kHz mono: 16000 samples/sec * 2 bytes/sample * 0.1s = 3200 bytes
_REALTIME_CHUNK_BYTES = 3200


class StartRequest(BaseModel):
  context_paths: list[str] = []


class SaveRequest(BaseModel):
  output_path: str


_LIVE_TRANSCRIPT_PATH = Path("/tmp/mt-live-transcript.md")


def _new_session() -> dict[str, Any]:
  """Return a fresh session state dict."""
  return {
    "active": False,
    "start_time": None,
    "context": [],
    "context_paths": [],
    "context_chunks": [],
    "transcript_chunks": [],
    "action_items": [],
    "total_cost": 0.0,
    "summary": None,
    "ws_clients": set(),
    "_ws_queue": [],
    "_thread": None,
    "recorder": None,
    "audio_path": None,
    "_realtime_streamer": None,
  }


def _load_context_files(paths: list[str]) -> list[str]:
  """Read context files and return their contents."""
  chunks: list[str] = []
  for p in paths:
    path = Path(p)
    if path.exists() and path.is_file():
      chunks.append(path.read_text(encoding="utf-8"))
  return chunks


def _elapsed(session: dict[str, Any]) -> float:
  """Return elapsed seconds since recording started."""
  if not session["active"] or session["start_time"] is None:
    return 0.0
  return time.time() - session["start_time"]


def _recording_loop(
  session: dict[str, Any],
  engine_name: str,
  language: str,
  chunk_duration: int = 30,
) -> None:
  """Background thread: record audio, chunk, transcribe, queue results."""
  from meeting_transcriber.chunker import chunk_audio
  from meeting_transcriber.engines import get_engine
  from meeting_transcriber.recorder import Recorder

  timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
  audio_path = Path(tempfile.mkdtemp()) / f"live-{timestamp}.wav"

  try:
    engine = get_engine(engine_name)
  except Exception as e:
    print(f"[recording] Engine init error: {e}")
    session["_ws_queue"].append({"type": "error", "text": f"Engine error: {e}"})
    return

  try:
    recorder = Recorder()
    recorder.start(audio_path)
  except Exception as e:
    print(f"[recording] Recorder start error: {e}")
    session["_ws_queue"].append({"type": "error", "text": f"Recorder error: {e}"})
    return

  session["audio_path"] = audio_path
  session["recorder"] = recorder
  print(f"[recording] Started. Audio: {audio_path}, Engine: {engine_name}")

  chunk_index = 0
  while session["active"]:
    time.sleep(chunk_duration)
    if not session["active"]:
      break
    try:
      chunks = chunk_audio(audio_path, chunk_duration=chunk_duration, overlap=2)
      print(f"[recording] Chunks: {len(chunks)}, new from index {chunk_index}")
      _transcribe_new_chunks(
        session,
        engine,
        chunks,
        chunk_index,
        chunk_duration,
        language,
        context_chunks=session.get("context_chunks", []),
      )
      chunk_index = len(chunks)
    except Exception as e:
      print(f"[recording] Error: {e}")
      session["_ws_queue"].append({"type": "error", "text": str(e)})


def _recording_loop_realtime(
  session: dict[str, Any],
  language: str,
) -> None:
  """Background thread: stream PCM audio to Qwen3-ASR realtime WebSocket."""
  import numpy as np
  import sounddevice as sd

  from meeting_transcriber.engines.qwen_realtime import QwenRealtimeStreamer

  def on_partial(text: str) -> None:
    session["_ws_queue"].append(
      {
        "type": "transcript_partial",
        "text": text,
      }
    )

  def on_final(text: str) -> None:
    elapsed = _elapsed(session)
    mm = int(elapsed) // 60
    ss = int(elapsed) % 60
    timestamp_str = f"{mm:02d}:{ss:02d}"

    session["transcript_chunks"].append(text)
    # Estimate cost from elapsed time
    duration_min = elapsed / 60.0
    session["total_cost"] = duration_min * 0.0054

    session["_ws_queue"].append(
      {
        "type": "transcript",
        "timestamp": timestamp_str,
        "text": text,
      }
    )
    session["_ws_queue"].append(
      {
        "type": "cost",
        "value": session["total_cost"],
      }
    )

    # Append to live transcript file for Claude Code CLI
    try:
      with open(_LIVE_TRANSCRIPT_PATH, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp_str}] {text}\n")
    except Exception:
      pass

  def on_error(text: str) -> None:
    session["_ws_queue"].append({"type": "error", "text": text})

  # Initialize streamer
  streamer = QwenRealtimeStreamer(
    language=language,
    on_partial=on_partial,
    on_final=on_final,
    on_error=on_error,
  )
  session["_realtime_streamer"] = streamer

  try:
    streamer.start()
  except Exception as e:
    print(f"[realtime] Streamer start error: {e}")
    session["_ws_queue"].append({"type": "error", "text": f"Realtime error: {e}"})
    return

  print("[realtime] Started. Streaming PCM to Qwen3-ASR realtime.")

  # Accumulate PCM from sounddevice callback
  audio_buffer = bytearray()
  buffer_lock = threading.Lock()

  def audio_callback(
    indata: np.ndarray,
    frame_count: int,
    time_info: object,
    status: object,
  ) -> None:
    # Convert float32 [-1,1] to int16 PCM
    pcm = (indata * 32767).astype(np.int16).tobytes()
    with buffer_lock:
      audio_buffer.extend(pcm)

  try:
    stream = sd.InputStream(
      samplerate=16000,
      channels=1,
      dtype="float32",
      callback=audio_callback,
    )
    stream.start()
  except Exception as e:
    print(f"[realtime] Audio stream error: {e}")
    session["_ws_queue"].append({"type": "error", "text": f"Audio error: {e}"})
    streamer.stop()
    return

  # Send audio chunks at regular intervals
  try:
    while session["active"] and streamer.is_running:
      time.sleep(_REALTIME_SEND_INTERVAL)
      with buffer_lock:
        if len(audio_buffer) >= _REALTIME_CHUNK_BYTES:
          chunk = bytes(audio_buffer[:_REALTIME_CHUNK_BYTES])
          del audio_buffer[:_REALTIME_CHUNK_BYTES]
        elif len(audio_buffer) > 0:
          chunk = bytes(audio_buffer)
          audio_buffer.clear()
        else:
          continue
      streamer.send_audio(chunk)
  finally:
    stream.stop()
    stream.close()
    streamer.stop()
    session["_realtime_streamer"] = None


def _transcribe_new_chunks(
  session: dict[str, Any],
  engine: Any,
  chunks: list[Path],
  chunk_index: int,
  chunk_duration: int,
  language: str,
  context_chunks: list | None = None,
) -> None:
  """Transcribe chunks from chunk_index onward and queue WS messages."""
  for i in range(chunk_index, len(chunks)):
    result = engine.transcribe_file(chunks[i], language=language)
    offset_secs = i * chunk_duration
    mm = offset_secs // 60
    ss = offset_secs % 60
    timestamp_str = f"{mm:02d}:{ss:02d}"

    session["transcript_chunks"].append(result.full_text)
    session["total_cost"] += result.cost
    session["_ws_queue"].append(
      {
        "type": "transcript",
        "timestamp": timestamp_str,
        "text": result.full_text,
      }
    )
    session["_ws_queue"].append(
      {
        "type": "cost",
        "value": session["total_cost"],
      }
    )

    pass  # Coaching is now triggered manually via /api/coach


def _load_session_context_chunks(session: dict[str, Any]) -> None:
  """Load structured context chunks from session's context_paths."""
  if not session["context_paths"]:
    return
  from meeting_transcriber.prompter import load_context

  session["context_chunks"] = load_context(session["context_paths"])


def _run_prompter(
  session: dict[str, Any],
  transcript_text: str,
  context_chunks: list,
) -> None:
  """Run coaching prompt pipeline on transcript text.

  Detects questions, matches context, generates prompt cards,
  and detects action items. Errors are logged but never propagated.
  """
  from meeting_transcriber.prompter import (
    detect_action_items,
    detect_questions,
    generate_prompt_card,
    match_context,
  )

  try:
    questions = detect_questions(transcript_text)
    for question in questions:
      matches = match_context(question.keywords, context_chunks)
      card_text = generate_prompt_card(question, matches)
      session["_ws_queue"].append(
        {
          "type": "coaching_nano",
          "text": card_text,
        }
      )
  except Exception:
    pass  # detect_questions already handles errors internally

  try:
    items = detect_action_items(transcript_text)
    for item in items:
      session["action_items"].append(
        {
          "text": item.text,
          "owner": item.owner,
          "deadline": item.deadline,
        }
      )
      session["_ws_queue"].append(
        {
          "type": "action_item",
          "text": item.text,
          "owner": item.owner,
          "deadline": item.deadline,
        }
      )
  except Exception:
    pass  # detect_action_items already handles errors internally


def _build_meeting_markdown(
  session: dict[str, Any],
  engine_name: str,
) -> str:
  """Build complete meeting notes as a markdown document."""
  now = datetime.now()
  lines = [
    "# Meeting Notes",
    "",
    "## Metadata",
    f"- **Date**: {now.strftime('%Y-%m-%d %H:%M')}",
    f"- **Engine**: {engine_name}",
    f"- **Cost**: ${session['total_cost']:.4f}",
    "",
  ]

  # Playbook coverage (context)
  if session["context"]:
    lines.append("## Playbook")
    lines.append("")
    lines.append("\n\n".join(session["context"]))
    lines.append("")

  # Summary
  if session.get("summary"):
    lines.append(session["summary"])
    lines.append("")

  # Action items
  if session["action_items"]:
    lines.append("## Action Items")
    lines.append("")
    for item in session["action_items"]:
      if isinstance(item, dict):
        lines.append(f"- [ ] {item['text']}")
      else:
        lines.append(f"- [ ] {item}")
    lines.append("")

  # Full transcript
  lines.append("## Transcript")
  lines.append("")
  lines.append("\n".join(session["transcript_chunks"]))
  lines.append("")

  return "\n".join(lines)


def create_app(
  context_paths: list[str] | None = None,
  engine_name: str = "openai",
  language: str = "zh",
  record: bool = False,
  chunk_duration: int = 5,
) -> FastAPI:
  """Factory: create a configured FastAPI application."""
  app = FastAPI(title="Meeting Transcriber")
  session = _new_session()

  # Pre-load context if provided via CLI
  if context_paths:
    session["context"] = _load_context_files(context_paths)
    session["context_paths"] = list(context_paths)

  # Store config on app state for access in routes
  app.state.engine_name = engine_name
  app.state.language = language
  app.state.record = record
  app.state.chunk_duration = chunk_duration
  app.state.session = session

  # --- Static files ---
  if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

  @app.get("/")
  async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")

  @app.post("/api/start")
  async def start_session(body: StartRequest | None = None) -> JSONResponse:
    if session["active"]:
      return JSONResponse(
        {"error": "Already recording"},
        status_code=409,
      )
    session["active"] = True
    session["start_time"] = time.time()
    session["transcript_chunks"] = []
    session["action_items"] = []
    session["total_cost"] = 0.0
    session["_ws_queue"] = []

    # Reset live transcript file for Claude Code CLI
    _LIVE_TRANSCRIPT_PATH.write_text("", encoding="utf-8")

    # Load additional context from request
    if body and body.context_paths:
      extra = _load_context_files(body.context_paths)
      session["context"] = session["context"] + extra
      session["context_paths"] = session["context_paths"] + list(body.context_paths)

    # Load structured context chunks for coaching prompt engine
    _load_session_context_chunks(session)

    # Start recording in background thread
    if app.state.record:
      use_realtime = app.state.engine_name == "qwen"
      if use_realtime:
        thread = threading.Thread(
          target=_recording_loop_realtime,
          args=(session, app.state.language),
          daemon=True,
        )
      else:
        thread = threading.Thread(
          target=_recording_loop,
          args=(session, app.state.engine_name, app.state.language, app.state.chunk_duration),
          daemon=True,
        )
      session["_thread"] = thread
      thread.start()

    return JSONResponse({"session_id": str(int(session["start_time"]))})

  @app.post("/api/stop")
  async def stop_session() -> JSONResponse:
    if not session["active"]:
      return JSONResponse(
        {"error": "Not recording"},
        status_code=404,
      )
    duration = _elapsed(session)
    session["active"] = False

    # Stop realtime streamer if active
    if session.get("_realtime_streamer"):
      session["_realtime_streamer"].stop()
      session["_realtime_streamer"] = None

    if session.get("recorder"):
      session["recorder"].stop()
    if session.get("_thread"):
      session["_thread"].join(timeout=5)

    session["start_time"] = None
    return JSONResponse(
      {
        "status": "stopped",
        "duration": round(duration, 1),
      }
    )

  @app.post("/api/context/upload")
  async def upload_context(file: UploadFile) -> JSONResponse:
    """Upload a context/playbook file from the browser."""
    content = (await file.read()).decode("utf-8", errors="replace")
    if not content.strip():
      return JSONResponse(
        {"error": "Empty file"},
        status_code=400,
      )
    session["context"].append(content)
    return JSONResponse(
      {
        "filename": file.filename,
        "length": len(content),
      }
    )

  @app.get("/api/status")
  async def get_status() -> JSONResponse:
    return JSONResponse(
      {
        "recording": session["active"],
        "duration": round(_elapsed(session), 1),
        "cost": round(session["total_cost"], 4),
      }
    )

  @app.post("/api/summarize")
  async def summarize_session() -> JSONResponse:
    from meeting_transcriber.summarizer import summarize

    text = "\n".join(session["transcript_chunks"])
    if not text:
      return JSONResponse(
        {"error": "No transcript to summarize"},
        status_code=400,
      )

    # Build playbook from context if available
    playbook_text = "\n\n".join(session["context"]) if session["context"] else None

    summary_md = summarize(text, playbook=playbook_text)
    session["summary"] = summary_md

    # Queue summary to WebSocket clients
    session["_ws_queue"].append({"type": "summary", "text": summary_md})

    return JSONResponse(
      {
        "summary": summary_md,
        "action_items": session["action_items"],
      }
    )

  @app.post("/api/coach")
  async def coach_me() -> JSONResponse:
    """On-demand coaching: analyze recent transcript against playbook."""
    from meeting_transcriber.prompter import generate_coaching_strategy, load_context

    # Grab last ~10 seconds of transcript (last few chunks)
    recent = session["transcript_chunks"][-5:] if session["transcript_chunks"] else []
    recent_text = "\n".join(recent)
    if not recent_text.strip():
      return JSONResponse(
        {"error": "No recent transcript to analyze"},
        status_code=400,
      )

    # Load context chunks if not already loaded
    if not session.get("context_chunks") and session.get("context_paths"):
      session["context_chunks"] = load_context(session["context_paths"])

    # Also use raw context text for the LLM
    playbook_text = "\n\n".join(session["context"]) if session["context"] else ""

    strategy = generate_coaching_strategy(recent_text, playbook_text)

    # Queue to WebSocket
    session["_ws_queue"].append({"type": "coaching_nano", "text": strategy})

    return JSONResponse({"strategy": strategy, "source": "nano"})

  @app.post("/api/coach/opus")
  async def coach_opus() -> JSONResponse:
    """On-demand deep coaching via Claude Code CLI (Opus)."""
    from meeting_transcriber.opus_coach import run_opus_coaching

    recent = session["transcript_chunks"][-10:] if session["transcript_chunks"] else []
    recent_text = "\n".join(recent)
    if not recent_text.strip():
      return JSONResponse(
        {"error": "No recent transcript to analyze"},
        status_code=400,
      )

    playbook_text = "\n\n".join(session["context"]) if session["context"] else ""

    def on_result(text: str) -> None:
      session["_ws_queue"].append({"type": "coaching_opus", "text": text})

    run_opus_coaching(recent_text, playbook_text, on_result)

    return JSONResponse({"status": "processing", "source": "opus"})

  @app.get("/api/transcript/live")
  async def get_live_transcript() -> JSONResponse:
    """Get recent transcript chunks for Claude Code CLI."""
    return JSONResponse(
      {
        "chunks": session["transcript_chunks"],
        "recent": session["transcript_chunks"][-5:],
        "context": session["context"],
        "file": str(_LIVE_TRANSCRIPT_PATH),
      }
    )

  @app.post("/api/coach/push")
  async def push_coaching(body: dict) -> JSONResponse:
    """External coaching push — allows Claude Code CLI to send coaching cards."""
    text = body.get("text", "")
    if not text.strip():
      return JSONResponse({"error": "Empty text"}, status_code=400)
    session["_ws_queue"].append({"type": "coaching_opus", "text": text})
    return JSONResponse({"status": "ok"})

  @app.post("/api/save")
  async def save_notes(body: SaveRequest) -> JSONResponse:
    text = "\n".join(session["transcript_chunks"])
    if not text:
      return JSONResponse(
        {"error": "No transcript to save"},
        status_code=400,
      )

    md = _build_meeting_markdown(session, app.state.engine_name)
    out = Path(body.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    return JSONResponse({"path": str(out)})

  @app.post("/api/export")
  async def export_notes() -> JSONResponse:
    text = "\n".join(session["transcript_chunks"])
    if not text:
      return JSONResponse(
        {"error": "No transcript to export"},
        status_code=400,
      )

    md = _build_meeting_markdown(session, app.state.engine_name)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"meeting-{timestamp}.md"
    return JSONResponse(
      {
        "markdown": md,
        "suggested_filename": filename,
      }
    )

  @app.websocket("/ws")
  async def websocket_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    session["ws_clients"].add(ws)
    try:
      while True:
        # Drain queued messages from the recording thread
        await _drain_ws_queue(session)
        # Listen for incoming messages with short timeout
        try:
          await asyncio.wait_for(ws.receive_text(), timeout=0.5)
          await ws.send_json(
            {
              "type": "status",
              "recording": session["active"],
            }
          )
        except asyncio.TimeoutError:
          continue
    except WebSocketDisconnect:
      session["ws_clients"].discard(ws)

  return app


def _queue_context(session: dict[str, Any]) -> None:
  """Queue context messages for WebSocket broadcast."""
  for text in session["context"]:
    session["_ws_queue"].append({"type": "context", "text": text})


async def _drain_ws_queue(session: dict[str, Any]) -> None:
  """Send all queued messages to every connected WebSocket client."""
  while session["_ws_queue"]:
    msg = session["_ws_queue"].pop(0)
    for client in list(session["ws_clients"]):
      try:
        await client.send_json(msg)
      except Exception:
        session["ws_clients"].discard(client)

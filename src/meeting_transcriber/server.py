"""FastAPI server for Meeting Transcriber Web UI."""

import asyncio
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

STATIC_DIR = Path(__file__).parent / "static"


class StartRequest(BaseModel):
  context_paths: list[str] = []


class SaveRequest(BaseModel):
  output_path: str


def _new_session() -> dict[str, Any]:
  """Return a fresh session state dict."""
  return {
    "active": False,
    "start_time": None,
    "context": [],
    "transcript_chunks": [],
    "action_items": [],
    "total_cost": 0.0,
    "ws_clients": set(),
    "_ws_queue": [],
    "_thread": None,
    "recorder": None,
    "audio_path": None,
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

  engine = get_engine(engine_name)
  recorder = Recorder()
  recorder.start(audio_path)
  session["audio_path"] = audio_path
  session["recorder"] = recorder

  chunk_index = 0
  while session["active"]:
    time.sleep(chunk_duration)
    if not session["active"]:
      break
    try:
      chunks = chunk_audio(audio_path, chunk_duration=chunk_duration, overlap=2)
      _transcribe_new_chunks(
        session, engine, chunks, chunk_index, chunk_duration, language,
      )
      chunk_index = len(chunks)
    except Exception as e:
      session["_ws_queue"].append({"type": "error", "text": str(e)})


def _transcribe_new_chunks(
  session: dict[str, Any],
  engine: Any,
  chunks: list[Path],
  chunk_index: int,
  chunk_duration: int,
  language: str,
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
    session["_ws_queue"].append({
      "type": "transcript",
      "timestamp": timestamp_str,
      "text": result.full_text,
    })
    session["_ws_queue"].append({
      "type": "cost",
      "value": session["total_cost"],
    })


def create_app(
  context_paths: list[str] | None = None,
  engine_name: str = "openai",
  language: str = "zh",
  record: bool = False,
) -> FastAPI:
  """Factory: create a configured FastAPI application."""
  app = FastAPI(title="Meeting Transcriber")
  session = _new_session()

  # Pre-load context if provided via CLI
  if context_paths:
    session["context"] = _load_context_files(context_paths)

  # Store config on app state for access in routes
  app.state.engine_name = engine_name
  app.state.language = language
  app.state.record = record
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

    # Load additional context from request
    if body and body.context_paths:
      extra = _load_context_files(body.context_paths)
      session["context"] = session["context"] + extra

    # Push context to WebSocket clients
    _queue_context(session)

    # Start recording in background thread
    if app.state.record:
      thread = threading.Thread(
        target=_recording_loop,
        args=(session, app.state.engine_name, app.state.language),
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

    if session.get("recorder"):
      session["recorder"].stop()
    if session.get("_thread"):
      session["_thread"].join(timeout=5)

    session["start_time"] = None
    return JSONResponse({
      "status": "stopped",
      "duration": round(duration, 1),
    })

  @app.get("/api/status")
  async def get_status() -> JSONResponse:
    return JSONResponse({
      "recording": session["active"],
      "duration": round(_elapsed(session), 1),
      "cost": round(session["total_cost"], 4),
    })

  @app.post("/api/summarize")
  async def summarize_session() -> JSONResponse:
    text = "\n".join(session["transcript_chunks"])
    if not text:
      return JSONResponse(
        {"error": "No transcript to summarize"},
        status_code=400,
      )
    return JSONResponse({
      "summary": f"Summary of {len(session['transcript_chunks'])} chunks",
      "action_items": session["action_items"],
    })

  @app.post("/api/save")
  async def save_notes(body: SaveRequest) -> JSONResponse:
    text = "\n".join(session["transcript_chunks"])
    if not text:
      return JSONResponse(
        {"error": "No transcript to save"},
        status_code=400,
      )
    out = Path(body.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return JSONResponse({"path": str(out)})

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
          data = await asyncio.wait_for(ws.receive_text(), timeout=0.5)
          await ws.send_json({
            "type": "status",
            "recording": session["active"],
          })
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

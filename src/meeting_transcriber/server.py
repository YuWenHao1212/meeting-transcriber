"""FastAPI server for Meeting Transcriber Web UI."""

import time
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


def create_app(
  context_paths: list[str] | None = None,
  engine_name: str = "openai",
  language: str = "zh",
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

    # Load additional context from request
    if body and body.context_paths:
      extra = _load_context_files(body.context_paths)
      session["context"] = session["context"] + extra

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
        data = await ws.receive_text()
        # Echo back for now; real impl would process commands
        await ws.send_json({"type": "status", "recording": session["active"]})
    except WebSocketDisconnect:
      session["ws_clients"].discard(ws)

  return app

"""FastAPI server for Meeting Transcriber Web UI."""

import asyncio
import shutil
import subprocess
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
  stereo: bool = False


class SaveRequest(BaseModel):
  output_path: str


# Multi-output device name created in Audio MIDI Setup
_MULTI_OUTPUT_DEVICE = "多重輸出裝置"


def _has_switch_audio() -> bool:
  """Check if SwitchAudioSource CLI is available."""
  return shutil.which("SwitchAudioSource") is not None


def _get_current_output() -> str | None:
  """Get current system audio output device name."""
  if not _has_switch_audio():
    return None
  try:
    result = subprocess.run(
      ["SwitchAudioSource", "-c", "-t", "output"],
      capture_output=True, text=True, timeout=5,
    )
    return result.stdout.strip() if result.returncode == 0 else None
  except Exception:
    return None


def _set_output_device(name: str) -> bool:
  """Switch system audio output to the given device."""
  if not _has_switch_audio():
    return False
  try:
    result = subprocess.run(
      ["SwitchAudioSource", "-s", name, "-t", "output"],
      capture_output=True, text=True, timeout=5,
    )
    return result.returncode == 0
  except Exception:
    return False


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


def _make_on_partial(session: dict[str, Any], speaker: str | None = None) -> callable:
  """Create an on_partial callback, optionally tagged with a speaker label."""
  def on_partial(text: str) -> None:
    msg: dict[str, Any] = {
      "type": "transcript_partial",
      "text": text,
    }
    if speaker:
      msg["speaker"] = speaker
    session["_ws_queue"].append(msg)
  return on_partial


def _make_on_final(session: dict[str, Any], speaker: str | None = None) -> callable:
  """Create an on_final callback, optionally tagged with a speaker label."""
  def on_final(text: str) -> None:
    elapsed = _elapsed(session)
    mm = int(elapsed) // 60
    ss = int(elapsed) % 60
    timestamp_str = f"{mm:02d}:{ss:02d}"

    # Store with speaker label prefix for coaching context
    chunk_text = f"[{speaker}] {text}" if speaker else text
    session["transcript_chunks"].append(chunk_text)

    # Estimate cost from elapsed time
    duration_min = elapsed / 60.0
    session["total_cost"] = duration_min * 0.0054

    msg: dict[str, Any] = {
      "type": "transcript",
      "timestamp": timestamp_str,
      "text": text,
    }
    if speaker:
      msg["speaker"] = speaker
    session["_ws_queue"].append(msg)
    session["_ws_queue"].append(
      {
        "type": "cost",
        "value": session["total_cost"],
      }
    )

    # Append to live transcript file for Claude Code CLI
    try:
      with open(_LIVE_TRANSCRIPT_PATH, "a", encoding="utf-8") as f:
        prefix = f"[{speaker}]" if speaker else ""
        f.write(f"[{timestamp_str}]{prefix} {text}\n")
    except Exception:
      pass
  return on_final


def _make_on_error(session: dict[str, Any]) -> callable:
  """Create an on_error callback."""
  def on_error(text: str) -> None:
    session["_ws_queue"].append({"type": "error", "text": text})
  return on_error


def _recording_loop_realtime(
  session: dict[str, Any],
  language: str,
  stereo: bool = False,
) -> None:
  """Background thread: stream PCM audio to Qwen3-ASR realtime WebSocket.

  When stereo=True, opens a 3-channel stream from an Aggregate Device where:
    Ch1-2 = BlackHole (system audio = other side)
    Ch3   = built-in mic (our side)
  Each side is sent to a separate QwenRealtimeStreamer instance.
  """
  import numpy as np
  import sounddevice as sd

  from meeting_transcriber.engines.qwen_realtime import QwenRealtimeStreamer

  if stereo:
    # Find the Aggregate Device for stereo input
    aggregate_dev = None
    for i, d in enumerate(sd.query_devices()):
      if "聚集" in d["name"] or "Aggregate" in d["name"].lower():
        if d["max_input_channels"] >= 3:
          aggregate_dev = i
          break
    if aggregate_dev is None:
      session["_ws_queue"].append({
        "type": "error",
        "text": "找不到聚合裝置（需要至少 3 個輸入通道）。請在 Audio MIDI Setup 建立聚合裝置。",
      })
      return

    # Two streamers: mic (我方) and system audio (對方)
    streamer_mine = QwenRealtimeStreamer(
      language=language,
      on_partial=_make_on_partial(session, "我方"),
      on_final=_make_on_final(session, "我方"),
      on_error=_make_on_error(session),
    )
    streamer_theirs = QwenRealtimeStreamer(
      language=language,
      on_partial=_make_on_partial(session, "對方"),
      on_final=_make_on_final(session, "對方"),
      on_error=_make_on_error(session),
    )
    session["_realtime_streamer"] = [streamer_mine, streamer_theirs]

    try:
      streamer_mine.start()
      streamer_theirs.start()
    except Exception as e:
      print(f"[realtime-stereo] Streamer start error: {e}")
      session["_ws_queue"].append({"type": "error", "text": f"Realtime error: {e}"})
      return

    print("[realtime-stereo] Started. Streaming 3-ch PCM (Ch1-2=對方, Ch3=我方).")

    # Separate buffers for mic (our side) and system audio (other side)
    buffer_mine = bytearray()
    buffer_theirs = bytearray()
    buffer_lock = threading.Lock()

    def audio_callback(
      indata: np.ndarray,
      frame_count: int,
      time_info: object,
      status: object,
    ) -> None:
      # indata shape: (frames, 3) — Ch1-2 = BlackHole (對方), Ch3 = mic (我方)
      theirs = indata[:, 0]  # BlackHole Ch1 (mono is enough for ASR)
      mine = indata[:, 2]    # Built-in mic
      pcm_mine = (mine * 32767).astype(np.int16).tobytes()
      pcm_theirs = (theirs * 32767).astype(np.int16).tobytes()
      with buffer_lock:
        buffer_mine.extend(pcm_mine)
        buffer_theirs.extend(pcm_theirs)

    try:
      stream = sd.InputStream(
        device=aggregate_dev,
        samplerate=16000,
        channels=3,
        dtype="float32",
        callback=audio_callback,
      )
      stream.start()
    except Exception as e:
      print(f"[realtime-stereo] Audio stream error: {e}")
      session["_ws_queue"].append({"type": "error", "text": f"Audio error: {e}"})
      streamer_mine.stop()
      streamer_theirs.stop()
      return

    # Send audio chunks at regular intervals
    try:
      while session["active"] and streamer_mine.is_running and streamer_theirs.is_running:
        time.sleep(_REALTIME_SEND_INTERVAL)
        with buffer_lock:
          # Our side (mic, Ch3)
          if len(buffer_mine) >= _REALTIME_CHUNK_BYTES:
            chunk_mine = bytes(buffer_mine[:_REALTIME_CHUNK_BYTES])
            del buffer_mine[:_REALTIME_CHUNK_BYTES]
          elif len(buffer_mine) > 0:
            chunk_mine = bytes(buffer_mine)
            buffer_mine.clear()
          else:
            chunk_mine = None

          # Other side (BlackHole, Ch1)
          if len(buffer_theirs) >= _REALTIME_CHUNK_BYTES:
            chunk_theirs = bytes(buffer_theirs[:_REALTIME_CHUNK_BYTES])
            del buffer_theirs[:_REALTIME_CHUNK_BYTES]
          elif len(buffer_theirs) > 0:
            chunk_theirs = bytes(buffer_theirs)
            buffer_theirs.clear()
          else:
            chunk_theirs = None

        if chunk_mine:
          streamer_mine.send_audio(chunk_mine)
        if chunk_theirs:
          streamer_theirs.send_audio(chunk_theirs)
    finally:
      stream.stop()
      stream.close()
      streamer_mine.stop()
      streamer_theirs.stop()
      session["_realtime_streamer"] = None

  else:
    # Mono mode — original behavior
    streamer = QwenRealtimeStreamer(
      language=language,
      on_partial=_make_on_partial(session),
      on_final=_make_on_final(session),
      on_error=_make_on_error(session),
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
  """Build meeting notes as a single markdown document.

  Structure: Summary first (main content), Transcript appended at the end.
  """
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

  # Summary first (includes action items as part of LLM output)
  if session.get("summary"):
    lines.append(session["summary"])
    lines.append("")

  # Transcript appended at the end
  lines.append("---")
  lines.append("")
  lines.append("## Transcript")
  lines.append("")
  lines.append("\n".join(session["transcript_chunks"]))
  lines.append("")

  return "\n".join(lines)


_PLAYBOOK_SEARCH_DIRS = [
  Path.home() / "Library/Mobile Documents/iCloud~md~obsidian/Documents/FLUX Vault",
]


def _resolve_playbook_path(filename: str | None) -> Path | None:
  """Search known directories for a file matching the uploaded filename."""
  if not filename:
    return None
  for search_dir in _PLAYBOOK_SEARCH_DIRS:
    if not search_dir.is_dir():
      continue
    matches = list(search_dir.rglob(filename))
    if len(matches) == 1:
      return matches[0]
    if len(matches) > 1:
      # Multiple matches — return the most recently modified
      return max(matches, key=lambda p: p.stat().st_mtime)
  return None


def _get_save_directory(session: dict[str, Any]) -> Path:
  """Determine save directory from playbook path or fallback."""
  # Try CLI context_paths first
  for p in session.get("context_paths", []):
    path = Path(p)
    if path.exists():
      return path.parent

  # Fallback
  fallback = Path.home() / "Documents" / "meetings"
  fallback.mkdir(parents=True, exist_ok=True)
  return fallback


def create_app(
  context_paths: list[str] | None = None,
  engine_name: str = "openai",
  language: str = "zh",
  record: bool = False,
  chunk_duration: int = 5,
  stereo: bool = False,
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
  app.state.stereo = stereo
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

    # Determine stereo mode: frontend toggle overrides CLI flag
    use_stereo = (body.stereo if body else False) or app.state.stereo

    # Auto-switch system output to multi-output device for stereo
    if use_stereo:
      prev = _get_current_output()
      session["_prev_output_device"] = prev
      if prev and prev != _MULTI_OUTPUT_DEVICE:
        ok = _set_output_device(_MULTI_OUTPUT_DEVICE)
        if not ok:
          return JSONResponse(
            {"error": f"無法切換系統輸出到「{_MULTI_OUTPUT_DEVICE}」，請確認 Audio MIDI Setup 已建立多重輸出裝置，並已安裝 SwitchAudioSource (brew install switchaudio-osx)"},
            status_code=500,
          )

    session["active"] = True
    session["start_time"] = time.time()
    session["transcript_chunks"] = []
    session["action_items"] = []
    session["total_cost"] = 0.0
    session["_ws_queue"] = []
    session["_stereo"] = use_stereo
    session["summary"] = ""
    session["_summary_cursor"] = 0

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
          args=(session, app.state.language, use_stereo),
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

    return JSONResponse({"session_id": str(int(session["start_time"])), "stereo": use_stereo})

  @app.post("/api/stop")
  async def stop_session() -> JSONResponse:
    if not session["active"]:
      return JSONResponse(
        {"error": "Not recording"},
        status_code=404,
      )
    duration = _elapsed(session)
    session["active"] = False

    # Stop realtime streamer(s) if active
    rs = session.get("_realtime_streamer")
    if rs:
      if isinstance(rs, list):
        for s in rs:
          s.stop()
      else:
        rs.stop()
      session["_realtime_streamer"] = None

    if session.get("recorder"):
      session["recorder"].stop()
    if session.get("_thread"):
      session["_thread"].join(timeout=5)

    # Restore previous audio output device after stereo session
    prev = session.pop("_prev_output_device", None)
    if prev and session.get("_stereo"):
      _set_output_device(prev)

    session["start_time"] = None
    session["_stereo"] = False
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

    # Try to find the original file path by searching known directories
    resolved_path = _resolve_playbook_path(file.filename)
    if resolved_path:
      session["context_paths"].append(str(resolved_path))
      print(f"[playbook] Resolved '{file.filename}' → {resolved_path}")
    else:
      print(f"[playbook] Could not resolve path for '{file.filename}'")

    return JSONResponse(
      {
        "filename": file.filename,
        "length": len(content),
        "resolved_path": str(resolved_path) if resolved_path else None,
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
    from meeting_transcriber.summarizer import summarize, summarize_incremental

    all_chunks = session["transcript_chunks"]
    cursor = session.get("_summary_cursor", 0)
    new_chunks = all_chunks[cursor:]

    if not new_chunks:
      return JSONResponse(
        {"error": "No new transcript to summarize"},
        status_code=400,
      )

    new_text = "\n".join(new_chunks)
    playbook_text = "\n\n".join(session["context"]) if session["context"] else None
    existing_summary = session.get("summary", "")

    if existing_summary:
      # Incremental: only process new chunks, merge into existing summary
      summary_md = summarize_incremental(
        new_transcript=new_text,
        existing_summary=existing_summary,
        playbook=playbook_text,
      )
    else:
      # First time: summarize everything
      summary_md = summarize(new_text, playbook=playbook_text)

    session["summary"] = summary_md
    session["_summary_cursor"] = len(all_chunks)

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
    """Quick coaching via Anthropic Sonnet."""
    from meeting_transcriber.coach import run_quick_coaching

    recent = session["transcript_chunks"][-15:] if session["transcript_chunks"] else []
    recent_text = "\n".join(recent)
    if not recent_text.strip():
      return JSONResponse(
        {"error": "No recent transcript to analyze"},
        status_code=400,
      )

    playbook_text = "\n\n".join(session["context"]) if session["context"] else ""

    def on_result(text: str) -> None:
      session["_ws_queue"].append({"type": "coaching_nano", "text": text})

    run_quick_coaching(recent_text, playbook_text, on_result)

    return JSONResponse({"status": "processing", "source": "sonnet"})

  @app.post("/api/coach/opus")
  async def coach_opus() -> JSONResponse:
    """Deep coaching via Anthropic Opus."""
    from meeting_transcriber.coach import run_deep_coaching

    recent = session["transcript_chunks"][-30:] if session["transcript_chunks"] else []
    recent_text = "\n".join(recent)
    if not recent_text.strip():
      return JSONResponse(
        {"error": "No recent transcript to analyze"},
        status_code=400,
      )

    playbook_text = "\n\n".join(session["context"]) if session["context"] else ""
    print(f"[coach-opus] Sending {len(recent)} chunks to LLM:\n{recent_text[:500]}")

    def on_result(text: str) -> None:
      session["_ws_queue"].append({"type": "coaching_opus", "text": text})

    run_deep_coaching(recent_text, playbook_text, on_result)

    return JSONResponse({"status": "processing", "source": "opus"})

  @app.post("/api/open-audio-midi")
  async def open_audio_midi() -> JSONResponse:
    """Open macOS Audio MIDI Setup app."""
    try:
      subprocess.Popen(["open", "-a", "Audio MIDI Setup"])
      return JSONResponse({"status": "opened"})
    except Exception as e:
      return JSONResponse({"error": str(e)}, status_code=500)

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
  async def save_notes() -> JSONResponse:
    text = "\n".join(session["transcript_chunks"])
    if not text:
      return JSONResponse(
        {"error": "No transcript to save"},
        status_code=400,
      )

    save_dir = _get_save_directory(session)
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M")
    filename = f"{timestamp}-meeting-notes.md"
    out = save_dir / filename

    md = _build_meeting_markdown(session, app.state.engine_name)
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

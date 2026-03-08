"""Transcript formatting utilities for markdown output."""

from meeting_transcriber.models import TranscriptResult


def _format_timestamp(seconds: float) -> str:
  """Format seconds as [MM:SS] or [HH:MM:SS] if >= 1 hour."""
  total_seconds = int(seconds)
  hours = total_seconds // 3600
  minutes = (total_seconds % 3600) // 60
  secs = total_seconds % 60

  if hours > 0:
    return f"[{hours:02d}:{minutes:02d}:{secs:02d}]"
  return f"[{minutes:02d}:{secs:02d}]"


def _format_duration(seconds: float) -> str:
  """Format duration as human-readable string (e.g., '2m 5s', '1h 30m')."""
  total_seconds = int(seconds)
  hours = total_seconds // 3600
  minutes = (total_seconds % 3600) // 60
  secs = total_seconds % 60

  parts: list[str] = []
  if hours > 0:
    parts.append(f"{hours}h")
  if minutes > 0:
    parts.append(f"{minutes}m")
  if secs > 0 or not parts:
    parts.append(f"{secs}s")
  return " ".join(parts)


def _segment_minute(start: float) -> int:
  """Return the minute bucket for a segment's start time."""
  return int(start) // 60


def transcript_to_markdown(result: TranscriptResult) -> str:
  """Format a TranscriptResult as markdown with timestamps.

  Each segment renders as:
    [MM:SS] Speaker: text   (if speaker is set)
    [MM:SS] text             (if no speaker)

  Segments are grouped by minute with blank lines between groups.
  Includes a header with duration and engine info.
  """
  lines: list[str] = []

  # Header
  duration_str = _format_duration(result.duration)
  lines.append(f"# Transcript ({duration_str}, engine: {result.engine})")
  lines.append("")

  if not result.segments:
    return "\n".join(lines)

  prev_minute: int | None = None

  for segment in result.segments:
    current_minute = _segment_minute(segment.start)

    # Blank line between different minute groups
    if prev_minute is not None and current_minute != prev_minute:
      lines.append("")

    timestamp = _format_timestamp(segment.start)

    if segment.speaker:
      lines.append(f"{timestamp} {segment.speaker}: {segment.text}")
    else:
      lines.append(f"{timestamp} {segment.text}")

    prev_minute = current_minute

  lines.append("")  # trailing newline
  return "\n".join(lines)


def meeting_notes_header(
  title: str,
  date: str,
  duration: float,
  attendees: list[str] | None = None,
) -> str:
  """Format a markdown header with meeting metadata.

  Args:
    title: Meeting title.
    date: Date string (e.g., '2026-03-08').
    duration: Duration in seconds.
    attendees: Optional list of attendee names.

  Returns:
    Markdown-formatted header string.
  """
  duration_str = _format_duration(duration)
  lines: list[str] = [
    f"# {title}",
    "",
    f"- **Date:** {date}",
    f"- **Duration:** {duration_str}",
  ]

  if attendees:
    lines.append(f"- **Attendees:** {', '.join(attendees)}")

  lines.append("")
  return "\n".join(lines)

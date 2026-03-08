"""Tests for formats.py — transcript formatting utilities."""

from meeting_transcriber.formats import meeting_notes_header, transcript_to_markdown
from meeting_transcriber.models import Segment, TranscriptResult


class TestTranscriptToMarkdown:
  """Tests for transcript_to_markdown()."""

  def test_segments_with_timestamps(self):
    """Each segment should render with [MM:SS] timestamp."""
    result = TranscriptResult(
      segments=[
        Segment(start=0.0, end=5.0, text="Hello everyone."),
        Segment(start=5.0, end=12.0, text="Let's get started."),
      ],
      full_text="Hello everyone. Let's get started.",
      duration=12.0,
      engine="openai",
    )
    md = transcript_to_markdown(result)
    assert "[00:00]" in md
    assert "[00:05]" in md
    assert "Hello everyone." in md
    assert "Let's get started." in md

  def test_segments_with_speaker_labels(self):
    """Segments with speakers should render as [MM:SS] Speaker: text."""
    result = TranscriptResult(
      segments=[
        Segment(start=0.0, end=5.0, text="Hi there.", speaker="Alice"),
        Segment(start=5.0, end=10.0, text="Hey Alice.", speaker="Bob"),
      ],
      full_text="Hi there. Hey Alice.",
      duration=10.0,
      engine="openai",
    )
    md = transcript_to_markdown(result)
    assert "Alice:" in md
    assert "Bob:" in md
    assert "Hi there." in md
    assert "Hey Alice." in md

  def test_segments_without_speakers(self):
    """Segments without speakers should render as [MM:SS] text (no colon)."""
    result = TranscriptResult(
      segments=[
        Segment(start=0.0, end=5.0, text="Some speech."),
      ],
      full_text="Some speech.",
      duration=5.0,
      engine="openai",
    )
    md = transcript_to_markdown(result)
    # Should NOT have a dangling colon when no speaker
    lines = [line for line in md.splitlines() if "Some speech." in line]
    assert len(lines) == 1
    # Pattern: [00:00] Some speech.  (no "None:" or ": Some speech.")
    assert "None" not in lines[0]
    # No colon between timestamp and text
    assert "] Some speech." in lines[0]

  def test_timestamp_format_mm_ss(self):
    """Timestamps under 1 hour should be [MM:SS]."""
    result = TranscriptResult(
      segments=[
        Segment(start=65.0, end=70.0, text="One minute in."),
        Segment(start=599.0, end=605.0, text="Almost ten minutes."),
      ],
      full_text="One minute in. Almost ten minutes.",
      duration=605.0,
      engine="openai",
    )
    md = transcript_to_markdown(result)
    assert "[01:05]" in md
    assert "[09:59]" in md

  def test_timestamp_format_hh_mm_ss(self):
    """Timestamps >= 1 hour should be [HH:MM:SS]."""
    result = TranscriptResult(
      segments=[
        Segment(start=3661.0, end=3670.0, text="Over an hour."),
      ],
      full_text="Over an hour.",
      duration=3670.0,
      engine="openai",
    )
    md = transcript_to_markdown(result)
    assert "[01:01:01]" in md

  def test_grouped_by_minute(self):
    """Segments in different minutes should be separated by blank lines."""
    result = TranscriptResult(
      segments=[
        Segment(start=0.0, end=10.0, text="First minute."),
        Segment(start=30.0, end=40.0, text="Still first minute."),
        Segment(start=60.0, end=70.0, text="Second minute."),
        Segment(start=120.0, end=130.0, text="Third minute."),
      ],
      full_text="First minute. Still first minute. Second minute. Third minute.",
      duration=130.0,
      engine="openai",
    )
    md = transcript_to_markdown(result)
    # Find the transcript body (after header)
    # Segments in same minute should NOT have blank line between them
    # Segments in different minutes should have blank line between them
    lines = md.splitlines()
    transcript_lines = [line for line in lines if line.startswith("[")]
    assert len(transcript_lines) == 4

    # Check blank line separates different-minute groups
    body = md[md.index("[00:00]") :]
    # Between "Still first minute." and "Second minute." there should be a blank line
    assert "Still first minute.\n\n[01:00]" in body
    # Between same-minute segments, no blank line
    assert "First minute.\n[00:30]" in body

  def test_header_includes_duration_and_engine(self):
    """Output should include a header with duration and engine info."""
    result = TranscriptResult(
      segments=[
        Segment(start=0.0, end=5.0, text="Test."),
      ],
      full_text="Test.",
      duration=125.0,
      engine="openai",
    )
    md = transcript_to_markdown(result)
    assert "openai" in md.lower()
    assert "2m" in md or "2:05" in md or "125" in md

  def test_empty_segments(self):
    """Empty segments should produce minimal output."""
    result = TranscriptResult(segments=[], full_text="", duration=0.0)
    md = transcript_to_markdown(result)
    assert isinstance(md, str)


class TestMeetingNotesHeader:
  """Tests for meeting_notes_header()."""

  def test_all_fields(self):
    """Header with all fields should include title, date, duration, attendees."""
    header = meeting_notes_header(
      title="Weekly Sync",
      date="2026-03-08",
      duration=1800.0,
      attendees=["Alice", "Bob", "Charlie"],
    )
    assert "Weekly Sync" in header
    assert "2026-03-08" in header
    assert "30" in header  # 1800s = 30 minutes
    assert "Alice" in header
    assert "Bob" in header
    assert "Charlie" in header

  def test_minimal_fields_no_attendees(self):
    """Header without attendees should still render title, date, duration."""
    header = meeting_notes_header(
      title="Quick Chat",
      date="2026-03-08",
      duration=300.0,
    )
    assert "Quick Chat" in header
    assert "2026-03-08" in header
    assert "5" in header  # 300s = 5 minutes
    # Should not crash or show "None"
    assert "None" not in header

  def test_header_is_markdown(self):
    """Header should use markdown formatting (# heading)."""
    header = meeting_notes_header(
      title="Test Meeting",
      date="2026-03-08",
      duration=600.0,
    )
    assert header.startswith("#")

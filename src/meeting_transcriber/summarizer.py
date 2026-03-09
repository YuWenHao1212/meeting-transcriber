"""Meeting summarization and transcript cleaning via Claude Code CLI.

Uses `claude -p` (subscription-based, no API billing).
"""

import re

from meeting_transcriber.claude_cli import call_claude_cli

DEFAULT_TEMPLATE = """\
## Meeting Summary
(2-3 sentence overview)

## Key Decisions
- (bullet points)

## Action Items
- [ ] (action) — **Owner**: (name) | **Deadline**: (date if mentioned)

## Key Discussions
- (bullet points of main topics discussed)

## Follow-ups
- (items requiring future attention)\
"""

PLAYBOOK_COVERAGE_TEMPLATE = """\

## Playbook 覆蓋率
✅ (objective) — 在 [timestamp] 討論
❌ (objective) — 未討論到\
"""

PLAYBOOK_COVERAGE_INSTRUCTIONS = (
  "\n\n"
  "## Playbook 覆蓋率 Cross-Reference Instructions\n\n"
  "A pre-meeting playbook with objectives is provided. "
  "Cross-reference the transcript with the pre-meeting objectives.\n\n"
  "Add a '## Playbook 覆蓋率' section at the end of the output.\n"
  "For each objective in the playbook:\n"
  "- If addressed in the transcript, use: ✅ (objective) — 在 [timestamp] 討論\n"
  "  Include the approximate timestamp from the transcript where it was discussed.\n"
  "- If NOT addressed, use: ❌ (objective) — 未討論到\n\n"
  "Output the Playbook 覆蓋率 section in Traditional Chinese (繁體中文).\n"
  "Note which objectives were addressed and which were not addressed."
)


def _build_summarize_prompt(
  transcript: str,
  playbook: str | None = None,
  template: str | None = None,
) -> str:
  """Build the full prompt for summarization."""
  output_structure = template if template else DEFAULT_TEMPLATE
  parts = [
    "You are an expert meeting note-taker. "
    "Analyze the provided meeting transcript and produce structured meeting notes.\n\n"
    f"Use this output structure:\n\n{output_structure}\n\n"
    "Guidelines:\n"
    "- Meeting Summary should be 2-3 concise sentences\n"
    "- Key Decisions: list concrete decisions made during the meeting\n"
    "- Action Items: include owner and deadline if mentioned in the transcript\n"
    "- Key Discussions: summarize main discussion points as bullet points\n"
    "- Follow-ups: items that need future attention or were left unresolved\n"
    "- Output in markdown format",
  ]
  if playbook:
    parts[0] += PLAYBOOK_COVERAGE_INSTRUCTIONS
    parts.append(f"\n---\n\n## Pre-Meeting Playbook\n\n{playbook}")
  parts.append(f"\n---\n\n## Transcript\n\n{transcript}")
  return "\n".join(parts)


def summarize(
  transcript: str,
  playbook: str | None = None,
  template: str | None = None,
) -> str:
  """Summarize a meeting transcript using Claude Code CLI.

  Args:
    transcript: The meeting transcript text.
    playbook: Optional pre-meeting playbook with objectives.
    template: Optional custom output template.

  Returns:
    Markdown-formatted meeting summary string.
  """
  prompt = _build_summarize_prompt(transcript, playbook=playbook, template=template)
  return call_claude_cli(prompt, timeout=180)


def summarize_incremental(
  new_transcript: str,
  existing_summary: str,
  playbook: str | None = None,
) -> str:
  """Incrementally update an existing summary with new transcript segments.

  Args:
    new_transcript: Only the new, unprocessed transcript text.
    existing_summary: The previously generated summary.
    playbook: Optional pre-meeting playbook for context.

  Returns:
    Updated markdown-formatted meeting summary string.
  """
  system = (
    "You are an expert meeting note-taker performing an incremental update.\n\n"
    "You are given:\n"
    "1. The EXISTING summary from earlier in the meeting\n"
    "2. NEW transcript segments that have not been summarized yet\n"
    "3. Optionally, a pre-meeting playbook for context\n\n"
    "Your task: produce an UPDATED summary that incorporates the new transcript.\n"
    "- Merge new information into the existing structure (don't duplicate)\n"
    "- Append new decisions, action items, discussions as needed\n"
    "- Update the Playbook 覆蓋率 section if playbook is provided\n"
    "- Output the COMPLETE updated summary (not just the diff)\n"
    "- Output in markdown format, in Traditional Chinese (繁體中文)\n"
  )
  if playbook:
    system += PLAYBOOK_COVERAGE_INSTRUCTIONS

  parts = [system]
  if playbook:
    parts.append(f"\n---\n\n## Pre-Meeting Playbook\n\n{playbook}")
  parts.append(f"\n---\n\n## Existing Summary\n\n{existing_summary}")
  parts.append(f"\n---\n\n## New Transcript Segments\n\n{new_transcript}")

  return call_claude_cli("\n".join(parts), timeout=180)


# --- Clean Transcript ---

CLEAN_SYSTEM = (
  "You are an expert transcript editor. "
  "Clean up a raw ASR (Automatic Speech Recognition) transcript.\n\n"
  "Your tasks:\n"
  "1. **Fix ASR errors**: correct misrecognized words based on context and the playbook\n"
  "2. **Remove filler words**: remove excessive 嗯、嗯嗯、嗯嗯嗯、哦、啊、呃、對對對 etc.\n"
  "3. **Merge duplicates**: combine repeated/stuttered phrases into one clean sentence\n"
  "4. **Fix broken sentences**: merge fragments that were split by ASR into complete sentences\n"
  "5. **Preserve speaker labels**: keep [我方] and [對方] labels exactly as-is\n"
  "6. **Simplify timestamps**: convert verbose timestamps like "
  "[0:00:03.520000 - 0:00:06.080000] to simple [MM:SS] format, e.g. [00:03]. "
  "Use only the start time, drop the end time and microseconds.\n"
  "7. **Preserve meaning**: do NOT rewrite, paraphrase, or add content. Only clean.\n"
  "8. **Fix domain terms**: use the playbook/context to correct technical terms, "
  "proper nouns, and jargon that ASR may have misrecognized\n"
  "9. **NEVER swap speakers**: [我方] must stay [我方], [對方] must stay [對方]. "
  "Do NOT change, swap, or reassign speaker labels under any circumstances.\n\n"
  "Output format:\n"
  "- Each line: [MM:SS] [speaker] text\n"
  "- You may merge adjacent lines from the same speaker if they form one sentence\n"
  "- Output in the same language as the original transcript\n"
  "- Output ONLY the cleaned transcript lines. "
  "Do NOT include any preamble, explanation, markdown code fences, or commentary.\n"
)

_CLEAN_CHUNK_LINES = 100
_CLEAN_OVERLAP_LINES = 10


def _build_clean_prompt(
  chunk: str,
  playbook: str | None,
  context_before: str | None = None,
  context_after: str | None = None,
) -> str:
  """Build the full prompt for cleaning a transcript chunk."""
  system = CLEAN_SYSTEM
  if playbook:
    system += (
      "\n## Reference Context (Playbook)\n\n"
      "Use this playbook to identify correct terminology, proper nouns, "
      "and domain-specific terms. Fix any ASR misrecognitions accordingly.\n"
    )

  context_note = ""
  if context_before or context_after:
    context_note = (
      "\nIMPORTANT: Context lines (marked with [CONTEXT]) are provided for "
      "continuity reference only. Do NOT include them in your output. "
      "Only output the cleaned version of the MAIN SECTION lines.\n"
    )

  parts = [system]
  if playbook:
    parts.append(f"\n---\n\n## Playbook (for term reference)\n\n{playbook}")

  transcript_section = ""
  if context_before:
    transcript_section += f"[CONTEXT - before]\n{context_before}\n\n"
  transcript_section += f"[MAIN SECTION - clean this]\n{chunk}"
  if context_after:
    transcript_section += f"\n\n[CONTEXT - after]\n{context_after}"

  parts.append(f"\n---\n\n## Raw Transcript\n{context_note}\n{transcript_section}")
  return "\n".join(parts)


def _strip_code_fences(text: str) -> str:
  """Remove markdown code fences and preamble from LLM output."""
  text = re.sub(r'^(?:Here is the cleaned transcript[:\s]*\n*)', '', text, flags=re.IGNORECASE)
  text = re.sub(r'^```\w*\n?', '', text)
  text = re.sub(r'\n?```\s*$', '', text)
  return text.strip()


def clean_transcript(
  transcript: str,
  playbook: str | None = None,
) -> str:
  """Clean a raw ASR transcript using Claude Code CLI.

  Splits long transcripts into chunks with overlap for context continuity.

  Args:
    transcript: The raw transcript text with timestamps and speaker labels.
    playbook: Optional pre-meeting playbook for term correction context.

  Returns:
    Cleaned transcript string in the same format.
  """
  lines = [line for line in transcript.split("\n") if line.strip()]

  # Short enough to process in one go
  if len(lines) <= _CLEAN_CHUNK_LINES + _CLEAN_OVERLAP_LINES:
    print(f"[clean] Processing {len(lines)} lines in one pass...")
    prompt = _build_clean_prompt(transcript, playbook)
    return _strip_code_fences(call_claude_cli(prompt, timeout=120))

  # Split into chunks with overlap context
  cleaned_parts = []
  total_chunks = (len(lines) + _CLEAN_CHUNK_LINES - 1) // _CLEAN_CHUNK_LINES

  for i in range(0, len(lines), _CLEAN_CHUNK_LINES):
    chunk_lines = lines[i : i + _CLEAN_CHUNK_LINES]
    chunk = "\n".join(chunk_lines)

    before_start = max(0, i - _CLEAN_OVERLAP_LINES)
    context_before = "\n".join(lines[before_start:i]) if i > 0 else None

    after_end = min(len(lines), i + _CLEAN_CHUNK_LINES + _CLEAN_OVERLAP_LINES)
    after_start = i + _CLEAN_CHUNK_LINES
    context_after = "\n".join(lines[after_start:after_end]) if after_start < len(lines) else None

    chunk_num = i // _CLEAN_CHUNK_LINES + 1
    print(f"[clean] Processing chunk {chunk_num}/{total_chunks}...")
    prompt = _build_clean_prompt(chunk, playbook, context_before, context_after)
    result = _strip_code_fences(call_claude_cli(prompt, timeout=120))
    cleaned_parts.append(result)

  return "\n".join(cleaned_parts)

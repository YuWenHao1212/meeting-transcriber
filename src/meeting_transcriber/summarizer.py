"""Meeting summarization via Anthropic Claude."""

import anthropic

DEFAULT_MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 4096

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


def _build_system_prompt(
  playbook: str | None = None,
  template: str | None = None,
) -> str:
  """Build the system prompt for the summarization request."""
  output_structure = template if template else DEFAULT_TEMPLATE
  base = (
    "You are an expert meeting note-taker. "
    "Analyze the provided meeting transcript and produce structured meeting notes.\n\n"
    "Use this output structure:\n\n"
    f"{output_structure}\n\n"
    "Guidelines:\n"
    "- Meeting Summary should be 2-3 concise sentences\n"
    "- Key Decisions: list concrete decisions made during the meeting\n"
    "- Action Items: include owner and deadline if mentioned in the transcript\n"
    "- Key Discussions: summarize main discussion points as bullet points\n"
    "- Follow-ups: items that need future attention or were left unresolved\n"
    "- Output in markdown format"
  )
  if playbook:
    base += PLAYBOOK_COVERAGE_INSTRUCTIONS
  return base


def _build_user_message(
  transcript: str,
  playbook: str | None = None,
) -> str:
  """Build the user message containing transcript and optional playbook."""
  parts = []
  if playbook:
    parts.append(f"## Pre-Meeting Playbook\n\n{playbook}")
  parts.append(f"## Transcript\n\n{transcript}")
  return "\n\n---\n\n".join(parts)


INCREMENTAL_SYSTEM = (
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


def summarize(
  transcript: str,
  playbook: str | None = None,
  template: str | None = None,
  model: str = DEFAULT_MODEL,
) -> str:
  """Summarize a meeting transcript using Anthropic Claude.

  Args:
    transcript: The meeting transcript text.
    playbook: Optional pre-meeting playbook with objectives.
    template: Optional custom output template (replaces default structure).
    model: Anthropic model to use.

  Returns:
    Markdown-formatted meeting summary string.
  """
  client = anthropic.Anthropic()
  response = client.messages.create(
    model=model,
    max_tokens=MAX_TOKENS,
    system=_build_system_prompt(playbook=playbook, template=template),
    messages=[
      {"role": "user", "content": _build_user_message(transcript, playbook=playbook)},
    ],
  )
  return response.content[0].text


def summarize_incremental(
  new_transcript: str,
  existing_summary: str,
  playbook: str | None = None,
  model: str = DEFAULT_MODEL,
) -> str:
  """Incrementally update an existing summary with new transcript segments.

  Args:
    new_transcript: Only the new, unprocessed transcript text.
    existing_summary: The previously generated summary.
    playbook: Optional pre-meeting playbook for context.
    model: Anthropic model to use.

  Returns:
    Updated markdown-formatted meeting summary string.
  """
  parts = []
  if playbook:
    parts.append(f"## Pre-Meeting Playbook\n\n{playbook}")
  parts.append(f"## Existing Summary\n\n{existing_summary}")
  parts.append(f"## New Transcript Segments\n\n{new_transcript}")
  user_msg = "\n\n---\n\n".join(parts)

  system = INCREMENTAL_SYSTEM
  if playbook:
    system += PLAYBOOK_COVERAGE_INSTRUCTIONS

  client = anthropic.Anthropic()
  response = client.messages.create(
    model=model,
    max_tokens=MAX_TOKENS,
    system=system,
    messages=[{"role": "user", "content": user_msg}],
  )
  return response.content[0].text

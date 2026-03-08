/* Meeting Transcriber — Web UI client */

let ws = null;
let timerInterval = null;
let startTime = null;

// --- DOM references ---
const timerEl = document.getElementById("timer");
const statusBadge = document.getElementById("status-badge");
const costEl = document.getElementById("cost");
const transcriptEl = document.getElementById("transcript");
const coachingNanoEl = document.getElementById("coaching-nano");
const coachingOpusEl = document.getElementById("coaching-opus");
const coachingNanoPanel = document.getElementById("coaching-nano-panel");
const coachingOpusPanel = document.getElementById("coaching-opus-panel");
const actionList = document.getElementById("action-list");
const actionItems = document.getElementById("action-items");
const btnStart = document.getElementById("btn-start");
const btnPause = document.getElementById("btn-pause");
const btnEnd = document.getElementById("btn-end");
const btnCoachNano = document.getElementById("btn-coach-nano");
const btnCoachOpus = document.getElementById("btn-coach-opus");
const btnCoachBoth = document.getElementById("btn-coach-both");
const btnSummarize = document.getElementById("btn-summarize");
const btnSave = document.getElementById("btn-save");

// --- Simple Markdown renderer ---
function renderMarkdown(md) {
  let html = escapeHtml(md);

  // Horizontal rules
  html = html.replace(/^---$/gm, "<hr>");

  // Headers (must come before bold processing)
  html = html.replace(/^### (.+)$/gm, "<h3>$1</h3>");
  html = html.replace(/^## (.+)$/gm, "<h2>$1</h2>");
  html = html.replace(/^# (.+)$/gm, "<h1>$1</h1>");

  // Bold and italic
  html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/\*(.+?)\*/g, "<em>$1</em>");

  // Inline code
  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");

  // Wikilinks [[target|display]] or [[target]]
  html = html.replace(/\[\[([^\]|]+)\|([^\]]+)\]\]/g, '<span class="wikilink">$2</span>');
  html = html.replace(/\[\[([^\]]+)\]\]/g, '<span class="wikilink">$1</span>');

  // Links [text](url)
  html = html.replace(
    /\[([^\]]+)\]\(([^)]+)\)/g,
    '<a href="$2" target="_blank" rel="noopener">$1</a>',
  );

  // Blockquotes
  html = html.replace(/^&gt; (.+)$/gm, "<blockquote>$1</blockquote>");

  // Tables — detect lines with | separators
  html = renderTables(html);

  // Unordered lists (- item)
  html = html.replace(/^- (.+)$/gm, "<li>$1</li>");
  html = html.replace(/((?:<li>.*<\/li>\n?)+)/g, "<ul>$1</ul>");

  // Paragraphs: wrap remaining bare lines
  html = html
    .split("\n")
    .map((line) => {
      const trimmed = line.trim();
      if (!trimmed) return "";
      if (/^<(h[1-6]|ul|ol|li|blockquote|hr|table|thead|tbody|tr|th|td|pre|div)/.test(trimmed)) {
        return line;
      }
      return `<p>${line}</p>`;
    })
    .join("\n");

  // Clean up empty paragraphs
  html = html.replace(/<p>\s*<\/p>/g, "");

  return html;
}

function renderTables(html) {
  const lines = html.split("\n");
  const result = [];
  let tableRows = [];
  let inTable = false;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    const isTableRow = line.startsWith("|") && line.endsWith("|") && line.includes("|");
    const isSeparator = /^\|[-:|  ]+\|$/.test(line);

    if (isTableRow) {
      if (!inTable) {
        inTable = true;
        tableRows = [];
      }
      if (!isSeparator) {
        const cells = line
          .split("|")
          .slice(1, -1)
          .map((c) => c.trim());
        tableRows.push(cells);
      }
    } else {
      if (inTable) {
        result.push(buildTable(tableRows));
        inTable = false;
        tableRows = [];
      }
      result.push(lines[i]);
    }
  }
  if (inTable) {
    result.push(buildTable(tableRows));
  }
  return result.join("\n");
}

function buildTable(rows) {
  if (rows.length === 0) return "";
  let html = "<table>";
  // First row is header
  html += "<thead><tr>";
  for (const cell of rows[0]) {
    html += `<th>${cell}</th>`;
  }
  html += "</tr></thead>";
  // Remaining rows
  if (rows.length > 1) {
    html += "<tbody>";
    for (let i = 1; i < rows.length; i++) {
      html += "<tr>";
      for (const cell of rows[i]) {
        html += `<td>${cell}</td>`;
      }
      html += "</tr>";
    }
    html += "</tbody>";
  }
  html += "</table>";
  return html;
}

// --- WebSocket ---
function connectWebSocket() {
  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  const url = `${protocol}//${location.host}/ws`;
  ws = new WebSocket(url);

  ws.onopen = () => {
    console.log("WebSocket connected");
  };

  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    handleMessage(msg);
  };

  ws.onclose = () => {
    console.log("WebSocket disconnected, reconnecting in 2s...");
    setTimeout(connectWebSocket, 2000);
  };

  ws.onerror = (err) => {
    console.error("WebSocket error:", err);
    ws.close();
  };
}

function handleMessage(msg) {
  switch (msg.type) {
    case "transcript":
      appendTranscript(msg.timestamp || "", msg.text || "", msg.speaker || "");
      break;
    case "transcript_partial":
      updatePartialTranscript(msg.text || "", msg.speaker || "");
      break;
    case "coaching_nano":
      appendCoachingNano(msg.text || "");
      break;
    case "coaching_opus":
      appendCoachingOpus(msg.text || "");
      break;
    case "action_item":
      appendActionItem(msg.text || "");
      break;
    case "cost":
      updateCost(msg.value || 0);
      break;
    case "summary":
      showSummary(msg.text || "");
      break;
    case "context":
      appendContext(msg.filename || "Playbook", msg.text || "");
      break;
    case "error":
      appendError(msg.text || "Unknown error");
      break;
    case "status":
      break;
  }
}

// --- Transcript ---
function appendTranscript(timestamp, text, speaker) {
  hasTranscript = true;
  btnCoachNano.disabled = false;
  btnCoachOpus.disabled = false;
  btnCoachBoth.disabled = false;
  clearPlaceholder(transcriptEl);
  removePartialLine();
  const line = document.createElement("div");
  line.className = "transcript-line";
  let speakerHtml = "";
  if (speaker) {
    const cls = speaker === "\u6211\u65b9" ? "speaker-mine" : "speaker-theirs";
    speakerHtml = `<span class="speaker-badge ${cls}">${escapeHtml(speaker)}</span>`;
  }
  line.innerHTML = `<span class="timestamp">${timestamp}</span>${speakerHtml}<span class="text">${escapeHtml(text)}</span>`;
  transcriptEl.appendChild(line);
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
}

function updatePartialTranscript(text, speaker) {
  clearPlaceholder(transcriptEl);
  let partial = transcriptEl.querySelector(".transcript-line-partial");
  if (!partial) {
    partial = document.createElement("div");
    partial.className = "transcript-line transcript-line-partial";
    transcriptEl.appendChild(partial);
  }
  let speakerHtml = "";
  if (speaker) {
    const cls = speaker === "\u6211\u65b9" ? "speaker-mine" : "speaker-theirs";
    speakerHtml = `<span class="speaker-badge ${cls}">${escapeHtml(speaker)}</span>`;
  }
  partial.innerHTML = `<span class="timestamp">...</span>${speakerHtml}<span class="text">${escapeHtml(text)}</span>`;
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
}

function removePartialLine() {
  const partial = transcriptEl.querySelector(".transcript-line-partial");
  if (partial) {
    partial.remove();
  }
}

// --- Coaching (Nano) ---
function appendCoachingNano(text) {
  coachingNanoPanel.classList.remove("hidden");
  clearPlaceholder(coachingNanoEl);
  const card = document.createElement("div");
  card.className = "coaching-card";
  card.innerHTML = `<div class="label">Quick Coach</div><div>${escapeHtml(text)}</div>`;
  coachingNanoEl.insertBefore(card, coachingNanoEl.firstChild);
}

// --- Toggle coaching panels ---
function togglePanel(which) {
  const panel = which === "nano" ? coachingNanoPanel : coachingOpusPanel;
  panel.classList.toggle("hidden");
}

// --- Coaching (Opus) ---
function appendCoachingOpus(text) {
  coachingOpusPanel.classList.remove("hidden");
  clearPlaceholder(coachingOpusEl);
  const card = document.createElement("div");
  card.className = "coaching-card summary-card";
  card.innerHTML = `<div class="label">Deep Coach</div><div class="context-rendered">${renderMarkdown(text)}</div>`;
  coachingOpusEl.insertBefore(card, coachingOpusEl.firstChild);
}

// --- Action Items ---
function appendActionItem(text) {
  actionItems.classList.add("visible");
  const li = document.createElement("li");
  li.textContent = text;
  actionList.appendChild(li);
}

// --- Cost ---
function updateCost(value) {
  costEl.textContent = `$${value.toFixed(4)}`;
}

// --- Timer ---
function startTimer() {
  startTime = Date.now();
  timerInterval = setInterval(() => {
    const elapsed = Math.floor((Date.now() - startTime) / 1000);
    const h = String(Math.floor(elapsed / 3600)).padStart(2, "0");
    const m = String(Math.floor((elapsed % 3600) / 60)).padStart(2, "0");
    const s = String(elapsed % 60).padStart(2, "0");
    timerEl.textContent = `${h}:${m}:${s}`;
  }, 1000);
}

function stopTimer() {
  clearInterval(timerInterval);
  timerInterval = null;
}

// --- Audio MIDI Setup ---
async function openAudioMidi() {
  await fetch("/api/open-audio-midi", { method: "POST" });
}

// --- Button handlers ---
async function startRecording() {
  const stereo = document.getElementById("chk-stereo").checked;
  const resp = await fetch("/api/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ context_paths: [], stereo }),
  });

  if (!resp.ok) {
    const err = await resp.json();
    alert(err.error || "Failed to start");
    return;
  }

  setUI("recording");
  startTimer();
}

async function pauseRecording() {
  if (sessionState === "paused") {
    // Resume
    const resp = await fetch("/api/resume", { method: "POST" });
    if (!resp.ok) {
      const err = await resp.json();
      alert(err.error || "Failed to resume");
      return;
    }
    setUI("recording");
    startTimer();
  } else {
    // Pause
    const resp = await fetch("/api/pause", { method: "POST" });
    if (!resp.ok) {
      const err = await resp.json();
      alert(err.error || "Failed to pause");
      return;
    }
    removePartialLine();
    setUI("paused");
    stopTimer();
  }
}

async function endRecording() {
  const resp = await fetch("/api/stop", { method: "POST" });
  if (!resp.ok) {
    const err = await resp.json();
    alert(err.error || "Failed to end");
    return;
  }

  removePartialLine();
  setUI("idle");
  stopTimer();
}

async function summarize() {
  btnSummarize.disabled = true;
  btnSummarize.textContent = "Summarizing...";
  try {
    const resp = await fetch("/api/summarize", { method: "POST" });
    if (!resp.ok) {
      const err = await resp.json();
      alert(err.error || "Nothing to summarize");
      return;
    }
    const data = await resp.json();
    showSummary(data.summary);
    if (data.action_items && data.action_items.length > 0) {
      data.action_items.forEach((item) => appendActionItem(item));
    }
  } finally {
    btnSummarize.disabled = false;
    btnSummarize.textContent = "Summarize";
  }
}

async function save() {
  btnSave.disabled = true;
  btnSave.textContent = "Saving...";
  try {
    const resp = await fetch("/api/save", { method: "POST" });
    if (!resp.ok) {
      const err = await resp.json();
      alert(err.error || "Failed to save");
      return;
    }
    const data = await resp.json();
    alert(`Saved to: ${data.path}`);
  } finally {
    btnSave.disabled = false;
    btnSave.textContent = "Save";
  }
}

// --- Coach (Nano — on-demand) ---
async function coachNano() {
  btnCoachNano.disabled = true;
  btnCoachNano.textContent = "Thinking...";
  try {
    const resp = await fetch("/api/coach", { method: "POST" });
    if (!resp.ok) {
      const err = await resp.json();
      alert(err.error || "No recent transcript");
      return;
    }
    // Result pushed via WebSocket as coaching_nano
  } finally {
    btnCoachNano.disabled = false;
    btnCoachNano.textContent = "Quick Coach";
  }
}

// --- Coach (Opus — on-demand) ---
async function coachOpus() {
  btnCoachOpus.disabled = true;
  btnCoachOpus.textContent = "Thinking...";
  try {
    const resp = await fetch("/api/coach/opus", { method: "POST" });
    if (!resp.ok) {
      const err = await resp.json();
      alert(err.error || "Deep coaching unavailable");
      return;
    }
    // Result pushed via WebSocket as coaching_opus
  } finally {
    btnCoachOpus.disabled = false;
    btnCoachOpus.textContent = "Deep Coach";
  }
}

// --- Coach (Both — fire Nano + Opus in parallel) ---
async function coachBoth() {
  coachNano();
  coachOpus();
}

// --- Summary ---
function showSummary(markdown) {
  const section = document.getElementById("summary-section");
  const editor = document.getElementById("summary-editor");
  section.classList.remove("hidden");
  editor.value = markdown;
}

// --- UI helpers ---
let hasTranscript = false;

// state: "idle" | "recording" | "paused"
let sessionState = "idle";

function setUI(state) {
  sessionState = state;
  const hasData = hasTranscript;

  btnStart.disabled = state !== "idle";
  btnPause.disabled = state === "idle";
  btnPause.textContent = state === "paused" ? "Resume" : "Pause";
  btnEnd.disabled = state === "idle";
  btnCoachNano.disabled = state === "idle" && !hasData;
  btnCoachOpus.disabled = state === "idle" && !hasData;
  btnCoachBoth.disabled = state === "idle" && !hasData;
  btnSummarize.disabled = false;
  btnSave.disabled = !hasData;

  if (state === "recording") {
    statusBadge.textContent = "Recording";
    statusBadge.className = "badge badge-recording";
  } else if (state === "paused") {
    statusBadge.textContent = "Paused";
    statusBadge.className = "badge badge-paused";
  } else {
    statusBadge.textContent = "Idle";
    statusBadge.className = "badge badge-idle";
  }
}

function clearPlaceholder(el) {
  const ph = el.querySelector(".placeholder");
  if (ph) ph.remove();
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

// --- Errors ---
function appendError(text) {
  clearPlaceholder(transcriptEl);
  const line = document.createElement("div");
  line.className = "transcript-line error-line";
  line.innerHTML = `<span class="timestamp">ERROR</span><span class="text">${escapeHtml(text)}</span>`;
  transcriptEl.appendChild(line);
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
}

// --- Context upload ---
const contextListEl = document.getElementById("context-list");

async function uploadContextFiles(files) {
  for (const file of files) {
    const form = new FormData();
    form.append("file", file);

    try {
      const resp = await fetch("/api/context/upload", {
        method: "POST",
        body: form,
      });

      if (!resp.ok) {
        const text = await resp.text();
        console.error("Upload failed:", resp.status, text);
        try {
          const err = JSON.parse(text);
          alert(err.detail || err.error || `Upload failed (${resp.status})`);
        } catch {
          alert(`Upload failed (${resp.status})`);
        }
        continue;
      }

      appendContext(file.name, await readFileText(file));
    } catch (e) {
      console.error("Upload error:", e);
      alert(`Upload error: ${e.message}`);
    }
  }
  document.getElementById("context-file").value = "";
}

function readFileText(file) {
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.readAsText(file);
  });
}

function appendContext(filename, text) {
  clearPlaceholder(contextListEl);
  const card = document.createElement("div");
  card.className = "context-card";
  card.innerHTML = `<div class="context-filename">${escapeHtml(filename)}</div><div class="context-rendered">${renderMarkdown(text)}</div>`;
  contextListEl.appendChild(card);
}

// --- Init ---
connectWebSocket();

/* Meeting Transcriber — Web UI client */

let ws = null;
let timerInterval = null;
let startTime = null;

// --- DOM references ---
const timerEl = document.getElementById("timer");
const statusBadge = document.getElementById("status-badge");
const costEl = document.getElementById("cost");
const transcriptEl = document.getElementById("transcript");
const coachingEl = document.getElementById("coaching");
const actionList = document.getElementById("action-list");
const actionItems = document.getElementById("action-items");
const btnStart = document.getElementById("btn-start");
const btnStop = document.getElementById("btn-stop");
const btnSummarize = document.getElementById("btn-summarize");
const btnSave = document.getElementById("btn-save");

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
      appendTranscript(msg.timestamp || "", msg.text || "");
      break;
    case "coaching":
      appendCoaching(msg.text || "");
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
    case "status":
      // Status update from server
      break;
  }
}

// --- Transcript ---
function appendTranscript(timestamp, text) {
  clearPlaceholder(transcriptEl);
  const line = document.createElement("div");
  line.className = "transcript-line";
  line.innerHTML = `<span class="timestamp">${timestamp}</span><span class="text">${escapeHtml(text)}</span>`;
  transcriptEl.appendChild(line);
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
}

// --- Coaching ---
function appendCoaching(text) {
  clearPlaceholder(coachingEl);
  const card = document.createElement("div");
  card.className = "coaching-card";
  card.innerHTML = `<div class="label">Suggestion</div><div>${escapeHtml(text)}</div>`;
  coachingEl.insertBefore(card, coachingEl.firstChild);
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

// --- Button handlers ---
async function startRecording() {
  const resp = await fetch("/api/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ context_paths: [] }),
  });

  if (!resp.ok) {
    const err = await resp.json();
    alert(err.error || "Failed to start");
    return;
  }

  setRecordingUI(true);
  startTimer();
}

async function stopRecording() {
  const resp = await fetch("/api/stop", { method: "POST" });
  if (!resp.ok) {
    const err = await resp.json();
    alert(err.error || "Failed to stop");
    return;
  }

  setRecordingUI(false);
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
  const path = prompt("Save path:", "meeting-notes.md");
  if (!path) return;

  const resp = await fetch("/api/save", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ output_path: path }),
  });

  if (!resp.ok) {
    const err = await resp.json();
    alert(err.error || "Failed to save");
    return;
  }

  const data = await resp.json();
  alert(`Saved to: ${data.path}`);
}

// --- Summary ---
function showSummary(markdown) {
  clearPlaceholder(coachingEl);
  const card = document.createElement("div");
  card.className = "coaching-card summary-card";
  card.innerHTML = `<div class="label">Summary</div><pre class="summary-text">${escapeHtml(markdown)}</pre>`;
  coachingEl.insertBefore(card, coachingEl.firstChild);
}

// --- UI helpers ---
function setRecordingUI(recording) {
  btnStart.disabled = recording;
  btnStop.disabled = !recording;
  btnSummarize.disabled = recording;
  btnSave.disabled = recording;

  if (recording) {
    statusBadge.textContent = "Recording";
    statusBadge.className = "badge badge-recording";
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

      const data = await resp.json();
      appendContext(file.name, await readFileText(file));
    } catch (e) {
      console.error("Upload error:", e);
      alert(`Upload error: ${e.message}`);
    }
  }
  // Reset input so same file can be re-selected
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
  card.innerHTML = `<div class="context-filename">${escapeHtml(filename)}</div><pre class="context-text">${escapeHtml(text)}</pre>`;
  contextListEl.appendChild(card);
}

// --- Init ---
connectWebSocket();

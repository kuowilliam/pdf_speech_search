const state = {
  asrModels: [],
  selectedModelId: window.localStorage.getItem("pdfSpeechSearch.asrModelId") || "",
  audioContext: null,
  mediaStream: null,
  worklet: null,
  silentGain: null,
  websocket: null,
  recording: false,
  finalizing: false,
  searchTimer: null,
  recorderTimer: null,
  modelsPollTimer: null,
  recorderStartedAt: 0,
  waveformLevels: [],
  searchRequestId: 0,
  lastQuery: "",
  results: [],
  activeKey: "",
};

const els = {
  statusLine: document.querySelector("#statusLine"),
  autoSearch: document.querySelector("#autoSearch"),
  modelOptions: document.querySelector("#modelOptions"),
  transcript: document.querySelector("#transcript"),
  captureBar: document.querySelector("#captureBar"),
  micBtn: document.querySelector("#micBtn"),
  recorderMeter: document.querySelector("#recorderMeter"),
  waveform: document.querySelector("#waveform"),
  recordTimer: document.querySelector("#recordTimer"),
  clearBtn: document.querySelector("#clearBtn"),
  searchBtn: document.querySelector("#searchBtn"),
  results: document.querySelector("#results"),
  resultCount: document.querySelector("#resultCount"),
  viewerTitle: document.querySelector("#viewerTitle"),
  viewerPage: document.querySelector("#viewerPage"),
  pdfViewer: document.querySelector("#pdfViewer"),
  openTab: document.querySelector("#openTab"),
};

function setControls(mode) {
  const recording = mode === "recording";
  const modelUnavailable = mode === "idle" && !selectedModel()?.available;
  const disabled = mode === "connecting" || mode === "finalizing" || modelUnavailable;
  els.micBtn.disabled = disabled;
  els.micBtn.classList.toggle("recording", recording);
  els.micBtn.setAttribute("aria-label", recording ? "Stop recording" : "Start recording");
}

function setStatus(text) {
  els.statusLine.textContent = text;
}

function selectedModel() {
  return state.asrModels.find((model) => model.id === state.selectedModelId) || null;
}

function modelStatusClass(model) {
  if (model.download_status === "downloading") {
    return "downloading";
  }
  if (model.download_status === "error") {
    return "error";
  }
  if (model.available) {
    return "ready";
  }
  if (model.installed && !model.runtime_available) {
    return "error";
  }
  return "missing";
}

function downloadIcon() {
  return `
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 3v12"></path>
      <path d="M7 10l5 5 5-5"></path>
      <path d="M5 21h14"></path>
    </svg>
  `;
}

function renderModelOptions() {
  els.modelOptions.innerHTML = "";
  for (const model of state.asrModels) {
    const option = document.createElement("div");
    option.role = "button";
    option.tabIndex = state.recording || state.finalizing ? -1 : 0;
    option.className = `modelOption ${state.selectedModelId === model.id ? "selected" : ""}`;
    option.classList.toggle("disabled", state.recording || state.finalizing);
    option.title = model.reason || model.model;

    const dot = document.createElement("span");
    dot.className = `modelDot ${modelStatusClass(model)}`;

    const text = document.createElement("span");
    text.className = "modelText";

    const label = document.createElement("span");
    label.className = "modelLabel";
    label.textContent = model.label;

    const detail = document.createElement("span");
    detail.className = "modelDetail";
    if (model.download_status === "downloading") {
      detail.textContent = "Downloading";
    } else if (model.download_status === "error") {
      detail.textContent = "Error";
    } else {
      detail.textContent = model.detail;
    }

    text.append(label, detail);
    option.append(dot, text);
    const selectOption = () => {
      if (state.recording || state.finalizing) {
        return;
      }
      state.selectedModelId = model.id;
      window.localStorage.setItem("pdfSpeechSearch.asrModelId", model.id);
      renderModelOptions();
      if (!model.available) {
        setStatus(model.reason || `${model.label} needs download`);
      } else if (!state.recording && !state.finalizing) {
        setStatus(`${model.label} ready`);
      }
      setControls("idle");
    };
    option.addEventListener("click", selectOption);
    option.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        selectOption();
      }
    });

    if (!model.installed && model.download_status !== "downloading") {
      const download = document.createElement("button");
      download.type = "button";
      download.className = "modelDownload";
      download.setAttribute("aria-label", `Download ${model.label}`);
      download.innerHTML = downloadIcon();
      download.addEventListener("click", (event) => {
        event.stopPropagation();
        downloadAsrModel(model.id);
      });
      option.append(download);
    }

    els.modelOptions.appendChild(option);
  }
}

function updateModelPolling() {
  const downloading = state.asrModels.some((model) => model.download_status === "downloading");
  if (downloading && !state.modelsPollTimer) {
    state.modelsPollTimer = window.setInterval(() => loadAsrModels({ silent: true }), 1800);
  } else if (!downloading && state.modelsPollTimer) {
    window.clearInterval(state.modelsPollTimer);
    state.modelsPollTimer = null;
  }
}

async function loadAsrModels({ silent = false } = {}) {
  const response = await fetch("/api/asr/models");
  if (!response.ok) {
    throw new Error("ASR model status failed");
  }
  const payload = await response.json();
  state.asrModels = payload.models;
  const selectedStillExists = state.asrModels.some((model) => model.id === state.selectedModelId);
  if (!selectedStillExists) {
    state.selectedModelId = payload.default_model_id;
    window.localStorage.setItem("pdfSpeechSearch.asrModelId", state.selectedModelId);
  }
  renderModelOptions();
  updateModelPolling();
  if (!silent && selectedModel()?.available && !state.recording && !state.finalizing) {
    setStatus(`${selectedModel().label} ready`);
  }
  setControls(state.recording ? "recording" : state.finalizing ? "finalizing" : "idle");
  return payload;
}

async function downloadAsrModel(modelId) {
  const model = state.asrModels.find((item) => item.id === modelId);
  setStatus(`Downloading ${model?.label || "model"}`);
  const response = await fetch(`/api/asr/models/${encodeURIComponent(modelId)}/download`, {
    method: "POST",
  });
  if (!response.ok) {
    setStatus("Download failed");
    return;
  }
  const payload = await response.json();
  state.asrModels = payload.models;
  renderModelOptions();
  updateModelPolling();
}

function initWaveform() {
  els.waveform.innerHTML = "";
  state.waveformLevels = Array.from({ length: 42 }, () => 0);
  for (let i = 0; i < state.waveformLevels.length; i += 1) {
    const bar = document.createElement("span");
    bar.className = "waveBar";
    els.waveform.appendChild(bar);
  }
}

function formatDuration(seconds) {
  const minutes = Math.floor(seconds / 60);
  const rest = Math.floor(seconds % 60)
    .toString()
    .padStart(2, "0");
  return `${minutes}:${rest}`;
}

function updateRecordTimer() {
  if (!state.recorderStartedAt) {
    els.recordTimer.textContent = "0:00";
    return;
  }
  const elapsed = (Date.now() - state.recorderStartedAt) / 1000;
  els.recordTimer.textContent = formatDuration(elapsed);
}

function setMeterMode(mode) {
  els.recorderMeter.classList.toggle("active", mode === "active");
  els.recorderMeter.classList.toggle("finalizing", mode === "finalizing");
  els.captureBar.classList.toggle("recording", mode === "active" || mode === "finalizing");
  els.captureBar.classList.toggle("finalizing", mode === "finalizing");
}

function resetMeter() {
  window.clearInterval(state.recorderTimer);
  state.recorderTimer = null;
  state.recorderStartedAt = 0;
  els.recordTimer.textContent = "0:00";
  state.waveformLevels = state.waveformLevels.map(() => 0);
  renderWaveform();
  setMeterMode("hidden");
}

function startMeter() {
  state.recorderStartedAt = Date.now();
  updateRecordTimer();
  setMeterMode("active");
  window.clearInterval(state.recorderTimer);
  state.recorderTimer = window.setInterval(updateRecordTimer, 250);
}

function finalizeMeter() {
  window.clearInterval(state.recorderTimer);
  state.recorderTimer = null;
  updateRecordTimer();
  setMeterMode("finalizing");
}

function renderWaveform() {
  const bars = els.waveform.querySelectorAll(".waveBar");
  bars.forEach((bar, index) => {
    const level = state.waveformLevels[index] ?? 0;
    const height = Math.max(2, Math.round(2 + level * 24));
    bar.style.height = `${height}px`;
    bar.style.opacity = String(0.35 + Math.min(0.65, level * 0.9));
  });
}

function updateWaveformFromPcm(buffer) {
  const samples = new Int16Array(buffer);
  if (samples.length === 0) {
    return;
  }
  let sum = 0;
  for (let i = 0; i < samples.length; i += 1) {
    const normalized = samples[i] / 32768;
    sum += normalized * normalized;
  }
  const rms = Math.sqrt(sum / samples.length);
  const level = Math.min(1, Math.pow(rms * 8, 0.75));
  state.waveformLevels.push(level);
  state.waveformLevels = state.waveformLevels.slice(-42);
  renderWaveform();
}

function resetConversation() {
  window.clearTimeout(state.searchTimer);
  state.searchTimer = null;
  state.searchRequestId += 1;
  state.lastQuery = "";
  state.activeKey = "";
  els.transcript.value = "";
  renderResults([]);
  els.viewerTitle.textContent = "PDF";
  els.viewerPage.textContent = "No page selected";
  els.openTab.href = "#";
  els.pdfViewer.removeAttribute("src");
}

function transcriptWindow() {
  const text = els.transcript.value.trim();
  if (text.length <= 900) {
    return text;
  }
  return text.slice(text.length - 900);
}

function appendTranscript(text, isFinal = true) {
  const current = els.transcript.value.trim();
  const separator = current ? " " : "";
  els.transcript.value = `${current}${separator}${text}`.trim();
  els.transcript.scrollTop = els.transcript.scrollHeight;
  if (isFinal) {
    transcriptChanged(true);
  }
}

function transcriptChanged(force = false) {
  if (force || els.autoSearch.checked) {
    scheduleSearch();
  }
}

function scheduleSearch() {
  window.clearTimeout(state.searchTimer);
  state.searchTimer = window.setTimeout(() => runSearch(), 250);
}

async function loadStatus() {
  const response = await fetch("/api/status");
  if (!response.ok) {
    throw new Error("Status request failed");
  }
  const status = await response.json();
  setStatus(`${status.pdfs} PDFs / ${status.pages} pages`);
  return status;
}

function renderResults(results) {
  state.results = results;
  els.resultCount.textContent = String(results.length);
  els.results.innerHTML = "";

  if (results.length === 0) {
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.textContent = "No matches";
    els.results.appendChild(empty);
    return;
  }

  for (const result of results) {
    const key = `${result.doc_id}:${result.page}`;
    const button = document.createElement("button");
    button.type = "button";
    button.className = `result${state.activeKey === key ? " active" : ""}`;
    button.dataset.key = key;

    const top = document.createElement("div");
    top.className = "resultTop";

    const title = document.createElement("div");
    title.className = "resultTitle";
    title.textContent = `${result.pdf_name} p.${result.page}`;

    const score = document.createElement("div");
    score.className = "resultScore";
    score.textContent = result.score.toFixed(3);

    const snippet = document.createElement("div");
    snippet.className = "resultSnippet";
    snippet.textContent = result.snippet;

    top.append(title, score);
    button.append(top, snippet);
    button.addEventListener("click", () => openResult(result));
    els.results.appendChild(button);
  }
}

function openResult(result) {
  const url = `/pdf/${result.doc_id}#page=${result.page}`;
  state.activeKey = `${result.doc_id}:${result.page}`;
  els.viewerTitle.textContent = result.pdf_name;
  els.viewerPage.textContent = `Page ${result.page}`;
  els.openTab.href = url;
  els.pdfViewer.src = url;
  renderResults(state.results);
}

async function runSearch() {
  const query = transcriptWindow();
  if (!query || query === state.lastQuery) {
    return;
  }
  state.lastQuery = query;
  const requestId = state.searchRequestId + 1;
  state.searchRequestId = requestId;
  setStatus("Searching slides");

  const response = await fetch("/api/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, top_k: 7 }),
  });
  if (!response.ok) {
    setStatus("Search failed");
    return;
  }
  const payload = await response.json();
  if (requestId !== state.searchRequestId) {
    return;
  }
  renderResults(payload.results);
  if (payload.results.length > 0) {
    openResult(payload.results[0]);
  }
  if (!state.recording && !state.finalizing) {
    setStatus("Matches updated");
  }
}

async function startStreamingAsr(modelId) {
  const model = selectedModel();
  if (!model?.available) {
    setStatus(model?.reason || `${model?.label || "Model"} needs download`);
    setControls("idle");
    return;
  }
  resetConversation();
  resetMeter();
  setControls("connecting");
  state.recording = false;
  state.finalizing = false;
  setStatus(`Connecting ${model.label}`);

  const stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    },
  });
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  const websocket = new WebSocket(
    `${protocol}://${window.location.host}/ws/asr/${encodeURIComponent(modelId)}`,
  );
  websocket.binaryType = "arraybuffer";

  websocket.onmessage = (event) => {
    const message = JSON.parse(event.data);
    if (message.type === "loading") {
      setStatus(message.message);
    } else if (message.type === "ready") {
      const loadedModel = message.model ? ` (${message.model})` : "";
      setStatus(`Listening${loadedModel}`);
      state.recording = true;
      startMeter();
      setControls("recording");
    } else if (message.type === "transcript") {
      if (message.final) {
        appendTranscript(message.text, true);
        if (state.finalizing) {
          setStatus("Finalizing transcription");
        }
      } else {
        setStatus(message.text);
      }
    } else if (message.type === "error") {
      closeStreaming(message.message);
    } else if (message.type === "done") {
      closeStreaming("Transcription finished");
    }
  };
  websocket.onerror = () => {
    closeStreaming(`${model.label} connection failed`);
  };
  websocket.onclose = () => {
    if (state.websocket === websocket) {
      closeStreaming(state.finalizing ? "Transcription finished" : `${model.label} connection closed`);
    }
  };

  const audioContext = new AudioContext();
  await audioContext.audioWorklet.addModule("/static/pcm-worklet.js");
  const source = audioContext.createMediaStreamSource(stream);
  const worklet = new AudioWorkletNode(audioContext, "pcm-worklet", {
    processorOptions: { targetRate: 16000 },
  });
  worklet.port.onmessage = (event) => {
    updateWaveformFromPcm(event.data);
    if (websocket.readyState === WebSocket.OPEN) {
      websocket.send(event.data);
    }
  };
  const silentGain = audioContext.createGain();
  silentGain.gain.value = 0;
  source.connect(worklet);
  worklet.connect(silentGain);
  silentGain.connect(audioContext.destination);

  state.mediaStream = stream;
  state.audioContext = audioContext;
  state.worklet = worklet;
  state.silentGain = silentGain;
  state.websocket = websocket;
}

async function start() {
  await startStreamingAsr(state.selectedModelId);
}

function stopAudioPipeline() {
  if (state.worklet) {
    state.worklet.disconnect();
    state.worklet = null;
  }
  if (state.silentGain) {
    state.silentGain.disconnect();
    state.silentGain = null;
  }
  if (state.audioContext) {
    state.audioContext.close();
    state.audioContext = null;
  }
  if (state.mediaStream) {
    for (const track of state.mediaStream.getTracks()) {
      track.stop();
    }
    state.mediaStream = null;
  }
}

function stopRecording() {
  if (!state.recording) {
    return;
  }
  state.recording = false;
  state.finalizing = true;
  stopAudioPipeline();
  finalizeMeter();
  if (state.websocket?.readyState === WebSocket.OPEN) {
    state.websocket.send("stop");
  }
  setControls("finalizing");
  setStatus("Finalizing transcription");
}

function closeStreaming(statusText = "Stopped") {
  const websocket = state.websocket;
  state.websocket = null;
  state.recording = false;
  state.finalizing = false;
  stopAudioPipeline();
  if (websocket && websocket.readyState === WebSocket.OPEN) {
    websocket.close();
  }
  resetMeter();
  setControls("idle");
  setStatus(statusText);
  renderModelOptions();
}

els.micBtn.addEventListener("click", () => {
  if (state.recording) {
    stopRecording();
    return;
  }
  start().catch((error) => {
    closeStreaming(error.message);
  });
});
els.clearBtn.addEventListener("click", () => {
  resetConversation();
});
els.searchBtn.addEventListener("click", () => {
  state.lastQuery = "";
  runSearch();
});
els.transcript.addEventListener("input", () => transcriptChanged(false));

loadAsrModels().catch((error) => setStatus(error.message));
loadStatus().catch((error) => setStatus(error.message));
initWaveform();
renderResults([]);

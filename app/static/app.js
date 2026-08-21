(() => {
  const startBtn = document.getElementById("startBtn");
  const stopBtn = document.getElementById("stopBtn");
  const downloadTxtBtn = document.getElementById("downloadTxtBtn");
  const downloadJsonBtn = document.getElementById("downloadJsonBtn");
  const statusEl = document.getElementById("status");
  const timerEl = document.getElementById("timer");
  const transcriptEl = document.getElementById("transcript");
  const errorBox = document.getElementById("errorBox");
  const indGpu = document.getElementById("ind-gpu");
  const indWhisper = document.getElementById("ind-whisper");

  const TARGET_SAMPLE_RATE = 16000;

  let audioContext = null;
  let mediaStream = null;
  let sourceNode = null;
  let processorNode = null;
  let ws = null;
  let recording = false;
  let timerInterval = null;
  let startTime = null;
  let segments = [];
  let partialEl = null; // DOM node currently showing the live "partial" line

  function showError(msg) {
    errorBox.textContent = msg;
    errorBox.hidden = false;
  }

  function clearError() {
    errorBox.hidden = true;
    errorBox.textContent = "";
  }

  function setStatus(text) {
    statusEl.textContent = text;
    statusEl.classList.toggle("rec", text === "Recording");
  }

  function formatTime(totalSeconds) {
    const h = Math.floor(totalSeconds / 3600).toString().padStart(2, "0");
    const m = Math.floor((totalSeconds % 3600) / 60).toString().padStart(2, "0");
    const s = Math.floor(totalSeconds % 60).toString().padStart(2, "0");
    return `${h}:${m}:${s}`;
  }

  function startTimer() {
    startTime = Date.now();
    timerInterval = setInterval(() => {
      timerEl.textContent = formatTime((Date.now() - startTime) / 1000);
    }, 500);
  }

  function stopTimer() {
    if (timerInterval) clearInterval(timerInterval);
    timerInterval = null;
  }

  async function loadHealth() {
    try {
      const res = await fetch("/api/health");
      const data = await res.json();
      indGpu.textContent = `GPU: ${data.gpu_available ? "Available" : "Unavailable"}`;
      indGpu.classList.toggle("ok", data.gpu_available);
      indGpu.classList.toggle("warn", !data.gpu_available);

      indWhisper.textContent = `Local STT: ${data.whisper_loaded ? `Loaded (${data.whisper_device})` : "Not loaded"}`;
      indWhisper.classList.toggle("ok", !!data.whisper_loaded);
      indWhisper.classList.toggle("warn", !data.whisper_loaded);

    } catch {
      indGpu.textContent = "GPU: unknown";
      indWhisper.textContent = "Whisper: unknown";
    }
  }

  function renderFinalSegment(seg) {
    if (transcriptEl.querySelector(".placeholder")) {
      transcriptEl.innerHTML = "";
    }

    if (partialEl) {
      partialEl.remove();
      partialEl = null;
    }

    const div = document.createElement("div");
    div.className = "segment";

    const meta = document.createElement("span");
    meta.className = "meta";
    meta.textContent = `${seg.start}s - ${seg.end}s · ${seg.language}`;

    const text = document.createElement("span");
    text.className = "text";
    text.textContent = seg.text;

    div.appendChild(meta);
    div.appendChild(text);

    transcriptEl.appendChild(div);
    transcriptEl.scrollTop = transcriptEl.scrollHeight;
  }

  function renderPartialSegment(seg) {
    if (transcriptEl.querySelector(".placeholder")) {
      transcriptEl.innerHTML = "";
    }

    if (!partialEl) {
      partialEl = document.createElement("div");
      partialEl.className = "segment partial";

      const meta = document.createElement("span");
      meta.className = "meta";

      const text = document.createElement("span");
      text.className = "text";

      partialEl.appendChild(meta);
      partialEl.appendChild(text);

      transcriptEl.appendChild(partialEl);
    }

    partialEl.querySelector(".meta").textContent =
      `${seg.start}s - ${seg.end}s · live`;

    partialEl.querySelector(".text").textContent = seg.text;

    transcriptEl.scrollTop = transcriptEl.scrollHeight;
  }

  function resetTranscriptUI() {
    transcriptEl.innerHTML = '<div class="placeholder">Listening...</div>';
    segments = [];
    partialEl = null;
    downloadTxtBtn.disabled = true;
    downloadJsonBtn.disabled = true;
  }

  function downsampleTo16k(float32Data, inputSampleRate) {
    if (inputSampleRate === TARGET_SAMPLE_RATE) {
      return float32Data;
    }
    const ratio = inputSampleRate / TARGET_SAMPLE_RATE;
    const outLength = Math.round(float32Data.length / ratio);
    const result = new Float32Array(outLength);
    for (let i = 0; i < outLength; i++) {
      const srcIndex = i * ratio;
      const idxLow = Math.floor(srcIndex);
      const idxHigh = Math.min(idxLow + 1, float32Data.length - 1);
      const frac = srcIndex - idxLow;
      result[i] = float32Data[idxLow] * (1 - frac) + float32Data[idxHigh] * frac;
    }
    return result;
  }

  function floatTo16BitPCM(float32Data) {
    const buffer = new ArrayBuffer(float32Data.length * 2);
    const view = new DataView(buffer);
    for (let i = 0; i < float32Data.length; i++) {
      let s = Math.max(-1, Math.min(1, float32Data[i]));
      s = s < 0 ? s * 0x8000 : s * 0x7fff;
      view.setInt16(i * 2, s, true);
    }
    return buffer;
  }

  function connectWebSocket() {
    return new Promise((resolve, reject) => {
      const protocol = location.protocol === "https:" ? "wss:" : "ws:";
      ws = new WebSocket(`${protocol}//${location.host}/ws`);
      ws.binaryType = "arraybuffer";

      ws.onopen = () => resolve();
      ws.onerror = () => reject(new Error("Could not connect to the server."));

      ws.onmessage = (event) => {
        let msg;
        try {
          msg = JSON.parse(event.data);
        } catch {
          return;
        }
        handleServerMessage(msg);
      };

      ws.onclose = () => {
        if (recording) {
          showError("Connection to the server was lost.");
          stopRecording(true);
        }
      };
    });
  }

  function handleServerMessage(msg) {
    switch (msg.type) {
      case "segment":
        segments.push(msg.segment);
        renderFinalSegment(msg.segment);
        break;
      case "partial":
        renderPartialSegment(msg.segment);
        break;
      case "status":
        setStatus(msg.message.includes("Processing") ? "Processing" : recording ? "Recording" : msg.message);
        break;
      case "warning":
        showError(msg.message);
        break;
      case "error":
        showError(msg.message);
        break;
      case "done":
        finalizeAfterStop();
        break;
    }
  }

  async function startRecording() {
    clearError();
    startBtn.disabled = true;

    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
      });
    } catch (err) {
      showError("Microphone permission was denied. Please allow microphone access and try again.");
      startBtn.disabled = false;
      return;
    }

    try {
      await connectWebSocket();
    } catch (err) {
      showError("Could not connect to the backend server.");
      startBtn.disabled = false;
      mediaStream.getTracks().forEach((t) => t.stop());
      return;
    }

    resetTranscriptUI();
    const langSelect = document.getElementById("langSelect");
    const lang = langSelect ? langSelect.value : "auto";
    ws.send(JSON.stringify({ type: "start", lang: lang }));

    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    sourceNode = audioContext.createMediaStreamSource(mediaStream);

    // ScriptProcessorNode is deprecated but is still the simplest way to
    // get raw PCM samples without extra worklet files - fine for a POC.
    processorNode = audioContext.createScriptProcessor(4096, 1, 1);
    processorNode.onaudioprocess = (event) => {
      if (!recording) return;
      const input = event.inputBuffer.getChannelData(0);
      const downsampled = downsampleTo16k(input, audioContext.sampleRate);
      const pcm16 = floatTo16BitPCM(downsampled);
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(pcm16);
      }
    };

    sourceNode.connect(processorNode);
    processorNode.connect(audioContext.destination);

    recording = true;
    setStatus("Recording");
    startTimer();
    stopBtn.disabled = false;
  }

  function stopRecording(dueToError = false) {
    recording = false;
    stopTimer();
    stopBtn.disabled = true;

    if (processorNode) {
      processorNode.disconnect();
      processorNode.onaudioprocess = null;
      processorNode = null;
    }
    if (sourceNode) {
      sourceNode.disconnect();
      sourceNode = null;
    }
    if (audioContext) {
      audioContext.close();
      audioContext = null;
    }
    if (mediaStream) {
      mediaStream.getTracks().forEach((t) => t.stop());
      mediaStream = null;
    }

    if (dueToError) {
      startBtn.disabled = false;
      setStatus("Error");
      return;
    }

    if (ws && ws.readyState === WebSocket.OPEN) {
      setStatus("Processing");
      ws.send(JSON.stringify({ type: "stop" }));
    } else {
      finalizeAfterStop();
    }
  }

  function finalizeAfterStop() {
    setStatus("Finished");
    startBtn.disabled = false;
    downloadTxtBtn.disabled = segments.length === 0;
    downloadJsonBtn.disabled = segments.length === 0;
    if (ws) {
      ws.close();
      ws = null;
    }
  }

  function triggerDownload(url) {
    const a = document.createElement("a");
    a.href = url;
    a.click();
  }

  startBtn.addEventListener("click", startRecording);
  stopBtn.addEventListener("click", () => stopRecording(false));
  downloadTxtBtn.addEventListener("click", () => triggerDownload("/api/download/txt"));
  downloadJsonBtn.addEventListener("click", () => triggerDownload("/api/download/json"));

  resetTranscriptUI();
  transcriptEl.innerHTML = '<div class="placeholder">Press START to begin.</div>';
  loadHealth();
})();

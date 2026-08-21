import re, sys, pathlib

def patch(path, replacements):
    p = pathlib.Path(path)
    if not p.exists():
        print(f"SKIP (not found): {path}")
        return
    text = p.read_text()
    for old, new in replacements:
        count = text.count(old)
        if count != 1:
            print(f"WARNING: expected 1 match in {path}, found {count} - skipping this replacement (file may already be patched or differs from expected).")
            continue
        text = text.replace(old, new)
    p.write_text(text)
    print(f"Patched: {path}")

# ---------------------------------------------------------------- stt.py
patch("app/stt.py", [
    (
'''    def _run(self, audio_float32, beam_size: int):
        try:
            segments, info = self.model.transcribe(
                audio_float32,
                task="transcribe",
                language=None,  # auto-detect among si/ta/en (and anything else)
                beam_size=beam_size,
                vad_filter=False,  # we already did VAD segmentation upstream
                condition_on_previous_text=False,
            )''',
'''    def _run(self, audio_float32, beam_size: int, language: str | None = None):
        try:
            segments, info = self.model.transcribe(
                audio_float32,
                task="transcribe",
                # Auto-detecting language independently on every short
                # (often 1-3s) VAD segment is unreliable - a wrong guess
                # (e.g. Sinhala misdetected as Malay/Indonesian) sends the
                # WHOLE segment through the wrong-language decoder,
                # producing fluent-looking garbage instead of an error.
                # Lock a language for the session whenever you know it.
                language=language,
                beam_size=beam_size,
                vad_filter=False,  # we already did VAD segmentation upstream
                condition_on_previous_text=False,
            )'''
    ),
    (
'''    def transcribe(self, audio_float32, sample_rate: int = 16000):
        """Full-quality final decode."""
        return self._run(audio_float32, beam_size=5)

    def transcribe_fast(self, audio_float32, sample_rate: int = 16000):
        """Fast, lower-quality decode for live 'partial' previews only."""
        return self._run(audio_float32, beam_size=1)''',
'''    def transcribe(self, audio_float32, sample_rate: int = 16000, language: str | None = None):
        """Full-quality final decode."""
        return self._run(audio_float32, beam_size=5, language=language)

    def transcribe_fast(self, audio_float32, sample_rate: int = 16000, language: str | None = None):
        """Fast, lower-quality decode for live 'partial' previews only."""
        return self._run(audio_float32, beam_size=1, language=language)'''
    ),
])

# ---------------------------------------------------------------- main.py
patch("app/main.py", [
    ('VAD_MIN_SEGMENT_MS = int(os.environ.get("VAD_MIN_SEGMENT_MS", "400"))',
     'VAD_MIN_SEGMENT_MS = int(os.environ.get("VAD_MIN_SEGMENT_MS", "600"))'),
    ('SPEAKER_MIN_ENROLL_SECONDS = float(os.environ.get("SPEAKER_MIN_ENROLL_SECONDS", "0.6"))',
     'SPEAKER_MIN_ENROLL_SECONDS = float(os.environ.get("SPEAKER_MIN_ENROLL_SECONDS", "1.2"))'),
    (
'''async def _finalize_segment(pcm_bytes, start_s, end_s, speaker_mgr: SpeakerManager, loop, websocket):
    """Runs the full-quality transcription + speaker ID for a closed
    segment, appends it to the session transcript, and sends it to the
    client. Guards against GPU OOM without killing the session."""
    audio = pcm16_bytes_to_float32(pcm_bytes)
    duration = max(0.0, end_s - start_s)

    try:
        text, language = await loop.run_in_executor(None, stt_engine.transcribe, audio)''',
'''async def _finalize_segment(pcm_bytes, start_s, end_s, speaker_mgr: SpeakerManager, loop, websocket, forced_language=None):
    """Runs the full-quality transcription + speaker ID for a closed
    segment, appends it to the session transcript, and sends it to the
    client. Guards against GPU OOM without killing the session."""
    audio = pcm16_bytes_to_float32(pcm_bytes)
    duration = max(0.0, end_s - start_s)

    try:
        text, language = await loop.run_in_executor(None, stt_engine.transcribe, audio, 16000, forced_language)'''
    ),
    (
'''async def _send_partial(segmenter: Segmenter, loop, websocket):
    """Fast, low-quality preview of whatever's currently being spoken.
    Never written to the saved transcript - just replaced client-side
    once the real (final) segment arrives."""
    peek = segmenter.peek_in_progress()
    if peek is None:
        return
    pcm_bytes, start_s, current_s = peek
    audio = pcm16_bytes_to_float32(pcm_bytes)
    try:
        text, language = await loop.run_in_executor(None, stt_engine.transcribe_fast, audio)''',
'''async def _send_partial(segmenter: Segmenter, loop, websocket, forced_language=None):
    """Fast, low-quality preview of whatever's currently being spoken.
    Never written to the saved transcript - just replaced client-side
    once the real (final) segment arrives."""
    peek = segmenter.peek_in_progress()
    if peek is None:
        return
    pcm_bytes, start_s, current_s = peek
    audio = pcm16_bytes_to_float32(pcm_bytes)
    try:
        text, language = await loop.run_in_executor(None, stt_engine.transcribe_fast, audio, 16000, forced_language)'''
    ),
    (
'''    segmenter = _new_segmenter()
    speaker_mgr = _new_speaker_manager()

    # Only one partial-preview decode in flight at a time (one active
    # session, one GPU - no point queueing several).
    partial_in_flight = False
    last_partial_time = 0.0

    async def maybe_send_partial():
        nonlocal partial_in_flight, last_partial_time
        if partial_in_flight:
            return
        now = loop.time()
        if now - last_partial_time < PARTIAL_UPDATE_INTERVAL_S:
            return
        last_partial_time = now
        partial_in_flight = True
        try:
            await _send_partial(segmenter, loop, websocket)
        finally:
            partial_in_flight = False''',
'''    segmenter = _new_segmenter()
    speaker_mgr = _new_speaker_manager()

    # Session-locked transcription language. None means "auto-detect per
    # segment" (kept as an option, not recommended for si/ta). Any other
    # value ("si"/"ta"/"en") skips per-segment language ID entirely -
    # this is what fixes the si/ta wrong-language garbage.
    forced_language = None

    # Only one partial-preview decode in flight at a time (one active
    # session, one GPU - no point queueing several).
    partial_in_flight = False
    last_partial_time = 0.0

    async def maybe_send_partial():
        nonlocal partial_in_flight, last_partial_time
        if partial_in_flight:
            return
        now = loop.time()
        if now - last_partial_time < PARTIAL_UPDATE_INTERVAL_S:
            return
        last_partial_time = now
        partial_in_flight = True
        try:
            await _send_partial(segmenter, loop, websocket, forced_language)
        finally:
            partial_in_flight = False'''
    ),
    (
'''                msg_type = control.get("type")
                if msg_type == "start":
                    segmenter = _new_segmenter()
                    speaker_mgr = _new_speaker_manager()
                    session_transcript.clear()
                    last_partial_time = 0.0
                    await websocket.send_json({"type": "status", "message": "Recording started."})

                elif msg_type == "stop":
                    await websocket.send_json({"type": "status", "message": "Processing final audio..."})
                    tail = segmenter.flush()
                    if tail:
                        pcm_bytes, start_s, end_s = tail
                        await _finalize_segment(pcm_bytes, start_s, end_s, speaker_mgr, loop, websocket)
                    await websocket.send_json({"type": "done"})''',
'''                msg_type = control.get("type")
                if msg_type == "start":
                    segmenter = _new_segmenter()
                    speaker_mgr = _new_speaker_manager()
                    session_transcript.clear()
                    last_partial_time = 0.0
                    # "lang" comes from the client's language dropdown:
                    # "si"/"ta"/"en" locks that language for the whole
                    # session; "auto" (or omitted) falls back to
                    # per-segment auto-detect.
                    lang = (control.get("lang") or "auto").strip().lower()
                    forced_language = None if lang == "auto" else lang
                    await websocket.send_json({"type": "status", "message": "Recording started."})

                elif msg_type == "stop":
                    await websocket.send_json({"type": "status", "message": "Processing final audio..."})
                    tail = segmenter.flush()
                    if tail:
                        pcm_bytes, start_s, end_s = tail
                        await _finalize_segment(pcm_bytes, start_s, end_s, speaker_mgr, loop, websocket, forced_language)
                    await websocket.send_json({"type": "done"})'''
    ),
    (
'''                for pcm_bytes, start_s, end_s in completed:
                    await _finalize_segment(pcm_bytes, start_s, end_s, speaker_mgr, loop, websocket)''',
'''                for pcm_bytes, start_s, end_s in completed:
                    await _finalize_segment(pcm_bytes, start_s, end_s, speaker_mgr, loop, websocket, forced_language)'''
    ),
])

# ---------------------------------------------------------------- index.html
patch("app/static/index.html", [
    (
'''    <div class="controls">
      <button id="startBtn">START</button>
      <button id="stopBtn" disabled>STOP</button>
    </div>''',
'''    <div class="controls">
      <label for="langSelect" style="margin-right:8px;">Language:</label>
      <select id="langSelect">
        <option value="si">Sinhala</option>
        <option value="ta">Tamil</option>
        <option value="en">English</option>
        <option value="auto">Auto-detect (not recommended for si/ta)</option>
      </select>
      <button id="startBtn">START</button>
      <button id="stopBtn" disabled>STOP</button>
    </div>'''
    ),
])

# ---------------------------------------------------------------- app.js
patch("app/static/app.js", [
    (
'''    ws.send(JSON.stringify({ type: "start" }));''',
'''    const langSelect = document.getElementById("langSelect");
    const lang = langSelect ? langSelect.value : "auto";
    ws.send(JSON.stringify({ type: "start", lang: lang }));'''
    ),
])

# ---------------------------------------------------------------- run.sh
patch("run.sh", [
    (
'''source venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000''',
'''source venv/bin/activate

# The pip-installed nvidia-cublas/nvidia-cudnn wheels ship their .so files
# inside the venv instead of the system CUDA path - without this,
# ctranslate2/faster-whisper fails per-segment with
# "Library libcublas.so.12 is not found or cannot be loaded".
export LD_LIBRARY_PATH="$PWD/venv/lib/python3.12/site-packages/nvidia/cublas/lib:$PWD/venv/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH"

uvicorn app.main:app --host 0.0.0.0 --port 8000'''
    ),
])

print("Done. If any WARNING lines appeared above, paste them here plus that file's content.")

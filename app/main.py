import asyncio
import json
import os
import logging
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles

from app.audio import Segmenter, pcm16_bytes_to_float32
from app.stt import create_stt_engine, GPUMemoryError
from app.transcript import format_txt, format_json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("court-stt")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"


def _load_dotenv(path: Path):
    """Tiny .env loader so we don't need an extra dependency for this."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv(BASE_DIR.parent / ".env")

# --- Whisper config ---
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large-v3")
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cuda")
WHISPER_COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "float16")
WHISPER_MODEL_DIR = os.environ.get("WHISPER_MODEL_DIR") or None

# --- VAD / segmentation config ---
VAD_THRESHOLD = float(os.environ.get("VAD_THRESHOLD", "0.5"))
VAD_PADDING_MS = int(os.environ.get("VAD_PADDING_MS", "300"))
VAD_TRIGGER_RATIO = float(os.environ.get("VAD_TRIGGER_RATIO", "0.9"))
VAD_MIN_SEGMENT_MS = int(os.environ.get("VAD_MIN_SEGMENT_MS", "600"))
VAD_MAX_SEGMENT_MS = int(os.environ.get("VAD_MAX_SEGMENT_MS", "20000"))

# --- Live partial-transcript config ---
PARTIAL_UPDATE_INTERVAL_S = float(os.environ.get("PARTIAL_UPDATE_INTERVAL_S", "0.8"))

app = FastAPI(title="Local Court STT POC")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Loaded lazily on startup so the process can still boot (and report a
# clear error over the websocket / health endpoint) even if the
# model/GPU/embedder isn't ready.
stt_engine = None
stt_load_warning = None
# Single-user POC: one in-memory transcript "session" is enough - no DB,
# no auth, no multi-tenant concerns per the project scope.
session_transcript: list = []


@app.on_event("startup")
async def load_models():
    global stt_engine, stt_load_warning

    loop = asyncio.get_event_loop()

    logger.info(
        "Loading Sinhala STT model '%s' on %s...",
        WHISPER_MODEL,
        WHISPER_DEVICE,
    )

    stt_engine, stt_load_warning = await loop.run_in_executor(
        None,
        create_stt_engine,
        WHISPER_MODEL,
        WHISPER_DEVICE,
        WHISPER_COMPUTE_TYPE,
        WHISPER_MODEL_DIR,
    )

    if stt_load_warning:
        logger.warning(stt_load_warning)
    else:
        logger.info("Sinhala STT model loaded on %s.", stt_engine.device)


def _gpu_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/health")
async def health():
    return {
        "gpu_available": _gpu_available(),
        "whisper_loaded": stt_engine is not None,
        "whisper_device": stt_engine.device if stt_engine else None,
        "whisper_warning": stt_load_warning,
    }


@app.get("/api/download/txt")
async def download_txt():
    body = format_txt(session_transcript)
    return PlainTextResponse(
        body,
        headers={"Content-Disposition": "attachment; filename=transcript.txt"},
    )


@app.get("/api/download/json")
async def download_json():
    body = format_json(session_transcript)
    return Response(
        content=body,
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=transcript.json"},
    )


def _new_segmenter() -> Segmenter:
    return Segmenter(
        vad_threshold=VAD_THRESHOLD,
        padding_ms=VAD_PADDING_MS,
        trigger_ratio=VAD_TRIGGER_RATIO,
        min_segment_ms=VAD_MIN_SEGMENT_MS,
        max_segment_ms=VAD_MAX_SEGMENT_MS,
    )




async def _finalize_segment(
    pcm_bytes,
    start_s,
    end_s,
    loop,
    websocket,
    forced_language=None,
):
    """Run full-quality local Sinhala STT and send the transcript."""
    audio = pcm16_bytes_to_float32(pcm_bytes)

    try:
        text, language = await loop.run_in_executor(
            None,
            stt_engine.transcribe,
            audio,
            16000,
            forced_language,
        )
    except GPUMemoryError:
        logger.warning("GPU out of memory - skipped one segment.")
        await websocket.send_json({
            "type": "error",
            "message": "GPU memory insufficient - skipped a segment.",
        })
        return
    except Exception:
        logger.exception("Transcription failed")
        await websocket.send_json({
            "type": "error",
            "message": "Transcription failed for a segment.",
        })
        return

    if not text:
        return

    segment = {
        "start": round(start_s, 2),
        "end": round(end_s, 2),
        "text": text,
        "language": language,
    }

    session_transcript.append(segment)

    await websocket.send_json({
        "type": "segment",
        "segment": segment,
    })


async def _send_partial(segmenter: Segmenter, loop, websocket, forced_language=None):
    """Fast, low-quality preview of whatever's currently being spoken.
    Never written to the saved transcript - just replaced client-side
    once the real (final) segment arrives."""
    peek = segmenter.peek_in_progress()
    if peek is None:
        return
    pcm_bytes, start_s, current_s = peek
    audio = pcm16_bytes_to_float32(pcm_bytes)
    try:
        text, language = await loop.run_in_executor(None, stt_engine.transcribe_fast, audio, 16000, forced_language)
    except Exception:
        return
    if not text:
        return
    await websocket.send_json({
        "type": "partial",
        "segment": {
            "start": round(start_s, 2),
            "end": round(current_s, 2),
            "text": text,
            "language": language,
        },
    })


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    loop = asyncio.get_event_loop()

    if stt_engine is None:
        await websocket.send_json({"type": "error", "message": "Model is still loading, please wait and retry."})
        await websocket.close()
        return

    if stt_load_warning:
        await websocket.send_json({"type": "warning", "message": stt_load_warning})
    segmenter = _new_segmenter()
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
            partial_in_flight = False

    try:
        while True:
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                break

            text_payload = message.get("text")
            bytes_payload = message.get("bytes")

            if text_payload is not None:
                try:
                    control = json.loads(text_payload)
                except json.JSONDecodeError:
                    continue

                msg_type = control.get("type")
                if msg_type == "start":
                    segmenter = _new_segmenter()
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
                        await _finalize_segment(pcm_bytes, start_s, end_s, loop, websocket, forced_language)
                    await websocket.send_json({"type": "done"})

            elif bytes_payload is not None:
                try:
                    completed = segmenter.add_audio(bytes_payload)
                except Exception:
                    logger.exception("Error processing audio chunk")
                    await websocket.send_json({"type": "error", "message": "Audio processing error."})
                    continue

                for pcm_bytes, start_s, end_s in completed:
                    await _finalize_segment(pcm_bytes, start_s, end_s, loop, websocket, forced_language)

                if not completed:
                    # nothing finalized this chunk - maybe emit a live
                    # preview of the sentence still being spoken
                    asyncio.create_task(maybe_send_partial())

    except WebSocketDisconnect:
        logger.info("Client disconnected.")
    except Exception:
        logger.exception("Unexpected WebSocket error")
        try:
            await websocket.send_json({"type": "error", "message": "Unexpected server error."})
        except Exception:
            pass

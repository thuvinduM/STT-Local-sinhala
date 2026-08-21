"""
Thin wrapper around faster-whisper for local, offline transcription.

Language is auto-detected per segment (Whisper large-v3 supports Sinhala,
Tamil and English natively) rather than forced, since a speaker could
switch languages between turns. Task is always "transcribe", never
"translate".

Two decode modes:
- transcribe(): full-quality final decode (beam_size=5), used once a
  segment is finalized (VAD closed it, or the session stopped).
- transcribe_fast(): low-beam, no-context decode used only for the
  "partial" live preview of a segment that's still being spoken. It never
  becomes part of the saved transcript - it's just replaced by the final
  decode once the segment closes.
"""

import logging
from faster_whisper import WhisperModel

from app.transcript import dedupe_repeated_phrases

logger = logging.getLogger("court-stt.stt")


class GPUMemoryError(RuntimeError):
    """Raised when a transcription fails specifically due to GPU OOM, so
    callers can tell that apart from other failures and keep the session
    alive instead of crashing it."""


def _is_oom_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda oom" in msg or "cublas_status_alloc_failed" in msg


class STTEngine:
    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
        download_root: str | None = None,
    ):
        self.device = device
        self.compute_type = compute_type
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            download_root=download_root or None,
        )

    def _run(self, audio_float32, beam_size: int, language: str | None = None):
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
            )
            text = " ".join(seg.text.strip() for seg in segments if seg.text.strip())
            text = dedupe_repeated_phrases(text.strip())
            language = getattr(info, "language", None) or "unknown"
            return text, language
        except Exception as e:
            if _is_oom_error(e):
                raise GPUMemoryError(str(e)) from e
            raise

    def transcribe(self, audio_float32, sample_rate: int = 16000, language: str | None = None):
        """Full-quality final decode."""
        return self._run(audio_float32, beam_size=5, language=language)

    def transcribe_fast(self, audio_float32, sample_rate: int = 16000, language: str | None = None):
        """Fast, lower-quality decode for live 'partial' previews only."""
        return self._run(audio_float32, beam_size=1, language=language)


def create_stt_engine(model_size: str, device: str, compute_type: str, download_root: str | None):
    """Tries the requested device first, falls back to CPU if the GPU
    path fails to load (missing/incompatible CUDA libs, no GPU, etc.),
    so the app still starts up and is testable. Returns (engine, warning)."""
    try:
        engine = STTEngine(model_size, device=device, compute_type=compute_type, download_root=download_root)
        return engine, None
    except Exception as e:
        warning = (
            f"Could not load Whisper on '{device}' ({e}). "
            "Falling back to CPU - transcription will be much slower."
        )
        engine = STTEngine(model_size, device="cpu", compute_type="int8", download_root=download_root)
        return engine, warning

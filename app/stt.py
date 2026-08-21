"""
Sinhala STT backend for court-stt.

Uses the already-working local Sinhala Whisper-small model from
dfcc_voicebot. The existing STTEngine interface is preserved so the
rest of court-stt does not need to change.
"""

import logging

import librosa
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

from app.transcript import dedupe_repeated_phrases


logger = logging.getLogger("court-stt.stt")


MODEL_PATH = "/home/thuvindu/dfcc_voicebot/models/whisper-small-si"
MODEL_DEVICE = "cuda"


class GPUMemoryError(RuntimeError):
    """Raised when transcription fails because of GPU memory."""


def _is_oom_error(exc: Exception) -> bool:
    msg = str(exc).lower()

    return (
        "out of memory" in msg
        or "cuda oom" in msg
        or "cublas_status_alloc_failed" in msg
    )


class STTEngine:

    def __init__(
        self,
        model_size: str = "whisper-small-si",
        device: str = "cuda",
        compute_type: str = "float16",
        download_root: str | None = None,
    ):
        self.device = device
        self.compute_type = compute_type

        if device != "cuda":
            raise RuntimeError(
                "Sinhala STT requires CUDA. "
                "The local whisper-small-si model is configured for GPU."
            )

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available.")

        logger.info("Loading Sinhala STT model from %s", MODEL_PATH)

        self.processor = AutoProcessor.from_pretrained(
            MODEL_PATH
        )

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to("cuda")

        self.model.eval()

        logger.info(
            "Sinhala STT ready on %s (%s)",
            torch.cuda.get_device_name(0),
            torch.__version__,
        )

    def _run(
        self,
        audio_float32,
        beam_size: int,
        language: str | None = None,
    ):
        try:
            # Existing court-stt audio pipeline provides float32 audio.
            # Normalize/resample here defensively.
            audio_float32 = audio_float32.astype("float32")

            inputs = self.processor(
                audio_float32,
                sampling_rate=16000,
                return_tensors="pt",
            )

            input_features = inputs.input_features.to(
                "cuda",
                dtype=torch.float16,
            )

            attention_mask = inputs.get("attention_mask")

            if attention_mask is not None:
                attention_mask = attention_mask.to("cuda")

            with torch.inference_mode():

                generated_ids = self.model.generate(
                    input_features=input_features,
                    attention_mask=attention_mask,
                    language="si",
                    task="transcribe",
                    max_new_tokens=440,

                    # beam_size is kept in the public API for compatibility.
                    # The local Sinhala model is optimized for direct
                    # Sinhala transcription.
                    num_beams=max(1, beam_size),
                )

            text = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )[0].strip()

            text = dedupe_repeated_phrases(text)

            return text, "si"

        except Exception as e:

            if _is_oom_error(e):
                raise GPUMemoryError(str(e)) from e

            raise

    def transcribe(
        self,
        audio_float32,
        sample_rate: int = 16000,
        language: str | None = None,
    ):
        """
        Full-quality final transcription.
        """

        if sample_rate != 16000:
            audio_float32 = librosa.resample(
                audio_float32.astype("float32"),
                orig_sr=sample_rate,
                target_sr=16000,
            )

        return self._run(
            audio_float32,
            beam_size=5,
            language="si",
        )

    def transcribe_fast(
        self,
        audio_float32,
        sample_rate: int = 16000,
        language: str | None = None,
    ):
        """
        Faster transcription for live partial previews.
        """

        if sample_rate != 16000:
            audio_float32 = librosa.resample(
                audio_float32.astype("float32"),
                orig_sr=sample_rate,
                target_sr=16000,
            )

        return self._run(
            audio_float32,
            beam_size=1,
            language="si",
        )


def create_stt_engine(
    model_size: str,
    device: str,
    compute_type: str,
    download_root: str | None,
):
    """
    Create the local Sinhala STT engine.

    Important:
    We intentionally do NOT fall back to CPU because this backend is
    designed to use the existing CUDA environment from dfcc_voicebot.
    """

    try:

        engine = STTEngine(
            model_size=model_size,
            device="cuda",
            compute_type="float16",
            download_root=download_root,
        )

        return engine, None

    except Exception as e:

        warning = (
            f"Could not load local Sinhala STT model on GPU: {e}"
        )

        logger.exception(warning)

        return None, warning

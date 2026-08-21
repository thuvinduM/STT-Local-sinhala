"""
Silero VAD wrapper - drop-in replacement for webrtcvad.Vad, exposing the
same is_speech(frame_bytes, sample_rate) interface used by app/audio.py.

Silero is a small neural VAD (~2MB) that models what speech actually
sounds like, instead of just measuring frame energy like webrtcvad does.
That makes it far more robust to steady background noise (fans, AC hum,
courtroom murmur/chatter) which is what was causing false speech triggers.

Runs on CPU in real time - no GPU needed for this.
"""

import logging

import numpy as np
import torch
from silero_vad import load_silero_vad

logger = logging.getLogger("court-stt.vad_silero")

SILERO_FRAME_SAMPLES_16K = 512  # 32ms at 16kHz

_model = None


def _get_model():
    global _model
    if _model is None:
        logger.info("Loading Silero VAD model (CPU, cached after first load)")
        _model = load_silero_vad()
        _model.eval()
    return _model


class SileroVAD:
    """Same interface as webrtcvad.Vad: .is_speech(frame_bytes, sample_rate) -> bool"""

    def __init__(self, threshold: float = 0.5):
        self.model = _get_model()
        self.threshold = threshold
        try:
            self.model.reset_states()
        except AttributeError:
            pass

    def is_speech(self, frame_bytes: bytes, sample_rate: int) -> bool:
        if sample_rate != 16000:
            raise ValueError("SileroVAD wrapper only supports 16kHz")

        audio = np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        if len(audio) < SILERO_FRAME_SAMPLES_16K:
            audio = np.pad(audio, (0, SILERO_FRAME_SAMPLES_16K - len(audio)))
        elif len(audio) > SILERO_FRAME_SAMPLES_16K:
            audio = audio[:SILERO_FRAME_SAMPLES_16K]

        tensor = torch.from_numpy(audio)
        with torch.no_grad():
            prob = self.model(tensor, sample_rate).item()

        return prob > self.threshold

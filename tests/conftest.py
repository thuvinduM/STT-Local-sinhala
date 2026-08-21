import numpy as np

SAMPLE_RATE = 16000


def tone_pcm16(ms: int, amplitude: int, freq: int = 440) -> bytes:
    """Generate a mono PCM16 sine tone (or silence if amplitude=0) for
    feeding into the Segmenter in tests."""
    n = int(SAMPLE_RATE * ms / 1000)
    t = np.arange(n)
    wave = (amplitude * np.sin(2 * np.pi * freq * t / SAMPLE_RATE)).astype(np.int16)
    return wave.tobytes()


def voice_float32(freq: int, seconds: float = 1.0, amplitude: float = 0.5) -> np.ndarray:
    """A distinguishable synthetic 'voice' (just a sine wave at a given
    pitch) for diarization tests - see tests/test_diarization.py for why
    real embeddings aren't used here."""
    t = np.linspace(0, seconds, int(SAMPLE_RATE * seconds))
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)

"""
Speaker diarization logic tests.

IMPORTANT / HONEST LIMITATION: these tests inject a fake embedder (a
deterministic FFT-magnitude "fingerprint" of a synthetic tone) instead of
loading the real SpeechBrain ECAPA model, because this sandbox has no
network access to download it and no real multi-speaker audio fixtures
to test with. This validates the SpeakerManager's *matching/clustering
logic* (thresholds, prototypes, min-duration guard) correctly - it does
NOT validate real-world embedding quality on real voices.

Before trusting diarization on real recordings, run a manual test on
your Ubuntu server with 2-3 real speakers - see README "Testing" section.
"""

import numpy as np

from app.diarization import SpeakerManager
from tests.conftest import voice_float32


class FakeEmbedder:
    """Deterministic stand-in that gives different 'pitches' different
    embeddings, and the same pitch (+ small noise) the same embedding -
    same relationship real speaker embeddings have to different voices."""

    def embed(self, wav: np.ndarray) -> np.ndarray:
        spec = np.abs(np.fft.rfft(wav, n=512))
        return spec / (np.linalg.norm(spec) + 1e-8)


def noisy(wav: np.ndarray, rng: np.random.Generator, scale: float = 0.01) -> np.ndarray:
    return (wav + rng.normal(0, scale, wav.shape)).astype(np.float32)


def test_stable_labels_across_a_full_session():
    """The scenario from the spec: A,B,A,C,B,A -> Speaker 1,2,1,3,2,1"""
    rng = np.random.default_rng(1)
    A, B, C = voice_float32(150), voice_float32(300), voice_float32(500)
    mgr = SpeakerManager(embedder=FakeEmbedder(), match_threshold=0.8, min_enroll_seconds=0.3)

    sequence = [A, B, A, C, B, A]
    labels = [mgr.identify(noisy(v, rng), duration_seconds=0.5)[0] for v in sequence]

    assert labels == ["Speaker 1", "Speaker 2", "Speaker 1", "Speaker 3", "Speaker 2", "Speaker 1"]
    assert mgr.speaker_count == 3


def test_short_utterance_does_not_create_phantom_speaker():
    rng = np.random.default_rng(2)
    A = voice_float32(150)
    weird_short_blip = voice_float32(320)  # would look like a different speaker if trusted
    mgr = SpeakerManager(embedder=FakeEmbedder(), match_threshold=0.8, min_enroll_seconds=0.6)

    mgr.identify(noisy(A, rng), duration_seconds=1.0)
    mgr.identify(noisy(weird_short_blip, rng), duration_seconds=0.2)  # too short to enroll
    mgr.identify(noisy(A, rng), duration_seconds=1.0)

    assert mgr.speaker_count == 1


def test_similar_voices_do_not_get_merged_below_threshold():
    rng = np.random.default_rng(3)
    A, B = voice_float32(150), voice_float32(170)  # close in pitch but distinguishable
    mgr = SpeakerManager(embedder=FakeEmbedder(), match_threshold=0.999, min_enroll_seconds=0.3)

    label_a = mgr.identify(noisy(A, rng, scale=0.001), duration_seconds=1.0)[0]
    label_b = mgr.identify(noisy(B, rng, scale=0.001), duration_seconds=1.0)[0]

    assert label_a != label_b, "a high match threshold should keep close-but-different voices separate"


def test_reset_clears_speakers_for_a_new_session():
    rng = np.random.default_rng(4)
    A = voice_float32(150)
    mgr = SpeakerManager(embedder=FakeEmbedder(), match_threshold=0.8, min_enroll_seconds=0.3)
    mgr.identify(noisy(A, rng), duration_seconds=1.0)
    assert mgr.speaker_count == 1

    mgr.reset()
    assert mgr.speaker_count == 0
    label, _ = mgr.identify(noisy(A, rng), duration_seconds=1.0)
    assert label == "Speaker 1", "labels must restart from Speaker 1 on a new session"


def test_missing_embedder_falls_back_to_single_speaker_gracefully():
    mgr = SpeakerManager(embedder=None)
    label, confidence = mgr.identify(np.zeros(1600, dtype=np.float32), duration_seconds=1.0)
    assert label == "Speaker 1"
    assert confidence == 0.0

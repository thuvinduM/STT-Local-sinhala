"""
Session-level speaker diarization.

This is intentionally NOT a full offline pipeline like pyannote.audio's
speaker-diarization-3.1 - that pipeline re-processes the whole recording
at once (segmentation + embedding + agglomerative clustering over
everything), which doesn't fit a live "transcribe as you go" POC, and its
embedding checkpoints are gated on HuggingFace (need to accept a license
and pass an HF token). For a *live* session, an online/incremental
approach is the practical choice.

What this does use, though, is the same *quality* of embedding model
pyannote itself relies on: SpeechBrain's ECAPA-TDNN speaker encoder
(speechbrain/spkrec-ecapa-voxceleb), which is free to download (no gated
license/token) and is a meaningfully stronger embedding than the older
GE2E model Resemblyzer uses. If SpeechBrain isn't installed or fails to
load (e.g. no internet on first run), we fall back to Resemblyzer so the
app still starts - same fallback pattern used for the Whisper GPU/CPU
switch.

SpeakerManager (the actual diarization logic):
- Each speaker is represented by a small set of embedding "prototypes"
  (not just one running average), so a voice's natural variation
  (pitch, volume, distance from mic) doesn't fool the matcher.
- A segment is matched to a speaker if its embedding is close enough to
  that speaker's prototypes (cosine similarity >= match_threshold).
  Below that, a new speaker is created.
- Very short utterances produce unreliable embeddings, so segments
  shorter than min_enroll_seconds are never allowed to *create* a new
  speaker - they're assigned to the closest existing speaker instead (or
  become Speaker 1 if this is the very first segment of the session).
- We deliberately do NOT auto-merge two already-created speakers later on
  - similar voices merging incorrectly is worse than an occasional extra
  speaker label, so that stays a manual/known limitation (see README).
"""

import logging
import numpy as np

logger = logging.getLogger("court-stt.diarization")

EMBEDDING_SAMPLE_RATE = 16000


class _SpeechBrainEmbedder:
    """ECAPA-TDNN embeddings via SpeechBrain, run on CPU so GPU memory
    stays free for Whisper."""

    def __init__(self, savedir: str = ".cache/speechbrain"):
        from speechbrain.inference.speaker import EncoderClassifier

        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=savedir,
            run_opts={"device": "cpu"},
        )

    def embed(self, audio_float32: np.ndarray) -> np.ndarray:
        import torch

        with torch.no_grad():
            tensor = torch.from_numpy(audio_float32).float().unsqueeze(0)
            emb = self.model.encode_batch(tensor)
            return emb.squeeze().cpu().numpy()


class _ResemblyzerEmbedder:
    """Fallback embedder if SpeechBrain isn't available."""

    def __init__(self):
        from resemblyzer import VoiceEncoder

        self.encoder = VoiceEncoder(device="cpu")

    def embed(self, audio_float32: np.ndarray) -> np.ndarray:
        return self.encoder.embed_utterance(audio_float32)


def create_embedder():
    """Returns (embedder, backend_name, warning)."""
    try:
        return _SpeechBrainEmbedder(), "speechbrain-ecapa", None
    except Exception as e:
        warning = f"Could not load SpeechBrain ECAPA embedder ({e}); falling back to Resemblyzer."
        logger.warning(warning)
        try:
            return _ResemblyzerEmbedder(), "resemblyzer", warning
        except Exception as e2:
            raise RuntimeError(f"No speaker embedding backend available: {e2}") from e2


class Speaker:
    def __init__(self, label: str, embedding: np.ndarray, max_prototypes: int = 8):
        self.label = label
        self.max_prototypes = max_prototypes
        self.prototypes = [embedding]

    @property
    def centroid(self) -> np.ndarray:
        return np.mean(self.prototypes, axis=0)

    def add_prototype(self, embedding: np.ndarray):
        self.prototypes.append(embedding)
        if len(self.prototypes) > self.max_prototypes:
            self.prototypes.pop(0)  # simple ring buffer, oldest first

    def similarity(self, embedding: np.ndarray) -> float:
        # Compare against both the centroid (stable average) and the
        # single closest prototype (captures natural variation) and take
        # the more generous of the two - a returning voice only needs to
        # be close to *one* thing it's said before, not the average of
        # everything.
        centroid_sim = _cosine(embedding, self.centroid)
        proto_sim = max(_cosine(embedding, p) for p in self.prototypes)
        return max(centroid_sim, proto_sim)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


class SpeakerManager:
    def __init__(
        self,
        embedder=None,
        match_threshold: float = 0.72,
        min_enroll_seconds: float = 0.6,
        max_prototypes: int = 8,
    ):
        self.embedder = embedder
        self.match_threshold = match_threshold
        self.min_enroll_seconds = min_enroll_seconds
        self.max_prototypes = max_prototypes
        self._speakers = []  # list[Speaker]
        self._next_id = 1

    def reset(self):
        self._speakers = []
        self._next_id = 1

    def identify(self, audio_float32: np.ndarray, duration_seconds: float):
        """Returns (label, confidence). confidence is the cosine
        similarity to the matched/created speaker's prototypes (1.0 for a
        brand new speaker's very first segment)."""
        if self.embedder is None or audio_float32.size == 0:
            return self._new_label(), 0.0

        try:
            embedding = self.embedder.embed(audio_float32)
        except Exception:
            logger.exception("Embedding failed, using best-effort label")
            return (self._speakers[-1].label if self._speakers else self._new_label()), 0.0

        if not self._speakers:
            self._speakers.append(Speaker(self._new_label(), embedding, self.max_prototypes))
            return self._speakers[-1].label, 1.0

        sims = [s.similarity(embedding) for s in self._speakers]
        best_idx = int(np.argmax(sims))
        best_sim = sims[best_idx]

        too_short_to_enroll = duration_seconds < self.min_enroll_seconds

        if best_sim >= self.match_threshold or too_short_to_enroll:
            speaker = self._speakers[best_idx]
            if not too_short_to_enroll:
                speaker.add_prototype(embedding)
            return speaker.label, best_sim

        new_speaker = Speaker(self._new_label(), embedding, self.max_prototypes)
        self._speakers.append(new_speaker)
        return new_speaker.label, best_sim

    def _new_label(self) -> str:
        label = f"Speaker {self._next_id}"
        self._next_id += 1
        return label

    @property
    def speaker_count(self) -> int:
        return len(self._speakers)

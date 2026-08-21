"""
Handles incoming raw PCM16 mono 16kHz audio from the browser.

Segmentation uses the standard webrtcvad "ring buffer" pattern instead of
a single silence-frame counter: we look at the ratio of speech/silence
frames inside a short rolling window before deciding to open or close a
segment. This is what makes short in-sentence pauses (a breath, "um")
not split one sentence into two, while a real pause still closes the
segment cleanly.

Segments are still non-overlapping (cut on silence), which is what keeps
Whisper output free of "hello hello hello" duplication - each chunk of
audio is only ever transcribed once.
"""

import collections
import numpy as np
import webrtcvad

SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2  # bytes per sample (16-bit PCM)


class Segmenter:
    def __init__(
        self,
        vad_aggressiveness: int = 2,
        frame_ms: int = 20,
        padding_ms: int = 300,
        trigger_ratio: float = 0.9,
        min_segment_ms: int = 400,
        max_segment_ms: int = 20000,
    ):
        self.vad = webrtcvad.Vad(vad_aggressiveness)
        self.frame_ms = frame_ms
        self.frame_bytes = int(SAMPLE_RATE * frame_ms / 1000) * SAMPLE_WIDTH

        num_padding_frames = max(1, padding_ms // frame_ms)
        self.trigger_ratio = trigger_ratio
        self.min_frames = max(1, min_segment_ms // frame_ms)
        self.max_frames = max(1, max_segment_ms // frame_ms)

        # Ring buffer of (frame_bytes, is_speech) used to smooth VAD
        # decisions, per the classic webrtcvad streaming example.
        self._ring = collections.deque(maxlen=num_padding_frames)

        self._byte_buffer = bytearray()
        self._triggered = False
        self._voiced_frames = []  # frames belonging to the current segment
        self._frames_seen = 0
        self._segment_start_frame = 0

    def _frame_seconds(self, frame_count: int) -> float:
        return frame_count * self.frame_ms / 1000.0

    def add_audio(self, data: bytes):
        """Feed raw PCM16 bytes in. Returns a list of completed segments,
        each as (pcm_bytes, start_seconds, end_seconds)."""
        completed = []
        self._byte_buffer.extend(data)

        while len(self._byte_buffer) >= self.frame_bytes:
            frame = bytes(self._byte_buffer[: self.frame_bytes])
            del self._byte_buffer[: self.frame_bytes]

            try:
                is_speech = self.vad.is_speech(frame, SAMPLE_RATE)
            except Exception:
                is_speech = False

            if not self._triggered:
                self._ring.append((frame, is_speech))
                num_voiced = len([f for f, s in self._ring if s])
                if self._ring.maxlen and num_voiced > self.trigger_ratio * self._ring.maxlen:
                    self._triggered = True
                    self._segment_start_frame = self._frames_seen - len(self._ring) + 1
                    # carry the ring buffer's frames into the segment so we
                    # don't lose the first syllable that triggered us
                    self._voiced_frames = [f for f, _ in self._ring]
                    self._ring.clear()
            else:
                self._voiced_frames.append(frame)
                self._ring.append((frame, is_speech))
                num_unvoiced = len([f for f, s in self._ring if not s])
                if self._ring.maxlen and num_unvoiced > self.trigger_ratio * self._ring.maxlen:
                    seg = self._finalize_segment()
                    if seg:
                        completed.append(seg)
                elif len(self._voiced_frames) >= self.max_frames:
                    seg = self._finalize_segment()
                    if seg:
                        completed.append(seg)
                    # keep listening immediately as a continuation of speech
                    self._triggered = True
                    self._segment_start_frame = self._frames_seen + 1

            self._frames_seen += 1

        return completed

    def _finalize_segment(self):
        seg_bytes = b"".join(self._voiced_frames)
        start_frame = self._segment_start_frame
        end_frame = self._frames_seen
        self._voiced_frames = []
        self._triggered = False
        self._ring.clear()

        if len(seg_bytes) < self.min_frames * self.frame_bytes:
            return None

        return seg_bytes, self._frame_seconds(start_frame), self._frame_seconds(end_frame)

    def peek_in_progress(self):
        """Non-destructive look at the currently-accumulating segment, for
        generating a fast 'partial' preview while the speaker is still
        talking. Returns (pcm_bytes, start_seconds, current_seconds) or
        None if nothing is currently in progress / it's still too short
        to be worth previewing."""
        if not self._triggered or len(self._voiced_frames) < self.min_frames:
            return None
        seg_bytes = b"".join(self._voiced_frames)
        return seg_bytes, self._frame_seconds(self._segment_start_frame), self._frame_seconds(self._frames_seen)

    def flush(self):
        """Call when the user hits STOP - returns the trailing in-progress
        segment (if any) even though no trailing silence was seen."""
        if not self._triggered or len(self._voiced_frames) < self.min_frames:
            self._voiced_frames = []
            self._triggered = False
            self._ring.clear()
            return None
        return self._finalize_segment()


def pcm16_bytes_to_float32(data: bytes) -> np.ndarray:
    if len(data) == 0:
        return np.zeros(0, dtype=np.float32)
    audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    return audio

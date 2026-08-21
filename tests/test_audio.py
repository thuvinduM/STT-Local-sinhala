"""
VAD/segmentation tests (requirement: short pauses don't split sentences,
long pauses do, noise blips are discarded, max-duration force-cuts work,
and in-progress audio can be peeked at for partial previews).

Uses webrtcvad directly (it's a lightweight pure-C-extension VAD with no
model download, so these run for real - no stubbing needed).
"""

from app.audio import Segmenter
from tests.conftest import tone_pcm16


def test_short_pause_does_not_split_sentence():
    seg = Segmenter(padding_ms=300, trigger_ratio=0.9, min_segment_ms=300, max_segment_ms=20000)
    completed = []
    for chunk in [tone_pcm16(400, 5000), tone_pcm16(150, 0), tone_pcm16(400, 5000), tone_pcm16(600, 0)]:
        completed.extend(seg.add_audio(chunk))
    assert len(completed) == 1, "a 150ms pause should not split one sentence into two"


def test_long_pause_creates_clean_boundary():
    seg = Segmenter(padding_ms=300, trigger_ratio=0.9, min_segment_ms=300, max_segment_ms=20000)
    completed = []
    for chunk in [tone_pcm16(400, 5000), tone_pcm16(800, 0), tone_pcm16(400, 5000), tone_pcm16(600, 0)]:
        completed.extend(seg.add_audio(chunk))
    assert len(completed) == 2, "an 800ms pause should cleanly split into two segments"


def test_short_noise_blip_is_discarded():
    seg = Segmenter(padding_ms=300, trigger_ratio=0.9, min_segment_ms=300, max_segment_ms=20000)
    completed = []
    for chunk in [tone_pcm16(80, 5000), tone_pcm16(600, 0)]:
        completed.extend(seg.add_audio(chunk))
    assert completed == [], "an 80ms blip should be discarded as noise, not become a segment"


def test_max_duration_force_cuts_long_speech():
    seg = Segmenter(padding_ms=300, trigger_ratio=0.9, min_segment_ms=300, max_segment_ms=1000)
    completed = []
    for _ in range(4):
        completed.extend(seg.add_audio(tone_pcm16(400, 5000)))  # 1.6s of continuous speech, no pauses
    assert len(completed) >= 1, "continuous speech longer than max_segment_ms must still get force-cut"


def test_peek_in_progress_returns_audio_while_still_speaking():
    seg = Segmenter(padding_ms=300, trigger_ratio=0.9, min_segment_ms=300, max_segment_ms=20000)
    seg.add_audio(tone_pcm16(1500, 5000))
    peek = seg.peek_in_progress()
    assert peek is not None
    pcm_bytes, start_s, current_s = peek
    assert len(pcm_bytes) > 0
    assert current_s > start_s


def test_flush_returns_trailing_segment_on_stop():
    seg = Segmenter(padding_ms=300, trigger_ratio=0.9, min_segment_ms=300, max_segment_ms=20000)
    seg.add_audio(tone_pcm16(500, 5000))  # speaking, no trailing silence yet (user hits STOP)
    tail = seg.flush()
    assert tail is not None
    pcm_bytes, start_s, end_s = tail
    assert end_s > start_s


def test_timestamps_are_session_relative_not_arrival_time():
    seg = Segmenter(padding_ms=300, trigger_ratio=0.9, min_segment_ms=300, max_segment_ms=20000)
    completed = []
    # 1s silence, then speech - the segment's start time should reflect
    # its position in the audio stream, not when add_audio() was called.
    completed.extend(seg.add_audio(tone_pcm16(1000, 0)))
    completed.extend(seg.add_audio(tone_pcm16(500, 5000)))
    completed.extend(seg.add_audio(tone_pcm16(600, 0)))
    assert len(completed) == 1
    _, start_s, _ = completed[0]
    assert start_s > 0.5, "segment should start around the 1s mark, not at t=0"

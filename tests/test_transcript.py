from app.transcript import format_txt, format_json, dedupe_repeated_phrases
import json


def test_format_txt_matches_expected_layout():
    segments = [
        {"speaker": "Speaker 1", "text": "The court is now in session.", "start": 0.0, "end": 2.0, "language": "en"},
        {"speaker": "Speaker 2", "text": "Thank you, Your Honour.", "start": 2.0, "end": 3.5, "language": "en"},
    ]
    out = format_txt(segments)
    assert out == "Speaker 1:\nThe court is now in session.\n\nSpeaker 2:\nThank you, Your Honour.\n"


def test_format_txt_skips_empty_text():
    segments = [
        {"speaker": "Speaker 1", "text": "  ", "start": 0.0, "end": 1.0, "language": "en"},
        {"speaker": "Speaker 2", "text": "Hello.", "start": 1.0, "end": 2.0, "language": "en"},
    ]
    out = format_txt(segments)
    assert "Speaker 1" not in out
    assert "Hello." in out


def test_format_json_round_trips_fields():
    segments = [
        {"speaker": "Speaker 1", "start": 3.2, "end": 7.8, "text": "The court is now in session.", "language": "en"},
    ]
    parsed = json.loads(format_json(segments))
    assert parsed[0]["speaker"] == "Speaker 1"
    assert parsed[0]["start"] == 3.2
    assert parsed[0]["text"] == "The court is now in session."


def test_dedupe_collapses_adjacent_repeats():
    assert dedupe_repeated_phrases("Thank you thank you thank you your honour.") == "Thank you your honour."
    assert dedupe_repeated_phrases("please continue please continue") == "please continue"


def test_dedupe_leaves_non_adjacent_repeats_alone():
    text = "I said yes and later I said yes again"
    assert dedupe_repeated_phrases(text) == text


def test_dedupe_leaves_clean_text_unchanged():
    text = "The court is now in session."
    assert dedupe_repeated_phrases(text) == text

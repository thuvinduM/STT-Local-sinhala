# court-stt (local STT + diarization POC)

Local, offline transcription (Sinhala / Tamil / English) with session-
stable "Speaker 1 / Speaker 2 / ..." labels, using faster-whisper
(large-v3) on your GPU and an online speaker-embedding diarizer. No
Docker, no DB, no cloud APIs.

## 1. Upload to your Ubuntu server
```
scp court-stt-v2.zip user@SERVER_IP:~/
```

## 2. Extract
```
ssh user@SERVER_IP
unzip court-stt-v2.zip
cd court-stt
```

## 3. Install
```
chmod +x install.sh run.sh
./install.sh
```
Creates a local `venv/`. Your system Python, driver, and CUDA install are
untouched.

## 4. Run
```
./run.sh
```
First run downloads: Whisper large-v3 (~3GB) and the SpeechBrain ECAPA
speaker-embedding model (~80MB, no HuggingFace token/license needed).
Needs internet access once.

## 5. Open
```
http://SERVER_IP:8000
```
**START** → allow mic → speak → **STOP** → **DOWNLOAD TXT/JSON**.
The header shows GPU / Whisper / Diarization status once the models load.

## Diarization approach (what changed and why)
Speaker labels come from an online **SpeakerManager** (`app/diarization.py`):
each segment gets a voice embedding, compared against a few stored
"prototype" embeddings per known speaker (not just one running average -
this absorbs natural pitch/volume variation). Above a similarity
threshold → same speaker; below it → new speaker. Utterances shorter
than `SPEAKER_MIN_ENROLL_SECONDS` can never *create* a new speaker (too
unreliable to embed) - they attach to the closest existing one instead.

Embeddings come from SpeechBrain's ECAPA-TDNN (`speechbrain/spkrec-ecapa-
voxceleb`) - the same embedding quality pyannote.audio itself is built
on, but freely downloadable (no gated license/token). Falls back to
Resemblyzer automatically if SpeechBrain can't load.

**Why not the full pyannote.audio pipeline:** it's designed to
diarize a whole finished recording at once (segment the entire file,
embed, cluster globally), which doesn't fit a live "transcribe as you
go" session. Its own embedding checkpoints also require accepting a
license and a HuggingFace token. If you later want an offline,
higher-accuracy *second pass* after a recording, running the pyannote
pipeline on the saved audio is a reasonable future addition - not
included here, to keep this POC simple.

We deliberately do **not** auto-merge two already-created speakers later
in a session - wrongly merging two different people is worse than an
occasional extra speaker label. Known limitation, not a bug.

## What else improved in this version
- **VAD**: switched from a single silence-counter to the standard
  webrtcvad rolling-window pattern (`VAD_PADDING_MS`/`VAD_TRIGGER_RATIO`)
  - short in-sentence pauses no longer split a sentence, long pauses
  still cut cleanly, sub-`VAD_MIN_SEGMENT_MS` noise blips are discarded.
- **Partial/final transcript**: while a sentence is still being spoken, a
  fast low-quality "partial" preview streams in (grey/italic); it's
  replaced by the real high-quality (`beam_size=5`) transcription once
  the segment closes. Partials are never saved to the transcript/downloads.
- **Duplicate-phrase cleanup**: Whisper's occasional "thank you thank you
  thank you" hallucination on repetitive/noisy audio is collapsed
  (`app/transcript.py: dedupe_repeated_phrases`) - only collapses
  back-to-back repeats, never legitimately repeated words spoken apart.
- **GPU OOM guard**: a segment that hits CUDA out-of-memory is skipped
  with an on-screen warning instead of crashing the session.
- **Status/health**: `/api/health` + header indicators for GPU / Whisper /
  Diarization; UI status now shows Ready / Recording / Processing /
  Finished / Error.
- **More config** in `.env.example` - VAD window/ratio, speaker match
  threshold, min-enroll duration, prototype count, partial-update interval.

## Testing
```
pip install -r requirements-dev.txt
pytest tests/
```
18 tests covering VAD segmentation (short/long pauses, noise blips, max-
duration cuts, timestamp accuracy), the SpeakerManager's clustering logic
(including the exact "A,B,A,C,B,A → Speaker 1,2,1,3,2,1" scenario), and
transcript formatting/dedup. All pass without a GPU or network.

**Honest limitation**: the diarization tests use a synthetic fake
embedder (distinguishable "voices" via pitch), not the real SpeechBrain
model - this sandbox has no network or real multi-speaker audio to test
against. This validates the matching/clustering *logic* correctly, not
real-world voice-embedding accuracy. **Please test with 2-3 real
speakers on your server before relying on it** - diarization can still
struggle with very similar voices, overlapping speech, short utterances,
heavy background noise, or a speaker moving far from the mic.

## Notes
- If `webrtcvad` fails to build: `sudo apt install python3-dev build-essential`.
- Single session at a time, in-memory only, no auth - don't expose this
  to the open internet.

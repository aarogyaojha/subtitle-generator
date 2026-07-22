"""
Orchestrator — single entry point: raw audio/video in, finished
.srt/.vtt out.

Runs in order:
  1. Audio extraction (if input is video)
  2. vad.py     — find speech regions
  3. asr.py     — transcribe them
  4. diarize.py — identify speakers
  5. merge      — attach speaker labels to transcript
  6. format.py  — segment into readable cues, write output

All config (language, hardware tier, reading-speed limits,
num_speakers if known) threads through from one config.yaml — not
re-specified per stage.

Fails fast: a missing HF token should surface before time is spent
on transcription, not after.
"""
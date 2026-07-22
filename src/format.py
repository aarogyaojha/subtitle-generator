"""
Formatting stage — turns labeled, timestamped text into readable
subtitle cues, and writes .srt/.vtt.

Enforces readability constraints (values live in config.yaml, not
hardcoded — they vary by standard and by language):
  - max characters per line
  - max lines per cue (2 is standard)
  - max reading speed in characters/second (CPS) — the real
    constraint; a cue must stay on screen at least
    len(text) / max_cps seconds, regardless of actual speaking pace
  - min/max cue duration
  - min gap between consecutive cues

Forces a cue break on every speaker change (via diarize.py's
output), so one subtitle box never silently merges two people's
lines.

Input:  labeled, timestamped transcript (asr.py + diarize.py merged).
Output: .srt or .vtt file.
"""
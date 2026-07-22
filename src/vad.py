"""
Voice Activity Detection (VAD) stage — runs before transcription.

Finds the time ranges in the audio that actually contain speech,
and filters out silence before any of it reaches the ASR model.

Why this exists: Whisper hallucinates text during silence — it has
no "this is silence, output nothing" training signal, so it falls
back to repeating whatever phrase it just generated. Removing
silence before it reaches Whisper is the most effective known fix.

Model: Silero VAD (open-source, lightweight, CPU-friendly).

Input:  path to audio (mono, 16kHz).
Output: list of (start_time, end_time) tuples marking speech
        regions, in seconds.
"""
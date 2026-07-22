"""
Speaker diarization stage — "who spoke when," independent of what
was said.

Uses pyannote.audio (speaker-diarization-3.1 + segmentation-3.0).
Both are gated on Hugging Face — requires an HF account, accepting
the license on both model pages, and a valid HF token (`hf auth
login` or HF_TOKEN env var) or this will fail to load.

Note: diarization is meaningfully less accurate than transcription —
expect real errors here even on a clean transcript, especially
around overlapping speech.

Input:  path to audio file.
Output: speaker segments — {start, end, speaker_id}, speaker_id
        being an anonymous label (SPEAKER_00, SPEAKER_01, ...).

Also holds merge_with_transcript(): attaches a speaker_id to every
asr.py transcript entry, based on which speaker segment overlaps
it most.
"""
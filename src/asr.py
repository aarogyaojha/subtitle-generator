"""
Transcription stage.

Runs the speech regions found by vad.py through faster-whisper
(Whisper large-v3, CTranslate2 backend) to produce timestamped text.

Hardware-tier aware: model size/quantization comes from config.yaml,
not hardcoded here — this is what lets the same pipeline run on a
16GB Colab T4 (large-v3, int8, ~3GB VRAM) and on a constrained local
machine (smaller model, int8, batch_size=1 on a 4GB-class card)
without touching this file.

Input:  audio file + speech regions from vad.py.
Output: segments — {start, end, text, confidence/avg_logprob,
        word-level timing if available}. Confidence matters
        downstream: the correction stage should only touch
        low-confidence segments, not rewrite everything.
"""
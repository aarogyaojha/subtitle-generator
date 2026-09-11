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

import logging
import math
from pathlib import Path
from typing import Any, Optional, Union
import faster_whisper
import torch

from src.config_utils import DEFAULT_CONFIG_PATH, load_stage_config

logger = logging.getLogger(__name__)

# Default ASR settings when config.yaml does not define an `asr:` section:
# - model_size: "large-v3" (standard Whisper large-v3)
# - compute_type: "int8" (8-bit quantization per README hardware tier spec)
# - device: "cuda" if CUDA is available, else "cpu"
# - language: "ne"
DEFAULT_MODEL_SIZE = "large-v3"
DEFAULT_COMPUTE_TYPE = "int8"
DEFAULT_LANGUAGE = "ne"

_CACHED_MODEL: Optional[faster_whisper.WhisperModel] = None
_CACHED_MODEL_KEY: Optional[tuple[str, str, str]] = None


def _load_asr_config(config_path: Union[str, Path] = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """
    Read ASR configuration settings if present in config.yaml, otherwise return defaults.

    Args:
        config_path: Path to the configuration YAML file.

    Returns:
        Dictionary containing model_size, compute_type, device, and language.
    """
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    asr_settings: dict[str, Any] = {
        "model_size": DEFAULT_MODEL_SIZE,
        "compute_type": DEFAULT_COMPUTE_TYPE,
        "device": default_device,
        "language": DEFAULT_LANGUAGE,
    }

    return load_stage_config(
        config_path=config_path,
        section_name="asr",
        defaults=asr_settings,
        top_level_keys=["language"],
        stage_label="ASR",
    )


def load_asr_model(
    model_size_or_path: str = DEFAULT_MODEL_SIZE,
    device: Optional[str] = None,
    compute_type: str = DEFAULT_COMPUTE_TYPE,
) -> faster_whisper.WhisperModel:
    """
    Load or return cached faster-whisper WhisperModel.

    Args:
        model_size_or_path: Whisper model size name or local path (e.g. 'large-v3').
        device: Device to load model onto ('cuda', 'cpu', 'auto'). If None, selects 'cuda'
            if torch.cuda.is_available(), else 'cpu'.
        compute_type: Quantization / compute type (e.g. 'int8', 'float16', 'float32').

    Returns:
        The cached or newly instantiated WhisperModel.
    """
    global _CACHED_MODEL, _CACHED_MODEL_KEY

    target_device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    model_key = (model_size_or_path, target_device, compute_type)

    if _CACHED_MODEL is not None and _CACHED_MODEL_KEY == model_key:
        return _CACHED_MODEL

    logger.info(
        "Loading faster-whisper model: %s (device=%s, compute_type=%s)",
        model_size_or_path,
        target_device,
        compute_type,
    )
    _CACHED_MODEL = faster_whisper.WhisperModel(
        model_size_or_path=model_size_or_path,
        device=target_device,
        compute_type=compute_type,
    )
    _CACHED_MODEL_KEY = model_key
    return _CACHED_MODEL


def transcribe(
    audio_path: Union[str, Path],
    speech_regions: Optional[list[tuple[float, float]]] = None,
    config_path: Union[str, Path] = DEFAULT_CONFIG_PATH,
    model_size_or_path: Optional[str] = None,
    device: Optional[str] = None,
    compute_type: Optional[str] = None,
    language: Optional[str] = None,
    task: str = "transcribe",
    word_timestamps: bool = True,
) -> list[dict[str, Any]]:
    """
    Transcribe speech regions from an audio file using faster-whisper.

    Note:
        Setting speech_regions=None (whole-file mode) is a convenience addition
        beyond the original module spec, which expects speech regions from vad.py.

    Args:
        audio_path: Path to the input audio file.
        speech_regions: Optional list of (start, end) time ranges in seconds detected by VAD.
            If an empty list [] is passed, returns [] immediately (silence detected).
            If None, transcribes the whole audio file.
        config_path: Path to config.yaml for model and runtime settings.
        model_size_or_path: Optional override for model size (e.g., 'large-v3').
        device: Optional override for compute device ('cuda', 'cpu', 'auto').
        compute_type: Optional override for quantization ('int8', 'float16').
        language: Optional language code override (e.g., 'ne', 'ja').
        task: Task to execute ('transcribe' or 'translate'). Defaults to 'transcribe'.
        word_timestamps: Whether to extract word-level timing. Defaults to True.

    Returns:
        List of segment dictionaries containing:
            - "start": float start timestamp in seconds.
            - "end": float end timestamp in seconds.
            - "text": stripped transcribed text.
            - "confidence": confidence score in [0, 1] derived from exp(avg_logprob).
            - "avg_logprob": raw average log-probability of tokens.
            - "no_speech_prob": probability that the segment contains no speech.
            - "words": list of word timing dictionaries with 'word', 'start', 'end', 'probability'.

    Raises:
        FileNotFoundError: If audio_path does not exist.
        ValueError: If task is not 'transcribe' or 'translate'.
    """
    if task not in ("transcribe", "translate"):
        raise ValueError(f"Invalid task '{task}'. Expected 'transcribe' or 'translate'.")

    audio_file = Path(audio_path)
    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")

    # If VAD produced an empty speech regions list, there is no speech to transcribe
    if speech_regions is not None and len(speech_regions) == 0:
        logger.info("Empty speech regions provided. Skipping transcription.")
        return []

    cfg_settings = _load_asr_config(config_path)
    final_model_size = model_size_or_path if model_size_or_path is not None else cfg_settings["model_size"]
    final_device = device if device is not None else cfg_settings["device"]
    final_compute_type = compute_type if compute_type is not None else cfg_settings["compute_type"]
    final_language = language if language is not None else cfg_settings["language"]

    model = load_asr_model(
        model_size_or_path=final_model_size,
        device=final_device,
        compute_type=final_compute_type,
    )

    transcribe_kwargs: dict[str, Any] = {
        "language": final_language,
        "task": task,
        "word_timestamps": word_timestamps,
    }

    if speech_regions is not None:
        clip_timestamps = [float(coord) for region in speech_regions for coord in region]
        transcribe_kwargs["clip_timestamps"] = clip_timestamps
        logger.info("Running ASR (%s) with VAD clip boundaries: %s", task, speech_regions)
    else:
        logger.info("Running ASR (%s) on entire audio file (whole-file mode).", task)

    raw_segments, _ = model.transcribe(str(audio_file), **transcribe_kwargs)

    formatted_segments: list[dict[str, Any]] = []
    for seg in raw_segments:
        confidence = (
            float(math.exp(seg.avg_logprob))
            if seg.avg_logprob is not None and not math.isnan(seg.avg_logprob)
            else 1.0
        )
        # Clamp confidence to [0.0, 1.0]
        confidence = max(0.0, min(1.0, confidence))

        words_list: list[dict[str, Any]] = []
        if seg.words:
            for w in seg.words:
                words_list.append(
                    {
                        "word": w.word,
                        "start": float(w.start),
                        "end": float(w.end),
                        "probability": float(w.probability),
                    }
                )

        formatted_segments.append(
            {
                "start": float(seg.start),
                "end": float(seg.end),
                "text": seg.text.strip(),
                "confidence": confidence,
                "avg_logprob": float(seg.avg_logprob) if seg.avg_logprob is not None else 0.0,
                "no_speech_prob": float(seg.no_speech_prob) if seg.no_speech_prob is not None else 0.0,
                "words": words_list,
            }
        )

    return formatted_segments


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python src/asr.py <path_to_audio_file> [start,end ...]")
        sys.exit(1)

    input_audio_path = sys.argv[1]
    speech_clips: Optional[list[tuple[float, float]]] = None

    if len(sys.argv) > 2:
        speech_clips = []
        for pair in sys.argv[2:]:
            start_str, end_str = pair.split(",")
            speech_clips.append((float(start_str), float(end_str)))

    print(f"Transcribing: {input_audio_path}")
    results = transcribe(input_audio_path, speech_regions=speech_clips)
    print(f"Transcription resulted in {len(results)} segment(s):")
    for idx, segment in enumerate(results, 1):
        print(
            f"  {idx}. [{segment['start']:.2f}s -> {segment['end']:.2f}s] "
            f"text='{segment['text']}' confidence={segment['confidence']:.4f}"
        )
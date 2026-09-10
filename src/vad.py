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

import logging
from pathlib import Path
from typing import Union
import torch
import torchaudio

from src.config_utils import DEFAULT_CONFIG_PATH, load_stage_config

logger = logging.getLogger(__name__)

# Sane default thresholds for Silero VAD:
# - threshold (0.5): Speech probability threshold above which a frame is classified as speech.
# - min_speech_duration_ms (250): Shorter speech chunks than this are discarded to avoid false positives.
# - min_silence_duration_ms (100): Silence chunks shorter than this will not split speech into separate segments.
# - speech_pad_ms (30): Padding in ms added to both sides of each detected speech segment.
DEFAULT_THRESHOLD = 0.5
DEFAULT_MIN_SPEECH_DURATION_MS = 250
DEFAULT_MIN_SILENCE_DURATION_MS = 100
DEFAULT_SPEECH_PAD_MS = 30
SAMPLE_RATE = 16000

_MODEL = None
_UTILS = None


def load_vad_model():
    """Load or return cached Silero VAD model and utilities via torch.hub."""
    global _MODEL, _UTILS
    if _MODEL is None or _UTILS is None:
        _MODEL, _UTILS = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
            trust_repo=True,
        )
    return _MODEL, _UTILS


def _load_thresholds_from_config(config_path: Union[str, Path] = DEFAULT_CONFIG_PATH) -> dict:
    """Read VAD configuration thresholds if present in config.yaml, otherwise return defaults."""
    thresholds = {
        "threshold": DEFAULT_THRESHOLD,
        "min_speech_duration_ms": DEFAULT_MIN_SPEECH_DURATION_MS,
        "min_silence_duration_ms": DEFAULT_MIN_SILENCE_DURATION_MS,
        "speech_pad_ms": DEFAULT_SPEECH_PAD_MS,
    }
    return load_stage_config(
        config_path=config_path,
        section_name="vad",
        defaults=thresholds,
        stage_label="VAD",
    )


def _load_audio(audio_path: Path) -> tuple[torch.Tensor, int]:
    """Load audio file into a waveform tensor and sampling rate."""
    try:
        if hasattr(torchaudio, "load_with_torchcodec"):
            return torchaudio.load_with_torchcodec(str(audio_path))
        return torchaudio.load(str(audio_path))
    except Exception as e:
        if "No audio frames were decoded" in str(e):
            return torch.empty(1, 0, dtype=torch.float32), SAMPLE_RATE
        raise


def get_speech_regions(
    audio_path: Union[str, Path],
    config_path: Union[str, Path] = DEFAULT_CONFIG_PATH,
    threshold: float | None = None,
    min_speech_duration_ms: int | None = None,
    min_silence_duration_ms: int | None = None,
    speech_pad_ms: int | None = None,
) -> list[tuple[float, float]]:
    """
    Detect speech regions in an audio file using Silero VAD.

    Args:
        audio_path: Path to the input audio file (mono, 16kHz).
        config_path: Path to config.yaml to load optional VAD settings.
        threshold: Optional override for speech probability threshold.
        min_speech_duration_ms: Optional override for minimum speech chunk duration.
        min_silence_duration_ms: Optional override for minimum silence duration.
        speech_pad_ms: Optional override for padding applied to speech segments.

    Returns:
        List of (start_time, end_time) tuples in seconds marking speech regions.

    Raises:
        FileNotFoundError: If the specified audio_path does not exist.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Load thresholds from config or use defaults
    cfg_thresholds = _load_thresholds_from_config(config_path)
    vad_threshold = threshold if threshold is not None else cfg_thresholds["threshold"]
    vad_min_speech_ms = (
        min_speech_duration_ms
        if min_speech_duration_ms is not None
        else cfg_thresholds["min_speech_duration_ms"]
    )
    vad_min_silence_ms = (
        min_silence_duration_ms
        if min_silence_duration_ms is not None
        else cfg_thresholds["min_silence_duration_ms"]
    )
    vad_speech_pad_ms = (
        speech_pad_ms
        if speech_pad_ms is not None
        else cfg_thresholds["speech_pad_ms"]
    )

    # Load audio
    waveform, sample_rate = _load_audio(audio_path)

    # Flatten to 1D tensor for Silero VAD
    waveform = waveform.squeeze()
    if waveform.ndim == 0 or waveform.numel() == 0:
        return []

    # Convert to mono if multi-channel
    if waveform.ndim > 1 and waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0)

    # Resample to 16kHz if needed
    if sample_rate != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=SAMPLE_RATE)
        waveform = resampler(waveform)

    model, utils = load_vad_model()
    get_speech_timestamps_fn = utils[0]

    speech_timestamps = get_speech_timestamps_fn(
        waveform,
        model,
        threshold=vad_threshold,
        sampling_rate=SAMPLE_RATE,
        min_speech_duration_ms=vad_min_speech_ms,
        min_silence_duration_ms=vad_min_silence_ms,
        speech_pad_ms=vad_speech_pad_ms,
        return_seconds=True,
    )

    return [(float(seg["start"]), float(seg["end"])) for seg in speech_timestamps]


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python src/vad.py <path_to_audio_file>")
        sys.exit(1)

    input_audio_path = sys.argv[1]
    print(f"Running VAD on: {input_audio_path}")
    speech_regions = get_speech_regions(input_audio_path)
    print(f"Detected {len(speech_regions)} speech region(s):")
    for idx, (start, end) in enumerate(speech_regions, 1):
        print(f"  {idx}. {start:.2f}s - {end:.2f}s (duration: {end - start:.2f}s)")
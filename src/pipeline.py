"""
Orchestrator — single entry point: raw audio/video in, finished
.srt/.vtt out.

Runs in order:
  1. Fail-fast HF pre-flight check (verifies auth before compute)
  2. Audio extraction (if input is video)
  3. vad.py     — find speech regions
  4. asr.py     — transcribe them
  5. diarize.py — identify speakers
  6. merge      — attach speaker labels to transcript
  7. format.py  — segment into readable cues, write output

Hardware-tier aware: reads hardware_tier from config.yaml and maps it
to explicit overrides for ASR (model size and quantization).
Stage-specific settings (VAD thresholds, language, num_speakers,
reading-speed limits) thread through automatically from config.yaml via
each stage's own config loader.
"""

import logging
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Optional, Union

from src.config_utils import DEFAULT_CONFIG_PATH, PROJECT_ROOT, load_stage_config
from src import vad, asr, diarize, format as fmt

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {
    ".mp4",
    ".mkv",
    ".avi",
    ".mov",
    ".flv",
    ".webm",
    ".wmv",
    ".m4v",
}

DEFAULT_HARDWARE_TIER = "colab"

# Hardware tier mapping:
# - "colab": Whisper large-v3 with INT8 quantization (~3GB VRAM).
#   Note: The colab tier comment in config.yaml mentions a "+ correction stage".
#   No LLM-assisted context correction module exists yet in v1; this is out-of-scope
#   future work and is intentionally not implemented here.
# - "local": Whisper medium with INT8 quantization (~1.5GB VRAM).
#   Reasoning: Whisper medium with INT8 quantization uses ~1.5GB VRAM, fitting
#   comfortably within the 4GB local GPU constraint specified in README.md alongside
#   Silero VAD (~100MB) and pyannote diarization (~1-1.5GB).
#   Note on batch_size: In faster-whisper 1.2.1, WhisperModel.transcribe() does not
#   support a batch_size parameter, so batch_size is not passed here.
HARDWARE_TIER_MAP: dict[str, dict[str, Any]] = {
    "colab": {
        "model_size_or_path": "large-v3",
        "compute_type": "int8",
    },
    "local": {
        "model_size_or_path": "medium",
        "compute_type": "int8",
    },
}


def _load_hardware_tier(config_path: Union[str, Path] = DEFAULT_CONFIG_PATH) -> str:
    """Read hardware_tier from config.yaml, defaulting to 'colab'."""
    cfg = load_stage_config(
        config_path=config_path,
        defaults={"hardware_tier": DEFAULT_HARDWARE_TIER},
        top_level_keys=["hardware_tier"],
        stage_label="Pipeline",
    )
    return str(cfg.get("hardware_tier", DEFAULT_HARDWARE_TIER))


def get_hardware_tier_overrides(tier: str) -> dict[str, Any]:
    """
    Map a hardware tier string to stage override parameters.

    Args:
        tier: Hardware tier identifier ('colab', 'local').

    Returns:
        Dictionary of keyword arguments for asr.transcribe().
    """
    tier_lower = tier.lower()
    if tier_lower in HARDWARE_TIER_MAP:
        return dict(HARDWARE_TIER_MAP[tier_lower])
    logger.warning("Unrecognized hardware tier '%s'. Falling back to 'colab'.", tier)
    return dict(HARDWARE_TIER_MAP["colab"])


def is_video_file(path: Union[str, Path]) -> bool:
    """Check if the given file path has a common video file extension."""
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def extract_audio_from_video(
    video_path: Union[str, Path],
    output_audio_path: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Extract audio from a video file using ffmpeg to 16kHz mono 16-bit PCM WAV.

    Args:
        video_path: Path to the input video file.
        output_audio_path: Optional destination path. If not provided, a temporary
            file is created.

    Returns:
        Path to the extracted .wav audio file.

    Raises:
        FileNotFoundError: If video_path does not exist or ffmpeg executable is not found.
        RuntimeError: If ffmpeg execution fails.
    """
    video_file = Path(video_path).resolve()
    if not video_file.exists():
        raise FileNotFoundError(f"Video file not found: {video_file}")

    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError(
            "ffmpeg executable not found in PATH. Please install ffmpeg to process video files."
        )

    if output_audio_path is None:
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        target_path = Path(temp_file.name)
        temp_file.close()
    else:
        target_path = Path(output_audio_path).resolve()
        target_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_file),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(target_path),
    ]

    logger.info("Extracting audio from %s to %s via ffmpeg", video_file, target_path)
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode != 0:
        if target_path.exists() and output_audio_path is None:
            target_path.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg audio extraction failed with error:\n{result.stderr}")

    return target_path


def run_pipeline(
    input_path: Union[str, Path],
    config_path: Union[str, Path] = DEFAULT_CONFIG_PATH,
    output_dir: Optional[Union[str, Path]] = None,
    hardware_tier: Optional[str] = None,
    include_speaker: bool = True,
) -> dict[str, Any]:
    """
    Execute the end-to-end subtitle generation pipeline.

    Orchestrates:
      1. Fail-fast HF pre-flight check (raises immediately if HF auth/token is invalid).
      2. Audio extraction (if input is video).
      3. VAD speech region detection (vad.get_speech_regions).
      4. ASR transcription (asr.transcribe) with hardware-tier model configuration.
      5. Speaker diarization and transcript merging (diarize.diarize & diarize.merge_with_transcript).
      6. Subtitle cue formatting and file export (format.create_cues, format.write_srt, format.write_vtt).

    Args:
        input_path: Path to input audio or video file.
        config_path: Path to config.yaml (passed through to each stage call).
        output_dir: Directory where .srt and .vtt files are saved. Defaults to PROJECT_ROOT / "output".
        hardware_tier: Optional hardware tier override ('colab', 'local'). If None, reads from config_path.
        include_speaker: Whether to include speaker labels in output subtitle files.

    Returns:
        Summary dictionary containing:
            - "input_path": Path of the original input file.
            - "audio_path": Path of the resolved audio file used for processing.
            - "srt_path": Path to the generated .srt file.
            - "vtt_path": Path to the generated .vtt file.
            - "cues": List of formatted subtitle cue dictionaries.
            - "cue_count": Total number of cues produced.
            - "speech_regions": List of (start, end) tuples from VAD.
            - "segments": List of merged transcript segment dictionaries.
            - "speaker_turns": List of diarization speaker turn dictionaries.
            - "hardware_tier": The hardware tier applied for this run.

    Raises:
        FileNotFoundError: If input_path does not exist.
        Exception: Gating/auth errors from pyannote pre-flight check or stage failures.
    """
    input_file = Path(input_path).resolve()
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Stage 0: Fail-fast pre-flight check for Hugging Face diarization authentication.
    # Attempting to load the diarization pipeline first surfaces missing tokens or license
    # acceptance errors immediately before heavy VAD or ASR compute is initiated.
    # We resolve the configured model_name and device to ensure the pre-flight check
    # validates the exact model that will be used during execution.
    logger.info("Performing pre-flight HF diarization pipeline check...")
    diarize_cfg = diarize._load_diarize_config(config_path)
    diarize.load_diarization_pipeline(
        model_name=diarize_cfg["model_name"],
        device=diarize_cfg["device"],
    )

    # Stage 1: Audio extraction if input is video.
    temp_audio_created = False
    resolved_audio_path: Path

    if is_video_file(input_file):
        resolved_audio_path = extract_audio_from_video(input_file)
        temp_audio_created = True
    else:
        resolved_audio_path = input_file

    try:
        # Explicit consistency guarantee:
        # The exact same resolved_audio_path variable (the extracted temp WAV when input is video,
        # or the original path when input is already audio) is passed to all three downstream
        # stages: vad.get_speech_regions(), asr.transcribe(), and diarize.diarize().
        # No stage accidentally references the original input path after extraction.

        # Stage 2: Voice Activity Detection (VAD)
        logger.info("Running VAD on %s", resolved_audio_path)
        speech_regions = vad.get_speech_regions(
            audio_path=resolved_audio_path,
            config_path=config_path,
        )
        logger.info("VAD detected %d speech region(s)", len(speech_regions))

        # Stage 3: ASR Transcription
        active_tier = hardware_tier if hardware_tier is not None else _load_hardware_tier(config_path)
        asr_overrides = get_hardware_tier_overrides(active_tier)
        logger.info("Running ASR with hardware tier '%s' (overrides: %s)", active_tier, asr_overrides)
        transcript_segments = asr.transcribe(
            audio_path=resolved_audio_path,
            speech_regions=speech_regions,
            config_path=config_path,
            **asr_overrides,
        )
        logger.info("ASR produced %d transcript segment(s)", len(transcript_segments))

        # Stage 4: Speaker Diarization & Merge
        logger.info("Running speaker diarization on %s", resolved_audio_path)
        speaker_turns = diarize.diarize(
            audio_path=resolved_audio_path,
            config_path=config_path,
        )
        logger.info("Diarization produced %d speaker turn(s)", len(speaker_turns))

        logger.info("Merging transcript segments with speaker diarization turns")
        merged_segments = diarize.merge_with_transcript(
            transcript_segments=transcript_segments,
            diarization_turns=speaker_turns,
        )

        # Stage 5: Subtitle Formatting & Export
        logger.info("Formatting transcript segments into subtitle cues")
        cues = fmt.create_cues(
            transcript_segments=merged_segments,
            config_path=config_path,
        )

        out_directory = Path(output_dir).resolve() if output_dir is not None else (PROJECT_ROOT / "output")
        out_directory.mkdir(parents=True, exist_ok=True)

        srt_destination = out_directory / f"{input_file.stem}.srt"
        vtt_destination = out_directory / f"{input_file.stem}.vtt"

        fmt.write_srt(cues=cues, output_path=srt_destination, include_speaker=include_speaker)
        fmt.write_vtt(cues=cues, output_path=vtt_destination, include_speaker=include_speaker)

        summary = {
            "input_path": str(input_file),
            "audio_path": str(resolved_audio_path),
            "srt_path": str(srt_destination),
            "vtt_path": str(vtt_destination),
            "cues": cues,
            "cue_count": len(cues),
            "speech_regions": speech_regions,
            "segments": merged_segments,
            "speaker_turns": speaker_turns,
            "hardware_tier": active_tier,
        }
        return summary

    finally:
        # Clean up temporary extracted audio file if one was created
        if temp_audio_created and resolved_audio_path.exists():
            try:
                resolved_audio_path.unlink()
                logger.debug("Cleaned up temporary extracted audio file: %s", resolved_audio_path)
            except Exception as e:
                logger.warning("Failed to remove temporary audio file %s: %s", resolved_audio_path, e)
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

Fallback & Tie-Breaking Policies for merge_with_transcript():
- Fallback: If an ASR segment has zero overlap with any diarization turn
  (or if no diarization turns exist), its speaker_id is assigned to
  fallback_speaker (defaults to "UNKNOWN").
- Tie-breaking: If multiple speakers have identical positive overlap
  duration with a segment, ties are broken deterministically by selecting
  the speaker with the earliest overlapping turn start timestamp, followed
  by alphabetical ordering of speaker_id.
"""

import logging
from pathlib import Path
from typing import Any, Optional, Union
import torch
from pyannote.audio import Pipeline

from src.config_utils import DEFAULT_CONFIG_PATH, load_stage_config

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "pyannote/speaker-diarization-3.1"

_CACHED_PIPELINE: Optional[Pipeline] = None
_CACHED_PIPELINE_KEY: Optional[tuple[str, str]] = None


def _load_diarize_config(config_path: Union[str, Path] = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """
    Read diarization configuration settings if present in config.yaml, otherwise return defaults.

    Args:
        config_path: Path to the configuration YAML file.

    Returns:
        Dictionary containing num_speakers, min_speakers, max_speakers, device, and model_name.
    """
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    diarize_settings: dict[str, Any] = {
        "num_speakers": None,
        "min_speakers": None,
        "max_speakers": None,
        "device": default_device,
        "model_name": DEFAULT_MODEL_NAME,
    }

    return load_stage_config(
        config_path=config_path,
        section_name="diarize",
        defaults=diarize_settings,
        top_level_keys=["num_speakers"],
        stage_label="diarization",
    )


def load_diarization_pipeline(
    model_name: str = DEFAULT_MODEL_NAME,
    device: Optional[str] = None,
) -> Pipeline:
    """
    Load or return cached pyannote.audio speaker diarization Pipeline.

    Args:
        model_name: Hugging Face model identifier for the diarization pipeline.
        device: Device to load model onto ('cuda', 'cpu'). If None, selects 'cuda'
            if torch.cuda.is_available(), else 'cpu'.

    Returns:
        The cached or newly instantiated Pipeline on the specified device.
    """
    global _CACHED_PIPELINE, _CACHED_PIPELINE_KEY

    target_device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    pipeline_key = (model_name, target_device)

    if _CACHED_PIPELINE is not None and _CACHED_PIPELINE_KEY == pipeline_key:
        return _CACHED_PIPELINE

    logger.info(
        "Loading pyannote diarization pipeline: %s (device=%s)",
        model_name,
        target_device,
    )
    pipeline = Pipeline.from_pretrained(model_name)
    pipeline.to(torch.device(target_device))

    _CACHED_PIPELINE = pipeline
    _CACHED_PIPELINE_KEY = pipeline_key
    return _CACHED_PIPELINE


def diarize(
    audio_path: Union[str, Path],
    config_path: Union[str, Path] = DEFAULT_CONFIG_PATH,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    device: Optional[str] = None,
    model_name: Optional[str] = None,
) -> list[dict[str, Any]]:
    """
    Run speaker diarization on an audio file using pyannote.audio.

    Args:
        audio_path: Path to the input audio file.
        config_path: Path to config.yaml for optional diarization settings.
        num_speakers: Optional exact number of speakers override.
        min_speakers: Optional minimum number of speakers override.
        max_speakers: Optional maximum number of speakers override.
        device: Optional compute device override ('cuda', 'cpu').
        model_name: Optional pipeline model identifier override.

    Returns:
        List of speaker segment dictionaries sorted by start time, each containing:
            - "start": float start timestamp in seconds.
            - "end": float end timestamp in seconds.
            - "speaker_id": string speaker label (e.g. 'SPEAKER_00').

    Raises:
        FileNotFoundError: If audio_path does not exist.
    """
    audio_file = Path(audio_path)
    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")

    cfg_settings = _load_diarize_config(config_path)
    final_model_name = model_name if model_name is not None else cfg_settings["model_name"]
    final_device = device if device is not None else cfg_settings["device"]
    final_num_speakers = num_speakers if num_speakers is not None else cfg_settings["num_speakers"]
    final_min_speakers = min_speakers if min_speakers is not None else cfg_settings["min_speakers"]
    final_max_speakers = max_speakers if max_speakers is not None else cfg_settings["max_speakers"]

    pipeline = load_diarization_pipeline(
        model_name=final_model_name,
        device=final_device,
    )

    pipeline_kwargs: dict[str, Any] = {}
    if final_num_speakers is not None:
        pipeline_kwargs["num_speakers"] = final_num_speakers
    if final_min_speakers is not None:
        pipeline_kwargs["min_speakers"] = final_min_speakers
    if final_max_speakers is not None:
        pipeline_kwargs["max_speakers"] = final_max_speakers

    logger.info(
        "Running diarization on %s with parameters: %s",
        audio_file,
        pipeline_kwargs,
    )

    diarization_output = pipeline(str(audio_file), **pipeline_kwargs)

    # Handle pyannote 4.x DiarizeOutput vs pyannote 3.x Annotation
    annotation = (
        diarization_output.speaker_diarization
        if hasattr(diarization_output, "speaker_diarization")
        else diarization_output
    )

    speaker_segments: list[dict[str, Any]] = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        speaker_segments.append(
            {
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker_id": str(speaker),
            }
        )

    speaker_segments.sort(key=lambda seg: (seg["start"], seg["end"], seg["speaker_id"]))
    return speaker_segments


def merge_with_transcript(
    transcript_segments: list[dict[str, Any]],
    diarization_turns: list[dict[str, Any]],
    fallback_speaker: str = "UNKNOWN",
) -> list[dict[str, Any]]:
    """
    Attach a speaker_id to each ASR transcript segment based on maximum time overlap
    with diarization speaker turns.

    For each transcript segment:
    1. Calculates temporal overlap with each diarization turn.
    2. Sums total overlap duration per speaker across all turns.
    3. If maximum overlap > 0, assigns the speaker_id of the speaker with the greatest
       total overlap duration.
    4. Tie-breaking rule: If multiple speakers have identical positive overlap duration,
       the tie is broken deterministically by selecting the speaker whose overlapping turn
       starts earliest in the audio, followed by alphabetical order of speaker_id.
    5. Fallback rule: If the segment overlaps no speaker turn (or if diarization_turns
       is empty), the segment is assigned fallback_speaker (default 'UNKNOWN').

    Args:
        transcript_segments: List of transcript segment dictionaries (from asr.py),
            each containing at least 'start' and 'end' float timestamps.
        diarization_turns: List of diarization turn dictionaries, each containing
            'start', 'end', and 'speaker_id'.
        fallback_speaker: Speaker label to assign when an ASR segment has zero overlap
            with any diarization turn. Defaults to 'UNKNOWN'.

    Returns:
        A new list of transcript segment dictionaries with 'speaker_id' assigned.
    """
    merged_segments: list[dict[str, Any]] = []

    for seg in transcript_segments:
        seg_start = float(seg["start"])
        seg_end = float(seg["end"])

        # Map speaker_id -> {"total_overlap": float, "first_turn_start": float}
        speaker_stats: dict[str, dict[str, float]] = {}

        for turn in diarization_turns:
            turn_start = float(turn["start"])
            turn_end = float(turn["end"])
            spk_id = str(turn["speaker_id"])

            overlap_start = max(seg_start, turn_start)
            overlap_end = min(seg_end, turn_end)
            overlap = max(0.0, overlap_end - overlap_start)

            if overlap > 0.0:
                if spk_id not in speaker_stats:
                    speaker_stats[spk_id] = {
                        "total_overlap": overlap,
                        "first_turn_start": turn_start,
                    }
                else:
                    speaker_stats[spk_id]["total_overlap"] += overlap
                    speaker_stats[spk_id]["first_turn_start"] = min(
                        speaker_stats[spk_id]["first_turn_start"],
                        turn_start,
                    )

        if speaker_stats:
            # Deterministic selection:
            # - Primary: Maximum total overlap duration (descending)
            # - Tie-break 1: Earliest turn start timestamp (ascending)
            # - Tie-break 2: Alphabetical speaker_id (ascending)
            best_speaker = min(
                speaker_stats.keys(),
                key=lambda s: (
                    -speaker_stats[s]["total_overlap"],
                    speaker_stats[s]["first_turn_start"],
                    s,
                ),
            )
            assigned_speaker = best_speaker
        else:
            assigned_speaker = fallback_speaker

        updated_seg = dict(seg)
        updated_seg["speaker_id"] = assigned_speaker
        merged_segments.append(updated_seg)

    return merged_segments


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python src/diarize.py <path_to_audio_file>")
        sys.exit(1)

    input_audio_path = sys.argv[1]
    print(f"Running speaker diarization on: {input_audio_path}")
    turns = diarize(input_audio_path)
    print(f"Diarization resulted in {len(turns)} speaker turn(s):")
    for idx, turn in enumerate(turns, 1):
        print(
            f"  {idx}. [{turn['start']:.2f}s -> {turn['end']:.2f}s] "
            f"speaker={turn['speaker_id']}"
        )
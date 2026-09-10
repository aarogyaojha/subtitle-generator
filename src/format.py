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

import logging
from pathlib import Path
import textwrap
from typing import Any, Optional, Union

from src.config_utils import DEFAULT_CONFIG_PATH, load_stage_config

logger = logging.getLogger(__name__)

# Default formatting settings matching config.yaml
DEFAULT_MAX_CHARS_PER_LINE = 42
DEFAULT_MAX_LINES_PER_CUE = 2
DEFAULT_MAX_CPS = 17.0
DEFAULT_MIN_CUE_DURATION = 0.8
DEFAULT_MAX_CUE_DURATION = 7.0
DEFAULT_MIN_GAP_BETWEEN_CUES = 0.1


def _load_format_config(config_path: Union[str, Path] = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """
    Read subtitle formatting configuration settings if present in config.yaml,
    otherwise return default values.

    Args:
        config_path: Path to the configuration YAML file.

    Returns:
        Dictionary containing max_chars_per_line, max_lines_per_cue, max_cps,
        min_cue_duration, max_cue_duration, and min_gap_between_cues.
    """
    format_settings: dict[str, Any] = {
        "max_chars_per_line": DEFAULT_MAX_CHARS_PER_LINE,
        "max_lines_per_cue": DEFAULT_MAX_LINES_PER_CUE,
        "max_cps": DEFAULT_MAX_CPS,
        "min_cue_duration": DEFAULT_MIN_CUE_DURATION,
        "max_cue_duration": DEFAULT_MAX_CUE_DURATION,
        "min_gap_between_cues": DEFAULT_MIN_GAP_BETWEEN_CUES,
    }

    return load_stage_config(
        config_path=config_path,
        section_name="format",
        defaults=format_settings,
        top_level_keys=[
            "max_chars_per_line",
            "max_lines_per_cue",
            "max_cps",
            "min_cue_duration",
            "max_cue_duration",
            "min_gap_between_cues",
        ],
        stage_label="Format",
    )


def format_timestamp(seconds: float, format_type: str = "srt") -> str:
    """
    Convert a floating-point time offset in seconds into standard subtitle timecode format.

    Args:
        seconds: Time offset in seconds (e.g. 75.345).
        format_type: Output timecode standard ('srt' for HH:MM:SS,mmm or 'vtt' for HH:MM:SS.mmm).

    Returns:
        Formatted timecode string.
    """
    if seconds < 0.0:
        seconds = 0.0

    total_ms = int(round(seconds * 1000))
    ms = total_ms % 1000
    total_sec = total_ms // 1000
    sec = total_sec % 60
    total_min = total_sec // 60
    minute = total_min % 60
    hours = total_min // 60

    sep = "," if format_type.lower() == "srt" else "."
    return f"{hours:02d}:{minute:02d}:{sec:02d}{sep}{ms:03d}"


def wrap_text(
    text: str,
    max_chars_per_line: int = DEFAULT_MAX_CHARS_PER_LINE,
    max_lines_per_cue: int = DEFAULT_MAX_LINES_PER_CUE,
) -> list[str]:
    """
    Break text into lines conforming to maximum line length and line count limits.

    Args:
        text: Subtitle cue text to wrap.
        max_chars_per_line: Maximum character width per line.
        max_lines_per_cue: Maximum allowed lines per subtitle box.

    Returns:
        List of wrapped text lines.
    """
    clean_text = " ".join(text.strip().split())
    if not clean_text:
        return []

    lines = textwrap.wrap(
        clean_text,
        width=max_chars_per_line,
        break_long_words=True,
        break_on_hyphens=False,
    )

    if not lines:
        return [clean_text]

    # If lines exceed max_lines_per_cue, combine remaining lines into the last line
    if len(lines) > max_lines_per_cue:
        first_lines = lines[: max_lines_per_cue - 1]
        last_line = " ".join(lines[max_lines_per_cue - 1 :])
        lines = first_lines + [last_line]

    return lines


def _split_words_into_cue_chunks(
    words: list[dict[str, Any]],
    max_chars: int,
    max_duration: float,
) -> list[list[dict[str, Any]]]:
    """
    Split a list of word dictionaries into chunks that fit within character and duration constraints.

    Args:
        words: List of word dicts with 'word', 'start', and 'end'.
        max_chars: Maximum character count allowed in a single cue box.
        max_duration: Maximum duration in seconds allowed for a single cue.

    Returns:
        List of word chunks (each chunk is a list of word dicts).
    """
    chunks: list[list[dict[str, Any]]] = []
    current_chunk: list[dict[str, Any]] = []
    current_chars = 0
    current_start = 0.0

    for word_info in words:
        word_text = str(word_info.get("word", "")).strip()
        if not word_text:
            continue

        w_start = float(word_info.get("start", 0.0))
        w_end = float(word_info.get("end", w_start))
        added_len = len(word_text) + (1 if current_chunk else 0)

        should_split = False
        if current_chunk:
            if current_chars + added_len > max_chars:
                should_split = True
            elif (w_end - current_start) > max_duration:
                should_split = True

        if should_split and current_chunk:
            chunks.append(current_chunk)
            current_chunk = []
            current_chars = 0

        if not current_chunk:
            current_start = w_start

        current_chunk.append(word_info)
        current_chars += added_len

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def _split_text_into_cue_chunks(
    text: str,
    seg_start: float,
    seg_end: float,
    max_chars: int,
    max_duration: float,
) -> list[dict[str, Any]]:
    """
    Split a raw text string without word-level timestamps into chunks with interpolated timestamps.

    Args:
        text: Segment text.
        seg_start: Start timestamp in seconds.
        seg_end: End timestamp in seconds.
        max_chars: Maximum character count allowed per cue.
        max_duration: Maximum duration in seconds allowed per cue.

    Returns:
        List of sub-cue dictionaries containing 'start', 'end', and 'text'.
    """
    words = text.strip().split()
    if not words:
        return []

    # Group words into chunks by max_chars
    word_chunks: list[list[str]] = []
    curr_chunk: list[str] = []
    curr_len = 0

    for w in words:
        add_len = len(w) + (1 if curr_chunk else 0)
        if curr_chunk and curr_len + add_len > max_chars:
            word_chunks.append(curr_chunk)
            curr_chunk = []
            curr_len = 0
        curr_chunk.append(w)
        curr_len += add_len

    if curr_chunk:
        word_chunks.append(curr_chunk)

    total_chars = max(1, sum(len(w) for w in words))
    total_duration = max(0.0, seg_end - seg_start)

    sub_cues: list[dict[str, Any]] = []
    consumed_chars = 0

    for chunk in word_chunks:
        chunk_text = " ".join(chunk)
        chunk_chars = sum(len(w) for w in chunk)

        c_start = seg_start + (consumed_chars / total_chars) * total_duration
        consumed_chars += chunk_chars
        c_end = seg_start + (consumed_chars / total_chars) * total_duration

        dur = c_end - c_start
        if dur > max_duration and len(chunk) > 1:
            n_sub = int(dur // max_duration) + 1
            words_per_sub = max(1, len(chunk) // n_sub)
            sub_start = c_start
            for i in range(0, len(chunk), words_per_sub):
                sub_words = chunk[i : i + words_per_sub]
                sub_text = " ".join(sub_words)
                sub_sub_dur = (len(sub_text) / len(chunk_text)) * dur
                sub_cues.append(
                    {
                        "start": sub_start,
                        "end": sub_start + sub_sub_dur,
                        "text": sub_text,
                    }
                )
                sub_start += sub_sub_dur
        else:
            sub_cues.append(
                {
                    "start": c_start,
                    "end": c_end,
                    "text": chunk_text,
                }
            )

    return sub_cues


def create_cues(
    transcript_segments: list[dict[str, Any]],
    config_path: Union[str, Path] = DEFAULT_CONFIG_PATH,
    **overrides: Any,
) -> list[dict[str, Any]]:
    """
    Transform merged transcript segments into readability-constrained subtitle cues.

    Enforces:
    - Speaker boundary breaks (different speaker_id values trigger separate cues).
    - Character and line limits per cue.
    - Reading speed (CPS) duration extension (only extends, never shortens natural duration).
    - Min/max cue duration clamping.
    - Non-overlapping consecutive cues and minimum gap enforcement via a single chronological sweep.

    Args:
        transcript_segments: List of transcript segments (from diarize.merge_with_transcript),
            each containing 'start', 'end', 'text', and optionally 'speaker_id' and 'words'.
        config_path: Path to config.yaml.
        **overrides: Optional runtime overrides for format configuration parameters.

    Returns:
        List of formatted cue dictionaries, each containing 'index', 'start', 'end',
        'text', 'lines', and 'speaker_id'.
    """
    config = _load_format_config(config_path)
    config.update(overrides)

    max_chars_per_line = int(config.get("max_chars_per_line", DEFAULT_MAX_CHARS_PER_LINE))
    max_lines_per_cue = int(config.get("max_lines_per_cue", DEFAULT_MAX_LINES_PER_CUE))
    max_cps = float(config.get("max_cps", DEFAULT_MAX_CPS))
    min_cue_duration = float(config.get("min_cue_duration", DEFAULT_MIN_CUE_DURATION))
    max_cue_duration = float(config.get("max_cue_duration", DEFAULT_MAX_CUE_DURATION))
    min_gap = float(config.get("min_gap_between_cues", DEFAULT_MIN_GAP_BETWEEN_CUES))

    max_chars_per_cue = max_chars_per_line * max_lines_per_cue

    raw_cues: list[dict[str, Any]] = []

    for seg in transcript_segments:
        text = str(seg.get("text", "")).strip()
        if not text:
            continue

        seg_start = float(seg.get("start", 0.0))
        seg_end = float(seg.get("end", seg_start))
        speaker_id = seg.get("speaker_id")
        words = seg.get("words", [])

        if words:
            word_chunks = _split_words_into_cue_chunks(
                words=words,
                max_chars=max_chars_per_cue,
                max_duration=max_cue_duration,
            )
            for chunk in word_chunks:
                chunk_text = " ".join(str(w.get("word", "")).strip() for w in chunk).strip()
                if not chunk_text:
                    continue
                c_start = float(chunk[0].get("start", seg_start))
                c_end = float(chunk[-1].get("end", seg_end))
                raw_cues.append(
                    {
                        "start": c_start,
                        "end": max(c_start, c_end),
                        "text": chunk_text,
                        "speaker_id": speaker_id,
                    }
                )
        else:
            sub_chunks = _split_text_into_cue_chunks(
                text=text,
                seg_start=seg_start,
                seg_end=seg_end,
                max_chars=max_chars_per_cue,
                max_duration=max_cue_duration,
            )
            for sc in sub_chunks:
                sc_text = str(sc["text"]).strip()
                if not sc_text:
                    continue
                raw_cues.append(
                    {
                        "start": float(sc["start"]),
                        "end": float(sc["end"]),
                        "text": sc_text,
                        "speaker_id": speaker_id,
                    }
                )

    if not raw_cues:
        return []

    # Sort raw cues by start timestamp
    raw_cues.sort(key=lambda c: (c["start"], c["end"]))

    # Single chronological sweep: Apply CPS extension, clamp, and resolve gaps/overlaps
    # Constraint 1: CPS extension only increases duration: final_duration = max(natural_duration, len(text)/max_cps)
    # clamped to [min_cue_duration, max_cue_duration]
    for i in range(len(raw_cues)):
        cue = raw_cues[i]
        natural_duration = max(0.0, cue["end"] - cue["start"])
        text_len = len(cue["text"])
        cps_duration = (text_len / max_cps) if max_cps > 0 else 0.0

        # Duration only increases: max(natural_duration, cps_duration)
        target_duration = max(natural_duration, cps_duration, min_cue_duration)
        target_duration = min(target_duration, max_cue_duration)
        cue["end"] = cue["start"] + target_duration

        # Constraint 2: Overlap/gap resolution cascades chronologically to the next cue
        if i < len(raw_cues) - 1:
            next_cue = raw_cues[i + 1]
            if cue["end"] + min_gap > next_cue["start"]:
                # Cap cue[i].end so it respects min_gap before next_cue.start
                max_allowed_end = next_cue["start"] - min_gap
                if max_allowed_end >= cue["start"]:
                    cue["end"] = max_allowed_end
                else:
                    cue["end"] = max(cue["start"], next_cue["start"])

    # Final pass: Wrap text lines and assign 1-based sequential indices
    formatted_cues: list[dict[str, Any]] = []
    for idx, cue in enumerate(raw_cues, start=1):
        lines = wrap_text(
            text=cue["text"],
            max_chars_per_line=max_chars_per_line,
            max_lines_per_cue=max_lines_per_cue,
        )
        formatted_cues.append(
            {
                "index": idx,
                "start": cue["start"],
                "end": cue["end"],
                "text": cue["text"],
                "lines": lines,
                "speaker_id": cue.get("speaker_id"),
            }
        )

    return formatted_cues


def format_srt(cues: list[dict[str, Any]], include_speaker: bool = False) -> str:
    """
    Format a list of subtitle cues as a standard SubRip (.srt) string.

    For SRT, when include_speaker is True, speaker labels are formatted as plain-text
    prefixes (e.g. '[SPEAKER_00] text') rather than HTML/VTT tags.

    Args:
        cues: List of formatted subtitle cue dictionaries.
        include_speaker: If True, prefixes text with plain-text speaker label '[SPEAKER_XX]'.

    Returns:
        Complete .srt formatted string.
    """
    if not cues:
        return ""

    blocks: list[str] = []
    for cue in cues:
        idx = cue["index"]
        start_ts = format_timestamp(cue["start"], format_type="srt")
        end_ts = format_timestamp(cue["end"], format_type="srt")

        lines = list(cue.get("lines", [cue.get("text", "")]))
        if include_speaker and cue.get("speaker_id"):
            spk_label = f"[{cue['speaker_id']}]"
            if lines:
                lines[0] = f"{spk_label} {lines[0]}"
            else:
                lines = [spk_label]

        body = "\n".join(lines)
        blocks.append(f"{idx}\n{start_ts} --> {end_ts}\n{body}")

    return "\n\n".join(blocks) + "\n"


def format_vtt(cues: list[dict[str, Any]], include_speaker: bool = False) -> str:
    """
    Format a list of subtitle cues as a standard WebVTT (.vtt) string.

    For VTT, when include_speaker is True, speaker labels are formatted using standard
    WebVTT voice tags (e.g. '<v SPEAKER_00>text</v>').

    Args:
        cues: List of formatted subtitle cue dictionaries.
        include_speaker: If True, wraps cue text in WebVTT voice tags '<v SPEAKER_XX>'.

    Returns:
        Complete .vtt formatted string.
    """
    if not cues:
        return "WEBVTT\n"

    blocks: list[str] = ["WEBVTT"]
    for cue in cues:
        idx = cue["index"]
        start_ts = format_timestamp(cue["start"], format_type="vtt")
        end_ts = format_timestamp(cue["end"], format_type="vtt")

        lines = list(cue.get("lines", [cue.get("text", "")]))
        if include_speaker and cue.get("speaker_id"):
            spk_id = cue["speaker_id"]
            if lines:
                lines[0] = f"<v {spk_id}>{lines[0]}"
                lines[-1] = f"{lines[-1]}</v>"
            else:
                lines = [f"<v {spk_id}></v>"]

        body = "\n".join(lines)
        blocks.append(f"{idx}\n{start_ts} --> {end_ts}\n{body}")

    return "\n\n".join(blocks) + "\n"


def write_srt(
    cues: list[dict[str, Any]],
    output_path: Union[str, Path],
    include_speaker: bool = False,
) -> Path:
    """
    Write formatted subtitle cues to an .srt file on disk.

    Args:
        cues: List of formatted subtitle cue dictionaries.
        output_path: Destination path for the .srt file.
        include_speaker: If True, includes plain-text speaker labels.

    Returns:
        The Path to the written file.
    """
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    content = format_srt(cues=cues, include_speaker=include_speaker)
    out_file.write_text(content, encoding="utf-8")
    logger.info("Saved SRT subtitles to %s (%d cues)", out_file, len(cues))
    return out_file


def write_vtt(
    cues: list[dict[str, Any]],
    output_path: Union[str, Path],
    include_speaker: bool = False,
) -> Path:
    """
    Write formatted subtitle cues to a .vtt file on disk.

    Args:
        cues: List of formatted subtitle cue dictionaries.
        output_path: Destination path for the .vtt file.
        include_speaker: If True, includes WebVTT voice tags.

    Returns:
        The Path to the written file.
    """
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    content = format_vtt(cues=cues, include_speaker=include_speaker)
    out_file.write_text(content, encoding="utf-8")
    logger.info("Saved VTT subtitles to %s (%d cues)", out_file, len(cues))
    return out_file


def format_transcript(
    transcript_segments: list[dict[str, Any]],
    config_path: Union[str, Path] = DEFAULT_CONFIG_PATH,
    **overrides: Any,
) -> list[dict[str, Any]]:
    """
    Convenience alias for create_cues. Formats transcript segments into subtitle cues.

    Args:
        transcript_segments: Merged transcript segments.
        config_path: Path to config.yaml.
        **overrides: Configuration overrides.

    Returns:
        List of formatted subtitle cue dictionaries.
    """
    return create_cues(
        transcript_segments=transcript_segments,
        config_path=config_path,
        **overrides,
    )


if __name__ == "__main__":
    sample_segments = [
        {
            "start": 0.5,
            "end": 2.5,
            "text": "नमस्कार म आज नेपालको बारेमा कुरा गर्दैछु।",
            "speaker_id": "SPEAKER_00",
        },
        {
            "start": 2.8,
            "end": 4.5,
            "text": "धेरै धेरै धन्यवाद।",
            "speaker_id": "SPEAKER_01",
        },
    ]

    cues = create_cues(sample_segments)
    print("--- SRT OUTPUT ---")
    print(format_srt(cues, include_speaker=True))
    print("--- VTT OUTPUT ---")
    print(format_vtt(cues, include_speaker=True))
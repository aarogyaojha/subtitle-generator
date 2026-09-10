import logging
from pathlib import Path
import pytest
import yaml

from src.asr import transcribe
from src.diarize import diarize, merge_with_transcript
from src.format import (
    DEFAULT_MAX_CHARS_PER_LINE,
    DEFAULT_MAX_CPS,
    DEFAULT_MAX_CUE_DURATION,
    DEFAULT_MAX_LINES_PER_CUE,
    DEFAULT_MIN_CUE_DURATION,
    DEFAULT_MIN_GAP_BETWEEN_CUES,
    _load_format_config,
    create_cues,
    format_srt,
    format_timestamp,
    format_transcript,
    format_vtt,
    wrap_text,
    write_srt,
    write_vtt,
)
from src.vad import get_speech_regions

logger = logging.getLogger(__name__)


def test_format_timestamp_edge_cases():
    """Verify timestamp conversion for both SRT and VTT across edge cases."""
    # Zero seconds
    assert format_timestamp(0.0, "srt") == "00:00:00,000"
    assert format_timestamp(0.0, "vtt") == "00:00:00.000"

    # Sub-second with rounding
    assert format_timestamp(0.5, "srt") == "00:00:00,500"
    assert format_timestamp(0.5004, "vtt") == "00:00:00.500"
    assert format_timestamp(0.9996, "srt") == "00:00:01,000"

    # Normal offset
    assert format_timestamp(75.345, "srt") == "00:01:15,345"
    assert format_timestamp(75.345, "vtt") == "00:01:15.345"

    # Greater than 1 hour
    assert format_timestamp(3661.123, "srt") == "01:01:01,123"
    assert format_timestamp(3661.123, "vtt") == "01:01:01.123"

    # Large hours
    assert format_timestamp(360000.0, "srt") == "100:00:00,000"

    # Negative inputs clamped to zero
    assert format_timestamp(-5.0, "srt") == "00:00:00,000"


def test_wrap_text_line_length_and_lines_limit():
    """Verify that wrap_text wraps lines cleanly according to configured limits."""
    # Short text
    assert wrap_text("Hello world", max_chars_per_line=20, max_lines_per_cue=2) == ["Hello world"]

    # Multi-line within max_lines
    text = "The quick brown fox jumps over the lazy dog"
    lines = wrap_text(text, max_chars_per_line=20, max_lines_per_cue=3)
    assert len(lines) <= 3
    assert all(len(line) <= 20 for line in lines[:-1])
    assert " ".join(lines) == text

    # Empty or whitespace-only text
    assert wrap_text("") == []
    assert wrap_text("   \n\t  ") == []


def test_short_segment_produces_single_cue():
    """Verify that a single short segment produces exactly one cue with appropriate timestamps."""
    segment = [
        {
            "start": 1.0,
            "end": 3.0,
            "text": "नमस्ते साथीहरु।",
            "speaker_id": "SPEAKER_00",
        }
    ]
    cues = create_cues(segment)
    assert len(cues) == 1
    assert cues[0]["index"] == 1
    assert cues[0]["start"] == 1.0
    assert cues[0]["text"] == "नमस्ते साथीहरु।"
    assert cues[0]["speaker_id"] == "SPEAKER_00"


def test_long_segment_splits_into_multiple_cues_with_words():
    """Verify that an overly long segment with word timestamps splits into multiple cues."""
    words = [
        {"word": f"word{i}", "start": float(i), "end": float(i) + 0.8, "probability": 0.95}
        for i in range(25)
    ]
    segment = [
        {
            "start": 0.0,
            "end": 25.0,
            "text": " ".join(f"word{i}" for i in range(25)),
            "speaker_id": "SPEAKER_00",
            "words": words,
        }
    ]
    # max_chars_per_line=20, max_lines_per_cue=2 -> max 40 chars per cue
    cues = create_cues(
        segment,
        max_chars_per_line=20,
        max_lines_per_cue=2,
        max_cue_duration=6.0,
    )
    assert len(cues) > 1
    for c in cues:
        assert len(c["text"]) <= 40
        assert c["end"] - c["start"] <= 6.0


def test_long_segment_splits_into_multiple_cues_without_words():
    """Verify that an overly long segment without word timestamps splits cleanly via text chunking."""
    long_text = "This is a very long segment that should exceed the maximum character limits per cue and split."
    segment = [
        {
            "start": 1.0,
            "end": 10.0,
            "text": long_text,
            "speaker_id": "SPEAKER_01",
        }
    ]
    cues = create_cues(
        segment,
        max_chars_per_line=25,
        max_lines_per_cue=2,
    )
    assert len(cues) > 1
    # Check that text is preserved across cues
    reconstructed = " ".join(c["text"] for c in cues)
    assert reconstructed == long_text


def test_cps_duration_extension_on_fast_speech():
    """Verify that a cue with fast speech (high CPS) has its duration extended to satisfy max_cps."""
    # 34 characters spoken in 0.5s -> 68 CPS (exceeds max_cps=17.0)
    fast_text = "यस भिडियोमा स्वागत छ सबैजनालाई।"
    segment = [
        {
            "start": 1.0,
            "end": 1.5,  # 0.5s natural duration
            "text": fast_text,
            "speaker_id": "SPEAKER_00",
        }
    ]
    cues = create_cues(segment, max_cps=17.0, min_cue_duration=0.5, max_cue_duration=7.0)
    assert len(cues) == 1
    expected_min_duration = len(fast_text) / 17.0
    actual_duration = cues[0]["end"] - cues[0]["start"]
    assert actual_duration >= expected_min_duration - 1e-6
    assert cues[0]["end"] == pytest.approx(1.0 + expected_min_duration, rel=1e-5)


def test_ample_natural_duration_remains_unchanged():
    """
    Constraint 1 verification:
    Verify that a cue with already-ample natural duration is left unchanged
    and never shortened: final_duration = max(natural_duration, len(text) / max_cps).
    """
    short_text = "Hello"  # 5 chars -> len(text) / 17.0 = ~0.29s
    segment = [
        {
            "start": 2.0,
            "end": 6.0,  # 4.0s natural duration (already ample)
            "text": short_text,
            "speaker_id": "SPEAKER_00",
        }
    ]
    cues = create_cues(segment, max_cps=17.0, min_cue_duration=0.8, max_cue_duration=7.0)
    assert len(cues) == 1
    # Duration must remain 4.0s (not shortened to 0.8 or 0.29s)
    assert cues[0]["start"] == 2.0
    assert cues[0]["end"] == 6.0


def test_chronological_cascade_overlap_resolution_three_cues():
    """
    Constraint 2 verification:
    Verify that overlap/gap resolution processes cues in one chronological sweep,
    so extending an early cue's duration correctly cascades into re-checking
    and adjusting the gap against subsequent cues.
    """
    # Cue 1: 0.0s to 0.2s, 50 chars -> needs 50 / 10.0 = 5.0s (would extend to 5.0s if unconstrained)
    # Cue 2: starts at 3.0s to 3.5s, 10 chars -> needs 1.0s (would end at 4.0s)
    # Cue 3: starts at 4.5s to 5.5s
    segments = [
        {
            "start": 0.0,
            "end": 0.2,
            "text": "A" * 50,
            "speaker_id": "SPEAKER_00",
        },
        {
            "start": 3.0,
            "end": 3.5,
            "text": "B" * 10,
            "speaker_id": "SPEAKER_01",
        },
        {
            "start": 4.5,
            "end": 5.5,
            "text": "C" * 10,
            "speaker_id": "SPEAKER_00",
        },
    ]

    min_gap = 0.2
    cues = create_cues(
        segments,
        max_cps=10.0,
        min_cue_duration=0.5,
        max_cue_duration=7.0,
        min_gap_between_cues=min_gap,
    )

    assert len(cues) == 3

    # Cue 0 should be capped before Cue 1 start by min_gap
    assert cues[0]["end"] <= cues[1]["start"] - min_gap
    assert pytest.approx(cues[0]["end"], 0.01) == 3.0 - min_gap

    # Cue 1 should maintain min_gap before Cue 2 start
    assert cues[1]["end"] <= cues[2]["start"] - min_gap

    # All cues must have positive duration
    for c in cues:
        assert c["end"] >= c["start"]


def test_speaker_boundary_separation():
    """Verify that different speaker_id values are kept in separate cues."""
    segments = [
        {
            "start": 1.0,
            "end": 2.0,
            "text": "First speaker line.",
            "speaker_id": "SPEAKER_00",
        },
        {
            "start": 2.1,
            "end": 3.0,
            "text": "Second speaker line.",
            "speaker_id": "SPEAKER_01",
        },
    ]
    cues = create_cues(segments)
    assert len(cues) == 2
    assert cues[0]["speaker_id"] == "SPEAKER_00"
    assert cues[1]["speaker_id"] == "SPEAKER_01"


def test_srt_and_vtt_speaker_formatting_rules():
    """
    Constraint 3 verification:
    Verify that SRT include_speaker=True uses plain-text prefix '[SPEAKER_XX]',
    while VTT include_speaker=True uses WebVTT '<v SPEAKER_XX>' tags.
    """
    cues = [
        {
            "index": 1,
            "start": 1.0,
            "end": 2.5,
            "text": "नमस्ते।",
            "lines": ["नमस्ते।"],
            "speaker_id": "SPEAKER_01",
        }
    ]

    srt_out = format_srt(cues, include_speaker=True)
    assert "1\n00:00:01,000 --> 00:00:02,500\n[SPEAKER_01] नमस्ते।" in srt_out
    assert "<v" not in srt_out

    vtt_out = format_vtt(cues, include_speaker=True)
    assert "WEBVTT\n" in vtt_out
    assert "1\n00:00:01.000 --> 00:00:02.500\n<v SPEAKER_01>नमस्ते।</v>" in vtt_out
    assert "[SPEAKER_01]" not in vtt_out


def test_empty_input_returns_valid_empty_output(tmp_path: Path):
    """Verify that empty input produces empty cues and valid empty/header-only outputs."""
    cues = create_cues([])
    assert cues == []

    srt_str = format_srt([])
    assert srt_str == ""

    vtt_str = format_vtt([])
    assert vtt_str == "WEBVTT\n"

    srt_file = write_srt([], tmp_path / "empty.srt")
    assert srt_file.exists()
    assert srt_file.read_text(encoding="utf-8") == ""

    vtt_file = write_vtt([], tmp_path / "empty.vtt")
    assert vtt_file.exists()
    assert vtt_file.read_text(encoding="utf-8") == "WEBVTT\n"


def test_writers_create_files_and_directories(tmp_path: Path):
    """Verify write_srt and write_vtt create nested parent directories and valid files."""
    cues = [
        {
            "index": 1,
            "start": 0.5,
            "end": 2.0,
            "text": "परीक्षण सबटाइटल।",
            "lines": ["परीक्षण सबटाइटल।"],
            "speaker_id": "SPEAKER_00",
        }
    ]

    nested_srt = tmp_path / "sub" / "dir" / "test.srt"
    out_srt = write_srt(cues, nested_srt)
    assert out_srt.exists()
    content_srt = out_srt.read_text(encoding="utf-8")
    assert "00:00:00,500 --> 00:00:02,000" in content_srt
    assert "परीक्षण सबटाइटल।" in content_srt

    nested_vtt = tmp_path / "sub" / "dir" / "test.vtt"
    out_vtt = write_vtt(cues, nested_vtt)
    assert out_vtt.exists()
    content_vtt = out_vtt.read_text(encoding="utf-8")
    assert "WEBVTT" in content_vtt
    assert "00:00:00.500 --> 00:00:02.000" in content_vtt


def test_config_parsing_and_overrides(tmp_path: Path):
    """Verify parsing format section and top-level overrides from YAML."""
    custom_yaml = tmp_path / "format_config.yaml"
    cfg = {
        "max_chars_per_line": 35,
        "format": {
            "max_lines_per_cue": 1,
            "max_cps": 15.0,
            "min_cue_duration": 1.0,
            "max_cue_duration": 5.0,
            "min_gap_between_cues": 0.15,
        },
    }
    with open(custom_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    loaded = _load_format_config(custom_yaml)
    assert loaded["max_chars_per_line"] == 35
    assert loaded["max_lines_per_cue"] == 1
    assert loaded["max_cps"] == 15.0
    assert loaded["min_cue_duration"] == 1.0
    assert loaded["max_cue_duration"] == 5.0
    assert loaded["min_gap_between_cues"] == 0.15


def test_real_full_pipeline_formatting_integration(tmp_path: Path):
    """
    Real Integration Test:
    Execute VAD -> ASR -> Diarize -> Format on tests/fixtures/nepali_sample.wav.
    Write .srt and .vtt to tmp_path and print the file contents to stdout.
    """
    fixture_path = Path(__file__).resolve().parent / "fixtures" / "nepali_sample.wav"
    assert fixture_path.exists(), f"Missing fixture at {fixture_path}"

    # 1. VAD stage
    speech_regions = get_speech_regions(fixture_path)
    assert len(speech_regions) > 0, "VAD found no speech regions"

    # 2. ASR stage
    asr_segments = transcribe(fixture_path, speech_regions=speech_regions, word_timestamps=True)
    assert len(asr_segments) > 0, "ASR produced no segments"

    # 3. Diarize stage
    diarization_turns = diarize(fixture_path)
    merged_segments = merge_with_transcript(asr_segments, diarization_turns)
    assert len(merged_segments) == len(asr_segments)

    # 4. Format stage
    cues = create_cues(merged_segments)
    assert len(cues) > 0, "Format stage produced no cues"

    srt_file = write_srt(cues, tmp_path / "nepali_output.srt", include_speaker=True)
    vtt_file = write_vtt(cues, tmp_path / "nepali_output.vtt", include_speaker=True)

    assert srt_file.exists()
    assert vtt_file.exists()

    srt_content = srt_file.read_text(encoding="utf-8")
    vtt_content = vtt_file.read_text(encoding="utf-8")

    # Print contents to stdout for user inspection
    print("\n" + "=" * 60)
    print("REAL NEPALI INTEGRATION TEST - GENERATED SRT OUTPUT:")
    print("=" * 60)
    print(srt_content)
    print("=" * 60)
    print("REAL NEPALI INTEGRATION TEST - GENERATED VTT OUTPUT:")
    print("=" * 60)
    print(vtt_content)
    print("=" * 60)

    # Basic validations
    assert "00:00:" in srt_content
    assert "WEBVTT" in vtt_content
    assert any(cue["speaker_id"] is not None for cue in cues)

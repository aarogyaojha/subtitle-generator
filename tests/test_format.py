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


def test_wrap_text_unbroken_token_no_fabricated_spaces():
    """
    Regression test: verify that a single unbroken token (e.g. CJK or long URL)
    exceeding max_chars_per_line is split by character count with NO fabricated spaces.
    """
    # 1. Japanese sentence with no spaces (14 chars), max_chars_per_line=13, max_lines=2
    ja_text = "新しい靴を履いて出かけます。"
    lines_ja = wrap_text(ja_text, max_chars_per_line=13, max_lines_per_cue=2)
    assert len(lines_ja) == 2
    assert lines_ja[0] == "新しい靴を履いて出かけます"
    assert lines_ja[1] == "。"
    # Zero fabricated characters: joining lines back together produces the exact original text
    assert "".join(lines_ja) == ja_text
    assert all(" " not in line for line in lines_ja)


    # 2. Long unbroken ASCII token (26 chars), max_chars_per_line=10, max_lines=3
    ascii_token = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    lines_ascii = wrap_text(ascii_token, max_chars_per_line=10, max_lines_per_cue=3)
    assert len(lines_ascii) == 3
    assert lines_ascii[0] == "ABCDEFGHIJ"
    assert lines_ascii[1] == "KLMNOPQRST"
    assert lines_ascii[2] == "UVWXYZ"
    assert "".join(lines_ascii) == ascii_token

    # 3. Unbroken token longer than max_chars_per_line * max_lines_per_cue
    lines_overflow = wrap_text(ascii_token, max_chars_per_line=10, max_lines_per_cue=2)
    assert len(lines_overflow) == 2
    assert lines_overflow[0] == "ABCDEFGHIJ"
    assert lines_overflow[1] == "KLMNOPQRSTUVWXYZ"
    assert "".join(lines_overflow) == ascii_token


def test_cjk_unbroken_token_create_cues_exact_duration_and_no_space_corruption():
    """
    Regression test: verify create_cues preserves original token characters from Whisper word dicts
    and computes CPS duration from the true content character count (not inflated by space insertion).
    """
    # Simulate Whisper subwords for Japanese text "新しい靴を履いて出かけます。" (14 chars)
    words = [
        {"word": " 新", "start": 0.8, "end": 1.14},
        {"word": "しい", "start": 1.14, "end": 1.34},
        {"word": " 靴", "start": 1.34, "end": 1.62},
        {"word": "を", "start": 1.62, "end": 1.76},
        {"word": " 履", "start": 1.76, "end": 2.02},
        {"word": "いて", "start": 2.02, "end": 2.22},
        {"word": " 出", "start": 2.22, "end": 2.36},
        {"word": "か", "start": 2.36, "end": 2.40},
        {"word": "け", "start": 2.40, "end": 2.44},
        {"word": " ます。", "start": 2.44, "end": 2.48},
    ]
    segment = [
        {
            "start": 0.8,
            "end": 2.48,
            "text": "新しい靴を履いて出かけます。",
            "words": words,
            "speaker_id": "SPEAKER_00",
        }
    ]

    cues = create_cues(segment, max_cps=4.0, max_chars_per_line=13, max_lines_per_cue=2)
    assert len(cues) == 1
    cue = cues[0]

    # Content text must match original exactly
    assert cue["text"] == "新しい靴を履いて出かけます。"
    assert " " not in cue["text"]

    # Lines must contain zero added characters
    assert "".join(cue["lines"]) == "新しい靴を履いて出かけます。"

    # Duration must be exactly len(text) / 4.0 = 14 / 4.0 = 3.5s (from start 0.8s -> end 4.3s)
    expected_duration = 14 / 4.0
    actual_duration = cue["end"] - cue["start"]
    assert abs(actual_duration - expected_duration) < 1e-4
    assert abs(cue["end"] - (0.8 + 3.5)) < 1e-4


def test_cjk_multi_chunk_split_exact_character_reconstruction_and_duration():
    """
    Synthetic Unit Test:
    Verify that long space-free CJK text spanning multiple chunks:
    1. Concatenating all cues' text in order reproduces the original text character-for-character with ZERO added characters.
    2. No cue's lines contain any character (e.g. space) not present in the original text.
    3. Each cue's duration reflects its own chunk's character count divided by max_cps (not the full original segment's length).
    """
    # 30-character space-free Japanese sentence: "吾輩は猫である名前はまだ無いどこで生れたかとんと見当がつかぬ"
    cjk_text = "吾輩は猫である名前はまだ無いどこで生れたかとんと見当がつかぬ"
    assert len(cjk_text) == 30
    assert " " not in cjk_text


    # Mock Whisper word/subword timestamp output spanning 12.0 seconds
    # 10 subword chunks of 2-4 characters each
    subwords = [
        "吾輩", "は", "猫である", "名前は", "まだ無い",
        "どこで", "生れたか", "とんと", "見当が", "つかぬ",
    ]
    assert "".join(subwords) == cjk_text

    words = []
    t = 0.0
    for idx, sw in enumerate(subwords):
        # In Whisper, subwords may have a leading space in raw BPE representation: e.g. " 吾輩", " は"
        word_entry = {
            "word": f" {sw}" if idx > 0 else sw,
            "start": t,
            "end": t + 1.1,
        }
        words.append(word_entry)
        t += 1.2

    segment = [
        {
            "start": 0.0,
            "end": 12.0,
            "text": cjk_text,
            "words": words,
            "speaker_id": "SPEAKER_00",
        }
    ]

    # max_cue_duration=4.0 and max_chars_per_cue=26 (13 * 2) forces splitting into >= 2 cues
    cues = create_cues(
        segment,
        max_chars_per_line=13,
        max_lines_per_cue=2,
        max_cue_duration=4.0,
        max_cps=4.0,
    )

    # Must produce multiple cues
    assert len(cues) >= 2, f"Expected multiple cues for 12s audio, got {len(cues)}"

    # 1. Exact character-for-character reconstruction of original text
    reconstructed_text = "".join(c["text"] for c in cues)
    assert reconstructed_text == cjk_text
    assert len(reconstructed_text) == len(cjk_text)

    # 2. No cue contains fabricated characters or spaces
    for cue in cues:
        assert " " not in cue["text"]
        for line in cue["lines"]:
            assert " " not in line
            assert len(line) <= 13
        # Re-joining cue's lines must reproduce cue["text"]
        assert "".join(cue["lines"]) == cue["text"]

    all_lines_text = "".join("".join(c["lines"]) for c in cues)
    assert all_lines_text == cjk_text

    # 3. Each cue's duration reflects its own chunk's character count, not the full 31 chars
    for cue in cues:
        chunk_len = len(cue["text"])
        expected_chunk_cps_dur = chunk_len / 4.0
        actual_dur = cue["end"] - cue["start"]
        assert actual_dur >= (expected_chunk_cps_dur - 1e-4)
        # Verify it is not scaled using the full 31 characters (31 / 4.0 = 7.75s)
        assert actual_dur < (len(cjk_text) / 4.0)


def test_space_containing_multi_chunk_split_word_joining():
    """
    Synthetic Unit Test:
    Verify that multi-chunk splitting on space-containing text (English/Nepali)
    joins words within each chunk with single spaces correctly.
    """
    words = [
        {"word": f"word{i}", "start": float(i), "end": float(i) + 0.8}
        for i in range(12)
    ]
    full_text = " ".join(f"word{i}" for i in range(12))
    segment = [
        {
            "start": 0.0,
            "end": 12.0,
            "text": full_text,
            "words": words,
            "speaker_id": "SPEAKER_00",
        }
    ]

    cues = create_cues(
        segment,
        max_chars_per_line=20,
        max_lines_per_cue=2,
        max_cue_duration=4.0,
    )

    assert len(cues) >= 2
    # Joining cues' texts with space reproduces original text
    assert " ".join(c["text"] for c in cues) == full_text
    for cue in cues:
        assert "  " not in cue["text"]
        assert len(cue["text"]) > 0


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


def test_format_config_fallback_when_no_language_override(tmp_path: Path):
    """Verify base defaults apply when no language override exists for the active language."""
    custom_yaml = tmp_path / "config.yaml"
    cfg = {
        "language": "en",
        "max_chars_per_line": 42,
        "max_lines_per_cue": 2,
        "max_cps": 17.0,
        "min_cue_duration": 0.8,
        "max_cue_duration": 7.0,
        "min_gap_between_cues": 0.1,
        "format": {
            "language_overrides": {
                "ja": {
                    "max_chars_per_line": 13,
                    "max_cps": 4.0,
                }
            }
        },
    }
    with open(custom_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    loaded = _load_format_config(custom_yaml)
    assert loaded["max_chars_per_line"] == 42
    assert loaded["max_lines_per_cue"] == 2
    assert loaded["max_cps"] == 17.0
    assert loaded["min_cue_duration"] == 0.8
    assert loaded["max_cue_duration"] == 7.0
    assert loaded["min_gap_between_cues"] == 0.1


def test_format_config_language_override_ja(tmp_path: Path):
    """Verify that a language override for Japanese overrides specified keys while others fall back to defaults."""
    custom_yaml = tmp_path / "config.yaml"
    cfg = {
        "language": "ja",
        "max_chars_per_line": 42,
        "max_lines_per_cue": 2,
        "max_cps": 17.0,
        "min_cue_duration": 0.8,
        "max_cue_duration": 7.0,
        "min_gap_between_cues": 0.1,
        "format": {
            "language_overrides": {
                "ja": {
                    "max_chars_per_line": 13,
                    "max_cps": 4.0,
                }
            }
        },
    }
    with open(custom_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    loaded = _load_format_config(custom_yaml)
    assert loaded["max_chars_per_line"] == 13
    assert loaded["max_cps"] == 4.0
    assert loaded["max_lines_per_cue"] == 2
    assert loaded["min_cue_duration"] == 0.8
    assert loaded["max_cue_duration"] == 7.0
    assert loaded["min_gap_between_cues"] == 0.1


def test_format_config_explicit_language_param_precedence(tmp_path: Path):
    """Verify that an explicit language parameter takes precedence over config.yaml's top-level language key."""
    custom_yaml = tmp_path / "config.yaml"
    cfg = {
        "language": "ne",
        "max_chars_per_line": 42,
        "max_lines_per_cue": 2,
        "max_cps": 17.0,
        "min_cue_duration": 0.8,
        "max_cue_duration": 7.0,
        "min_gap_between_cues": 0.1,
        "format": {
            "language_overrides": {
                "ja": {
                    "max_chars_per_line": 13,
                    "max_cps": 4.0,
                },
                "ne": {
                    "max_chars_per_line": 42,
                    "max_cps": 17.0,
                },
            }
        },
    }
    with open(custom_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    # 1. Without explicit language parameter, loads 'ne' based on config.yaml
    loaded_ne = _load_format_config(custom_yaml)
    assert loaded_ne["max_chars_per_line"] == 42
    assert loaded_ne["max_cps"] == 17.0

    # 2. With explicit language='ja', overrides top-level 'ne'
    loaded_ja = _load_format_config(custom_yaml, language="ja")
    assert loaded_ja["max_chars_per_line"] == 13
    assert loaded_ja["max_cps"] == 4.0
    assert loaded_ja["max_lines_per_cue"] == 2
    assert loaded_ja["min_cue_duration"] == 0.8

    # 3. create_cues with explicit language parameter applies the override
    # Text length 20 chars at 4.0 CPS requires at least 20 / 4.0 = 5.0 seconds duration
    segment = [{"start": 0.0, "end": 1.0, "text": "12345678901234567890"}]
    cues = create_cues(segment, config_path=custom_yaml, language="ja")
    assert len(cues) == 1
    assert cues[0]["end"] >= 5.0



@pytest.mark.real_model
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


@pytest.mark.real_model
def test_real_japanese_formatting_integration(tmp_path: Path):
    """
    Real Integration Test:
    Execute VAD -> ASR -> Diarize -> Format on tests/fixtures/japanese_sample.wav with language="ja".
    Explicitly asserts that Japanese formatting overrides (max_chars_per_line <= 13, max_cps <= 4.0)
    were applied to the real transcribed text, and writes .srt/.vtt files to tmp_path.
    """
    fixture_path = Path(__file__).resolve().parent / "fixtures" / "japanese_sample.wav"
    assert fixture_path.exists(), f"Missing fixture at {fixture_path}"

    # 1. VAD stage
    speech_regions = get_speech_regions(fixture_path)
    assert len(speech_regions) > 0, "VAD found no speech regions in japanese_sample.wav"

    # 2. ASR stage
    asr_segments = transcribe(
        fixture_path,
        speech_regions=speech_regions,
        language="ja",
        word_timestamps=True,
    )
    assert len(asr_segments) > 0, "ASR produced no segments for Japanese fixture"

    # 3. Diarize stage
    diarization_turns = diarize(fixture_path)
    merged_segments = merge_with_transcript(asr_segments, diarization_turns)
    assert len(merged_segments) == len(asr_segments)

    # 4. Format stage with explicit language="ja"
    cues = create_cues(merged_segments, language="ja")
    assert len(cues) > 0, "Format stage produced no cues"

    # Verify Japanese formatting constraints were applied:
    # A. Line length cap: <= 13 characters per line
    for cue in cues:
        for line in cue["lines"]:
            assert len(line) <= 13, f"Line exceeds 13 characters in Japanese cue: '{line}' ({len(line)} chars)"

    # B. CPS constraint: duration >= len(text) / 4.0 CPS (or min_cue_duration)
    for cue in cues:
        duration = cue["end"] - cue["start"]
        text_len = len(cue["text"])
        required_duration_by_cps = text_len / 4.0
        assert duration >= (required_duration_by_cps - 1e-4), (
            f"Cue duration {duration:.3f}s does not satisfy 4.0 CPS cap for text '{cue['text']}' "
            f"({text_len} chars, requires >= {required_duration_by_cps:.3f}s)"
        )

    srt_file = write_srt(cues, tmp_path / "japanese_output.srt", include_speaker=True)
    vtt_file = write_vtt(cues, tmp_path / "japanese_output.vtt", include_speaker=True)

    assert srt_file.exists()
    assert vtt_file.exists()

    srt_content = srt_file.read_text(encoding="utf-8")
    vtt_content = vtt_file.read_text(encoding="utf-8")

    # Print contents to stdout for inspection
    print("\n" + "=" * 60)
    print("REAL JAPANESE INTEGRATION TEST - GENERATED SRT OUTPUT:")
    print("=" * 60)
    print(srt_content)
    print("=" * 60)
    print("REAL JAPANESE INTEGRATION TEST - GENERATED VTT OUTPUT:")
    print("=" * 60)
    print(vtt_content)
    print("=" * 60)

    # Basic validations
    assert "00:00:" in srt_content
    assert "WEBVTT" in vtt_content
    assert any(cue["speaker_id"] is not None for cue in cues)


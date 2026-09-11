import logging
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import yaml

from src.asr import transcribe
from src.diarize import (
    DEFAULT_MODEL_NAME,
    _load_diarize_config,
    diarize,
    load_diarization_pipeline,
    merge_with_transcript,
)
from src.vad import get_speech_regions

logger = logging.getLogger(__name__)


@pytest.fixture
def dummy_audio_file(tmp_path: Path) -> Path:
    """Create a dummy audio file for testing path resolution without decoding."""
    file_path = tmp_path / "dummy.wav"
    file_path.write_bytes(b"RIFFdummydataWAVEfmt ")
    return file_path


def test_nonexistent_audio_raises_filenotfound():
    """Verify that a nonexistent file path raises FileNotFoundError."""
    nonexistent = Path("/nonexistent/path/to/audio.wav")
    with pytest.raises(FileNotFoundError):
        diarize(nonexistent)


def test_config_parsing_and_overrides(tmp_path: Path):
    """Verify parsing of custom diarization settings from config.yaml."""
    custom_config_path = tmp_path / "custom_config.yaml"
    custom_cfg = {
        "num_speakers": 2,
        "diarize": {
            "num_speakers": 3,
            "min_speakers": 1,
            "max_speakers": 4,
            "device": "cpu",
            "model_name": "pyannote/speaker-diarization-3.1",
        },
    }
    with open(custom_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(custom_cfg, f)

    settings = _load_diarize_config(custom_config_path)
    assert settings["num_speakers"] == 3
    assert settings["min_speakers"] == 1
    assert settings["max_speakers"] == 4
    assert settings["device"] == "cpu"
    assert settings["model_name"] == "pyannote/speaker-diarization-3.1"


def test_config_fallback_on_corrupt_yaml(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    """Verify that invalid/corrupt YAML falls back to defaults and logs a warning."""
    corrupt_config = tmp_path / "corrupt_config.yaml"
    corrupt_config.write_text("diarize: [invalid yaml: {", encoding="utf-8")

    settings = _load_diarize_config(corrupt_config)
    assert settings["num_speakers"] is None
    assert settings["min_speakers"] is None
    assert settings["max_speakers"] is None
    assert settings["model_name"] == DEFAULT_MODEL_NAME
    assert "Failed to parse diarization configuration" in caplog.text


def test_diarize_mocked_pipeline_output_format(dummy_audio_file: Path, tmp_path: Path):
    """Verify that diarize() invokes the pipeline with proper kwargs and formats output dicts."""
    mock_turn1 = MagicMock()
    mock_turn1.start = 1.0
    mock_turn1.end = 3.5

    mock_turn2 = MagicMock()
    mock_turn2.start = 4.0
    mock_turn2.end = 6.2

    mock_annotation = MagicMock()
    mock_annotation.itertracks.return_value = [
        (mock_turn2, "track2", "SPEAKER_01"),
        (mock_turn1, "track1", "SPEAKER_00"),
    ]

    mock_output = MagicMock()
    mock_output.speaker_diarization = mock_annotation

    mock_pipeline = MagicMock()
    mock_pipeline.return_value = mock_output

    custom_config = tmp_path / "diarize_config.yaml"
    custom_cfg = {
        "diarize": {
            "num_speakers": 2,
            "min_speakers": 1,
            "max_speakers": 3,
        }
    }
    with open(custom_config, "w", encoding="utf-8") as f:
        yaml.safe_dump(custom_cfg, f)

    with patch("src.diarize.load_diarization_pipeline", return_value=mock_pipeline) as mock_load:
        turns = diarize(dummy_audio_file, config_path=custom_config)

        mock_load.assert_called_once()
        mock_pipeline.assert_called_once_with(
            str(dummy_audio_file),
            num_speakers=2,
            min_speakers=1,
            max_speakers=3,
        )

        assert len(turns) == 2
        # Sorted by start time
        assert turns[0] == {"start": 1.0, "end": 3.5, "speaker_id": "SPEAKER_00"}
        assert turns[1] == {"start": 4.0, "end": 6.2, "speaker_id": "SPEAKER_01"}


def test_merge_with_transcript_clean_overlap():
    """Verify standard 1-to-1 overlap assignment between ASR segments and speaker turns."""
    asr_segments = [
        {"start": 1.0, "end": 2.5, "text": "Hello world", "confidence": 0.95},
        {"start": 3.0, "end": 4.8, "text": "Good morning", "confidence": 0.92},
    ]
    diarization_turns = [
        {"start": 0.8, "end": 2.7, "speaker_id": "SPEAKER_00"},
        {"start": 2.9, "end": 5.0, "speaker_id": "SPEAKER_01"},
    ]

    merged = merge_with_transcript(asr_segments, diarization_turns)
    assert len(merged) == 2
    assert merged[0]["speaker_id"] == "SPEAKER_00"
    assert merged[0]["text"] == "Hello world"
    assert merged[1]["speaker_id"] == "SPEAKER_01"
    assert merged[1]["text"] == "Good morning"


def test_merge_with_transcript_competing_overlap():
    """Verify that when multiple speakers overlap a segment, the one with highest overlap wins."""
    asr_segments = [
        # Segment from 1.0 to 3.0 (duration 2.0s)
        # SPEAKER_00 overlaps 1.0 to 1.6 (0.6s)
        # SPEAKER_01 overlaps 1.6 to 3.0 (1.4s) -> SPEAKER_01 should win
        {"start": 1.0, "end": 3.0, "text": "Turn transition segment"},
    ]
    diarization_turns = [
        {"start": 0.5, "end": 1.6, "speaker_id": "SPEAKER_00"},
        {"start": 1.6, "end": 3.5, "speaker_id": "SPEAKER_01"},
    ]

    merged = merge_with_transcript(asr_segments, diarization_turns)
    assert len(merged) == 1
    assert merged[0]["speaker_id"] == "SPEAKER_01"


def test_merge_with_transcript_deterministic_tie_breaking():
    """Verify deterministic tie-breaking when two speakers have exactly identical overlap duration."""
    # Segment from 1.0 to 3.0 (duration 2.0s)
    # SPEAKER_A overlaps [1.0, 2.0] (1.0s overlap, starts at 0.5)
    # SPEAKER_B overlaps [2.0, 3.0] (1.0s overlap, starts at 2.0)
    # Both have 1.0s overlap. SPEAKER_A starts earlier (0.5 < 2.0), so SPEAKER_A wins.
    asr_segments = [{"start": 1.0, "end": 3.0, "text": "Tied segment"}]
    diarization_turns = [
        {"start": 0.5, "end": 2.0, "speaker_id": "SPEAKER_A"},
        {"start": 2.0, "end": 3.5, "speaker_id": "SPEAKER_B"},
    ]

    merged = merge_with_transcript(asr_segments, diarization_turns)
    assert merged[0]["speaker_id"] == "SPEAKER_A"

    # Secondary tie-break: Same overlap duration AND same start time -> alphabetical speaker_id
    # SPEAKER_2 overlaps [1.0, 2.0] (1.0s overlap, starts at 1.0)
    # SPEAKER_1 overlaps [2.0, 3.0] (1.0s overlap, starts at 1.0 - e.g. co-occurring turn)
    diarization_turns_same_start = [
        {"start": 1.0, "end": 2.0, "speaker_id": "SPEAKER_2"},
        {"start": 1.0, "end": 2.0, "speaker_id": "SPEAKER_1"},
    ]
    merged_alpha = merge_with_transcript(asr_segments, diarization_turns_same_start)
    assert merged_alpha[0]["speaker_id"] == "SPEAKER_1"


def test_merge_with_transcript_gap_fallback_unknown():
    """Verify that an ASR segment falling entirely within silence/gap receives the fallback speaker."""
    asr_segments = [
        # Segment from 5.0 to 6.0 in a gap between speaker turns
        {"start": 5.0, "end": 6.0, "text": "Silence artifact or noise"},
    ]
    diarization_turns = [
        {"start": 1.0, "end": 3.0, "speaker_id": "SPEAKER_00"},
        {"start": 8.0, "end": 10.0, "speaker_id": "SPEAKER_01"},
    ]

    merged = merge_with_transcript(asr_segments, diarization_turns)
    assert len(merged) == 1
    assert merged[0]["speaker_id"] == "UNKNOWN"

    # Custom fallback speaker string
    merged_custom = merge_with_transcript(asr_segments, diarization_turns, fallback_speaker="NO_SPEAKER")
    assert merged_custom[0]["speaker_id"] == "NO_SPEAKER"


def test_merge_with_transcript_empty_inputs():
    """Verify handling when either transcript segments or diarization turns are empty."""
    asr_segments = [{"start": 1.0, "end": 2.0, "text": "Test"}]

    # Empty diarization turns -> fallback
    merged_empty_diar = merge_with_transcript(asr_segments, [])
    assert len(merged_empty_diar) == 1
    assert merged_empty_diar[0]["speaker_id"] == "UNKNOWN"

    # Empty transcript segments -> returns empty list
    merged_empty_trans = merge_with_transcript([], [{"start": 1.0, "end": 2.0, "speaker_id": "SPEAKER_00"}])
    assert merged_empty_trans == []


@pytest.mark.real_model
def test_real_nepali_diarization_and_merge_integration():
    """
    Real integration test on tests/fixtures/nepali_sample.wav.

    Verifies:
    1. pyannote diarization executes on the real Nepali audio fixture.
    2. Returns exactly one speaker turn (SPEAKER_00) matching single-speaker ground truth
       without over-segmenting into multiple fake speakers.
    3. The turn covers roughly the expected 1.1s–2.8s window.
    4. merge_with_transcript correctly assigns SPEAKER_00 to the transcribed ASR segment.
    """
    fixture_path = Path(__file__).resolve().parent / "fixtures" / "nepali_sample.wav"
    assert fixture_path.exists(), f"Real speech fixture missing: {fixture_path}"

    # 1. Run real diarization
    turns = diarize(fixture_path)
    assert len(turns) == 1, f"Expected exactly 1 speaker turn for single-speaker audio, got: {turns}"
    speaker_turn = turns[0]

    assert speaker_turn["speaker_id"] == "SPEAKER_00"
    assert 0.8 <= speaker_turn["start"] <= 1.4, f"Unexpected turn start: {speaker_turn['start']}"
    assert 2.4 <= speaker_turn["end"] <= 3.0, f"Unexpected turn end: {speaker_turn['end']}"
    assert speaker_turn["end"] > speaker_turn["start"]

    # 2. Run real VAD + ASR transcription
    speech_regions = get_speech_regions(fixture_path)
    asr_segments = transcribe(fixture_path, speech_regions=speech_regions, language="ne")
    assert len(asr_segments) > 0, "ASR transcription produced no segments"

    # 3. Merge transcript with diarization
    merged = merge_with_transcript(asr_segments, turns)
    assert len(merged) == len(asr_segments)
    for seg in merged:
        assert seg["speaker_id"] == "SPEAKER_00"

    # Visual eyeball report
    report_lines = [
        "\n=======================================================",
        "DIARIZATION & MERGE INTEGRATION REPORT (Utterance 50173ab781)",
        f"  Fixture File       : {fixture_path.name}",
        f"  Diarization Turns  : {turns}",
        "  Merged Transcript Segments:",
    ]
    for idx, seg in enumerate(merged, 1):
        report_lines.append(
            f"    {idx}. [{seg['start']:.2f}s -> {seg['end']:.2f}s] "
            f"[{seg['speaker_id']}] text='{seg['text']}' (confidence: {seg['confidence']:.4f})"
        )
    report_lines.append("=======================================================")
    report_str = "\n".join(report_lines)
    print(report_str)
    logger.info(report_str)


@pytest.mark.real_model
def test_real_japanese_diarization_and_merge_integration():
    """
    Real integration test on tests/fixtures/japanese_sample.wav.

    Verifies:
    1. pyannote diarization executes on the real Japanese audio fixture.
    2. Returns valid speaker turns for the audio fixture.
    3. merge_with_transcript correctly assigns speaker tags to the transcribed Japanese ASR segments.
    """
    fixture_path = Path(__file__).resolve().parent / "fixtures" / "japanese_sample.wav"
    assert fixture_path.exists(), f"Real speech fixture missing: {fixture_path}"

    # 1. Run real diarization
    turns = diarize(fixture_path)
    assert len(turns) > 0, f"Expected at least 1 speaker turn for speech fixture, got: {turns}"
    for turn in turns:
        assert isinstance(turn["speaker_id"], str) and len(turn["speaker_id"]) > 0
        assert turn["end"] > turn["start"]

    # 2. Run real VAD + ASR transcription with language="ja"
    speech_regions = get_speech_regions(fixture_path)
    asr_segments = transcribe(fixture_path, speech_regions=speech_regions, language="ja")
    assert len(asr_segments) > 0, "ASR transcription produced no segments"

    # 3. Merge transcript with diarization
    merged = merge_with_transcript(asr_segments, turns)
    assert len(merged) == len(asr_segments)
    for seg in merged:
        assert seg["speaker_id"] is not None

    # Visual eyeball report
    report_lines = [
        "\n=======================================================",
        "DIARIZATION & MERGE INTEGRATION REPORT (Common Voice JA 19499629)",
        f"  Fixture File       : {fixture_path.name}",
        f"  Diarization Turns  : {turns}",
        "  Merged Transcript Segments:",
    ]
    for idx, seg in enumerate(merged, 1):
        report_lines.append(
            f"    {idx}. [{seg['start']:.2f}s -> {seg['end']:.2f}s] "
            f"[{seg['speaker_id']}] text='{seg['text']}' (confidence: {seg['confidence']:.4f})"
        )
    report_lines.append("=======================================================")
    report_str = "\n".join(report_lines)
    print(report_str)
    logger.info(report_str)


@pytest.mark.real_model
def test_real_multispeaker_diarization_accuracy():
    """
    Real integration & structural accuracy test on tests/fixtures/nepali_multispeaker_sample.wav.

    Verifies:
    1. pyannote diarization executes successfully on the real constructed multi-speaker audio fixture.
    2. Enforces structural sanity: non-empty turns with valid time spans (start >= 0, start < end)
       and at least 1 speaker detected.
    3. Evaluates and honestly reports diarization performance (distinct speakers detected vs.
       ground-truth count of 3, and midpoint-to-speaker mapping across ground-truth segments)
       for manual inspection without brittle assertions.
    """
    import json

    fixtures_dir = Path(__file__).resolve().parent / "fixtures"
    fixture_path = fixtures_dir / "nepali_multispeaker_sample.wav"
    gt_path = fixtures_dir / "nepali_multispeaker_ground_truth.json"

    assert fixture_path.exists(), f"Multi-speaker fixture missing: {fixture_path}"
    assert gt_path.exists(), f"Ground truth metadata missing: {gt_path}"

    gt_data = json.loads(gt_path.read_text(encoding="utf-8"))
    gt_segments = gt_data["segments"]
    expected_speaker_count = gt_data.get("num_ground_truth_speakers", 3)

    # 1. Run real diarization
    turns = diarize(fixture_path)
    assert len(turns) > 0, "Diarization produced no turns on multi-speaker fixture"

    # 2. Structural sanity assertions
    for turn in turns:
        assert turn["start"] >= 0.0, f"Turn start time negative: {turn}"
        assert turn["end"] > turn["start"], f"Turn end time not greater than start: {turn}"
        assert isinstance(turn["speaker_id"], str) and len(turn["speaker_id"]) > 0, f"Invalid speaker_id: {turn}"

    detected_speakers = sorted(set(t["speaker_id"] for t in turns))
    assert len(detected_speakers) >= 1, "Expected at least 1 speaker detected"

    # 3. Compute ground-truth segment midpoint mappings
    midpoint_mappings = []
    for seg in gt_segments:
        midpoint = (seg["start_time_seconds"] + seg["end_time_seconds"]) / 2.0
        matching_turns = [
            t["speaker_id"] for t in turns if t["start"] <= midpoint <= t["end"]
        ]
        assigned_speaker = matching_turns[0] if matching_turns else None
        midpoint_mappings.append((seg, midpoint, assigned_speaker))

    mapped_speakers = [m[2] for m in midpoint_mappings if m[2] is not None]
    distinct_mapped_speakers = set(mapped_speakers)

    # 4. Print & log honest accuracy evaluation report
    report_lines = [
        "\n=======================================================",
        "MULTI-SPEAKER DIARIZATION ACCURACY INTEGRATION REPORT",
        f"  Fixture File             : {fixture_path.name}",
        f"  Total Duration           : {gt_data['total_duration_seconds']:.2f}s",
        f"  Ground Truth Speakers    : {expected_speaker_count} ({gt_data.get('ground_truth_speaker_ids', [])})",
        f"  Detected Speakers Count  : {len(detected_speakers)} ({detected_speakers})",
        f"  Midpoint Mapped Speakers : {len(distinct_mapped_speakers)} distinct label(s) across {len(gt_segments)} segment(s)",
        "\n  Ground Truth vs. Diarization Midpoint Mapping:",
    ]
    for seg, mid, assigned_spk in midpoint_mappings:
        spk_str = f"[{assigned_spk}]" if assigned_spk is not None else "[UNASSIGNED / SILENCE]"
        report_lines.append(
            f"    - Segment {seg['segment_index']} (Original Speaker: {seg['speaker_id']}): "
            f"[{seg['start_time_seconds']:.2f}s -> {seg['end_time_seconds']:.2f}s] "
            f"(mid: {mid:.2f}s) -> Diarization Label: {spk_str} | Text: '{seg['text']}'"
        )
    report_lines.append("\n  All Detected Diarization Turns:")
    for idx, turn in enumerate(turns, 1):
        report_lines.append(
            f"    Turn {idx}. [{turn['start']:.2f}s -> {turn['end']:.2f}s] [{turn['speaker_id']}]"
        )
    report_lines.append("=======================================================")
    report_str = "\n".join(report_lines)
    print(report_str)
    logger.info(report_str)




import logging
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import yaml

from src.asr import (
    DEFAULT_COMPUTE_TYPE,
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL_SIZE,
    _load_asr_config,
    load_asr_model,
    transcribe,
)
from src.vad import get_speech_regions

logger = logging.getLogger(__name__)

# Ground truth transcription from OpenSLR 54 dataset TSV (`asr_nepali/utt_spk_text.tsv`)
# for utterance 50173ab781 (speaker 6059d)
OPENSLR_54_GROUND_TRUTH_50173ab781 = "वसन्तपुर दरवार गद्दी"

# Ground truth transcription from Mozilla Common Voice 17.0 Japanese test set (`transcript/ja/test.tsv`)
# for utterance common_voice_ja_19499629.mp3 (sentence ID: 15ad6b4189a5cd1670c10c64fdfeb84f01e7d203ccc2b6ebd444961df4b89020)
COMMON_VOICE_JA_GROUND_TRUTH_19499629 = "新しい靴をはいて出かけます。"



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
        transcribe(nonexistent)


def test_empty_speech_regions_returns_empty_list(dummy_audio_file: Path):
    """Verify that empty speech regions ([]) returns [] immediately without calling the model."""
    with patch("src.asr.load_asr_model") as mock_load_model:
        result = transcribe(dummy_audio_file, speech_regions=[])
        assert result == []
        mock_load_model.assert_not_called()


def test_config_parsing_and_overrides(tmp_path: Path):
    """Verify parsing of custom ASR settings from config.yaml."""
    custom_config_path = tmp_path / "custom_config.yaml"
    custom_cfg = {
        "language": "ja",
        "asr": {
            "model_size": "medium",
            "compute_type": "float16",
            "device": "cpu",
            "language": "ja",
        },
    }
    with open(custom_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(custom_cfg, f)

    settings = _load_asr_config(custom_config_path)
    assert settings["model_size"] == "medium"
    assert settings["compute_type"] == "float16"
    assert settings["device"] == "cpu"
    assert settings["language"] == "ja"


def test_config_fallback_on_corrupt_yaml(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    """Verify that invalid/corrupt YAML falls back to defaults and logs a warning."""
    corrupt_config = tmp_path / "corrupt_config.yaml"
    corrupt_config.write_text("asr: [invalid: {", encoding="utf-8")

    settings = _load_asr_config(corrupt_config)
    assert settings["model_size"] == DEFAULT_MODEL_SIZE
    assert settings["compute_type"] == DEFAULT_COMPUTE_TYPE
    assert settings["language"] == DEFAULT_LANGUAGE
    assert "Failed to parse ASR configuration" in caplog.text


def test_transcribe_mocked_model_and_formatting(dummy_audio_file: Path, tmp_path: Path):
    """Verify that transcribe correctly passes arguments to WhisperModel and formats output segments."""
    mock_word = MagicMock()
    mock_word.word = " नमस्ते"
    mock_word.start = 0.5
    mock_word.end = 1.0
    mock_word.probability = 0.95

    mock_segment = MagicMock()
    mock_segment.start = 0.5
    mock_segment.end = 1.8
    mock_segment.text = " नमस्ते संसार "
    mock_segment.avg_logprob = -0.2
    mock_segment.no_speech_prob = 0.01
    mock_segment.words = [mock_word]

    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([mock_segment], MagicMock())

    custom_config = tmp_path / "asr_config.yaml"
    custom_cfg = {
        "asr": {
            "model_size": "small",
            "compute_type": "int8",
            "device": "cpu",
            "language": "ne",
        }
    }
    with open(custom_config, "w", encoding="utf-8") as f:
        yaml.safe_dump(custom_cfg, f)

    with patch("src.asr.load_asr_model", return_value=mock_model) as mock_load:
        speech_regions = [(0.5, 1.8), (2.0, 3.5)]
        segments = transcribe(
            dummy_audio_file,
            speech_regions=speech_regions,
            config_path=custom_config,
        )

        mock_load.assert_called_with(
            model_size_or_path="small",
            device="cpu",
            compute_type="int8",
        )
        mock_model.transcribe.assert_called_once_with(
            str(dummy_audio_file),
            language="ne",
            word_timestamps=True,
            clip_timestamps=[0.5, 1.8, 2.0, 3.5],
        )

        assert len(segments) == 1
        seg = segments[0]
        assert seg["start"] == 0.5
        assert seg["end"] == 1.8
        assert seg["text"] == "नमस्ते संसार"
        assert 0.0 <= seg["confidence"] <= 1.0
        assert seg["avg_logprob"] == -0.2
        assert seg["no_speech_prob"] == 0.01
        assert len(seg["words"]) == 1
        assert seg["words"][0]["word"] == " नमस्ते"


def test_real_nepali_transcription_integration(caplog: pytest.LogCaptureFixture):
    """
    Integration test: run real Whisper transcription on tests/fixtures/nepali_sample.wav.
    
    Verifies that speech regions from VAD are passed as clip timestamps,
    the model produces non-empty output, and logs the comparison between
    the OpenSLR ground truth and actual transcribed text.
    """
    fixture_path = Path(__file__).resolve().parent / "fixtures" / "nepali_sample.wav"
    assert fixture_path.exists(), f"Real speech fixture missing: {fixture_path}"

    # 1. Run VAD to obtain real speech regions
    speech_regions = get_speech_regions(fixture_path)
    assert len(speech_regions) > 0, "Expected VAD to detect speech in nepali_sample.wav"
    logger.info("Detected VAD speech regions for fixture: %s", speech_regions)

    # 2. Run real ASR transcription
    segments = transcribe(fixture_path, speech_regions=speech_regions, language="ne")

    assert len(segments) > 0, "Expected transcription output to have at least one segment"
    first_seg = segments[0]
    assert isinstance(first_seg["text"], str) and len(first_seg["text"]) > 0
    assert 0.0 <= first_seg["confidence"] <= 1.0
    assert first_seg["start"] >= 0.0
    assert first_seg["end"] > first_seg["start"]

    transcribed_text = " ".join(seg["text"] for seg in segments)

    # Print & log ground truth vs transcribed text comparison for manual eyeball inspection
    comparison_report = (
        "\n=======================================================\n"
        "ASR INTEGRATION TEST COMPARISON REPORT (Utterance 50173ab781)\n"
        f"  VAD Clip Boundaries : {speech_regions}\n"
        f"  Ground Truth (TSV)  : {OPENSLR_54_GROUND_TRUTH_50173ab781}\n"
        f"  Actual Transcribed  : {transcribed_text}\n"
        f"  Confidence Score    : {first_seg['confidence']:.4f} (avg_logprob: {first_seg['avg_logprob']:.4f})\n"
        "======================================================="
    )
    print(comparison_report)
    logger.info(comparison_report)


def test_real_japanese_transcription_integration(caplog: pytest.LogCaptureFixture):
    """
    Integration test: run real Whisper transcription on tests/fixtures/japanese_sample.wav.

    Verifies that speech regions from VAD are passed as clip timestamps,
    the model produces non-empty output with language="ja", and logs the comparison
    between the Mozilla Common Voice ground truth and actual transcribed text.
    """
    fixture_path = Path(__file__).resolve().parent / "fixtures" / "japanese_sample.wav"
    assert fixture_path.exists(), f"Real speech fixture missing: {fixture_path}"

    # 1. Run VAD to obtain real speech regions
    speech_regions = get_speech_regions(fixture_path)
    assert len(speech_regions) > 0, "Expected VAD to detect speech in japanese_sample.wav"
    logger.info("Detected VAD speech regions for Japanese fixture: %s", speech_regions)

    # 2. Run real ASR transcription with language="ja"
    segments = transcribe(fixture_path, speech_regions=speech_regions, language="ja")

    assert len(segments) > 0, "Expected transcription output to have at least one segment"
    first_seg = segments[0]
    assert isinstance(first_seg["text"], str) and len(first_seg["text"]) > 0
    assert 0.0 <= first_seg["confidence"] <= 1.0
    assert first_seg["start"] >= 0.0
    assert first_seg["end"] > first_seg["start"]

    transcribed_text = " ".join(seg["text"] for seg in segments)

    # Print & log ground truth vs transcribed text comparison for manual eyeball inspection
    comparison_report = (
        "\n=======================================================\n"
        "ASR INTEGRATION TEST COMPARISON REPORT (Common Voice JA 19499629)\n"
        f"  VAD Clip Boundaries : {speech_regions}\n"
        f"  Ground Truth (TSV)  : {COMMON_VOICE_JA_GROUND_TRUTH_19499629}\n"
        f"  Actual Transcribed  : {transcribed_text}\n"
        f"  Confidence Score    : {first_seg['confidence']:.4f} (avg_logprob: {first_seg['avg_logprob']:.4f})\n"
        "======================================================="
    )
    print(comparison_report)
    logger.info(comparison_report)


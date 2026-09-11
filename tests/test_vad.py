import struct
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import yaml

from src.vad import (
    DEFAULT_MIN_SILENCE_DURATION_MS,
    DEFAULT_MIN_SPEECH_DURATION_MS,
    DEFAULT_SPEECH_PAD_MS,
    DEFAULT_THRESHOLD,
    _load_thresholds_from_config,
    get_speech_regions,
)


def _write_wav_file(file_path: Path, num_samples: int, sample_rate: int = 16000) -> Path:
    """Helper to write a mono 16-bit PCM WAV file containing zeros."""
    with wave.open(str(file_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        if num_samples > 0:
            raw_data = struct.pack(f"<{num_samples}h", *([0] * num_samples))
            wf.writeframes(raw_data)
        else:
            wf.writeframes(b"")
    return file_path


@pytest.fixture
def temp_audio_dir(tmp_path: Path) -> Path:
    """Fixture providing a temporary directory for audio fixtures."""
    return tmp_path


@pytest.fixture
def silent_audio_file(temp_audio_dir: Path) -> Path:
    """Creates a 2-second synthetic silent audio file (16kHz, mono)."""
    file_path = temp_audio_dir / "silence.wav"
    return _write_wav_file(file_path, num_samples=16000 * 2, sample_rate=16000)


@pytest.fixture
def empty_audio_file(temp_audio_dir: Path) -> Path:
    """Creates an empty 0-sample audio file (16kHz, mono)."""
    file_path = temp_audio_dir / "empty.wav"
    return _write_wav_file(file_path, num_samples=0, sample_rate=16000)


def test_nonexistent_audio_raises_filenotfound():
    """Verify that a nonexistent file path raises FileNotFoundError."""
    nonexistent = Path("/nonexistent/path/to/audio.wav")
    with pytest.raises(FileNotFoundError):
        get_speech_regions(nonexistent)


def test_empty_audio_returns_empty_list(empty_audio_file: Path):
    """Verify that an empty audio file returns an empty list without error."""
    regions = get_speech_regions(empty_audio_file)
    assert regions == []


def test_silent_audio_returns_empty_list(silent_audio_file: Path):
    """Verify that pure silent audio returns an empty list of speech regions."""
    mock_model = MagicMock()
    mock_get_speech_timestamps = MagicMock(return_value=[])
    mock_utils = (mock_get_speech_timestamps,)
    with patch("src.vad.load_vad_model", return_value=(mock_model, mock_utils)):
        regions = get_speech_regions(silent_audio_file)
        assert regions == []


def test_config_threshold_overrides_from_yaml(tmp_path: Path):
    """Verify that custom thresholds in config.yaml are properly parsed."""
    custom_config_path = tmp_path / "custom_config.yaml"
    custom_cfg = {
        "vad": {
            "threshold": 0.8,
            "min_speech_duration_ms": 400,
            "min_silence_duration_ms": 200,
            "speech_pad_ms": 50,
        }
    }
    with open(custom_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(custom_cfg, f)

    thresholds = _load_thresholds_from_config(custom_config_path)
    assert thresholds["threshold"] == 0.8
    assert thresholds["min_speech_duration_ms"] == 400
    assert thresholds["min_silence_duration_ms"] == 200
    assert thresholds["speech_pad_ms"] == 50


def test_config_fallback_on_corrupt_yaml(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    """Verify that invalid/corrupt YAML falls back to defaults and logs a warning."""
    corrupt_config = tmp_path / "corrupt_config.yaml"
    corrupt_config.write_text("vad: [invalid yaml: {", encoding="utf-8")

    thresholds = _load_thresholds_from_config(corrupt_config)
    assert thresholds["threshold"] == DEFAULT_THRESHOLD
    assert thresholds["min_speech_duration_ms"] == DEFAULT_MIN_SPEECH_DURATION_MS
    assert thresholds["min_silence_duration_ms"] == DEFAULT_MIN_SILENCE_DURATION_MS
    assert thresholds["speech_pad_ms"] == DEFAULT_SPEECH_PAD_MS
    assert "Failed to parse VAD configuration" in caplog.text


def test_get_speech_regions_respects_threshold_overrides_and_mock(silent_audio_file: Path, tmp_path: Path):
    """Verify get_speech_regions passes the configured and explicit overrides to Silero VAD."""
    mock_model = MagicMock()
    mock_get_speech_timestamps = MagicMock(return_value=[{"start": 0.5, "end": 1.5}])
    mock_utils = (mock_get_speech_timestamps,)

    custom_config_path = tmp_path / "custom_config.yaml"
    custom_cfg = {
        "vad": {
            "threshold": 0.75,
            "min_speech_duration_ms": 350,
            "min_silence_duration_ms": 150,
            "speech_pad_ms": 45,
        }
    }
    with open(custom_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(custom_cfg, f)

    with patch("src.vad.load_vad_model", return_value=(mock_model, mock_utils)):
        # 1. Test using config.yaml values
        regions = get_speech_regions(silent_audio_file, config_path=custom_config_path)
        assert regions == [(0.5, 1.5)]
        mock_get_speech_timestamps.assert_called_with(
            mock_get_speech_timestamps.call_args[0][0],  # waveform tensor
            mock_model,
            threshold=0.75,
            sampling_rate=16000,
            min_speech_duration_ms=350,
            min_silence_duration_ms=150,
            speech_pad_ms=45,
            return_seconds=True,
        )

        # 2. Test explicit function argument overrides
        get_speech_regions(
            silent_audio_file,
            config_path=custom_config_path,
            threshold=0.9,
            min_speech_duration_ms=500,
            min_silence_duration_ms=300,
            speech_pad_ms=60,
        )
        mock_get_speech_timestamps.assert_called_with(
            mock_get_speech_timestamps.call_args[0][0],
            mock_model,
            threshold=0.9,
            sampling_rate=16000,
            min_speech_duration_ms=500,
            min_silence_duration_ms=300,
            speech_pad_ms=60,
            return_seconds=True,
        )


@pytest.mark.real_model
def test_real_nepali_speech_returns_speech_regions():
    """Verify that get_speech_regions detects speech in the real Nepali test fixture."""
    fixture_path = Path(__file__).resolve().parent / "fixtures" / "nepali_sample.wav"
    assert fixture_path.exists(), f"Real speech fixture missing: {fixture_path}"

    regions = get_speech_regions(fixture_path)
    assert len(regions) > 0
    for start, end in regions:
        assert isinstance(start, (int, float))
        assert isinstance(end, (int, float))
        assert start >= 0
        assert end > start


@pytest.mark.real_model
def test_real_japanese_speech_returns_speech_regions():
    """Verify that get_speech_regions detects speech in the real Japanese test fixture."""
    fixture_path = Path(__file__).resolve().parent / "fixtures" / "japanese_sample.wav"
    assert fixture_path.exists(), f"Real speech fixture missing: {fixture_path}"

    regions = get_speech_regions(fixture_path)
    assert len(regions) > 0, "Expected VAD to detect speech in japanese_sample.wav"
    for start, end in regions:
        assert isinstance(start, (int, float))
        assert isinstance(end, (int, float))
        assert start >= 0
        assert end > start



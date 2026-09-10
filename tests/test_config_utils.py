from pathlib import Path
import pytest
import yaml

from src.config_utils import load_stage_config


def test_missing_file_returns_defaults(tmp_path: Path):
    """Verify that a nonexistent config file path returns defaults unchanged without error."""
    nonexistent = tmp_path / "nonexistent.yaml"
    defaults = {"param_a": 10, "param_b": "test"}
    result = load_stage_config(config_path=nonexistent, section_name="vad", defaults=defaults)
    assert result == defaults


def test_valid_section_override(tmp_path: Path):
    """Verify that section-level configuration overrides defaults."""
    config_file = tmp_path / "config.yaml"
    config_data = {
        "vad": {
            "threshold": 0.8,
            "min_speech_duration_ms": 500,
        }
    }
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(config_data, f)

    defaults = {
        "threshold": 0.5,
        "min_speech_duration_ms": 250,
        "speech_pad_ms": 30,
    }
    result = load_stage_config(config_path=config_file, section_name="vad", defaults=defaults)
    assert result["threshold"] == 0.8
    assert result["min_speech_duration_ms"] == 500
    assert result["speech_pad_ms"] == 30


def test_corrupt_yaml_fallback_and_logging(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    """Verify that malformed/corrupt YAML falls back to defaults and logs a warning."""
    config_file = tmp_path / "corrupt.yaml"
    config_file.write_text("vad: [malformed yaml: {", encoding="utf-8")

    defaults = {"threshold": 0.5}
    result = load_stage_config(
        config_path=config_file,
        section_name="vad",
        defaults=defaults,
        stage_label="VAD",
    )
    assert result == defaults
    assert "Failed to parse VAD configuration" in caplog.text


def test_top_level_vs_section_level_precedence(tmp_path: Path):
    """Verify precedence: section-level > top-level > defaults."""
    config_file = tmp_path / "config.yaml"
    config_data = {
        "language": "ja",
        "num_speakers": 2,
        "asr": {
            "model_size": "medium",
            "language": "ne",  # overrides top-level "ja"
        },
    }
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(config_data, f)

    defaults = {
        "model_size": "large-v3",
        "compute_type": "int8",
        "language": "en",
    }

    # 1. Section overrides top-level and defaults
    result_asr = load_stage_config(
        config_path=config_file,
        section_name="asr",
        defaults=defaults,
        top_level_keys=["language"],
    )
    assert result_asr["model_size"] == "medium"
    assert result_asr["compute_type"] == "int8"
    assert result_asr["language"] == "ne"

    # 2. Top-level overrides default when section doesn't define it
    diarize_defaults = {
        "num_speakers": None,
        "device": "cpu",
    }
    result_diarize = load_stage_config(
        config_path=config_file,
        section_name="diarize",
        defaults=diarize_defaults,
        top_level_keys=["num_speakers"],
    )
    assert result_diarize["num_speakers"] == 2
    assert result_diarize["device"] == "cpu"


def test_invalid_yaml_structure_fallback(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    """Verify that a YAML file containing a list instead of a mapping falls back to defaults."""
    config_file = tmp_path / "invalid_structure.yaml"
    config_file.write_text("- item1\n- item2\n", encoding="utf-8")

    defaults = {"key": "val"}
    result = load_stage_config(config_path=config_file, section_name="vad", defaults=defaults)
    assert result == defaults
    assert "Invalid configuration structure" in caplog.text

"""
Test suite for the Command-Line Interface (src/cli.py).

Covers:
  - Argument parsing and parameter passthrough to run_pipeline().
  - Flag toggle for --include-speaker / --no-speaker via BooleanOptionalAction.
  - Fail-fast validation on missing required positional arguments.
  - Distinct speaker count computation across multi-turn speaker transcripts.
  - Graceful exit and clean error messages on FileNotFoundError and Hugging Face gating errors.
  - Transparent propagation of unexpected exceptions.
  - Real end-to-end CLI execution on the Nepali audio fixture.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
import pytest

from src.cli import build_parser, main
from src.config_utils import DEFAULT_CONFIG_PATH, PROJECT_ROOT

FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures"
NEPALI_AUDIO_FIXTURE = FIXTURES_DIR / "nepali_sample.wav"


def test_build_parser_defaults():
    """Verify default parser settings."""
    parser = build_parser()
    args = parser.parse_args(["input_audio.wav"])

    assert args.input_file == "input_audio.wav"
    assert args.output_dir is None
    assert args.config == str(DEFAULT_CONFIG_PATH)
    assert args.hardware_tier is None
    assert args.language is None
    assert args.include_speaker is True


def test_build_parser_options_and_boolean_flag():
    """Verify custom CLI options, language override, and --no-speaker toggle."""
    parser = build_parser()
    args = parser.parse_args(
        [
            "video.mp4",
            "-o",
            "custom_output",
            "-c",
            "custom_config.yaml",
            "-t",
            "local",
            "-l",
            "ja",
            "--no-speaker",
        ]
    )

    assert args.input_file == "video.mp4"
    assert args.output_dir == "custom_output"
    assert args.config == "custom_config.yaml"
    assert args.hardware_tier == "local"
    assert args.language == "ja"
    assert args.include_speaker is False


def test_missing_required_input_argument():
    """Verify missing required positional input fails argument parsing."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


@patch("src.cli.run_pipeline")
def test_cli_dispatches_to_run_pipeline(mock_run_pipeline, tmp_path):
    """Verify main() parses arguments and dispatches to run_pipeline with correct types."""
    dummy_input = tmp_path / "audio.wav"
    dummy_out = tmp_path / "out"
    dummy_config = tmp_path / "cfg.yaml"

    mock_run_pipeline.return_value = {
        "cue_count": 5,
        "speaker_turns": [{"speaker_id": "SPEAKER_00"}],
        "srt_path": str(dummy_out / "audio.srt"),
        "vtt_path": str(dummy_out / "audio.vtt"),
    }

    exit_code = main(
        [
            str(dummy_input),
            "-o",
            str(dummy_out),
            "-c",
            str(dummy_config),
            "-t",
            "colab",
            "--include-speaker",
        ]
    )

    assert exit_code == 0
    mock_run_pipeline.assert_called_once_with(
        input_path=dummy_input,
        config_path=dummy_config,
        output_dir=dummy_out,
        hardware_tier="colab",
        language=None,
        include_speaker=True,
    )


@patch("src.cli.run_pipeline")
def test_cli_dispatches_with_language_override(mock_run_pipeline, tmp_path):
    """Verify main() forwards the --language / -l CLI flag to run_pipeline()."""
    dummy_input = tmp_path / "audio.wav"
    mock_run_pipeline.return_value = {
        "cue_count": 2,
        "speaker_turns": [],
        "srt_path": "output/audio.srt",
        "vtt_path": "output/audio.vtt",
    }

    exit_code = main([str(dummy_input), "-l", "ja"])
    assert exit_code == 0
    mock_run_pipeline.assert_called_once_with(
        input_path=dummy_input,
        config_path=DEFAULT_CONFIG_PATH,
        output_dir=None,
        hardware_tier=None,
        language="ja",
        include_speaker=True,
    )


@patch("src.cli.run_pipeline")
def test_distinct_speaker_count_computation(mock_run_pipeline, capsys):
    """
    Verify distinct speaker count is computed uniquely from speaker turns
    and not overcounted by multiple turns from the same speaker.
    """
    mock_run_pipeline.return_value = {
        "cue_count": 4,
        "speaker_turns": [
            {"speaker_id": "SPEAKER_00", "start": 0.0, "end": 1.0},
            {"speaker_id": "SPEAKER_00", "start": 1.2, "end": 2.0},
            {"speaker_id": "SPEAKER_01", "start": 2.2, "end": 3.0},
            {"speaker_id": "SPEAKER_00", "start": 3.1, "end": 4.0},
        ],
        "srt_path": "output/test.srt",
        "vtt_path": "output/test.vtt",
    }

    exit_code = main(["test.wav"])
    assert exit_code == 0

    captured = capsys.readouterr()
    assert "Total cues generated: 4" in captured.out
    # 4 turns, but only 2 unique speakers (SPEAKER_00 and SPEAKER_01)
    assert "Distinct speakers detected: 2" in captured.out


@patch("src.cli.run_pipeline")
def test_filenotfound_error_clean_exit(mock_run_pipeline, capsys):
    """Verify FileNotFoundError is caught cleanly and exits with status 1 without a raw traceback."""
    mock_run_pipeline.side_effect = FileNotFoundError("Input file not found: /path/to/missing.wav")

    exit_code = main(["/path/to/missing.wav"])
    assert exit_code == 1

    captured = capsys.readouterr()
    assert "Error: Input file not found: /path/to/missing.wav" in captured.err
    assert "Traceback" not in captured.err


@patch("src.cli.run_pipeline")
def test_hf_gated_repo_error_clean_exit(mock_run_pipeline, capsys):
    """Verify Hugging Face GatedRepoError / HfHubHTTPError exits with status 1 and clear instructions."""
    dummy_response = MagicMock()
    dummy_response.status_code = 403
    mock_run_pipeline.side_effect = GatedRepoError(
        "Gated model access required for pyannote/speaker-diarization-3.1",
        response=dummy_response,
    )

    exit_code = main(["sample.wav"])
    assert exit_code == 1

    captured = capsys.readouterr()
    assert "Hugging Face Authentication / Gated Access Error" in captured.err
    assert "hf auth login" in captured.err
    assert "Traceback" not in captured.err


@patch("src.cli.run_pipeline")
def test_unexpected_exception_propagates(mock_run_pipeline):
    """Verify genuinely unexpected exceptions propagate to the caller with traceback."""
    mock_run_pipeline.side_effect = RuntimeError("Unexpected internal CUDA error")

    with pytest.raises(RuntimeError, match="Unexpected internal CUDA error"):
        main(["sample.wav"])


@pytest.mark.skipif(
    not NEPALI_AUDIO_FIXTURE.exists(),
    reason="Real Nepali audio fixture required for full integration test",
)
def test_cli_real_end_to_end_integration(tmp_path, capsys):
    """
    Real CLI integration test: executes main() on tests/fixtures/nepali_sample.wav,
    verifying file creation and standard output reporting.
    """
    out_dir = tmp_path / "cli_subtitles_output"

    exit_code = main([str(NEPALI_AUDIO_FIXTURE), "-o", str(out_dir)])
    assert exit_code == 0

    captured = capsys.readouterr()
    print("\n" + "=" * 60)
    print("CLI REAL INTEGRATION TEST STDOUT:")
    print("=" * 60)
    print(captured.out)
    print("=" * 60 + "\n")

    assert "Subtitle Generator — Starting processing" in captured.out
    assert "Subtitle Generation Complete:" in captured.out
    assert "Total cues generated:" in captured.out
    assert "Distinct speakers detected:" in captured.out

    srt_file = out_dir / f"{NEPALI_AUDIO_FIXTURE.stem}.srt"
    vtt_file = out_dir / f"{NEPALI_AUDIO_FIXTURE.stem}.vtt"

    assert srt_file.exists(), f"Expected SRT file {srt_file} to exist"
    assert vtt_file.exists(), f"Expected VTT file {vtt_file} to exist"
    assert srt_file.stat().st_size > 0
    assert vtt_file.stat().st_size > 0

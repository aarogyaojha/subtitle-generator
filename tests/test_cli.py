import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from huggingface_hub.errors import GatedRepoError

from src.cli import build_parser, main
from src.config_utils import DEFAULT_CONFIG_PATH


NEPALI_AUDIO_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "nepali_sample.wav"


def test_build_parser_defaults():
    """Verify CLI argument parser has correct defaults for all optional arguments."""
    parser = build_parser()
    args = parser.parse_args(["audio.wav"])

    assert args.input_file == "audio.wav"
    assert args.output_dir is None
    assert args.config == str(DEFAULT_CONFIG_PATH)
    assert args.hardware_tier is None
    assert args.language is None
    assert args.include_speaker is True


def test_build_parser_options_and_boolean_flag():
    """Verify CLI parser correctly parses flags and handles BooleanOptionalAction."""
    parser = build_parser()

    # Test with speaker flag enabled
    args_with_speaker = parser.parse_args([
        "video.mp4",
        "-o", "custom_out",
        "-c", "custom_config.yaml",
        "-t", "local",
        "-l", "ne",
        "--include-speaker",
    ])
    assert args_with_speaker.input_file == "video.mp4"
    assert args_with_speaker.output_dir == "custom_out"
    assert args_with_speaker.config == "custom_config.yaml"
    assert args_with_speaker.hardware_tier == "local"
    assert args_with_speaker.language == "ne"
    assert args_with_speaker.include_speaker is True

    # Test with --no-speaker flag
    args_no_speaker = parser.parse_args(["audio.wav", "--no-speaker"])
    assert args_no_speaker.include_speaker is False


def test_missing_required_input_argument(capsys):
    """Verify CLI exits with code 2 and prints usage when input_file argument is missing."""
    with pytest.raises(SystemExit) as exc_info:
        main([])
    assert exc_info.value.code == 2


@patch("src.cli.run_pipeline")
def test_cli_dispatches_to_run_pipeline(mock_run_pipeline, tmp_path, capsys):
    """Verify main() dispatches parsed arguments to run_pipeline and prints summary."""
    mock_run_pipeline.return_value = {
        "hardware_tier": "local",
        "cue_count": 3,
        "speaker_turns": [
            {"speaker_id": "SPEAKER_00"},
            {"speaker_id": "SPEAKER_01"},
            {"speaker_id": "SPEAKER_00"},
        ],
        "srt_path": str(tmp_path / "audio.srt"),
        "vtt_path": str(tmp_path / "audio.vtt"),
    }

    exit_code = main([
        "audio.wav",
        "-o", str(tmp_path),
        "-c", "config.yaml",
        "-t", "local",
        "--no-speaker",
    ])

    assert exit_code == 0
    mock_run_pipeline.assert_called_once_with(
        input_path=Path("audio.wav"),
        config_path=Path("config.yaml"),
        output_dir=tmp_path,
        hardware_tier="local",
        language=None,
        include_speaker=False,
    )

    captured = capsys.readouterr()
    assert "Subtitle Generator — Starting processing for: audio.wav" in captured.out
    assert "Hardware tier override: local" in captured.out
    assert "Subtitle Generation Complete:" in captured.out
    assert "Total cues generated: 3" in captured.out
    assert "Distinct speakers detected: 2" in captured.out
    assert f"SRT output: {tmp_path / 'audio.srt'}" in captured.out
    assert f"VTT output: {tmp_path / 'audio.vtt'}" in captured.out


@patch("src.cli.run_pipeline")
def test_cli_dispatches_with_language_override(mock_run_pipeline, tmp_path):
    """Verify main() dispatches language override to run_pipeline."""
    mock_run_pipeline.return_value = {
        "cue_count": 1,
        "speaker_turns": [],
        "srt_path": str(tmp_path / "audio.srt"),
        "vtt_path": str(tmp_path / "audio.vtt"),
    }

    exit_code = main(["audio.wav", "-l", "ja"])

    assert exit_code == 0
    mock_run_pipeline.assert_called_once_with(
        input_path=Path("audio.wav"),
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
            {"speaker_id": "SPEAKER_01", "start": 2.5, "end": 3.5},
            {"speaker_id": "SPEAKER_00", "start": 4.0, "end": 5.0},
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
def test_hardware_tier_always_printed_in_summary(mock_run_pipeline, capsys):
    """Verify resolved hardware tier from run_pipeline is always printed in the summary even without -t flag."""
    mock_run_pipeline.return_value = {
        "hardware_tier": "local",
        "cue_count": 1,
        "speaker_turns": [],
        "srt_path": "output/test.srt",
        "vtt_path": "output/test.vtt",
    }

    exit_code = main(["test.wav"])
    assert exit_code == 0

    captured = capsys.readouterr()
    assert "Hardware tier: local" in captured.out


@patch("src.cli.run_pipeline")
def test_filenotfound_error_clean_exit(mock_run_pipeline, capsys):
    """Verify FileNotFoundError exits cleanly with status 1 and error message to stderr."""
    mock_run_pipeline.side_effect = FileNotFoundError("Input file not found: non_existent.wav")

    exit_code = main(["non_existent.wav"])
    assert exit_code == 1

    captured = capsys.readouterr()
    assert "Error: Input file not found: non_existent.wav" in captured.err
    assert "Traceback" not in captured.err


@patch("src.cli.run_pipeline")
def test_runtime_error_clean_exit(mock_run_pipeline, capsys):
    """Verify RuntimeError (e.g. ffmpeg extraction failure) exits cleanly with status 1 and no traceback."""
    mock_run_pipeline.side_effect = RuntimeError("ffmpeg audio extraction failed with error:\nInvalid data found")

    exit_code = main(["corrupted_video.mp4"])
    assert exit_code == 1

    captured = capsys.readouterr()
    assert "Error: ffmpeg audio extraction failed with error:\nInvalid data found" in captured.err
    assert "Traceback" not in captured.err


@patch("src.cli.run_pipeline")
def test_hf_gated_repo_error_clean_exit(mock_run_pipeline, capsys):
    """Verify Hugging Face GatedRepoError / HfHubHTTPError exits with status 1 and clear instructions."""
    mock_run_pipeline.side_effect = GatedRepoError("Access to model pyannote/speaker-diarization-3.1 is restricted.")

    exit_code = main(["audio.wav"])
    assert exit_code == 1

    captured = capsys.readouterr()
    assert "Hugging Face Authentication / Gated Access Error:" in captured.err
    assert "Please ensure you have accepted the model conditions on Hugging Face" in captured.err
    assert "Traceback" not in captured.err


@patch("src.cli.run_pipeline")
def test_unexpected_exception_propagates(mock_run_pipeline):
    """Verify genuinely unexpected exceptions (e.g. TypeError, ValueError) propagate with traceback."""
    mock_run_pipeline.side_effect = TypeError("Unexpected internal type error")

    with pytest.raises(TypeError, match="Unexpected internal type error"):
        main(["sample.wav"])


@pytest.mark.skipif(
    not NEPALI_AUDIO_FIXTURE.exists(),
    reason="Real Nepali audio fixture required for CLI end-to-end integration test",
)
def test_cli_real_end_to_end_integration(tmp_path, capsys):
    """
    Real end-to-end CLI integration test using tests/fixtures/nepali_sample.wav.

    Executes main() on real audio, verifies return code 0, verifies generated .srt/.vtt files,
    and checks CLI stdout summary formatting.
    """
    out_dir = tmp_path / "cli_subtitles_output"

    exit_code = main([
        str(NEPALI_AUDIO_FIXTURE),
        "-o", str(out_dir),
        "-t", "local",
        "-l", "ne",
        "--include-speaker",
    ])

    assert exit_code == 0

    captured = capsys.readouterr()
    assert "Subtitle Generator — Starting processing for:" in captured.out
    assert "Subtitle Generation Complete:" in captured.out
    assert "Total cues generated:" in captured.out
    assert "Distinct speakers detected:" in captured.out

    srt_file = out_dir / f"{NEPALI_AUDIO_FIXTURE.stem}.srt"
    vtt_file = out_dir / f"{NEPALI_AUDIO_FIXTURE.stem}.vtt"

    assert srt_file.exists(), f"Expected SRT file {srt_file} to exist"
    assert vtt_file.exists(), f"Expected VTT file {vtt_file} to exist"
    assert srt_file.stat().st_size > 0
    assert vtt_file.stat().st_size > 0

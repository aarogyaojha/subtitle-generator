"""
Test suite for pipeline orchestrator (src/pipeline.py).

Covers:
  - Fail-fast pre-flight check when Hugging Face credentials/pipeline fail.
  - Correct execution order and parameter routing across stages.
  - Hardware tier mapping ('colab' vs 'local') and ASR overrides.
  - Video vs. audio input detection and ffmpeg audio extraction/cleanup.
  - End-to-end integration test on real Nepali audio fixture.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from src.config_utils import DEFAULT_CONFIG_PATH, PROJECT_ROOT
from src.pipeline import (
    extract_audio_from_video,
    get_hardware_tier_overrides,
    is_video_file,
    run_pipeline,
)

FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures"
NEPALI_AUDIO_FIXTURE = FIXTURES_DIR / "nepali_sample.wav"
JAPANESE_AUDIO_FIXTURE = FIXTURES_DIR / "japanese_sample.wav"



def test_is_video_file():
    """Verify video file extension detection."""
    assert is_video_file("sample.mp4") is True
    assert is_video_file("video.MKV") is True
    assert is_video_file("clip.webm") is True
    assert is_video_file("audio.wav") is False
    assert is_video_file("recording.mp3") is False
    assert is_video_file("track.flac") is False


def test_hardware_tier_mapping():
    """Verify hardware tier mapping produces correct ASR overrides."""
    colab_overrides = get_hardware_tier_overrides("colab")
    assert colab_overrides == {
        "model_size_or_path": "large-v3",
        "compute_type": "int8",
    }

    local_overrides = get_hardware_tier_overrides("local")
    assert local_overrides == {
        "model_size_or_path": "medium",
        "compute_type": "int8",
    }

    # Unknown tier falls back to colab with a warning
    fallback_overrides = get_hardware_tier_overrides("unknown_tier")
    assert fallback_overrides == {
        "model_size_or_path": "large-v3",
        "compute_type": "int8",
    }


def test_nonexistent_input_raises_filenotfound(tmp_path):
    """Verify missing input path raises FileNotFoundError immediately."""
    fake_path = tmp_path / "nonexistent.wav"
    with pytest.raises(FileNotFoundError, match="Input file not found"):
        run_pipeline(input_path=fake_path)


@patch("src.pipeline.diarize.load_diarization_pipeline")
@patch("src.pipeline.vad.get_speech_regions")
@patch("src.pipeline.asr.transcribe")
def test_fail_fast_preflight_check_blocks_vad_and_asr(
    mock_transcribe,
    mock_get_speech_regions,
    mock_load_diarization,
    tmp_path,
):
    """
    Verify that an HF authentication / license gating failure in the pre-flight check
    raises immediately before any time is spent running VAD or ASR.
    """
    dummy_audio = tmp_path / "test.wav"
    dummy_audio.write_bytes(b"RIFF dummy audio content")

    mock_load_diarization.side_effect = RuntimeError(
        "Gated model pyannote/speaker-diarization-3.1 access denied. Please accept user conditions."
    )

    with pytest.raises(RuntimeError, match="Gated model.*access denied"):
        run_pipeline(input_path=dummy_audio)

    # Ensure preflight check was called
    mock_load_diarization.assert_called_once()

    # Crucial assertion: VAD and ASR must never have been called
    mock_get_speech_regions.assert_not_called()
    mock_transcribe.assert_not_called()


@patch("src.pipeline.fmt.write_vtt")
@patch("src.pipeline.fmt.write_srt")
@patch("src.pipeline.fmt.create_cues")
@patch("src.pipeline.diarize.merge_with_transcript")
@patch("src.pipeline.diarize.diarize")
@patch("src.pipeline.asr.transcribe")
@patch("src.pipeline.vad.get_speech_regions")
@patch("src.pipeline.diarize.load_diarization_pipeline")
def test_fail_fast_preflight_uses_configured_model_name_and_device(
    mock_load_diarization,
    mock_vad,
    mock_asr,
    mock_diarize,
    mock_merge,
    mock_create_cues,
    mock_write_srt,
    mock_write_vtt,
    tmp_path,
):
    """
    Verify that the pre-flight load_diarization_pipeline call receives the exact model_name
    and device resolved from config.yaml, rather than always using hardcoded defaults.
    """
    dummy_audio = tmp_path / "test.wav"
    dummy_audio.write_bytes(b"RIFF dummy audio content")

    custom_config_path = tmp_path / "custom_config.yaml"
    custom_config_path.write_text(
        """
diarize:
  model_name: "custom-org/custom-diarization-model"
  device: "cpu"
""",
        encoding="utf-8",
    )

    mock_vad.return_value = []
    mock_asr.return_value = []
    mock_diarize.return_value = []
    mock_merge.return_value = []
    mock_create_cues.return_value = []

    run_pipeline(
        input_path=dummy_audio,
        config_path=custom_config_path,
        output_dir=tmp_path / "out",
    )

    # Pre-flight call must receive configured model and device
    mock_load_diarization.assert_called_once_with(
        model_name="custom-org/custom-diarization-model",
        device="cpu",
    )


@patch("src.pipeline.fmt.write_vtt")
@patch("src.pipeline.fmt.write_srt")
@patch("src.pipeline.fmt.create_cues")
@patch("src.pipeline.diarize.merge_with_transcript")
@patch("src.pipeline.diarize.diarize")
@patch("src.pipeline.asr.transcribe")
@patch("src.pipeline.vad.get_speech_regions")
@patch("src.pipeline.diarize.load_diarization_pipeline")
def test_pipeline_execution_order_and_audio_path_consistency(
    mock_load_diarization,
    mock_vad,
    mock_asr,
    mock_diarize,
    mock_merge,
    mock_create_cues,
    mock_write_srt,
    mock_write_vtt,
    tmp_path,
):
    """
    Verify full mock pipeline execution order, hardware tier parameter passthrough,
    and guarantee that the exact same resolved audio path is provided to VAD, ASR, and Diarization.
    """
    dummy_audio = tmp_path / "recording.wav"
    dummy_audio.write_bytes(b"RIFF dummy audio content")
    out_dir = tmp_path / "output_subtitles"

    # Setup mock stage returns
    mock_vad.return_value = [(0.0, 2.0)]
    mock_asr.return_value = [
        {"start": 0.1, "end": 1.9, "text": "नमस्ते", "confidence": 0.95, "words": []}
    ]
    mock_diarize.return_value = [
        {"start": 0.0, "end": 2.0, "speaker_id": "SPEAKER_00"}
    ]
    mock_merge.return_value = [
        {
            "start": 0.1,
            "end": 1.9,
            "text": "नमस्ते",
            "confidence": 0.95,
            "speaker_id": "SPEAKER_00",
            "words": [],
        }
    ]
    mock_create_cues.return_value = [
        {
            "index": 1,
            "start": 0.1,
            "end": 1.9,
            "text": "नमस्ते",
            "lines": ["नमस्ते"],
            "speaker_id": "SPEAKER_00",
        }
    ]

    # Run with local hardware tier
    summary = run_pipeline(
        input_path=dummy_audio,
        output_dir=out_dir,
        hardware_tier="local",
        include_speaker=True,
    )

    # 1. Check pre-flight called
    mock_load_diarization.assert_called_once()

    # 2. Check VAD called with resolved audio path
    mock_vad.assert_called_once_with(
        audio_path=dummy_audio.resolve(),
        config_path=DEFAULT_CONFIG_PATH,
    )

    # 3. Check ASR called with identical audio path and local hardware overrides
    mock_asr.assert_called_once_with(
        audio_path=dummy_audio.resolve(),
        speech_regions=[(0.0, 2.0)],
        language=None,
        config_path=DEFAULT_CONFIG_PATH,
        model_size_or_path="medium",
        compute_type="int8",
    )

    # 4. Check Diarization called with identical audio path
    mock_diarize.assert_called_once_with(
        audio_path=dummy_audio.resolve(),
        config_path=DEFAULT_CONFIG_PATH,
    )

    # 5. Check Merge and Cue formatting called
    mock_merge.assert_called_once_with(
        transcript_segments=mock_asr.return_value,
        diarization_turns=mock_diarize.return_value,
    )
    mock_create_cues.assert_called_once_with(
        transcript_segments=mock_merge.return_value,
        config_path=DEFAULT_CONFIG_PATH,
        language=None,
    )

    # 6. Check SRT and VTT writers called
    mock_write_srt.assert_called_once_with(
        cues=mock_create_cues.return_value,
        output_path=out_dir / "recording.srt",
        include_speaker=True,
    )
    mock_write_vtt.assert_called_once_with(
        cues=mock_create_cues.return_value,
        output_path=out_dir / "recording.vtt",
        include_speaker=True,
    )

    # 7. Check summary dictionary structure
    assert summary["input_path"] == str(dummy_audio.resolve())
    assert summary["audio_path"] == str(dummy_audio.resolve())
    assert summary["srt_path"] == str(out_dir / "recording.srt")
    assert summary["vtt_path"] == str(out_dir / "recording.vtt")
    assert summary["cue_count"] == 1
    assert summary["hardware_tier"] == "local"


@patch("src.pipeline.subprocess.run")
@patch("src.pipeline.shutil.which")
def test_extract_audio_from_video_command_args(mock_which, mock_run, tmp_path):
    """Verify ffmpeg invocation arguments during video audio extraction."""
    mock_which.return_value = "/usr/bin/ffmpeg"
    mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

    dummy_video = tmp_path / "movie.mp4"
    dummy_video.write_bytes(b"dummy video data")
    out_wav = tmp_path / "extracted.wav"

    extracted = extract_audio_from_video(dummy_video, output_audio_path=out_wav)
    assert extracted == out_wav.resolve()

    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    assert cmd[0] == "ffmpeg"
    assert cmd[1] == "-y"
    assert cmd[2] == "-i"
    assert cmd[3] == str(dummy_video.resolve())
    assert "-vn" in cmd
    assert "-acodec" in cmd and "pcm_s16le" in cmd
    assert "-ar" in cmd and "16000" in cmd
    assert "-ac" in cmd and "1" in cmd


@patch("src.pipeline.extract_audio_from_video")
@patch("src.pipeline.fmt.write_vtt")
@patch("src.pipeline.fmt.write_srt")
@patch("src.pipeline.fmt.create_cues")
@patch("src.pipeline.diarize.merge_with_transcript")
@patch("src.pipeline.diarize.diarize")
@patch("src.pipeline.asr.transcribe")
@patch("src.pipeline.vad.get_speech_regions")
@patch("src.pipeline.diarize.load_diarization_pipeline")
def test_video_input_extracts_audio_and_cleans_up_temp_file(
    mock_load_diarization,
    mock_vad,
    mock_asr,
    mock_diarize,
    mock_merge,
    mock_create_cues,
    mock_write_srt,
    mock_write_vtt,
    mock_extract_audio,
    tmp_path,
):
    """
    Verify that when input is a video (.mp4):
    1. Audio is extracted to a temp WAV file.
    2. The exact same extracted WAV path is supplied to VAD, ASR, and Diarization.
    3. The temp WAV file is cleaned up after execution.
    """
    dummy_video = tmp_path / "clip.mp4"
    dummy_video.write_bytes(b"dummy video data")

    # Create a real temp wav file that extract_audio returns so we can verify cleanup
    temp_wav = tmp_path / "temp_extracted.wav"
    temp_wav.write_bytes(b"RIFF extracted wav")
    mock_extract_audio.return_value = temp_wav

    mock_vad.return_value = []
    mock_asr.return_value = []
    mock_diarize.return_value = []
    mock_merge.return_value = []
    mock_create_cues.return_value = []

    summary = run_pipeline(input_path=dummy_video, output_dir=tmp_path / "out")

    # Check extraction was triggered
    mock_extract_audio.assert_called_once_with(dummy_video.resolve())

    # Check all stages received the extracted temp wav, NOT the video path
    mock_vad.assert_called_once_with(audio_path=temp_wav, config_path=DEFAULT_CONFIG_PATH)
    mock_asr.assert_called_once()
    assert mock_asr.call_args[1]["audio_path"] == temp_wav
    mock_diarize.assert_called_once()
    assert mock_diarize.call_args[1]["audio_path"] == temp_wav

    # Check that temp_wav was cleaned up
    assert not temp_wav.exists()
    assert summary["input_path"] == str(dummy_video.resolve())


@patch("src.pipeline.fmt.write_vtt")
@patch("src.pipeline.fmt.write_srt")
@patch("src.pipeline.fmt.create_cues")
@patch("src.pipeline.diarize.merge_with_transcript")
@patch("src.pipeline.diarize.diarize")
@patch("src.pipeline.asr.transcribe")
@patch("src.pipeline.vad.get_speech_regions")
@patch("src.pipeline.diarize.load_diarization_pipeline")
def test_pipeline_language_override_passthrough(
    mock_load_diarization,
    mock_vad,
    mock_asr,
    mock_diarize,
    mock_merge,
    mock_create_cues,
    mock_write_srt,
    mock_write_vtt,
    tmp_path,
):
    """Verify that an explicit language parameter to run_pipeline() is forwarded to both ASR and format stages."""
    dummy_audio = tmp_path / "japanese.wav"
    dummy_audio.write_bytes(b"RIFF dummy audio content")

    mock_vad.return_value = [(0.0, 3.5)]
    mock_asr.return_value = [{"start": 0.0, "end": 3.5, "text": "テスト"}]
    mock_diarize.return_value = []
    mock_merge.return_value = [{"start": 0.0, "end": 3.5, "text": "テスト", "speaker_id": None}]
    mock_create_cues.return_value = []

    run_pipeline(
        input_path=dummy_audio,
        output_dir=tmp_path / "out",
        language="ja",
    )

    mock_asr.assert_called_once_with(
        audio_path=dummy_audio.resolve(),
        speech_regions=[(0.0, 3.5)],
        language="ja",
        config_path=DEFAULT_CONFIG_PATH,
        model_size_or_path="large-v3",
        compute_type="int8",
    )
    mock_create_cues.assert_called_once_with(
        transcript_segments=mock_merge.return_value,
        config_path=DEFAULT_CONFIG_PATH,
        language="ja",
    )


@patch("src.pipeline.fmt.write_vtt")
@patch("src.pipeline.fmt.write_srt")
@patch("src.pipeline.fmt.create_cues")
@patch("src.pipeline.diarize.merge_with_transcript")
@patch("src.pipeline.diarize.diarize")
@patch("src.pipeline.asr.transcribe")
@patch("src.pipeline.vad.get_speech_regions")
@patch("src.pipeline.diarize.load_diarization_pipeline")
def test_pipeline_default_language_fallback_passthrough(
    mock_load_diarization,
    mock_vad,
    mock_asr,
    mock_diarize,
    mock_merge,
    mock_create_cues,
    mock_write_srt,
    mock_write_vtt,
    tmp_path,
):
    """Verify that omitting language forwards language=None, allowing ASR and format to fall back to config.yaml."""
    dummy_audio = tmp_path / "default.wav"
    dummy_audio.write_bytes(b"RIFF dummy audio content")

    mock_vad.return_value = []
    mock_asr.return_value = []
    mock_diarize.return_value = []
    mock_merge.return_value = []
    mock_create_cues.return_value = []

    run_pipeline(
        input_path=dummy_audio,
        output_dir=tmp_path / "out",
    )

    mock_asr.assert_called_once_with(
        audio_path=dummy_audio.resolve(),
        speech_regions=[],
        language=None,
        config_path=DEFAULT_CONFIG_PATH,
        model_size_or_path="large-v3",
        compute_type="int8",
    )
    mock_create_cues.assert_called_once_with(
        transcript_segments=[],
        config_path=DEFAULT_CONFIG_PATH,
        language=None,
    )


@pytest.mark.skipif(
    not NEPALI_AUDIO_FIXTURE.exists(),
    reason="Real Nepali audio fixture required for full integration test",
)
def test_real_full_pipeline_nepali_integration(tmp_path):
    """
    Real end-to-end integration test: runs the complete pipeline on tests/fixtures/nepali_sample.wav
    using real model weights (no mocks), generates .srt and .vtt files, and verifies output format.
    """
    out_dir = tmp_path / "subtitles_output"

    summary = run_pipeline(
        input_path=NEPALI_AUDIO_FIXTURE,
        output_dir=out_dir,
        include_speaker=True,
    )

    srt_file = Path(summary["srt_path"])
    vtt_file = Path(summary["vtt_path"])

    assert srt_file.exists(), "SRT output file should exist"
    assert vtt_file.exists(), "VTT output file should exist"

    srt_content = srt_file.read_text(encoding="utf-8")
    vtt_content = vtt_file.read_text(encoding="utf-8")

    assert len(summary["cues"]) > 0, "Pipeline should generate at least one subtitle cue"
    assert summary["cue_count"] == len(summary["cues"])
    assert "-->" in srt_content
    assert "WEBVTT" in vtt_content

    # Print summary and output for inspection (visible with pytest -s)
    print("\n" + "=" * 60)
    print("PIPELINE END-TO-END INTEGRATION TEST RESULT")
    print("=" * 60)
    print(f"Input file:     {summary['input_path']}")
    print(f"Resolved audio: {summary['audio_path']}")
    print(f"Hardware tier:  {summary['hardware_tier']}")
    print(f"VAD regions:    {len(summary['speech_regions'])}")
    print(f"ASR segments:   {len(summary['segments'])}")
    print(f"Speaker turns:  {len(summary['speaker_turns'])}")
    print(f"Cue count:      {summary['cue_count']}")
    print(f"SRT Path:       {summary['srt_path']}")
    print(f"VTT Path:       {summary['vtt_path']}")
    print("\n--- GENERATED SRT CONTENT ---")
    print(srt_content)
    print("--- GENERATED VTT CONTENT ---")
    print(vtt_content)
    print("=" * 60 + "\n")


@pytest.mark.skipif(
    not JAPANESE_AUDIO_FIXTURE.exists(),
    reason="Real Japanese audio fixture required for full integration test",
)
def test_real_full_pipeline_japanese_integration(tmp_path):
    """
    Real end-to-end integration test: runs the complete pipeline on tests/fixtures/japanese_sample.wav
    with language="ja" using real model weights (no mocks), generates .srt and .vtt files,
    and verifies output format and Japanese subtitle formatting.
    """
    out_dir = tmp_path / "japanese_subtitles_output"

    summary = run_pipeline(
        input_path=JAPANESE_AUDIO_FIXTURE,
        output_dir=out_dir,
        language="ja",
        include_speaker=True,
    )

    srt_file = Path(summary["srt_path"])
    vtt_file = Path(summary["vtt_path"])

    assert srt_file.exists(), "SRT output file should exist"
    assert vtt_file.exists(), "VTT output file should exist"

    srt_content = srt_file.read_text(encoding="utf-8")
    vtt_content = vtt_file.read_text(encoding="utf-8")

    assert len(summary["cues"]) > 0, "Pipeline should generate at least one subtitle cue"
    assert summary["cue_count"] == len(summary["cues"])
    assert "-->" in srt_content
    assert "WEBVTT" in vtt_content

    # Verify Japanese formatting constraints
    for cue in summary["cues"]:
        for line in cue["lines"]:
            assert len(line) <= 13, f"Line exceeds 13 characters in Japanese cue: '{line}' ({len(line)} chars)"
        duration = cue["end"] - cue["start"]
        text_len = len(cue["text"])
        required_duration_by_cps = text_len / 4.0
        assert duration >= (required_duration_by_cps - 1e-4), (
            f"Cue duration {duration:.3f}s violates 4.0 CPS cap for text '{cue['text']}'"
        )

    # Print summary and output for inspection (visible with pytest -s)
    print("\n" + "=" * 60)
    print("PIPELINE END-TO-END JAPANESE INTEGRATION TEST RESULT")
    print("=" * 60)
    print(f"Input file:     {summary['input_path']}")
    print(f"Resolved audio: {summary['audio_path']}")
    print(f"Hardware tier:  {summary['hardware_tier']}")
    print(f"VAD regions:    {len(summary['speech_regions'])}")
    print(f"ASR segments:   {len(summary['segments'])}")
    print(f"Speaker turns:  {len(summary['speaker_turns'])}")
    print(f"Cue count:      {summary['cue_count']}")
    print(f"SRT Path:       {summary['srt_path']}")
    print(f"VTT Path:       {summary['vtt_path']}")
    print("\n--- GENERATED SRT CONTENT ---")
    print(srt_content)
    print("--- GENERATED VTT CONTENT ---")
    print(vtt_content)
    print("=" * 60 + "\n")


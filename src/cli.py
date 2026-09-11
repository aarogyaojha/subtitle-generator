"""
Command-line interface (CLI) for the Subtitle Generator pipeline.

Usage:
  python src/cli.py <input_file> [options]

Options:
  -o, --output-dir      Directory to save output .srt/.vtt files.
  -c, --config          Path to config.yaml configuration file.
  -t, --hardware-tier   Hardware tier override ('colab', 'local').
  -l, --language        Language code override (e.g. 'ne', 'ja').
  --include-speaker / --no-speaker
                        Toggle speaker labels in output subtitles (default: True).
"""

import argparse
from pathlib import Path
import sys
from typing import Optional

from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

from src.config_utils import DEFAULT_CONFIG_PATH
from src.pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    """
    Construct the command-line argument parser for the subtitle generator.

    Returns:
        Configured argparse.ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        prog="subgen",
        description="End-to-end multilingual AI subtitle generator pipeline.",
    )

    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input audio or video file to generate subtitles for.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save generated .srt and .vtt files (defaults to output/).",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to YAML configuration file (defaults to project config.yaml).",
    )
    parser.add_argument(
        "-t",
        "--hardware-tier",
        type=str,
        choices=["colab", "local"],
        default=None,
        help="Hardware tier override ('colab' for large-v3, 'local' for 4GB VRAM cards).",
    )
    parser.add_argument(
        "-l",
        "--language",
        type=str,
        default=None,
        help="Language code override (e.g. 'ne', 'ja'). If omitted, reads from config.yaml.",
    )
    parser.add_argument(
        "--speaker",
        "--include-speaker",
        dest="include_speaker",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include speaker identification labels in output subtitles (default: --include-speaker).",
    )

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """
    Main CLI entry point: parses arguments, invokes run_pipeline(), and reports results.

    Args:
        argv: Optional list of command-line arguments. Defaults to sys.argv[1:].

    Returns:
        Exit code: 0 on success, 1 on handled user/auth errors.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.input_file)
    config_path = Path(args.config)
    output_dir = Path(args.output_dir) if args.output_dir is not None else None

    print(f"Subtitle Generator — Starting processing for: {input_path}")
    if args.hardware_tier:
        print(f"Hardware tier override: {args.hardware_tier}")
    if args.language:
        print(f"Language override: {args.language}")

    try:
        summary = run_pipeline(
            input_path=input_path,
            config_path=config_path,
            output_dir=output_dir,
            hardware_tier=args.hardware_tier,
            language=args.language,
            include_speaker=args.include_speaker,
        )

        speaker_turns = summary.get("speaker_turns", [])
        distinct_speakers = len(set(turn["speaker_id"] for turn in speaker_turns))

        print("\nSubtitle Generation Complete:")
        print(f"  - Hardware tier: {summary.get('hardware_tier')}")
        print(f"  - Total cues generated: {summary.get('cue_count', 0)}")
        print(f"  - Distinct speakers detected: {distinct_speakers}")
        print(f"  - SRT output: {summary.get('srt_path')}")
        print(f"  - VTT output: {summary.get('vtt_path')}")
        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    except (GatedRepoError, HfHubHTTPError) as e:
        print(
            f"Hugging Face Authentication / Gated Access Error:\n{e}\n"
            "Please ensure you have accepted the model conditions on Hugging Face "
            "and run `hf auth login`.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())

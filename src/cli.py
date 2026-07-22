"""
Command-line entry point.

Usage: python cli.py <input_file> [options]

Parses arguments (input path, output path, language/hardware-tier
overrides), loads config.yaml, calls pipeline.py's run function.
Kept separate from pipeline logic so pipeline.py stays callable/
testable without going through a terminal.
"""
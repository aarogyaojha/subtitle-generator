"""
Configuration utilities for loading and merging pipeline stage settings from YAML.
"""

import logging
from pathlib import Path
from typing import Any, Optional, Union
import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_stage_config(
    config_path: Union[str, Path] = DEFAULT_CONFIG_PATH,
    section_name: Optional[str] = None,
    defaults: Optional[dict[str, Any]] = None,
    top_level_keys: Optional[list[str]] = None,
    stage_label: Optional[str] = None,
) -> dict[str, Any]:
    """
    Load configuration from a YAML file, merging top-level keys and/or a named stage section
    over the provided default dictionary.

    Precedence order:
    1. Base defaults (lowest precedence).
    2. Top-level keys specified in `top_level_keys` found in config.yaml.
    3. Section-level keys found under `section_name` (highest precedence, overrides top-level).

    If the file does not exist, returns a copy of defaults unchanged.
    If the file is corrupt or fails to parse, logs a warning and returns defaults unchanged.

    Args:
        config_path: Path to the YAML configuration file.
        section_name: Optional section key in config.yaml (e.g. "vad", "asr", "diarize").
        defaults: Base default dictionary for the stage.
        top_level_keys: Optional list of keys to read from the top-level YAML root.
        stage_label: Optional human-readable stage name for log messages (e.g. "VAD", "ASR").
            Defaults to section_name if not provided.

    Returns:
        A dictionary containing the resolved configuration values.
    """
    resolved: dict[str, Any] = dict(defaults) if defaults is not None else {}
    cfg_file = Path(config_path)

    if not cfg_file.exists():
        return resolved

    display_label = stage_label if stage_label is not None else (section_name or "stage")

    try:
        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        if not isinstance(cfg, dict):
            logger.warning(
                "Invalid configuration structure in %s. Falling back to default settings.",
                cfg_file,
            )
            return resolved

        # 1. Merge top-level keys if specified
        if top_level_keys:
            for key in top_level_keys:
                if key in cfg and cfg[key] is not None:
                    resolved[key] = cfg[key]

        # 2. Merge section-level keys if specified (overrides top-level)
        # Supports nested dot-separated paths (e.g. "format.language_overrides.ja")
        if section_name:
            section_cfg: Any = cfg
            for part in section_name.split("."):
                if isinstance(section_cfg, dict):
                    section_cfg = section_cfg.get(part, {})
                else:
                    section_cfg = {}
                    break
            if isinstance(section_cfg, dict):
                for key, val in section_cfg.items():
                    if val is not None:
                        resolved[key] = val

    except Exception as e:
        logger.warning(
            "Failed to parse %s configuration from %s: %s. Falling back to default settings.",
            display_label,
            cfg_file,
            e,
        )

    return resolved

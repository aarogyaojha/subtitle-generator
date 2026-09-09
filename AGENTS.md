# Project Standards & Agent Guidelines

## 1. Project Context

**Subtitle Generator** is an open-source AI subtitle generation pipeline engineered to solve common failure modes present in existing speech recognition tools: hallucinated or repeated text during silence, duplicate text at chunk boundaries, dropped text under background noise, and context-dependent word-sense ambiguity. The project targets multilingual subtitle generation (initially Nepali and Japanese) with configurable hardware tiers (from 4GB VRAM local GPUs to cloud/Colab instances).

The pipeline architecture consists of six sequential stages:

1. `src/vad.py` — Voice Activity Detection using Silero VAD to detect speech regions and filter out silence before ASR.
2. `src/asr.py` — Speech-to-text transcription via `faster-whisper` (Whisper large-v3 with INT8 quantization support).
3. `src/diarize.py` — Speaker diarization and alignment via `pyannote.audio`.
4. `src/format.py` — Subtitle cue segmentation, line-length, and CPS (characters per second) formatting into SRT/VTT.
5. `src/pipeline.py` — Central orchestrator executing the end-to-end pipeline with shared configuration and fail-fast validation.
6. `src/cli.py` — Command-line interface and argument parsing.

> **Implementation Note:** Each stage in this codebase is implemented one at a time upon explicit user approval. Do not attempt to implement unapproved or future stages all at once.

---

## 2. Environment Management

- **Conda Environment:** All operations and commands must strictly execute within the `subgen` conda environment:

  ```bash
  conda run -n subgen <command>
  ```

- **No External Installations:** Never install packages outside the `subgen` environment.
- **Dependency Management:** If a new dependency is required, explicitly add it to `environment.yml` and explain the rationale before installing it into `subgen`.

---

## 3. Code Style & Standards

- **PEP 8 Compliance:** Follow standard PEP 8 conventions for formatting, naming, and structure.
- **Type Annotations:** Full type hints (`typing` / built-in generics) are mandatory on all public functions, methods, and return values.
- **Google-Style Docstrings:** Use Google-style docstrings (`Args:`, `Returns:`, `Raises:`) that match existing module conventions.
- **Error Handling:** Never use bare `except Exception: pass` or silently swallow errors. Always log errors/warnings with context or re-raise exceptions appropriately.

---

## 4. Path Resolution

- **No Hardcoded Relative Paths:** Never hardcode working-directory-dependent paths like `"config.yaml"`.
- **Project Root Resolution:** Resolve paths relative to the repository root so modules execute reliably regardless of the caller's current working directory:

  ```python
  from pathlib import Path

  PROJECT_ROOT = Path(__file__).resolve().parent.parent
  DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
  ```

---

## 5. Scope Discipline

- **Strict Specification Adherence:** Implement strictly what a module's docstring and requirements specify.
- **No Speculative Surface:** Do not add speculative aliases, unused utility functions, extra public APIs, or unrequested features "just in case."

---

## 6. Testing & Fixtures

- **Test Suite Location:** Every pipeline stage must have a corresponding pytest file under `tests/` (e.g., `tests/test_vad.py`).
- **Synthetic Audio for Edge Cases:** Use deterministic, numpy-generated synthetic audio (e.g., pure silence, empty arrays, artificial waveforms) to test deterministic boundary conditions and error handling.
- **Real Audio Fixtures:** Stages requiring real speech accuracy (VAD, ASR, diarization) require at least one real audio fixture located in `tests/fixtures/`. If no real fixture is present in the repository, treat this as a flagged gap rather than attempting to fake speech recognition using synthetic tones.
- **Running Tests:**

  ```bash
  conda run -n subgen pytest tests/ -v
  ```

---

## 7. Logging & Output

- **Logging Module:** Use Python's standard `logging` module (`logger = logging.getLogger(__name__)`) for internal diagnostics, warnings, and error messages across all library modules.
- **No Bare `print()` in Library Code:** Do not use `print()` in library code. `src/cli.py` user-facing output and standalone `if __name__ == "__main__":` smoke tests are the only allowed exceptions.

---

## 8. Commit Conventions

- Follow the conventional commit format matching the repository history:
  - `feat:` for new features or stage implementations
  - `fix:` for bug fixes and corrections
  - `test:` for adding or updating tests
  - `docs:` for documentation updates
  - `chore:` for maintenance, environment, or configuration tasks

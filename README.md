# Subtitle Generator

[![GitHub Repository](https://img.shields.io/badge/GitHub-aarogyaojha%2Fsubtitle--generator-blue?logo=github)](https://github.com/aarogyaojha/subtitle-generator)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An open-source AI subtitle generation pipeline engineered to solve common failure modes present in existing speech recognition tools: hallucinated or repeated text during silence, duplicate text at chunk boundaries, dropped text under background noise, and context-dependent word-sense ambiguity.

**Repository URL:** [https://github.com/aarogyaojha/subtitle-generator](https://github.com/aarogyaojha/subtitle-generator)

---

## Key Features & Targeted Problems

Existing open-source ASR implementations frequently suffer from specific edge-case failures. Subtitle Generator targets these directly:

| Failure Mode / Feature | Pipeline Remedy | Status |
| --- | --- | --- |
| **Silence Hallucination / Repetition** | Silero VAD pre-filtering before ASR | Partial (drastically reduced) |
| **Chunk-Boundary Text Duplication** | Dynamic VAD-based audio batching | Solved |
| **Reading Speed & Line Overflow** | Language-specific deterministic CPS & line-length formatting (Nepali, Japanese, English) | Solved |
| **Multilingual Transcription** | `faster-whisper` large-v3 / medium with CJK character handling | Solved (Nepali & Japanese validated) |
| **English Translation Output** | Dual native + English subtitle export (`task="translate"`, `.srt`/`.vtt` + `.en.srt`/`.en.vtt`) | Solved |
| **Speaker Diarization Error** | Integrated `pyannote.audio` pipeline | Partial (tested on 3-speaker fixture: 2 distinct clusters detected, 2 speakers merged) |
| **Context-Dependent Word Sense** | Confidence-gated context correction pass | Planned / Future |
| **Noise-Induced Dropped Text** | Tunable VAD thresholds & hardware tier sensitivity | Under active research |

---

## Known Limitations & Research Backlog

### Speaker Diarization Limitation
- Passing `num_speakers` explicitly to `diarize.diarize()` (already a supported parameter) is the most reliable fix for pyannote's speaker-merging errors — but this only works when the speaker count is known ahead of time, as it was for our controlled test fixture.
- For the project's actual target use case — arbitrary movies, videos, podcasts — the speaker count is **not** known in advance, so this mitigation does not apply. pyannote must estimate the count itself in that case, which is a harder problem and where the speaker-merging error found during testing is most likely to occur in real usage.
- No clean fix currently exists for the unknown-speaker-count case. Tuning pyannote's clustering threshold is the likely next research avenue, but doing that properly requires a validation dataset separate from whatever fixture is used for regression testing (to avoid overfitting the threshold to one specific test case).
- **Status:** Open research item, not yet started.

### Context-Dependent Word-Sense Correction (v2 Direction Decided)
- **Direction chosen:** Research running a small local LLM (e.g. via `llama.cpp` or Ollama) to re-process low-confidence ASR segments (confidence score already computed in `src/asr.py`) using surrounding transcript context, rather than Whisper-only re-prompting or a cloud LLM API.
- **Real constraint to solve:** The `local` hardware tier already uses ~2GB of the target 4GB VRAM budget for VAD + ASR + diarization combined. Adding local LLM inference on top of that will likely require either a very small model (needs evaluation) or restricting this feature to the `colab`/cloud tier only until a small enough local model is validated to fit the 4GB budget.
- **Noted synergy:** The same local LLM investment is also a candidate upgrade path for translation quality (see below) — worth evaluating both together rather than building two separate LLM integrations independently.
- **Status:** Direction decided, not yet implemented.

### Translation Quality Upgrade Path
- Whisper's built-in `task="translate"` was chosen as the v1 approach (validated, working, zero new dependencies).
- If translation quality proves insufficient on more/longer real content, NLLB (text-based translation, reuses native-pass timestamps) or the local LLM being researched for word-sense correction above are both candidate upgrades.
- **Status:** Current approach validated and working; upgrade only if a real quality gap is found on more content.

---

## Planned Architecture (v1 Audio-Only)

```
Source Audio File
       │
       ▼
  [Silero VAD] ─────────────────► Filters silence before ASR model
       │
       ▼
 [Multilingual ASR] ────────────► faster-whisper (Whisper large-v3, INT8)
       │                           with prompt & context biasing
       ▼
 [pyannote.audio] ──────────────► Diarization (speaker segmentation & alignment)
       │
       ▼
 [Hallucination Filter] ────────► Repetition-count threshold + phrase filtering
       │
       ▼
 [Context-Aware Pass] ──────────► Confidence-gated correction for multi-sense words
       │
       ▼
 [Reading-Speed Formatter] ─────► CPS (Characters Per Second) & line limits
       │
       ▼
  SRT / VTT Subtitle File
```

---

## Languages Scope (v1)

- **Nepali (`ne`)**: Fully implemented and validated end-to-end on OpenSLR 54 dataset baseline fixtures.
- **Japanese (`ja`)**: Fully implemented and validated end-to-end with strict 13 CJK characters/line formatting and 4.0 CPS pacing on Mozilla Common Voice fixtures.
- **English (`en`) Translation**: Fully implemented dual-export translation pass (`task="translate"`), generating `.en.srt` and `.en.vtt` alongside native subtitles by default.

---

## CLI Usage

The command-line interface provides a single entry point to run the entire pipeline on audio or video files:

```bash
python -m src.cli <input_file> [options]
```

### Options
- `<input_file>`: Path to input audio (`.wav`, `.flac`, `.mp3`, etc.) or video (`.mp4`, `.mkv`, `.avi`, `.mov`, etc.).
- `-o, --output-dir`: Directory to save generated `.srt` and `.vtt` files (defaults to `output/`).
- `-c, --config`: Path to YAML configuration file (defaults to `config.yaml`).
- `-t, --hardware-tier`: Hardware tier override (`colab` for large-v3, `local` for 4GB VRAM cards).
- `-l, --language`: Language code override (e.g. `ne`, `ja`). If omitted, reads from `config.yaml`.
- `--include-speaker / --no-speaker`: Toggle speaker identification labels in output subtitles (default: `--include-speaker`).

> [!NOTE]
> For non-English source audio, Subtitle Generator automatically generates both native subtitles (`<input>.srt` / `<input>.vtt`) and English translation subtitles (`<input>.en.srt` / `<input>.en.vtt`) in a single pass.

### Example Invocation
```bash
conda run -n subgen python -m src.cli sample_video.mp4 -t local -l ja -o ./subtitles
```

---

## Hardware Support Tiers

Subtitle Generator supports configurable hardware profiles so it runs effectively on low-end local consumer GPUs as well as cloud infrastructure:

- **Low-End Local GPU (`local` tier):** Uses Whisper `medium` with INT8 quantization (`faster-whisper`), Silero VAD, and `pyannote.audio`. Validated on a 4GB RTX 3050 Laptop GPU, staying comfortably within ~2GB VRAM.
- **Capable / Cloud GPU (`colab` tier):** Uses Whisper `large-v3` with INT8 quantization (~3GB VRAM) for cloud or high-VRAM environments.

---

## Quickstart Setup

### Requirements
- **OS:** Windows / Linux
- **Conda / Mamba**
- **NVIDIA GPU** with compatible drivers (CUDA runtime wheels installed via PyTorch/CTranslate2)

### 1. Environment Creation & Dependencies
```powershell
# Set strict channel priority to avoid channel mixing issues (e.g., libglib crashes)
conda config --set channel_priority strict

# Create and activate environment
conda create -n subgen python=3.10 -y
conda activate subgen

# Install ffmpeg via conda-forge
conda install -c conda-forge ffmpeg -y

# Install PyTorch with CUDA 12.4 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install ASR and Diarization frameworks
pip install faster-whisper pyannote.audio
```

### 2. Hugging Face Authentication (For pyannote models)
Pyannote diarization models (`pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.0`) require a free license agreement on Hugging Face.
```powershell
hf auth login
```

---

## License

Distributed under the [MIT License](LICENSE).

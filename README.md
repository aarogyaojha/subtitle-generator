# Subtitle Generator

An open-source AI subtitle generation pipeline engineered to solve common failure modes present in existing speech recognition tools: hallucinated or repeated text during silence, duplicate text at chunk boundaries, dropped text under background noise, and context-dependent word-sense ambiguity.

---

## Key Features & Targeted Problems

Existing open-source ASR implementations frequently suffer from specific edge-case failures. Subtitle Generator targets these directly:

| Failure Mode | Pipeline Remedy | Status |
| --- | --- | --- |
| **Silence Hallucination / Repetition** | Silero VAD pre-filtering + prompt-priming + repetition thresholds | Partial (drastically reduced) |
| **Chunk-Boundary Text Duplication** | Dynamic VAD-based audio batching | Solved |
| **Reading Speed & Line Overflow** | Language-specific deterministic CPS & line-length formatting | Solved |
| **Speaker Diarization Error** | Integrated `pyannote.audio` pipeline | Partial (10–20% DER baseline) |
| **Context-Dependent Word Sense** | Confidence-gated context correction pass | Partial |
| **Noise-Induced Dropped Text** | Tunable VAD thresholds & hardware tier sensitivity | Under active research |

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

- **Nepali** (low-resource language script baseline using OpenSLR 54 dataset baseline)
- **Japanese** (high-resource script baseline using ReazonSpeech dataset / `kotoba-whisper-v2.0` optimizations)

---

## Hardware Support Tiers

Subtitle Generator supports configurable hardware profiles so it runs effectively on low-end local consumer GPUs as well as cloud infrastructure:

- **Low-End Local GPU (e.g., RTX 3050 4GB VRAM):** INT8 quantized models (`faster-whisper`), optimized batch sizes, CPU fallback options.
- **Capable / Cloud GPU (e.g., NVIDIA T4 16GB VRAM on Colab / AWS `g4dn.xlarge`):** `Whisper large-v3`, full `pyannote.audio` diarization pass, and LLM-assisted context correction.

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

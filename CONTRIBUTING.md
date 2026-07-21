# Contributing to Subtitle Generator

Thank you for your interest in contributing to Subtitle Generator! We welcome contributions to help make AI-generated subtitles accurate, resilient to noise, and context-aware.

---

## Getting Started

1. **Fork & Clone:** Fork the repository on GitHub and clone your fork locally.
2. **Environment Setup:** Follow the instructions in [README.md](README.md) to set up the `subgen` Conda environment on your system.
3. **Hardware Constraints:** Test your code under both low-VRAM (4GB local GPU) and high-VRAM (16GB Colab/Cloud T4) settings if possible.

---

## Development Guidelines

### Conda Channel Priority
Always ensure `conda config --set channel_priority strict` is set before installing binaries via Conda to avoid DLL/shared library conflicts (e.g., `libglib` mismatch between `defaults` and `conda-forge`).

### Hugging Face Models & Licenses
- When working with `pyannote.audio`, do not commit Hugging Face access tokens (`hf_...`) to code or configuration files. Use environment variables.
- When using Japanese dataset resources (ReazonSpeech), make sure your usage complies with Japanese Copyright Act Article 30-4.

### Commit Guidelines
We prefer clear, atomic commits that describe the specific problem or feature addressed:

- `feat(vad): add silero VAD silence filtering pass`
- `fix(asr): resolve chunk-boundary text duplication`
- `docs: update setup commands for huggingface-cli deprecation`

---

## Issue & Feature Requests

When reporting issues or proposing features, please include:
1. Environment specs (`nvidia-smi` output, PyTorch CUDA availability).
2. Audio language (Nepali, Japanese, or other).
3. Exact command ran and full error traceback log.

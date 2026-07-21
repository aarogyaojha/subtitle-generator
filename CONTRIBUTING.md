# Contributing to Subtitle Generator

Thank you for contributing to Subtitle Generator! We welcome contributions to make AI-generated subtitles accurate, resilient to background noise, and context-aware.

**Repository:** [https://github.com/aarogyaojha/subtitle-generator](https://github.com/aarogyaojha/subtitle-generator)

---

## 1. Development Environment Setup

1. **Fork the Repository:** Click **Fork** on [github.com/aarogyaojha/subtitle-generator](https://github.com/aarogyaojha/subtitle-generator).
2. **Clone your fork locally:**
   ```powershell
   git clone https://github.com/YOUR-USERNAME/subtitle-generator.git
   cd subtitle-generator
   ```
3. **Configure Upstream Remote:**
   ```powershell
   git remote add upstream https://github.com/aarogyaojha/subtitle-generator.git
   ```
4. **Environment Setup:** Follow the instructions in [README.md](README.md) to activate the `subgen` Conda environment and PyTorch/CUDA dependencies.

---

## 2. Branching Naming Conventions

Always create a new branch from `main` (or `master`) for your work. Use descriptive prefixes:

| Branch Type | Prefix Pattern | Example |
| --- | --- | --- |
| **New Features** | `feature/<short-description>` or `feat/<short-description>` | `feature/silero-vad-integration` |
| **Bug Fixes** | `bugfix/<short-description>` or `fix/<short-description>` | `fix/chunk-boundary-overlap` |
| **Documentation** | `docs/<short-description>` | `docs/add-colab-setup` |
| **Refactoring** | `refactor/<short-description>` | `refactor/hardware-tier-config` |
| **Experiments** | `experiment/<short-description>` | `experiment/nepali-fine-tune` |

---

## 3. Step-by-Step Pull Request (PR) Workflow

### Step 1: Sync with Upstream
Before starting work, update your local main branch:
```powershell
git checkout master
git fetch upstream
git merge upstream/master
```

### Step 2: Create a Working Branch
```powershell
git checkout -b feature/vad-silence-filter
```

### Step 3: Make Changes & Commit
Keep your commits focused and write descriptive commit messages:
```powershell
# Stage modified/new files
git add src/vad_filter.py

# Commit with clear prefix
git commit -m "feat(vad): add silero VAD silence filtering pass"
```

### Step 4: Push to Your Fork
```powershell
git push -u origin feature/vad-silence-filter
```

### Step 5: Open a Pull Request on GitHub
1. Navigate to [https://github.com/aarogyaojha/subtitle-generator](https://github.com/aarogyaojha/subtitle-generator).
2. You will see a prompt saying **"Compare & pull request"** for your branch.
3. Click it and fill out the PR template:
   - **Title:** Brief title summarizing your change (e.g., `feat: integrate Silero VAD pre-filtering pass`).
   - **Description:** 
     - What issue or goal does this address?
     - How was it tested (e.g., tested on RTX 3050 / Google Colab)?
     - Sample SRT output comparison (if applicable).
4. Click **Create Pull Request**.

---

## 4. Code & Commit Best Practices

- **Conda Strict Priority:** Always run `conda config --set channel_priority strict` before installing new packages.
- **Hugging Face Secrets:** **NEVER** commit Hugging Face write/read API tokens (`hf_...`). Pass tokens via environment variables.
- **Hardware Awareness:** Test your changes against both low-VRAM (4GB local card) and high-VRAM (Cloud T4) settings whenever possible.
- **Atomic Commits:** Prefer multiple small, logical commits over one massive commit.

# Contributing to Multimodal E-Commerce Review Analytics

Thank you for your interest in contributing to **Multimodal E-Commerce Review Analytics**! We welcome contributions from ML engineers, full-stack developers, and open-source enthusiasts.

This document outlines the development workflow, coding standards, and contribution guidelines for maintaining production-grade quality across our ML pipelines and web platform.

---

## 1. Architecture Overview

The repository is structured into distinct microservice components:

```text
ecommerce-review-analytics/
├── ai_engine/                # Python AI Service (FastAPI, PyTorch, PhoBERT, ResNet50)
│   ├── fusion/               # Cross-modal Fusion Engine (Trust Score calculation)
│   ├── image_processing/     # Vision pipeline (CLIP zero-shot, FeatureDenoiser, Defect Detection)
│   ├── llm_integration/      # Gemini LLM client fallback & budget management
│   ├── models/               # PyTorch models, baselines, and model registries
│   ├── recommendation/       # Zero-Shot Content-Based Reranker (PhoBERT vector similarity)
│   └── text_processing/      # Text cleaning, PhoBERT sentiment analysis, Isolation Forest spam filter
├── web_platform/
│   ├── backend/              # Node.js Express REST API & Supabase database controllers
│   └── frontend/             # Next.js 16 App Router UI dashboard & interactive charts
├── tests/                    # Pytest test suite for AI Engine pipelines
└── .github/workflows/        # CI/CD automated validation pipelines
```

---

## 2. Development Setup

### Prerequisites

- **Python:** `3.10` or higher
- **Node.js:** `v20.x` or higher
- **Package Managers:** `pip`, `npm`
- **Git:** Version `2.x`+

### Local Environment Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/ThanhChuong12/ecommerce-review-analytics.git
   cd ecommerce-review-analytics
   ```

2. **Configure the AI Engine (Python):**
   ```bash
   python -m venv .venv
   # Windows PowerShell
   .venv\Scripts\Activate.ps1
   # Linux/macOS
   source .venv/bin/activate

   pip install --upgrade pip
   pip install -r ai_engine/requirements.txt
   pip install pytest
   ```

3. **Configure the Web Backend (Node.js):**
   ```bash
   cd web_platform/backend
   npm install
   cp .env.example .env  # Configure Supabase credentials
   ```

4. **Configure the Web Frontend (Next.js):**
   ```bash
   cd web_platform/frontend
   npm install
   ```

---

## 3. Git Branching Strategy & Commit Standards

### Branch Naming Conventions

Prefix your branch name according to the scope of your changes:

- `feat/<short-description>`: New feature implementation (e.g., `feat/image-defect-detector`)
- `fix/<short-description>`: Bug fix (e.g., `fix/reranker-price-penalty`)
- `ml/<short-description>`: Model training, tuning, or architecture changes (e.g., `ml/phobert-head-tuning`)
- `refactor/<short-description>`: Code refactoring without behavioral changes
- `docs/<short-description>`: Documentation updates

### Commit Message Guidelines

We follow the **Conventional Commits** specification:

```text
<type>(<scope>): <short summary in imperative mood>

[optional body]
```

**Allowed types:**
- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation changes
- `style`: Code formatting, missing semi-colons, etc. (no functional change)
- `refactor`: Code change that neither fixes a bug nor adds a feature
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `chore`: Maintenance tasks or dependency updates

**Examples:**
```bash
git commit -m "feat(ai_engine): add zero-shot content-based reranker module"
git commit -m "fix(frontend): update trust score display label to preliminary reputation"
git commit -m "test(pipeline): add unit tests for FeatureDenoiser gaussian noise injection"
```

---

## 4. Coding & Machine Learning Standards

### Python & Machine Learning Guidelines

- **Style:** Adhere strictly to **PEP 8**. Use 4-space indentation.
- **Type Hints:** Use explicit static type annotations (`typing.List`, `typing.Dict`, `typing.Optional`, `torch.Tensor`, `np.ndarray`).
- **Docstrings:** Provide concise English docstrings for all public classes and functions. Avoid redundant visual dividers (e.g., `# ===...===`).
- **Reproducibility:** Fix random seeds explicitly when writing training or data split scripts:
  ```python
  import random
  import numpy as np
  import torch

  def set_seed(seed: int = 42) -> None:
      random.seed(seed)
      np.random.seed(seed)
      torch.manual_seed(seed)
      if torch.cuda.is_available():
          torch.cuda.manual_seed_all(seed)
  ```
- **Weights & Dataset Files:** **Do NOT commit binary model weights (`.pt`, `.pth`, `.bin`, `.onnx`) or raw scraped CSV files directly to Git**. Rely on project storage buckets or upload model checkpoints to GitHub Releases. Make sure `.gitignore` rules are respected.

### JavaScript / TypeScript & Web Guidelines

- **Framework Rules:** Follow standard Next.js App Router conventions.
- **Linting:** Run `npm run lint` before submitting frontend code to ensure compliance.
- **Component Modularity:** Keep UI components focused, clean, and reusable. Avoid inline styles; leverage utility CSS classes.

---

## 5. Testing & Quality Assurance

All new features or bug fixes must include unit test coverage.

### Running AI Engine Tests

Execute pytest from the root directory:

```bash
# Run full test suite
python -m pytest tests/ -v

# Run specific test module
python -m pytest tests/test_reranker.py -v
```

Ensure all tests pass cleanly before creating a pull request.

---

## 6. Submitting a Pull Request (PR)

1. **Rebase against `main`:** Keep your branch up to date:
   ```bash
   git fetch origin
   git rebase origin/main
   ```
2. **Run Quality Gates:**
   - Execute all unit tests (`python -m pytest tests/`).
   - Run frontend linter (`npm run lint`).
3. **Open Pull Request:**
   - Provide a clear, descriptive title following Conventional Commits.
   - Fill in the PR description detailing:
     - Motivation and context of the changes.
     - Summary of technical changes made.
     - Verification plan and test results.
4. **Code Review:** Address any feedback requested by project maintainers. Once approved, PRs will be squash-merged into `main`.

---

Thank you for contributing to **Multimodal E-Commerce Review Analytics**!

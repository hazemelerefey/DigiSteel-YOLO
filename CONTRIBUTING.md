<div align="center">
  <img src="assets/logo.png" alt="DigiSteel Logo" width="100" height="auto" />
</div>

# Contributing to DigiSteel-YOLO

**Read this before making any contributions.** This is a graduation project with a 12-week deadline and specific operating rules.

---

## Team Members & Responsibilities

| Member | WP | Role | Primary Branch |
|---|---|---|---|
| **Hazem Elerefy** | WP1 | Lead; Training pipeline, A2/A3 modules, ONNX export | `feat/hazem-wp1-pipeline` |
| **Youssef Sherif** | WP2 | Dataset preprocessing, model zoo, comparison tables | `feat/youssef-wp2-datasets` |
| **Mohamed Salah** | WP3 | Robustness perturbations, evaluation sweeps | `feat/mohamed-wp3-robustness` |
| **Moamen Esmat** | WP4 | Environment, GPU access, dashboards, Pareto plots | `feat/moamen-wp4-infrastructure` |
| **Mahmoud Hisham** | WP5 | Metrics, figures, written report integration | `feat/mahmoud-wp5-reporting` |

---

## Operating Principles

### 1. No-Guessing Protocol

- **Never interpolate or estimate a number.** If it's missing, mark it `NR` (Not Reported).
- **Every metric must trace back to a source:** run-ID for our own results, Table # for literature.
- **Discrepancies are flagged, never silently corrected.**

### 2. Audit Trail

All experimental outputs land in versioned directories:

```
runs/<run_id>/              # Training: config, weights, logs, tensorboard
  ├── weights/
  ├── args.yaml             # Exact hyperparameters used
  ├── results.csv
  └── RESULTS.md            # Summary + publish approval

evals/<run_id>/             # Evaluation: robustness sweeps
  ├── perturbation/
  │   ├── blur/
  │   │   ├── level_1/
  │   │   ├── level_2/
  │   │   └── level_3/
  │   └── ...
  └── robustness.csv        # Aggregated results
```

**Every run-ID must be traceable back to a specific `git commit`.** Tag the commit when publishing results.

### 3. Git Conventions

- **Branch naming:** `feat/[name]-[feature]`, `fix/[issue]`, `docs/[topic]`
  - Example: `feat/hazem-wp1-ghostconv`, `fix/voc-converter-class-id`, `docs/methodology`

- **Commit messages:** Atomic, descriptive
  ```bash
  git commit -m "Implement GhostConv weight-sharing layer"
  git commit -m "Add perturbation library (blur, noise, brightness, jpeg)"
  git commit -m "Fix VOC-to-YOLO converter class-id mapping"
  ```

- **Pull Requests:** Required before any merge to `develop`
  - Title: `[WP#] Short description`
  - Description: Link the issue, explain the change, cite any papers
  - Example: `[WP1] Implement A2 GhostConv weight-sharing per P01 §3.1`

- **Merging:** Never force-push. Use "Squash and merge" or "Create a merge commit" only.

- **Milestones:** Tag releases
  ```bash
  git tag v0.1-arch-freeze      # End of Week 4
  git tag v0.2-results-freeze   # End of Week 8
  git tag v1.0-submission       # Week 12
  ```

---

## Weekly Cadence

### Monday — Standup (30 min)

Each WP lead reports:
- What you completed last week
- What you're doing this week
- Any blockers

### Friday — Integration Check (60 min)

Hazem (WP1 lead) merges all PRs from the week into `develop` and runs:

```bash
bash scripts/run_all.sh  # Smoke test: 1-epoch training on 50-image subset
```

If the smoke test passes, the week's work is frozen in a **git tag** (`v0.X-week-N`).

### Milestones (Weeks 4, 8, 12)

In-person or video review with Dr. Tarek Ghoneimy.

---

## Code Standards

### Style

- **Python:** PEP 8 via `black` (line length 100)
- **Formatting:** Run `black .` before committing
- **Linting:** Run `ruff check .` to catch style issues

### Before Every Commit

```bash
black .                    # Auto-format
ruff check --fix .         # Fix linting issues
pytest -q                  # Run tests
git diff --staged          # Review changes
```

### Testing

- Every module in `digisteel/` must have a corresponding test in `tests/`
- Tests are run on every PR via GitHub Actions
- Minimum coverage: 70%

Example test structure:

```python
# tests/test_ghost_conv.py
import pytest
from digisteel.modules import GhostConv

def test_ghost_conv_param_count():
    """A2: single GhostConv block is 1/3 the params of 3 independent blocks."""
    layer = GhostConv(64, 64, 3)
    # Assert...

def test_ghost_conv_weight_sharing():
    """A2: forward pass through 3 stages reuses same weights."""
    # Assert...
```

---

## Documentation

### Docstrings

Every function, class, and module must have a docstring:

```python
def train_model(config_path: str, epochs: int = 200) -> dict:
    """Train YOLOv11n with the given config.
    
    Args:
        config_path: Path to YAML config file (e.g., 'configs/yolov11n_a2_a3.yaml')
        epochs: Number of training epochs (default: 200)
    
    Returns:
        Dictionary with keys: 'run_id', 'best_mAP@0.5', 'model_path'
    
    References:
        - Chapter 2, §2.7: Training protocol
        - Ultralytics YOLO docs: https://docs.ultralytics.com/modes/train/
    """
    ...
```

### Inline Comments

Use comments sparingly. Prefer clear code over comments. When needed:

```python
# A3: Inner-WIoU = λ · Inner-IoU + (1−λ) · WIoU_v3
# Reference: Zhang 2023 [33], Tong 2023 [32]
loss = lambda_w * inner_iou_loss + (1 - lambda_w) * wiou_loss
```

### README Files

Each subdirectory with a significant component should have a mini-README:

```
digisteel/perturbations/README.md
  - What: 4 perturbations (blur, noise, brightness, jpeg)
  - How: import and use the Albumentations wrapper
  - Why: Robustness evaluation suite (Gap 2)
```

---

## Handling Disagreement

If you disagree with a code review or merge decision:

1. **Comment respectfully** on the PR with your concern (cite papers / Chapter 2 if needed)
2. **Tag Hazem** (WP1 lead) if it's blocking
3. **Escalate to Dr. Ghoneimy** only if unresolved after 2 days

---

## Common Pitfalls & How to Avoid Them

| Pitfall | Symptom | Fix |
|---|---|---|
| Custom module used but not appearing in model info | `model.info()` doesn't show GhostConv | Did you patch `BboxLoss` *after* `YOLO(...)` and *before* `.train()`? |
| ONNX export gives different predictions | mAP differs by >1 pp | Recheck: are you exporting `best.pt` and using the same input preprocessing? |
| VOC class-id off-by-one | Predictions are shifted across classes | Always use `digisteel/data/<dataset>.py` as the single source of truth for class mappings. |
| Robustness images saved at wrong precision | JPEG artifacts blur the actual JPEG perturbation effect | Save as PNG for verification; for eval, pass as in-memory tensors only. |
| Forgot to seed random generators | Results are not reproducible | Set `seed=42` in every training script *and* in the YAML config. |

---

## Security

### Never commit:

- ❌ Datasets (too large, license restrictions)
- ❌ Trained weights (use `.gitignore`, re-train from config)
- ❌ API keys or tokens (use `.env`, which is `.gitignore`d)
- ❌ Kaggle credentials (use `~/.kaggle/kaggle.json` with `chmod 600`)

### Environment variables

If you need Kaggle, WandB, or GitHub tokens:

```bash
# .env (NEVER commit this file)
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
WANDB_API_KEY=your_wandb_key

# In Python, load with python-dotenv:
from dotenv import load_dotenv
import os
load_dotenv()
kaggle_user = os.getenv('KAGGLE_USERNAME')
```

---

## Questions?

- **Technical:** Ask on the team Slack/Discord, or open a GitHub issue
- **Process:** Ask Hazem (team lead)
- **Scope:** Ask Dr. Ghoneimy

---

## tl;dr: Before Every Push

```bash
# 1. Update your branch with latest develop
git checkout develop && git pull
git checkout your-feature-branch
git rebase develop

# 2. Format code
black .
ruff check --fix .

# 3. Run tests
pytest -q

# 4. Commit
git add .
git commit -m "[WP#] Your message (citing Chapter 2 if needed)"

# 5. Push and open PR
git push origin your-feature-branch
# Open PR on GitHub with title "[WP#] Description"

# 6. Wait for CI + peer review
# Never force-push to main or develop
```

Thank you for contributing to DigiSteel-YOLO. See you at Friday integration!

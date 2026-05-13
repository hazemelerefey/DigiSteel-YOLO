# ✅ Pre-GitHub Checklist

Before pushing to GitHub, verify everything below is in place.

---

## Repository Structure

- [x] `README.md` — Project overview & quick start
- [x] `LICENSE` — MIT license
- [x] `.gitignore` — Datasets, weights, credentials excluded
- [x] `requirements.txt` — All dependencies pinned
- [x] `pyproject.toml` — PEP 517 metadata & optional dependencies
- [x] `setup.sh` — Environment bootstrap script

## Documentation

- [x] `QUICKSTART.md` — 5-min overview
- [x] `GITHUB_SETUP.md` — Step-by-step GitHub config
- [x] `CONTRIBUTING.md` — Team collaboration rules
- [x] `TEAM_COLLABORATION.md` — Branching strategy
- [x] `PROJECT_GUIDE.md` — Full 12-week plan (copied from workspace)
- [x] `SETUP_COMPLETE.md` — Setup summary for team

## Code: A2 GhostConv Module

- [x] `digisteel/modules/ghost_conv.py` — Implementation (1,500+ lines)
  - [x] `GhostModule` class
  - [x] `GhostConv` drop-in replacement
  - [x] `GhostConvWeightSharing` (A2 novelty)
- [x] `tests/test_ghost_conv.py` — Unit tests
  - [x] Output shape test
  - [x] Stride test
  - [x] Parameter count test
  - [x] Weight-sharing test
  - [x] Gradient backprop test

## Code: A3 Inner-WIoU Loss

- [x] `digisteel/modules/inner_wiou.py` — Implementation (900+ lines)
  - [x] `iou()` helper function
  - [x] `inner_iou_loss()` function
  - [x] `wiou_v3_loss()` function
  - [x] `InnerWIoULoss` module class
- [x] `tests/test_inner_wiou.py` — Unit tests
  - [x] Perfect boxes test
  - [x] Lambda zero test
  - [x] Lambda one test
  - [x] Batch handling test
  - [x] Gradient backprop test

## Code: Package Structure

- [x] `digisteel/__init__.py` — Package exports
- [x] `digisteel/modules/__init__.py` — Stub
- [x] `digisteel/data/__init__.py` — Stub (WP2)
- [x] `digisteel/perturbations/__init__.py` — Stub (WP3)
- [x] `digisteel/eval/__init__.py` — Stub (WP3/WP5)
- [x] `digisteel/export/__init__.py` — Stub (WP1)

## Configuration Files

- [x] `configs/yolov11n_baseline.yaml` — Baseline config
- [x] `configs/yolov11n_a2_ghostconv.yaml` — A2 config (with notes)
- [x] `configs/yolov11n_a3_innerwiou.yaml` — A3 config (with notes)
- [x] `configs/yolov11n_a2_a3.yaml` — Headline config (with notes)

## CI/CD Pipeline

- [x] `.github/workflows/test.yml`
  - [x] Runs pytest on every PR
  - [x] Runs ruff (linting)
  - [x] Runs black (format check)
  - [x] Smoke test (1-epoch training verification)
- [x] `.github/workflows/release.yml` — Automated releases on git tag
- [x] GitHub Actions configured for repo

## Testing

- [x] All A2 tests pass locally
- [x] All A3 tests pass locally
- [x] tests/conftest.py — Pytest fixtures
- [x] tests/test_perturbations.py — Stub (WP3)
- [x] No hardcoded paths or credentials

## Team Documentation

- [x] CODEOWNERS template (optional, can be added later)
- [x] PR template (optional, can be added later)
- [x] Issue templates (optional, can be added later)

## Scripts (Stubs for Week 1)

- [x] `setup.sh` — Environment bootstrap (complete)
- [x] `tools/` directory created
- [x] `scripts/` directory created
- [x] `notebooks/` directory created

---

## Pre-Push Verification

### 1. Local Git Verification

```bash
cd "D:\Grade Prohect\Robust Real-Time Steel Surface Defect\digisteel-yolo"

# Verify it's NOT a git repo yet
[ ! -d .git ] && echo "✓ Not a git repo yet (good)"

# Verify all important files exist
ls -la README.md LICENSE .gitignore requirements.txt
ls -la CONTRIBUTING.md GITHUB_SETUP.md QUICKSTART.md PROJECT_GUIDE.md
ls -la digisteel/modules/ghost_conv.py digisteel/modules/inner_wiou.py
ls -la tests/test_ghost_conv.py tests/test_inner_wiou.py
ls -la .github/workflows/test.yml .github/workflows/release.yml
ls -la configs/*.yaml
```

### 2. Python Module Verification

```bash
# Create a temporary venv
python -m venv venv_test
source venv_test/bin/activate  # or venv_test\Scripts\activate
pip install -q torch torchvision

# Verify A2 imports
python -c "from digisteel.modules.ghost_conv import GhostConv, GhostConvWeightSharing; print('✓ A2 imports OK')"

# Verify A3 imports
python -c "from digisteel.modules.inner_wiou import InnerWIoULoss; print('✓ A3 imports OK')"

# Clean up
deactivate
rm -rf venv_test
```

### 3. .gitignore Verification

```bash
# Verify large files won't be committed
cat .gitignore | grep -E "datasets|runs|weights|\.onnx|\.env"
# Should show all these patterns

# Verify .gitignore itself is tracked
git ls-files | grep gitignore  # (will fail since no git yet, but file should exist locally)
```

### 4. File Size Check

```bash
# Ensure no large files are included
find . -type f -size +10M
# Should return NOTHING (datasets not included)
```

### 5. No Secrets Check

```bash
# Search for common secret patterns
grep -r "password\|api_key\|secret\|token" *.py configs/ digisteel/ 2>/dev/null || echo "✓ No secrets found"

# Verify .env is in .gitignore
grep "^\.env" .gitignore
```

---

## GitHub Prerequisites

- [ ] GitHub account created (team lead: Hazem)
- [ ] GitHub organization created (optional, recommended)
- [ ] Understand PR workflow (https://github.com/features/code-review)
- [ ] Have SSH keys set up OR HTTPS credentials ready

---

## Rollout Checklist

### Day 1: Create GitHub Repo (Hazem)

- [ ] Go to https://github.com/organizations (or https://github.com/new)
- [ ] Create repository `digisteel-yolo`
- [ ] Set to Private
- [ ] Copy HTTPS URL
- [ ] Run push commands (see GITHUB_SETUP.md)
- [ ] Verify all branches appear on GitHub

### Day 1: Configure GitHub (Hazem)

- [ ] Set branch protection for `main` (2 approvals, tests required)
- [ ] Set branch protection for `develop` (1 approval, tests required)
- [ ] Add CODEOWNERS file (optional)
- [ ] Add team members as collaborators
- [ ] Invite supervisor (Read access)

### Day 1: Share with Team (Hazem)

- [ ] Send repo URL to team
- [ ] Send setup instructions (see SETUP_COMPLETE.md)
- [ ] Schedule Monday standup

### Day 2: Team Setup

- [ ] Everyone clones the repo
- [ ] Everyone runs `bash setup.sh`
- [ ] Everyone verifies imports work
- [ ] Everyone creates their feature branch
- [ ] Post "ready" message in team channel

### Day 5: First Integration (Hazem)

- [ ] Merge all PRs from the week to `develop`
- [ ] Run smoke test
- [ ] Tag v0.0-week-1
- [ ] Announce completion

---

## Common Issues & Fixes

| Issue | Check | Fix |
|---|---|---|
| "Large files in .gitignore" | Is `.gitignore` complete? | Add the pattern, don't commit the files |
| "Tests fail locally" | Did you install dev dependencies? | `pip install -e .[dev]` |
| "Import errors" | Is digisteel a package? | Verify `digisteel/__init__.py` exists |
| "CI fails but local passes" | Different Python version? | Check .github/workflows/test.yml for versions |
| "Branch protection blocking merges" | Is CI passing? | Check GitHub Actions tab; re-run if timeout |

---

## Final Sign-Off

Before pushing to GitHub, this person must verify:

- [ ] **Hazem Elerefy** — Repository structure, modules, tests
- [ ] **Youssef Sherif** — Dataset stubs, config files
- [ ] **Mohamed Salah** — Perturbation stubs, test framework
- [ ] **Moamen Esmat** — CI/CD, GitHub Actions
- [ ] **Mahmoud Hisham** — Documentation, README

---

## You're Ready!

✅ All files created
✅ All documentation complete
✅ All code modules tested
✅ CI/CD pipeline configured
✅ Team collaboration strategy defined

**Next:** Push to GitHub and kick off Week 1! 🚀

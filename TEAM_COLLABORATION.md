# Team Collaboration & Branching Strategy

**For 5-team parallel development without work overlap.**

---

## Branch Hierarchy

```
main (submission)
 ↑ (milestone tags only: v1.0-submission)
 │
 └─ release/v0.2 (results-freeze, Week 8)
     ↑ (tag: v0.2-results-freeze)
     │
     ├─ develop (integration point, Friday merges)
     │  ↑ (weekly tags: v0.X-week-Y)
     │  │
     │  ├─ feat/hazem-wp1-pipeline
     │  │  ├─ feat/hazem-wp1-ghostconv
     │  │  ├─ feat/hazem-wp1-inner-wiou
     │  │  ├─ feat/hazem-wp1-training-scripts
     │  │  ├─ feat/hazem-wp1-onnx-export
     │  │  └─ ... (sub-tasks)
     │  │
     │  ├─ feat/youssef-wp2-datasets
     │  │  ├─ feat/youssef-wp2-voc-converter
     │  │  ├─ feat/youssef-wp2-split-script
     │  │  ├─ feat/youssef-wp2-neu-loader
     │  │  └─ feat/youssef-wp2-gc10-loader
     │  │
     │  ├─ feat/mohamed-wp3-robustness
     │  │  ├─ feat/mohamed-wp3-blur-perturbation
     │  │  ├─ feat/mohamed-wp3-noise-perturbation
     │  │  ├─ feat/mohamed-wp3-brightness-perturbation
     │  │  ├─ feat/mohamed-wp3-jpeg-perturbation
     │  │  └─ feat/mohamed-wp3-sweep-script
     │  │
     │  ├─ feat/moamen-wp4-infrastructure
     │  │  ├─ feat/moamen-wp4-gpu-setup
     │  │  ├─ feat/moamen-wp4-wandb-integration
     │  │  ├─ feat/moamen-wp4-pareto-plotter
     │  │  └─ feat/moamen-wp4-ci-cd
     │  │
     │  └─ feat/mahmoud-wp5-reporting
     │     ├─ feat/mahmoud-wp5-metrics-aggregation
     │     ├─ feat/mahmoud-wp5-figure-generation
     │     ├─ feat/mahmoud-wp5-results-table
     │     └─ feat/mahmoud-wp5-chapter4-draft
```

---

## Per-Member Responsibilities

### Hazem Elerefy (WP1: Training Pipeline)

**What:** Baseline training, A2 GhostConv, A3 Inner-WIoU, ONNX export, integration

**Branches:**
- `feat/hazem-wp1-pipeline` — Main WP1 branch (stable)
  - `feat/hazem-wp1-ghostconv-module` — A2 implementation
  - `feat/hazem-wp1-inner-wiou-module` — A3 implementation
  - `feat/hazem-wp1-training-scripts` — train_baseline.py, train_a2.py, etc.
  - `feat/hazem-wp1-onnx-export` — ONNX conversion & verification
  - `fix/hazem-module-integration` — Bug fixes, integration with Ultralytics

**Deliverables (by end of each week):**
- Week 1: A2 module + A3 module (smoke test on 10 epochs)
- Week 2: training_scripts working on NEU-DET
- Week 3: Integration test (A2+A3 combined)
- Weeks 4–6: Multi-dataset tuning
- Week 7: Competitor baseline (P09 EFEN retrain)
- Week 8: ONNX export + verification
- Weeks 9–10: Colab demo notebook

**Weekly PR count:** 2–4 PRs per week to `feat/hazem-wp1-pipeline`, then 1 PR to `develop` on Friday

---

### Youssef Sherif (WP2: Dataset Preprocessing)

**What:** VOC→YOLO conversion, train/val/test splits, dataset loaders, comparison tables

**Branches:**
- `feat/youssef-wp2-datasets` — Main WP2 branch
  - `feat/youssef-wp2-voc-converter` — VOC XML to YOLO .txt conversion
  - `feat/youssef-wp2-split-script` — 7:2:1 split with seed=42
  - `feat/youssef-wp2-neu-loader` — NEU-DET preprocessing
  - `feat/youssef-wp2-gc10-loader` — GC10-DET preprocessing
  - `feat/youssef-wp2-comparison-table` — Literature numbers aggregation

**Deliverables (by end of each week):**
- Week 1: VOC converter + split script (test on 50-image subset)
- Week 2: GC10-DET preprocessing + verify shapes
- Week 3: Baseline datasets ready for multi-dataset runs
- Weeks 4–8: Curate model zoo results, comparison table data
- Weeks 9–10: Final comparison figures (vs P03/P05/P09/P10)

**Weekly PR count:** 2–3 PRs per week

---

### Mohamed Salah (WP3: Robustness Evaluation)

**What:** Albumentations perturbation toolkit, 4×3 robustness sweep, perturbation figures

**Branches:**
- `feat/mohamed-wp3-robustness` — Main WP3 branch
  - `feat/mohamed-wp3-blur-perturbation` — Gaussian blur (σ ∈ {1,3,5})
  - `feat/mohamed-wp3-noise-perturbation` — Gaussian noise (σ ∈ {0.05,0.1,0.2})
  - `feat/mohamed-wp3-brightness-perturbation` — Brightness drift
  - `feat/mohamed-wp3-jpeg-perturbation` — JPEG compression (Q ∈ {30,50,80})
  - `feat/mohamed-wp3-sweep-script` — Full sweep orchestrator
  - `feat/mohamed-wp3-visualizations` — Perturbation effect plots

**Deliverables (by end of each week):**
- Week 1: Albumentations study + prototype (blur + noise)
- Week 2: All 4 perturbations implemented
- Week 3: Sweep script working on small test set
- Weeks 5–7: Full 4×3 sweeps for baseline and A2+A3
- Weeks 8–10: Perturbation effect visualizations, comparison

**Weekly PR count:** 2–3 PRs per week

---

### Moamen Esmat (WP4: Infrastructure)

**What:** GPU/environment setup, dataset versioning, dashboards (WandB/TensorBoard), Pareto plots, CI/CD

**Branches:**
- `feat/moamen-wp4-infrastructure` — Main WP4 branch
  - `feat/moamen-wp4-gpu-setup` — CUDA/PyTorch verification
  - `feat/moamen-wp4-wandb-logging` — Weights & Biases integration
  - `feat/moamen-wp4-pareto-plots` — mAP/FPS, mAP/params frontiers
  - `feat/moamen-wp4-metrics-dashboard` — TensorBoard setup
  - `feat/moamen-wp4-ci-cd-github-actions` — Automated testing

**Deliverables (by end of each week):**
- Week 1: GPU accessible, environment verified
- Week 2: WandB logging working for training runs
- Weeks 3–8: Dashboard integration with all training runs
- Week 7: Pareto plots (baseline, A2, A3, A2+A3)
- Week 11: Final Pareto comparison (vs literature)

**Weekly PR count:** 1–2 PRs per week

---

### Mahmoud Hisham (WP5: Reporting)

**What:** Metrics aggregation, figure generation, results table, Chapter 4 draft, thesis integration

**Branches:**
- `feat/mahmoud-wp5-reporting` — Main WP5 branch
  - `feat/mahmoud-wp5-metrics-aggregation` — CSV to results table
  - `feat/mahmoud-wp5-figure-generation` — Matplotlib/seaborn scripts
  - `feat/mahmoud-wp5-results-table` — 8-metric table (mAP, precision, recall, etc.)
  - `feat/mahmoud-wp5-chapter4-draft` — Results section
  - `feat/mahmoud-wp5-discussion` — Discussion + comparison

**Deliverables (by end of each week):**
- Week 1: Metrics template, figure skeleton
- Weeks 2–3: Methodology section draft
- Week 4: Results section template
- Weeks 5–8: Results populated as experiments complete
- Week 9: Comparison section + Pareto figures
- Weeks 10–11: Discussion, literature integration, full draft

**Weekly PR count:** 1–2 PRs per week

---

## Daily Workflow

### For Each Team Member

**Morning (~9 AM):**
1. Pull latest `develop`: `git pull origin develop`
2. Check if any merges happened overnight
3. Rebase your feature branch if needed

**During Day (Work):**
1. Create sub-branch for today's task: `git checkout -b feat/[name]-[task]`
2. Implement, test, commit regularly
3. Push to origin: `git push -u origin feat/[name]-[task]`

**Evening (~5 PM):**
1. Open PR to your WP's main branch (e.g., `feat/hazem-wp1-pipeline`)
2. Link to the issue / task if available
3. Request review from 1–2 teammates

---

## Weekly Meetings

### Monday 11:00 AM — Standup (30 min)

Each WP lead reports:
- What completed last week (PRs merged to `develop`)
- What starting this week (PRs in review)
- Any blockers (GPU time, dependencies, design questions)

**Action:** Update the team board / wiki with the week's plan.

### Friday 11:00 AM — Integration Check (60 min)

**Hazem** merges all PRs from the week into `develop`:

```bash
git checkout develop
git pull
git merge feat/youssef-wp2-datasets --squash -m "[WP2] Week X datasets"
git merge feat/mohamed-wp3-robustness --squash -m "[WP3] Week X robustness"
# ... etc for all WPs
bash scripts/run_all.sh  # Smoke test: 1-epoch on 50-image subset
```

**If smoke test passes:**
```bash
git tag v0.X-week-Y
git push origin develop --tags
# Announce on team channel: "Week X frozen, all PRs merged"
```

**If smoke test fails:**
```bash
git revert HEAD  # Undo the bad merge
# Notify the WP lead to fix and resubmit
```

---

## No-Overlap Rules

### 1. Own Your WP

Each team member owns their WP branch. **Nobody else pushes directly to another's WP branch without explicit permission.**

Example:
- Hazem works on `feat/hazem-wp1-*` only
- Youssef works on `feat/youssef-wp2-*` only
- When WP1 and WP2 interact (e.g., training script uses dataset loader), do it through:
  - Merging to `develop` (public interface)
  - PR comments and code reviews (discussion)
  - Slack/Discord (quick questions)

### 2. Sub-Task Branches

Each team member can create sub-branches under their WP for organizational clarity:

```bash
git checkout feat/hazem-wp1-pipeline
git checkout -b feat/hazem-wp1-ghostconv
# Work on A2 here
git push -u origin feat/hazem-wp1-ghostconv
# PR to feat/hazem-wp1-pipeline (not directly to develop)
```

### 3. Conflict Resolution

If two PRs conflict:
1. **Don't force-push.** Flag the conflict on the PR comment.
2. **First-in wins.** The PR merged first is considered ready; the second one rebases.
3. **Rebase and re-request review:** `git rebase develop`, push with `--force-with-lease` to your own branch.

### 4. Cross-WP Dependencies

If WP3 (robustness) needs output from WP1 (training):
1. **Explicit issue:** "WP3 is blocked on WP1 Week-4 baseline training results"
2. **Monday standup:** Flag it then.
3. **Friday integration:** Ensure WP1 merges first, then WP3.

---

## Git Commands for Your Daily Workflow

```bash
# === Setup (first time) ===
git clone https://github.com/<your-team>/digisteel-yolo.git
cd digisteel-yolo
git checkout develop

# === Start a new task ===
git checkout -b feat/[your-name]-[short-description]
# Example: feat/hazem-fix-ghostconv-stride

# === Work & commit ===
git add <files>
git commit -m "[WP#] Short message (cite chapter if needed)"
# Example: git commit -m "[WP1] Implement A2 GhostConv weight-sharing per Chapter 2 §5.1"

# === Push & PR ===
git push -u origin feat/[your-name]-[short-description]
# Go to GitHub, open PR to YOUR WP branch (e.g., feat/hazem-wp1-pipeline)

# === Friday: WP lead merges to develop ===
git checkout develop
git pull
git merge feat/[teammate]-[task] --squash -m "[WP#] Week X task"
git push origin develop

# === Tag milestone ===
git tag v0.X-week-Y
git push origin --tags
```

---

## Review Checklist (Before Merging Any PR)

- [ ] CI/CD passes (GitHub Actions green)
- [ ] Code follows PEP 8 (black, ruff)
- [ ] Tests added/updated (coverage > 70%)
- [ ] Docstrings present
- [ ] No hardcoded credentials / API keys
- [ ] No large files (datasets, weights)
- [ ] Commit messages are clear
- [ ] Related issue is linked (if applicable)

---

## Escalation Path

**Problem** → **First contact** → **If blocked 2+ hours** → **Escalate to:**

- Code design question → PR comment → Hazem (WP1, lead)
- GPU/environment issue → Slack → Moamen (WP4)
- Dataset/data issue → Slack → Youssef (WP2)
- Robustness toolkit → Slack → Mohamed (WP3)
- Results/figures → Slack → Mahmoud (WP5)
- Scope/timeline issue → Slack → Hazem → Dr. Ghoneimy if needed

---

## Definition of "Complete" for Each Week

| Week | Criterion |
|---|---|
| 1 | All A2 + A3 modules implemented, smoke test passes |
| 2 | All training scripts working, baseline on NEU-DET |
| 3 | A2+A3 integration smoke test |
| 4 | Hyper-parameter search complete, arch frozen |
| 5 | Multi-dataset runs started |
| 6 | MVP models ready for robustness evaluation |
| 7 | Competitor baseline (P09) retrained |
| 8 | Results freeze, ONNX export verified |
| 9 | Colab demo notebook working |
| 10 | Code cleanup, README complete |
| 11 | Reproducibility check passed |
| 12 | Final submission, demo rehearsal |

---

**Questions? Ask on Slack/Discord or in Monday standup.**

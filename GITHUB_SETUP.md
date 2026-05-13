# GitHub Setup Guide for DigiSteel-YOLO Team

**Follow this guide to set up the team GitHub repository and configure branch protection rules.**

---

## Step 1: Create the GitHub Organization/Repository

### Option A: Team organization (recommended for scale)

1. Go to https://github.com/organizations/new
2. Create organization: `digisteel-team` (or similar)
3. Verify email
4. Create repository inside the organization

### Option B: Personal account (if team account not available yet)

1. Go to https://github.com/new
2. Repository name: `digisteel-yolo`
3. Description: "Robust Real-Time Steel Surface Defect Detection"
4. Visibility: **Private** (until grading is complete, then public)

---

## Step 2: Push the Initial Repository

```bash
cd digisteel-yolo/
git init
git add .
git commit -m "Initial commit: repository scaffold, configs, modules stubs"
git branch -M main
git remote add origin https://github.com/<your-team>/digisteel-yolo.git
git push -u origin main
```

**Create initial branches:**

```bash
# develop branch (integration point)
git checkout -b develop
git push -u origin develop

# Protected main branch (submission only)
git checkout main
```

---

## Step 3: Configure GitHub Branch Protection Rules

Go to repository **Settings** → **Branches** → **Add rule**

### For `main` branch (production/submission)

- **Branch name pattern:** `main`
- ✅ **Require a pull request before merging**
  - Require approvals: 2 (or 1 if team is small)
  - Dismiss stale PR approvals when new commits are pushed: ✅
- ✅ **Require status checks to pass before merging**
  - Require branches to be up to date before merging: ✅
  - Status checks that must pass: `test`, `smoke-test`
- ✅ **Require code reviews from code owners** (optional)
- ✅ **Require deployment reviews** → None needed for Phase 1
- ✅ **Restrict who can push to matching branches**
  - Allow Hazem Elerefy only (team lead)

### For `develop` branch (integration)

- **Branch name pattern:** `develop`
- ✅ **Require a pull request before merging**
  - Require approvals: 1
  - Dismiss stale PR approvals: ✅
- ✅ **Require status checks to pass before merging**
  - Status checks: `test`
- ✅ **Allow specified actors to bypass required pull requests**
  - Hazem Elerefy (team lead) can bypass for emergency fixes

---

## Step 4: Add Team Members

Go to repository **Settings** → **Collaborators and teams**

1. Add each member as a **collaborator** with:
   - Hazem Elerefy: **Maintain** (can merge PRs)
   - Youssef Sherif: **Write** (can push to branches)
   - Mohamed Salah: **Write**
   - Moamen Esmat: **Write**
   - Mahmoud Hisham: **Write**

---

## Step 5: Create Per-Member Feature Branches

Hazem creates base branches for each work-package lead:

```bash
git checkout -b feat/hazem-wp1-pipeline
git push -u origin feat/hazem-wp1-pipeline

git checkout -b feat/youssef-wp2-datasets
git push -u origin feat/youssef-wp2-datasets

git checkout -b feat/mohamed-wp3-robustness
git push -u origin feat/mohamed-wp3-robustness

git checkout -b feat/moamen-wp4-infrastructure
git push -u origin feat/moamen-wp4-infrastructure

git checkout -b feat/mahmoud-wp5-reporting
git push -u origin feat/mahmoud-wp5-reporting
```

Each member can create sub-branches from their base branch:
```bash
git checkout feat/hazem-wp1-pipeline
git checkout -b feat/hazem-wp1-ghostconv-module
# work...
git push -u origin feat/hazem-wp1-ghostconv-module
# Open PR to feat/hazem-wp1-pipeline, then to develop
```

---

## Step 6: Set Up GitHub Actions

GitHub Actions workflows are already in `.github/workflows/`:
- `test.yml`: Runs tests on every PR and push to `develop`
- `release.yml`: Creates a release when you tag with `v*`

**Verify they're enabled:**

1. Go to **Actions** tab
2. Should see the workflows listed
3. First PR or push to `develop` should trigger them automatically

---

## Step 7: Invite Supervisor (Dr. Ghoneimy)

Add Dr. Tarek Ghoneimy as a **collaborator** with **Read** access:

```
Settings → Collaborators → Search by GitHub username
```

(Or add after he creates a GitHub account.)

---

## Step 8: Configure CODEOWNERS (Optional but Recommended)

Create `.github/CODEOWNERS`:

```
# CODEOWNERS file — auto-request reviews

# Architecture & modules (WP1)
/digisteel/modules/     @hazem-elerefy
/configs/               @hazem-elerefy

# Data & preprocessing (WP2)
/digisteel/data/        @youssef-sherif
/tools/                 @youssef-sherif

# Robustness & evaluation (WP3)
/digisteel/perturbations/  @mohamed-salah
/digisteel/eval/           @mohamed-salah

# Infrastructure & CI (WP4)
/.github/               @moamen-esmat
/scripts/               @moamen-esmat

# Reporting & figures (WP5)
/notebooks/             @mahmoud-hisham

# Everything else
*                       @hazem-elerefy
```

Commit this file to the repo.

---

## Step 9: Set Up GitHub Secrets (for automation)

Go to **Settings** → **Secrets and variables** → **Actions**

Optional (leave empty for Phase 1, use in Phase 2):
- `KAGGLE_USERNAME`: Your Kaggle username
- `KAGGLE_KEY`: Your Kaggle API key
- `WANDB_API_KEY`: Weights & Biases API key

These allow CI/CD to automatically download datasets and log to WandB. **Never hardcode credentials in the repo.**

---

## Step 10: Your First Feature Branch & PR

### 1. Clone the repo locally

```bash
git clone https://github.com/<your-team>/digisteel-yolo.git
cd digisteel-yolo
pip install -r requirements.txt
pip install -e .[dev]
```

### 2. Create a feature branch

```bash
git checkout develop
git pull
git checkout -b feat/hazem-wp1-test-first-commit
```

### 3. Make a small change

```bash
echo "# Test commit from Hazem" >> README.md
```

### 4. Commit and push

```bash
git add README.md
git commit -m "Test: First commit from Hazem"
git push -u origin feat/hazem-wp1-test-first-commit
```

### 5. Open a Pull Request

On GitHub, click **Compare & pull request** → Create PR to `develop`

### 6. Wait for CI to pass

GitHub Actions should run:
- Linting ✅
- Tests ✅
- Smoke test ✅

### 7. Approve & merge

Once CI passes, ask another team member to review. Merge via **Squash and merge** when ready.

### 8. Tag the first milestone

```bash
git checkout develop
git pull
git tag v0.0-skeleton
git push origin v0.0-skeleton
```

---

## Ongoing Friday Integration Workflow

Every Friday at 11 AM (example time):

1. **Hazem** pulls all feature branches
2. **Hazem** merges them one by one into `develop`
3. **Hazem** runs smoke test locally
4. If pass: Tag `v0.X-week-Y` and announce on team channel
5. If fail: Revert the offending PR and ask author to fix

```bash
git checkout develop
git pull
git merge feat/youssef-wp2-datasets --squash
git merge feat/mohamed-wp3-robustness --squash
# etc.

bash scripts/run_all.sh  # Smoke test

# If all pass:
git tag v0.1-week-1
git push origin develop --tags
```

---

## Common Gotchas

| Problem | Solution |
|---|---|
| "Permission denied (publickey)" when pushing | Add your SSH key to GitHub (Settings → SSH and GPG keys) |
| "Branch protection rule prevents merge" | Ensure all CI checks pass + approvals are met |
| "Merge conflict" | Rebase your branch: `git rebase develop`, resolve conflicts, force-push (only to your feature branch) |
| "Main branch broken" | Never push directly to main; use PRs. Only Hazem can merge. |
| "Forgot which branch I'm on" | `git branch -v` to see all branches and current one |

---

## Useful GitHub Commands

```bash
# List all branches
git branch -a

# Delete a local branch
git branch -D feat/old-branch

# Delete a remote branch
git push origin --delete feat/old-branch

# Create a PR from CLI (requires gh CLI tool)
gh pr create --base develop --head feat/my-feature --title "[WP1] My title" --body "Description"

# View PR status
gh pr view feat/my-feature

# Merge via CLI
gh pr merge feat/my-feature --squash
```

---

## Phase 1 Milestone Tags

Tag these at the end of each phase:

```
v0.0-skeleton              (Week 0: Initial repo structure)
v0.1-arch-freeze           (Week 4: A2 + A3 stable)
v0.2-results-freeze        (Week 8: All training complete)
v1.0-submission            (Week 12: Final submission)
```

---

## Questions?

- **GitHub-specific:** See https://docs.github.com/en/github
- **Team process:** Ask Hazem (team lead)
- **Supervisor questions:** Email Dr. Tarek Ghoneimy

---

**Ready to start?**

```bash
# 1. Clone
git clone https://github.com/<your-team>/digisteel-yolo.git

# 2. Set up environment
cd digisteel-yolo
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt

# 3. Create your feature branch
git checkout -b feat/<your-name>-<task>

# 4. Start Week 1 work!
```

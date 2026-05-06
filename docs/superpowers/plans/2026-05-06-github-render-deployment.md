# GitHub + Render Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Organise the repo, commit all pending changes including the trained model, push to a new public GitHub repo, and prepare the Render config for Docker deployment.

**Architecture:** The 6MB `stock_model.joblib` is committed directly to the repo (within GitHub's file limit). Render clones the repo and Docker copies the model in at build time — no runtime download needed. `render.yaml` is simplified accordingly.

**Tech Stack:** Git, GitHub CLI (`gh`), Docker, Render (Docker web service)

---

### Task 1: Remove `stock_model.joblib` from `.gitignore`

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Open `.gitignore` and remove the `stock_model.joblib` line**

Current content of `.gitignore`:
```
stock_model.joblib
__pycache__/
*.pyc
*.pyo
venv/
.env
.DS_Store
.pytest_cache/
tests/__pycache__/
.coverage
htmlcov/
.claude/
scratch/
data/*.parquet
data/*.feather
data/*.csv
```

Updated content (remove first line):
```
__pycache__/
*.pyc
*.pyo
venv/
.env
.DS_Store
.pytest_cache/
tests/__pycache__/
.coverage
htmlcov/
.claude/
scratch/
data/*.parquet
data/*.feather
data/*.csv
```

- [ ] **Step 2: Verify `stock_model.joblib` is now visible to git**

Run:
```bash
git status
```
Expected: `stock_model.joblib` now appears under "Untracked files" (not ignored).

---

### Task 2: Simplify `render.yaml` — remove MODEL_URL

**Files:**
- Modify: `render.yaml`

- [ ] **Step 1: Update `render.yaml` to remove the env var block**

Replace the entire file with:
```yaml
services:
  - type: web
    name: alphapulse
    runtime: docker
    dockerfilePath: ./Dockerfile
    dockerContext: .
```

- [ ] **Step 2: Verify**

Run:
```bash
cat render.yaml
```
Expected: 6-line YAML with no `envVars` block.

---

### Task 3: Commit all changes in one cleanup commit

**Files:**
- Stage: `.gitignore`, `render.yaml`, `app.py`, `data/5starlist.csv` (delete), `data/.gitkeep`, `stock_model.joblib`, `docs/superpowers/plans/2026-05-06-feature-expansion-lgbm.md`, `docs/superpowers/specs/2026-05-06-github-render-deployment-design.md`, `docs/superpowers/plans/2026-05-06-github-render-deployment.md`

- [ ] **Step 1: Stage all changes**

Run:
```bash
git add .gitignore render.yaml app.py data/.gitkeep stock_model.joblib \
  docs/superpowers/plans/2026-05-06-feature-expansion-lgbm.md \
  docs/superpowers/specs/2026-05-06-github-render-deployment-design.md \
  docs/superpowers/plans/2026-05-06-github-render-deployment.md
git rm data/5starlist.csv
```

- [ ] **Step 2: Verify staging area looks correct**

Run:
```bash
git status
```
Expected output (staged):
```
Changes to be committed:
  modified:   .gitignore
  modified:   app.py
  deleted:    data/5starlist.csv
  new file:   data/.gitkeep
  new file:   docs/superpowers/plans/2026-05-06-feature-expansion-lgbm.md
  new file:   docs/superpowers/plans/2026-05-06-github-render-deployment.md
  new file:   docs/superpowers/specs/2026-05-06-github-render-deployment-design.md
  modified:   render.yaml
  new file:   stock_model.joblib
```

- [ ] **Step 3: Commit**

Run:
```bash
git commit -m "$(cat <<'EOF'
chore: prepare repo for GitHub and Render deployment

- Remove stock_model.joblib from .gitignore; commit 6MB model directly
- Simplify render.yaml: remove MODEL_URL env var (model is in repo)
- Fix WATCHLIST to use yfinance ticker format (XXXX.HK / XXXX.SI)
- Remove data/5starlist.csv; add data/.gitkeep
- Add feature expansion plan and deployment design docs

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

Expected: `master (root-commit) xxxxxxx chore: prepare repo for GitHub and Render deployment`

---

### Task 4: Create GitHub repo and push

**Prerequisites:** `gh` CLI installed and authenticated (`gh auth status` should show a logged-in account).

- [ ] **Step 1: Check `gh` is authenticated**

Run:
```bash
gh auth status
```
Expected: shows `Logged in to github.com as simgsr`

- [ ] **Step 2: Create the public repo and push**

Run:
```bash
gh repo create alphapulse --public --source=. --remote=origin --push
```
Expected output includes:
```
✓ Created repository simgsr/alphapulse on GitHub
✓ Added remote https://github.com/simgsr/alphapulse.git
✓ Pushed commits to https://github.com/simgsr/alphapulse.git
```

- [ ] **Step 3: Verify remote and branch are set up**

Run:
```bash
git remote -v && git log --oneline -3
```
Expected: `origin` pointing to `https://github.com/simgsr/alphapulse.git`

---

### Task 5: Verify GitHub repo contents

- [ ] **Step 1: Open the repo in the browser to confirm files are present**

Run:
```bash
gh repo view simgsr/alphapulse --web
```
Expected: browser opens to the GitHub repo showing `app.py`, `stock_model.joblib`, `render.yaml`, `Dockerfile`, `requirements.txt`, `static/`, `tests/`.

---

### Task 6: Connect to Render (manual steps — instructions only)

This task is performed in the Render dashboard — no CLI commands.

- [ ] **Step 1:** Go to [https://render.com](https://render.com) and log in.
- [ ] **Step 2:** Click **"New +"** → **"Web Service"**.
- [ ] **Step 3:** Connect your GitHub account if not already connected.
- [ ] **Step 4:** Search for and select the `alphapulse` repo.
- [ ] **Step 5:** Render will detect `render.yaml` and pre-fill the config. Confirm:
  - **Name:** `alphapulse`
  - **Runtime:** Docker
  - **Dockerfile path:** `./Dockerfile`
- [ ] **Step 6:** Click **"Create Web Service"**. Render will clone the repo and build the Docker image (~3-5 minutes for first deploy).
- [ ] **Step 7:** Once deploy is green, open the service URL and confirm `/predict/9988.HK` returns a JSON response.

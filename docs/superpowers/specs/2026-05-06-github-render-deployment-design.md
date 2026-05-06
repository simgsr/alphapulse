# GitHub + Render Deployment Design

**Date:** 2026-05-06
**Project:** AlphaPulse — HKEX stock signal app (FastAPI + LightGBM)

## Goal

Organise the working tree, push to a new public GitHub repo, and prepare for Docker-based deployment on Render.

## Decisions

- **Model hosting:** `stock_model.joblib` (6MB) committed directly to the repo. Within GitHub's 100MB file limit; no Git LFS needed. Render clones the repo and the Docker build copies the model in.
- **`render.yaml`:** Remove the `MODEL_URL` env var — no runtime download required.
- **Data files:** `data/*.csv` stays gitignored (ticker CSVs are local training inputs). `data/.gitkeep` preserves the directory in the repo.
- **Commit strategy:** Single cleanup commit covering all pending changes.

## Changes

| File | Action |
|---|---|
| `.gitignore` | Remove `stock_model.joblib` line; keep `data/*.csv` |
| `render.yaml` | Remove `MODEL_URL` env var |
| `app.py` | Already fixed (yfinance ticker format + EOF) |
| `data/5starlist.csv` | Delete from tracking |
| `data/.gitkeep` | Add |
| `stock_model.joblib` | Add (6MB model) |
| `docs/superpowers/plans/2026-05-06-feature-expansion-lgbm.md` | Add |

## GitHub

- New public repo: `alphapulse` under `simgsr`
- Remote: `origin`
- Branch: `master`
- Command: `gh repo create alphapulse --public --source=. --remote=origin --push`

## Render

- Service type: Web (Docker)
- Config picked up from `render.yaml` automatically
- Manual step after push: connect repo in Render dashboard and trigger first deploy

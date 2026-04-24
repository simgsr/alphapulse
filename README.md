---
title: AlphaPulse Stock Predictor
emoji: 📈
colorFrom: green
colorTo: green
sdk: docker
app_port: 8000
---

# AlphaPulse Deployment Guide

This folder contains all the files needed to deploy the **AlphaPulse Stock Prediction AI** to [Render](https://render.com) or [Hugging Face Spaces](https://huggingface.co/spaces).

## Structure
- `app.py`: FastAPI backend that handles predictions and serves the frontend.
- `get_price_data.py`: Improved data fetching and feature engineering module.
- `stock_model.joblib`: The trained Random Forest model.
- `static/`: Frontend assets (HTML, CSS, JS).
- `requirements.txt`: Python dependencies.
- `Dockerfile`: Container configuration for easy deployment.

## Deployment Instructions

### 1. Deploy to Render
1. Create a new **Web Service** on Render.
2. Connect your GitHub repository containing these files.
3. Select **Docker** as the Runtime.
4. Render will automatically build the container and deploy it.

### 2. Deploy to Hugging Face Spaces
1. Create a new **Space**.
2. Select **Docker** as the SDK.
3. Upload all files in this folder to the Space repository.
4. Hugging Face will build and start the app automatically.

## Local Testing
To run locally:
```bash
pip install -r requirements.txt
python app.py
```
Then visit `http://localhost:8000`.

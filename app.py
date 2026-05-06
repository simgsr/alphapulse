from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import joblib
import os
import time
import urllib.request
import numpy as np
from get_price_data import fetch_latest_data, calculate_technical_indicators

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SIGNAL_MAP = {
    1: "UP > 3%",
    0: "STABLE",
    -1: "DOWN > 3%",
}

# Screener watchlist — yfinance format (XXXX.HK / XXXX.SI)
WATCHLIST = [
    # Singapore Stocks (.SI)
    "U96.SI", "HLPD.SI", "H18.SI", "OV8.SI", "EB5.SI", "D05.SI",

    # Hong Kong Stocks (.HK)
    "1698.HK", "1882.HK", "1997.HK", "2196.HK", "2313.HK", 
    "2331.HK", "3606.HK", "6110.HK", "6169.HK", "6178.HK", 
    "6690.HK", "6936.HK", "9899.HK", "9987.HK", "9988.HK", 
    "9999.HK", "1398.HK", "2525.HK", "0151.HK", "0168.HK", 
    "0382.HK", "0700.HK", "1513.HK", "2176.HK", "2669.HK", 
    "3660.HK", "3888.HK", "1171.HK", "1179.HK", "1969.HK", 
    "2648.HK", "2660.HK", "6826.HK", "9618.HK", "9979.HK", 
    "9922.HK", "3686.HK", "1876.HK", "1760.HK", "0839.HK", 
    "3088.HK", "2801.HK", "2839.HK", "3403.HK", "3188.HK", 
    "2822.HK"
]

_scan_cache: dict = {"data": None, "ts": 0.0}
SCAN_TTL = 3600  # seconds

_here = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_here, 'stock_model.joblib')

# Fallback to /tmp for read-only filesystems (HF Spaces)
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = '/tmp/stock_model.joblib'

if not os.path.exists(MODEL_PATH):
    _hf_repo = os.environ.get('HF_MODEL_REPO')
    _model_url = os.environ.get('MODEL_URL')
    if _hf_repo:
        from huggingface_hub import hf_hub_download
        print(f"Downloading model from HF Hub: {_hf_repo}", flush=True)
        MODEL_PATH = hf_hub_download(
            repo_id=_hf_repo,
            filename='stock_model.joblib',
            cache_dir='/tmp/hf_cache',
            token=os.environ.get('HF_TOKEN'),
        )
        print("Model download complete.", flush=True)
    elif _model_url:
        print("Downloading model from MODEL_URL…", flush=True)
        urllib.request.urlretrieve(_model_url, MODEL_PATH)
        print("Model download complete.", flush=True)

try:
    model_data = joblib.load(MODEL_PATH)
    model = model_data['model']
    feature_names = model_data['features']
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    feature_names = []


def build_prediction_response(ticker: str, mdl, features_array: np.ndarray, raw_data) -> dict:
    """Pure function: run inference and build the API response dict."""
    prediction = int(mdl.predict(features_array)[0])
    probabilities = mdl.predict_proba(features_array)[0].tolist()
    classes = mdl.classes_.tolist()
    prob_dict = {str(int(c)): round(p, 4) for c, p in zip(classes, probabilities)}

    confidence_up_3pct = round(prob_dict.get('1', 0.0), 4)
    confidence_down_3pct = round(prob_dict.get('-1', 0.0), 4)

    p_down = confidence_down_3pct
    edge_ratio = round(confidence_up_3pct / p_down, 2) if p_down > 0 else 99.0

    return {
        "ticker": ticker,
        "prediction": prediction,
        "signal": SIGNAL_MAP.get(prediction, "UNKNOWN"),
        "confidence_up_3pct": confidence_up_3pct,
        "confidence_down_3pct": confidence_down_3pct,
        "edge_ratio": edge_ratio,
        "probabilities": prob_dict,
        "current_price": float(raw_data['Adj_Close'].iloc[-1]),
        "last_updated": str(raw_data.index[-1].date()),
    }


def rank_scan_results(results: list, n: int = 5) -> list:
    """Return top-n results sorted by edge_ratio descending."""
    return sorted(results, key=lambda r: r["edge_ratio"], reverse=True)[:n]


@app.get("/predict/{ticker}")
async def predict(ticker: str):
    ticker = ticker.upper()
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded on server")

    data = fetch_latest_data(ticker)
    if data is None:
        raise HTTPException(status_code=404, detail=f"No data found for {ticker}")

    processed = calculate_technical_indicators(data)
    if processed.empty:
        raise HTTPException(status_code=400, detail="Insufficient data for analysis")

    latest_features = processed[feature_names].iloc[-1:].values
    return build_prediction_response(ticker, model, latest_features, data)


@app.get("/scan")
async def scan():
    global _scan_cache
    if _scan_cache["data"] is not None and (time.time() - _scan_cache["ts"]) < SCAN_TTL:
        return _scan_cache["data"]

    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded on server")

    results = []
    for ticker in WATCHLIST:
        try:
            data = fetch_latest_data(ticker)
            if data is None:
                continue
            processed = calculate_technical_indicators(data)
            if processed.empty:
                continue
            latest_features = processed[feature_names].iloc[-1:].values
            results.append(build_prediction_response(ticker, model, latest_features, data))
        except Exception:
            continue

    top5 = rank_scan_results(results, n=5)
    _scan_cache = {"data": top5, "ts": time.time()}
    return top5


app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

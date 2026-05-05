from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import joblib
import os
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
    2: "UP > 5%",
    1: "UP 3-5%",
    0: "STABLE",
    -1: "DOWN 3-5%",
    -2: "DOWN > 5%",
}

MODEL_PATH = 'stock_model.joblib'
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(os.path.dirname(__file__), 'stock_model.joblib')

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

    confidence_up_3pct = round(prob_dict.get('1', 0.0) + prob_dict.get('2', 0.0), 4)
    confidence_up_5pct = round(prob_dict.get('2', 0.0), 4)

    return {
        "ticker": ticker,
        "prediction": prediction,
        "signal": SIGNAL_MAP.get(prediction, "UNKNOWN"),
        "confidence_up_3pct": confidence_up_3pct,
        "confidence_up_5pct": confidence_up_5pct,
        "probabilities": prob_dict,
        "current_price": float(raw_data['Adj_Close'].iloc[-1]),
        "last_updated": str(raw_data.index[-1].date()),
    }


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


app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

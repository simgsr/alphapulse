from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import joblib
import os
from get_price_data import fetch_latest_data, calculate_technical_indicators

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
MODEL_PATH = 'stock_model.joblib'
if not os.path.exists(MODEL_PATH):
    # Try parent directory if not in current
    MODEL_PATH = os.path.join(os.path.dirname(__file__), 'stock_model.joblib')

try:
    model_data = joblib.load(MODEL_PATH)
    model = model_data['model']
    feature_names = model_data['features']
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.get("/predict/{ticker}")
async def predict(ticker: str):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded on server")
        
    data = fetch_latest_data(ticker)
    if data is None:
        raise HTTPException(status_code=404, detail=f"No data found for {ticker}")
        
    processed = calculate_technical_indicators(data)
    if processed.empty:
        raise HTTPException(status_code=400, detail="Insufficient data for analysis")
    
    latest_features = processed[feature_names].iloc[-1:].values
    
    prediction = int(model.predict(latest_features)[0])
    probabilities = model.predict_proba(latest_features)[0].tolist()
    
    classes = model.classes_.tolist()
    prob_dict = {str(c): p for c, p in zip(classes, probabilities)}
    
    signal = "STABLE/NEUTRAL"
    if prediction == 1: signal = "UP (>3%)"
    elif prediction == -1: signal = "DOWN (>3%)"
    
    return {
        "ticker": ticker.upper(),
        "prediction": prediction,
        "signal": signal,
        "probabilities": prob_dict,
        "current_price": float(data['Adj_Close'].iloc[-1]),
        "last_updated": str(data.index[-1].date())
    }

# Serve static files
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

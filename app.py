"""
Volatility Forecasting Engine (FastAPI)
------------------------------------------------------
Live production API. On each request:
  1. Pulls fresh price data from yfinance
  2. Computes returns + rolling volatility
  3. Loads the trained LSTM model + scalers
  4. Forecasts next-period volatility
  5. Returns it as JSON for the website to consume

Run locally with:
    uvicorn app:app --reload --port 8000

Then test at:
    http://localhost:8000/forecast-volatility?ticker=^NSEI
"""

import time
import traceback
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tensorflow import keras

SEQUENCE_LENGTH = 30
ROLLING_WINDOW = 21
TRADING_DAYS_PER_YEAR = 252
LOOKBACK_DAYS = 90  # extra buffer so rolling vol + sequence window both have enough data

# ---- Load trained model + scalers once at startup (not per-request — this is important for speed) ----
model = keras.models.load_model("lstm_volatility_model.keras")
return_scaler = joblib.load("return_scaler.pkl")
vol_scaler = joblib.load("vol_scaler.pkl")

app = FastAPI(title="Volatility Forecasting Engine")

# Allow your website (running on a different domain/port) to call this API from the browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your actual website domain once deployed
    allow_methods=["GET"],
    allow_headers=["*"],
)

# Simple in-memory cache so multiple users within a short window don't all trigger
# separate Yahoo Finance calls (protects you from rate limiting under real traffic)
_cache = {}
CACHE_SECONDS = 300  # 5 minutes


@app.get("/forecast-volatility")
def forecast_volatility(ticker: str = "^NSEI"):
    now = time.time()

    # ---- Serve from cache if recent enough ----
    if ticker in _cache and (now - _cache[ticker]["timestamp"]) < CACHE_SECONDS:
        return _cache[ticker]["result"]

    # ---- 1. Fetch fresh data ----
    try:
        end_date = datetime.today()
        start_date = end_date - timedelta(days=LOOKBACK_DAYS * 2)  # extra buffer for weekends/holidays
        data = yf.download(ticker, start=start_date, end=end_date, interval="1d", progress=False, auto_adjust=True)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch data: {e}")

    if data.empty or len(data) < SEQUENCE_LENGTH + ROLLING_WINDOW:
        raise HTTPException(
            status_code=502,
            detail=f"Not enough data returned for {ticker} to compute a forecast."
        )

    # yfinance sometimes returns MultiIndex columns (ticker, field) even for a single ticker
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    try:
        close = data["Close"]
        if isinstance(close, pd.DataFrame):  # can happen if columns weren't fully flattened
            close = close.iloc[:, 0]
        close = close.reset_index(drop=True)

        # ---- 2. Compute returns and rolling volatility ----
        log_returns = np.log(close / close.shift(1))
        rolling_vol = log_returns.rolling(window=ROLLING_WINDOW).std() * np.sqrt(TRADING_DAYS_PER_YEAR)

        df = pd.DataFrame({"LogReturn": log_returns, "RollingVol": rolling_vol}).dropna()

        if len(df) < SEQUENCE_LENGTH:
            raise HTTPException(
                status_code=502,
                detail="Not enough valid rows after computing returns/volatility."
            )

        # ---- 3. Take the most recent SEQUENCE_LENGTH days and scale using the TRAINING-fitted scalers ----
        recent = df.tail(SEQUENCE_LENGTH)
        r_scaled = return_scaler.transform(recent[["LogReturn"]]).flatten()
        v_scaled = vol_scaler.transform(recent[["RollingVol"]]).flatten()

        X = np.stack([r_scaled, v_scaled], axis=1).reshape(1, SEQUENCE_LENGTH, 2)

        # ---- 4. Predict and inverse-transform back to real volatility ----
        pred_scaled = model.predict(X, verbose=0)
        predicted_vol = float(vol_scaler.inverse_transform(pred_scaled).flatten()[0])

        result = {
            "ticker": ticker,
            "as_of_date": str(data.index[-1].date()),
            "last_close": round(float(close.iloc[-1]), 2),
            "predicted_volatility": round(predicted_vol, 4),
            "predicted_volatility_pct": f"{predicted_vol * 100:.2f}%",
            "current_realized_vol": round(float(df['RollingVol'].iloc[-1]), 4),
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()  # full traceback prints in the uvicorn terminal
        raise HTTPException(status_code=500, detail=f"Processing error: {e}")

    _cache[ticker] = {"result": result, "timestamp": now}
    return result


@app.get("/health")
def health():
    return {"status": "ok"}
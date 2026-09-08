#!/usr/bin/env python3
"""
BTC Autonomous Paper Trading Worker + Status API
================================================
- Runs the aggressive long-biased trend strategy continuously
- Exposes a simple HTTP endpoint so a local dashboard can read live state

Deploy on Railway / Fly.io.
Important: The trading loop starts automatically even when run under gunicorn.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from flask import Flask, jsonify
import json
import time
import threading
from datetime import datetime, timezone
from pathlib import Path
import os
import warnings
warnings.filterwarnings("ignore")

# ====================== STRATEGY CONFIG ======================
SYMBOL = "BTC-USD"
INTERVAL = "4h"
EMA_FAST, EMA_SLOW, EMA_FILTER = 12, 26, 50
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
RSI_LEN, RSI_LONG_MIN = 14, 48
ATR_LEN, ATR_STOP_MULT = 14, 2.0
TRAIL_ACTIVATE_R, TRAIL_ATR_MULT = 0.8, 1.6
RISK_PCT = 0.015
FEE_RT = 0.0004
INITIAL_EQUITY = 10_000.0
CHECK_INTERVAL = 300  # seconds

STATE_FILE = Path("paper_state.json")

# ====================== STATE ======================
def default_state():
    return {
        "equity": INITIAL_EQUITY,
        "position": 0,
        "entry_price": 0.0,
        "stop_price": 0.0,
        "orig_stop_pct": 0.0,
        "bars_held": 0,
        "last_bar_time": None,
        "trades": [],
        "equity_curve": [{"time": datetime.now(timezone.utc).isoformat(), "equity": INITIAL_EQUITY}],
        "last_update": datetime.now(timezone.utc).isoformat(),
        "status": "starting"
    }

def load_state():
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return default_state()

def save_state(state):
    state["last_update"] = datetime.now(timezone.utc).isoformat()
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

# ====================== STRATEGY LOGIC ======================
def fetch_and_prepare():
    df = yf.download(SYMBOL, period="60d", interval=INTERVAL, progress=False)
    if df.empty:
        return None
    df = df.dropna()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]

    df["ema_fast"] = EMAIndicator(df["close"], EMA_FAST).ema_indicator()
    df["ema_slow"] = EMAIndicator(df["close"], EMA_SLOW).ema_indicator()
    df["ema_filter"] = EMAIndicator(df["close"], EMA_FILTER).ema_indicator()

    macd = MACD(df["close"], window_slow=MACD_SLOW, window_fast=MACD_FAST, window_sign=MACD_SIGNAL)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()
    df["macd_hist_prev"] = df["macd_hist"].shift(1)

    df["rsi"] = RSIIndicator(df["close"], RSI_LEN).rsi()
    df["atr"] = AverageTrueRange(df["high"], df["low"], df["close"], ATR_LEN).average_true_range()
    return df.dropna()

def get_signal(row):
    uptrend = (row["ema_fast"] > row["ema_slow"] > row["ema_filter"]) and (row["close"] > row["ema_fast"])
    macd_bull = (row["macd"] > row["macd_signal"]) and (row["macd_hist"] > row["macd_hist_prev"]) and (row["macd_hist"] > 0)
    rsi_ok = row["rsi"] >= RSI_LONG_MIN
    return bool(uptrend and macd_bull and rsi_ok)

def run_once(state):
    df = fetch_and_prepare()
    if df is None or len(df) < 5:
        state["status"] = "data_error"
        return state

    latest = df.iloc[-1]
    bar_time = str(latest.name)
    price = float(latest["close"])
    atr = float(latest["atr"])

    if state.get("last_bar_time") == bar_time:
        state["status"] = "waiting_for_new_bar"
        state["last_price"] = price
        return state

    # Manage open position
    if state["position"] == 1:
        state["bars_held"] = state.get("bars_held", 0) + 1
        entry = state["entry_price"]
        stop = state["stop_price"]

        r_mult = (price - entry) / (entry - stop) if entry > stop else 0
        if r_mult >= TRAIL_ACTIVATE_R:
            new_stop = price - atr * TRAIL_ATR_MULT
            if new_stop > stop:
                state["stop_price"] = new_stop

        exit_now, reason = False, ""
        if price <= state["stop_price"]:
            exit_now, reason = True, "stop"
        elif latest["ema_fast"] < latest["ema_slow"]:
            exit_now, reason = True, "trend_break"
        elif state["bars_held"] >= 80:
            exit_now, reason = True, "time"

        if exit_now:
            raw = (price - entry) / entry
            pnl = raw - FEE_RT
            risk_d = state["equity"] * RISK_PCT
            size = risk_d / state["orig_stop_pct"] if state.get("orig_stop_pct", 0) > 1e-8 else 0
            dollar = size * pnl
            state["equity"] += dollar

            trade = {
                "time": bar_time,
                "side": "long",
                "entry": round(entry, 2),
                "exit": round(price, 2),
                "pnl_pct": round(pnl * 100, 3),
                "dollar": round(dollar, 2),
                "reason": reason,
                "bars": state["bars_held"],
                "equity_after": round(state["equity"], 2)
            }
            state.setdefault("trades", []).append(trade)
            state["position"] = 0
            state["bars_held"] = 0

    # New entry
    if state["position"] == 0:
        if get_signal(latest) and atr > 0:
            stop_dist = atr * ATR_STOP_MULT
            state["position"] = 1
            state["entry_price"] = price
            state["stop_price"] = price - stop_dist
            state["orig_stop_pct"] = stop_dist / price
            state["bars_held"] = 0

    state["last_bar_time"] = bar_time
    state.setdefault("equity_curve", []).append({
        "time": datetime.now(timezone.utc).isoformat(),
        "equity": state["equity"]
    })
    if len(state["equity_curve"]) > 3000:
        state["equity_curve"] = state["equity_curve"][-2000:]

    state["status"] = "ok"
    state["last_price"] = price
    return state

# ====================== BACKGROUND WORKER ======================
def worker_loop():
    print("Worker loop started", flush=True)
    while True:
        try:
            state = load_state()
            state = run_once(state)
            save_state(state)
            pos = "LONG" if state["position"] == 1 else "FLAT"
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {pos} | Equity ${state['equity']:.2f} | Trades {len(state.get('trades', []))} | status={state.get('status')}", flush=True)
        except Exception as e:
            print(f"Worker error: {e}", flush=True)
        time.sleep(CHECK_INTERVAL)

# ====================== FLASK STATUS API ======================
app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({
        "service": "BTC Paper Trader",
        "endpoints": {
            "/status": "Full current state",
            "/health": "Simple health check"
        }
    })

@app.route("/health")
def health():
    return jsonify({"status": "alive", "time": datetime.now(timezone.utc).isoformat()})

@app.route("/status")
def status():
    state = load_state()
    return jsonify(state)

# ====================== START WORKER THREAD (works with gunicorn too) ======================
_worker_started = False
_worker_lock = threading.Lock()

def start_worker_once():
    global _worker_started
    with _worker_lock:
        if not _worker_started:
            t = threading.Thread(target=worker_loop, daemon=True)
            t.start()
            _worker_started = True
            print("Background trading thread launched", flush=True)

# Start the trading loop as soon as this module is imported
# (this is what makes it work under gunicorn)
start_worker_once()

# ====================== MAIN (for local python btc_worker.py) ======================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting status server on port {port}", flush=True)
    app.run(host="0.0.0.0", port=port)

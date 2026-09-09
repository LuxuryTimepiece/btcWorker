#!/usr/bin/env python3
"""
BTC Autonomous Paper Trading Worker + Status API
================================================
- Aggressive long-biased 4H trend strategy
- Regime Filter (daily EMA50) – only trade in friendly/mixed regimes
- Risk Governor (drawdown + consecutive losses) – pause when risk is off
- Exposes /status and /health for the local dashboard and daily Agent

Deploy on Railway / Fly.io.
Trading loop starts automatically even under gunicorn.
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

# Regime + Risk Governor
DAILY_EMA_LEN = 50
MAX_DRAWDOWN_PCT = 0.08          # 8% from peak → RISK-OFF
MAX_CONSECUTIVE_LOSSES = 3

STATE_FILE = Path("paper_state.json")

# ====================== STATE ======================
def default_state():
    return {
        "equity": INITIAL_EQUITY,
        "peak_equity": INITIAL_EQUITY,
        "position": 0,
        "entry_price": 0.0,
        "stop_price": 0.0,
        "orig_stop_pct": 0.0,
        "bars_held": 0,
        "last_bar_time": None,
        "trades": [],
        "equity_curve": [{"time": datetime.now(timezone.utc).isoformat(), "equity": INITIAL_EQUITY}],
        "last_update": datetime.now(timezone.utc).isoformat(),
        "status": "starting",
        "regime": "unknown",
        "risk_state": "unknown",
        "agent_decision": "unknown",
        "last_price": None,
    }

def load_state():
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                state = json.load(f)
            # Ensure new keys exist after upgrades
            defaults = default_state()
            for k, v in defaults.items():
                if k not in state:
                    state[k] = v
            return state
        except Exception:
            pass
    return default_state()

def save_state(state):
    state["last_update"] = datetime.now(timezone.utc).isoformat()
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

# ====================== DATA HELPERS ======================
def _normalize_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    return df

def fetch_4h():
    df = yf.download(SYMBOL, period="60d", interval=INTERVAL, progress=False)
    if df.empty:
        return None
    df = _normalize_columns(df.dropna())
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

def fetch_daily():
    df = yf.download(SYMBOL, period="1y", interval="1d", progress=False)
    if df.empty:
        return None
    df = _normalize_columns(df.dropna())
    df["ema50"] = EMAIndicator(df["close"], DAILY_EMA_LEN).ema_indicator()
    return df.dropna()

# ====================== REGIME FILTER ======================
def evaluate_regime():
    """
    Returns (regime_str, detail_dict)
    FRIENDLY = price > daily EMA50 and EMA50 not clearly falling
    HOSTILE  = price < daily EMA50 or EMA50 clearly falling
    MIXED    = everything else
    """
    df = fetch_daily()
    if df is None or len(df) < DAILY_EMA_LEN + 5:
        return "unknown", {"error": "insufficient daily data"}

    last = df.iloc[-1]
    prev = df.iloc[-6] if len(df) >= 6 else df.iloc[0]  # ~1 week lookback for slope

    price = float(last["close"])
    ema50 = float(last["ema50"])
    ema50_prev = float(prev["ema50"])
    slope_up = ema50 >= ema50_prev

    above = price > ema50

    if above and slope_up:
        regime = "FRIENDLY"
    elif (not above) or (not slope_up and price < ema50 * 0.98):
        regime = "HOSTILE"
    else:
        regime = "MIXED"

    detail = {
        "daily_close": round(price, 2),
        "daily_ema50": round(ema50, 2),
        "ema50_slope_up": slope_up,
        "price_above_ema50": above,
    }
    return regime, detail

# ====================== RISK GOVERNOR ======================
def evaluate_risk(state):
    """
    Returns (risk_state_str, detail_dict)
    RISK-OFF if drawdown >= 8% or >= 3 consecutive losses
    """
    equity = float(state.get("equity", INITIAL_EQUITY))
    peak = float(state.get("peak_equity", equity))
    if equity > peak:
        peak = equity
        state["peak_equity"] = peak

    drawdown = (peak - equity) / peak if peak > 0 else 0.0

    trades = state.get("trades", [])
    consecutive_losses = 0
    for t in reversed(trades):
        if t.get("pnl_pct", 0) < 0:
            consecutive_losses += 1
        else:
            break

    risk_off = (drawdown >= MAX_DRAWDOWN_PCT) or (consecutive_losses >= MAX_CONSECUTIVE_LOSSES)
    risk_state = "RISK-OFF" if risk_off else "RISK-ON"

    detail = {
        "equity": round(equity, 2),
        "peak_equity": round(peak, 2),
        "drawdown_pct": round(drawdown * 100, 2),
        "consecutive_losses": consecutive_losses,
    }
    return risk_state, detail

# ====================== ENTRY SIGNAL ======================
def get_signal(row):
    uptrend = (row["ema_fast"] > row["ema_slow"] > row["ema_filter"]) and (row["close"] > row["ema_fast"])
    macd_bull = (
        row["macd"] > row["macd_signal"]
        and row["macd_hist"] > row["macd_hist_prev"]
        and row["macd_hist"] > 0
    )
    rsi_ok = row["rsi"] >= RSI_LONG_MIN
    return bool(uptrend and macd_bull and rsi_ok)

# ====================== MAIN LOGIC ======================
def run_once(state):
    df = fetch_4h()
    if df is None or len(df) < 5:
        state["status"] = "data_error"
        return state

    latest = df.iloc[-1]
    bar_time = str(latest.name)
    price = float(latest["close"])
    atr = float(latest["atr"])

    # Always refresh regime + risk (even on same bar)
    regime, regime_detail = evaluate_regime()
    risk_state, risk_detail = evaluate_risk(state)
    state["regime"] = regime
    state["risk_state"] = risk_state
    state["regime_detail"] = regime_detail
    state["risk_detail"] = risk_detail

    # Hard decision
    allow_new_entries = (regime in ("FRIENDLY", "MIXED")) and (risk_state == "RISK-ON")
    state["agent_decision"] = "CONTINUE" if allow_new_entries else "PAUSE"

    if state.get("last_bar_time") == bar_time:
        state["status"] = "waiting_for_new_bar"
        state["last_price"] = price
        return state

    # ----- Manage open position (always allowed to exit) -----
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

            # Update peak after equity change
            if state["equity"] > state.get("peak_equity", state["equity"]):
                state["peak_equity"] = state["equity"]

            trade = {
                "time": bar_time,
                "side": "long",
                "entry": round(entry, 2),
                "exit": round(price, 2),
                "pnl_pct": round(pnl * 100, 3),
                "dollar": round(dollar, 2),
                "reason": reason,
                "bars": state["bars_held"],
                "equity_after": round(state["equity"], 2),
            }
            state.setdefault("trades", []).append(trade)
            state["position"] = 0
            state["bars_held"] = 0

            # Re-evaluate risk after the closed trade
            risk_state, risk_detail = evaluate_risk(state)
            state["risk_state"] = risk_state
            state["risk_detail"] = risk_detail
            allow_new_entries = (regime in ("FRIENDLY", "MIXED")) and (risk_state == "RISK-ON")
            state["agent_decision"] = "CONTINUE" if allow_new_entries else "PAUSE"

    # ----- New entry (only if regime + risk allow) -----
    if state["position"] == 0 and allow_new_entries:
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
        "equity": state["equity"],
    })
    if len(state["equity_curve"]) > 3000:
        state["equity_curve"] = state["equity_curve"][-2000:]

    state["status"] = "ok"
    state["last_price"] = price
    return state

# ====================== BACKGROUND WORKER ======================
def worker_loop():
    print("Worker loop started (with Regime Filter + Risk Governor)", flush=True)
    while True:
        try:
            state = load_state()
            state = run_once(state)
            save_state(state)
            pos = "LONG" if state["position"] == 1 else "FLAT"
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] {pos} | "
                f"Eq ${state['equity']:.2f} | Trades {len(state.get('trades', []))} | "
                f"Regime={state.get('regime')} Risk={state.get('risk_state')} "
                f"Decision={state.get('agent_decision')} | {state.get('status')}",
                flush=True,
            )
        except Exception as e:
            print(f"Worker error: {e}", flush=True)
        time.sleep(CHECK_INTERVAL)

# ====================== FLASK STATUS API ======================
app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({
        "service": "BTC Paper Trader (Regime + Risk Governor)",
        "endpoints": {
            "/status": "Full current state including regime, risk, and agent decision",
            "/health": "Simple health check",
        },
    })

@app.route("/health")
def health():
    return jsonify({"status": "alive", "time": datetime.now(timezone.utc).isoformat()})

@app.route("/status")
def status():
    state = load_state()
    return jsonify(state)

# ====================== START WORKER THREAD ======================
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

start_worker_once()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting status server on port {port}", flush=True)
    app.run(host="0.0.0.0", port=port)

"""
Backtest 026211 (平安科技精选混合C) scoring engine vs. forward returns.
NAV-only signals — no news, no holdings (can't backfill those).
"""
import os, sys, json
from datetime import datetime, timedelta
from collections import defaultdict

for key in list(os.environ.keys()):
    if key.lower().endswith('_proxy'):
        os.environ.pop(key, None)

import akshare as ak
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fund_advisor import (
    compute_rsi, compute_ma_dev, classify_trend, classify_rsi, safe_fetch,
    generate_suggestion, MA_SHORT, MA_MEDIUM, MA_LONG, RSI_PERIOD, NAV_LOOKBACK,
)

CODE = "026211"
NAME = "平安科技精选混合C"
LOOKBACK_DAYS = 120  # fetch 120 days

def analyze_fund_nav_at(df_full: pd.DataFrame, target_idx: int) -> dict:
    """Compute NAV indicators as if today were df_full.iloc[target_idx].

    Uses only data up to target_idx (simulates real-time view).
    """
    # Slice from beginning to target_idx
    df = df_full.iloc[:target_idx + 1].tail(NAV_LOOKBACK).copy()
    # NAV is the second column (index 1)
    nav = pd.to_numeric(df.iloc[:, 1], errors='coerce').dropna()

    if len(nav) < 20:
        return None

    cur = float(nav.iloc[-1])
    prev1 = float(nav.iloc[-2]) if len(nav) >= 2 else cur
    prev5 = float(nav.iloc[-6]) if len(nav) >= 6 else cur
    prev20 = float(nav.iloc[-21]) if len(nav) >= 21 else cur

    ma5 = float(nav.rolling(MA_SHORT).mean().iloc[-1])
    ma20 = float(nav.rolling(MA_MEDIUM).mean().iloc[-1])
    ma60 = float(nav.rolling(MA_LONG).mean().iloc[-1]) if len(nav) >= MA_LONG else ma20

    rsi = compute_rsi(nav)
    trend = classify_trend(cur, ma5, ma20, ma60)

    return {
        "nav_current": round(cur, 4),
        "change_1d": round((cur/prev1 - 1)*100, 2),
        "change_5d": round((cur/prev5 - 1)*100, 2),
        "change_20d": round((cur/prev20 - 1)*100, 2),
        "ma5": round(ma5, 4), "ma20": round(ma20, 4), "ma60": round(ma60, 4),
        "dev_ma5_pct": round(compute_ma_dev(cur, ma5), 2),
        "dev_ma20_pct": round(compute_ma_dev(cur, ma20), 2),
        "dev_ma60_pct": round(compute_ma_dev(cur, ma60), 2),
        "rsi": round(rsi, 1), "rsi_status": classify_rsi(rsi),
        "trend": trend, "data_points": len(nav),
    }


def main():
    print(f"Backtesting {CODE} {NAME}...")
    print("=" * 60)

    # Fetch NAV history
    print("Fetching NAV history...")
    df_raw = safe_fetch(ak.fund_open_fund_info_em, symbol=CODE, indicator="单位净值走势")
    if df_raw is None or df_raw.empty:
        print("Failed to fetch NAV data")
        return

    cols = df_raw.columns.tolist()
    date_col, nav_col = cols[0], cols[1]
    df_raw[date_col] = pd.to_datetime(df_raw[date_col])
    df_raw = df_raw.sort_values(date_col).tail(LOOKBACK_DAYS)
    df_raw[nav_col] = pd.to_numeric(df_raw[nav_col], errors='coerce')
    df_raw = df_raw.dropna(subset=[nav_col])
    df_raw = df_raw.reset_index(drop=True)

    print(f"  Got {len(df_raw)} days, {df_raw[date_col].iloc[0].date()} ~ {df_raw[date_col].iloc[-1].date()}")

    # ── Run scoring for each day ──
    days = len(df_raw)
    results = []
    nav_series = pd.to_numeric(df_raw.iloc[:, 1], errors='coerce')

    for i in range(20, days):  # start from day 20 (need enough lookback)
        nav_data = analyze_fund_nav_at(df_raw, i)
        if nav_data is None:
            continue

        # Score using NAV only, no news/stocks/index
        sug = generate_suggestion(
            nav=nav_data, idx=None, stocks=None, news=None, fund_type="active"
        )
        score = sug["score"]
        action = sug["action"]
        nav_cur = nav_data["nav_current"]
        date = df_raw[date_col].iloc[i]

        # Compute forward returns (if enough future data)
        fwd_3d = None
        fwd_5d = None
        fwd_10d = None
        if i + 3 < days:
            fwd_3d = round((float(nav_series.iloc[i + 3]) / nav_cur - 1) * 100, 2)
        if i + 5 < days:
            fwd_5d = round((float(nav_series.iloc[i + 5]) / nav_cur - 1) * 100, 2)
        if i + 10 < days:
            fwd_10d = round((float(nav_series.iloc[i + 10]) / nav_cur - 1) * 100, 2)

        results.append({
            "date": str(date.date()),
            "nav": nav_cur,
            "score": score,
            "rsi": nav_data["rsi"],
            "trend": nav_data["trend"],
            "dev_ma20": nav_data["dev_ma20_pct"],
            "chg_5d": nav_data["change_5d"],
            "action": action,
            "fwd_3d": fwd_3d,
            "fwd_5d": fwd_5d,
            "fwd_10d": fwd_10d,
        })

    print(f"  Scored {len(results)} trading days")

    # ── Analyze signals ──
    buy_more = [r for r in results if r["action"] == "BUY_MORE"]
    hold = [r for r in results if r["action"] == "HOLD"]
    reduce = [r for r in results if r["action"] == "REDUCE"]

    print(f"\nSignal distribution:")
    print(f"  [BUY]  BUY_MORE: {len(buy_more)} ({100*len(buy_more)/len(results):.0f}%)")
    print(f"  [HOLD] HOLD:     {len(hold)} ({100*len(hold)/len(results):.0f}%)")
    print(f"  [SELL] REDUCE:   {len(reduce)} ({100*len(reduce)/len(results):.0f}%)")

    # Forward returns by signal
    print(f"\n{'='*60}")
    print("Forward Return Analysis")
    print(f"{'='*60}")

    for label, group in [("[BUY] BUY_MORE", buy_more), ("[HOLD] HOLD", hold), ("[SELL] REDUCE", reduce)]:
        if not group:
            continue
        print(f"\n{label} ({len(group)} signals):")
        for horizon, key in [("3-day", "fwd_3d"), ("5-day", "fwd_5d"), ("10-day", "fwd_10d")]:
            vals = [r[key] for r in group if r[key] is not None]
            if vals:
                avg = sum(vals) / len(vals)
                win_rate = sum(1 for v in vals if v > 0) / len(vals) * 100
                print(f"  {horizon}: avg {avg:+.2f}% | win rate {win_rate:.0f}% | n={len(vals)}")

    # ── Show all BUY_MORE signals in detail ──
    if buy_more:
        print(f"\n{'='*60}")
        print("BUY_MORE Signale Details")
        print(f"{'='*60}")
        print(f"{'Date':<12} {'Score':>6} {'RSI':>6} {'Trend':>12} {'MA20dev':>8} {'5dChg':>7} {'Fwd5d':>7} {'Fwd10d':>7}")
        print("-" * 70)
        for r in buy_more:
            print(f"{r['date']:<12} {r['score']:>6.1f} {r['rsi']:>6.1f} {r['trend']:>12} {r['dev_ma20']:>+7.1f}% {r['chg_5d']:>+6.1f}% {r['fwd_5d'] or '-':>7} {r['fwd_10d'] or '-':>7}")

    # ── Show all REDUCE signals ──
    if reduce:
        print(f"\n{'='*60}")
        print("REDUCE Signale Details")
        print(f"{'='*60}")
        print(f"{'Date':<12} {'Score':>6} {'RSI':>6} {'Trend':>12} {'MA20dev':>8} {'5dChg':>7} {'Fwd5d':>7} {'Fwd10d':>7}")
        print("-" * 70)
        for r in reduce:
            print(f"{r['date']:<12} {r['score']:>6.1f} {r['rsi']:>6.1f} {r['trend']:>12} {r['dev_ma20']:>+7.1f}% {r['chg_5d']:>+6.1f}% {r['fwd_5d'] or '-':>7} {r['fwd_10d'] or '-':>7}")


if __name__ == "__main__":
    main()

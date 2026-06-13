"""
Backtest 012700 (易方达证券公司ETF联接C) — index fund tracking 399975.
Has both NAV history AND index history → can run full scoring engine.
"""
import os, sys, json
from datetime import datetime, timedelta

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

CODE = "012700"
INDEX_CODE = "399975"
LOOKBACK_DAYS = 120

def analyze_nav_at(df_full: pd.DataFrame, target_idx: int) -> dict:
    df = df_full.iloc[:target_idx + 1].tail(NAV_LOOKBACK).copy()
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

def analyze_index_at(df_full: pd.DataFrame, target_idx: int) -> dict:
    df = df_full.iloc[:target_idx + 1].tail(100).copy()
    close = pd.to_numeric(df.iloc[:, 4], errors='coerce').dropna()  # close is col 4
    if len(close) < 20:
        return None
    cur = float(close.iloc[-1])
    prev = float(close.iloc[-2]) if len(close) >= 2 else cur
    ma20 = float(close.rolling(MA_MEDIUM).mean().iloc[-1])
    ma60 = float(close.rolling(MA_LONG).mean().iloc[-1]) if len(close) >= MA_LONG else ma20
    rsi = compute_rsi(close)
    chg_5d = float(cur / close.iloc[-6] - 1) * 100 if len(close) >= 6 else 0
    chg_20d = float(cur / close.iloc[-21] - 1) * 100 if len(close) >= 21 else 0
    return {
        "index_current": round(cur, 2),
        "change_1d": round((cur/prev - 1)*100, 2),
        "change_5d": round(chg_5d, 2),
        "change_20d": round(chg_20d, 2),
        "ma20": round(ma20, 2),
        "dev_ma20_pct": round(compute_ma_dev(cur, ma20), 2),
        "rsi": round(rsi, 1), "rsi_status": classify_rsi(rsi),
        "trend": classify_trend(cur, ma20, ma20, ma60),
    }

def main():
    print(f"Backtesting {CODE} (index fund, tracks {INDEX_CODE})")
    print("=" * 60)

    # Fetch NAV
    print("Fetching NAV...")
    df_nav = safe_fetch(ak.fund_open_fund_info_em, symbol=CODE, indicator="单位净值走势")
    df_nav[df_nav.columns[0]] = pd.to_datetime(df_nav[df_nav.columns[0]])
    df_nav = df_nav.sort_values(df_nav.columns[0]).tail(LOOKBACK_DAYS).reset_index(drop=True)
    print(f"  NAV: {len(df_nav)} days, {df_nav[df_nav.columns[0]].iloc[0].date()} ~ {df_nav[df_nav.columns[0]].iloc[-1].date()}")

    # Fetch index
    print(f"Fetching index {INDEX_CODE}...")
    df_idx = safe_fetch(ak.stock_zh_index_daily, symbol=f"sz{INDEX_CODE}")
    if df_idx is None or df_idx.empty:
        # Try sh prefix
        df_idx = safe_fetch(ak.stock_zh_index_daily, symbol=f"sh{INDEX_CODE}")
    df_idx['date'] = pd.to_datetime(df_idx['date'])
    df_idx = df_idx.sort_values('date').reset_index(drop=True)
    print(f"  Index: {len(df_idx)} days, {df_idx['date'].iloc[0].date()} ~ {df_idx['date'].iloc[-1].date()}")

    # Align dates — use NAV dates as reference
    days = len(df_nav)
    results = []
    nav_values = pd.to_numeric(df_nav.iloc[:, 1], errors='coerce')

    for i in range(20, days):
        nav_data = analyze_nav_at(df_nav, i)
        if nav_data is None:
            continue

        nav_date = df_nav[df_nav.columns[0]].iloc[i]

        # Find matching index bar (closest date <= nav_date)
        idx_mask = df_idx['date'] <= nav_date
        if not idx_mask.any():
            continue
        idx_pos = df_idx[idx_mask].index[-1]
        idx_data = analyze_index_at(df_idx, idx_pos)

        # Score with NAV + index (no news)
        sug = generate_suggestion(
            nav=nav_data, idx=idx_data, stocks=None, news=None, fund_type="index"
        )
        score = sug["score"]
        action = sug["action"]
        nav_cur = nav_data["nav_current"]

        # Forward returns
        fwd_3d = fwd_5d = fwd_10d = None
        if i + 3 < days:
            fwd_3d = round((float(nav_values.iloc[i + 3]) / nav_cur - 1) * 100, 2)
        if i + 5 < days:
            fwd_5d = round((float(nav_values.iloc[i + 5]) / nav_cur - 1) * 100, 2)
        if i + 10 < days:
            fwd_10d = round((float(nav_values.iloc[i + 10]) / nav_cur - 1) * 100, 2)

        results.append({
            "date": str(nav_date.date()),
            "nav": nav_cur,
            "score": score,
            "rsi": nav_data["rsi"],
            "idx_rsi": idx_data["rsi"] if idx_data else None,
            "trend": nav_data["trend"],
            "dev_ma20": nav_data["dev_ma20_pct"],
            "chg_5d": nav_data["change_5d"],
            "action": action,
            "bull": sug.get("reasons_bull", []),
            "bear": sug.get("reasons_bear", []),
            "fwd_3d": fwd_3d,
            "fwd_5d": fwd_5d,
            "fwd_10d": fwd_10d,
        })

    print(f"  Scored {len(results)} trading days")

    # Signal distribution
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
                best, worst = max(vals), min(vals)
                print(f"  {horizon}: avg {avg:+.2f}% | win {win_rate:.0f}% | best {best:+.2f}% | worst {worst:+.2f}% | n={len(vals)}")

    # Show BUY_MORE signals
    if buy_more:
        print(f"\n{'='*60}")
        print("BUY_MORE signals")
        print(f"{'='*60}")
        print(f"{'Date':<12} {'Score':>6} {'RSI':>6} {'IdxRSI':>7} {'Trend':>12} {'MA20dev':>8} {'5dChg':>7} {'Fwd5d':>7} {'Fwd10d':>7}")
        print("-" * 80)
        for r in buy_more:
            print(f"{r['date']:<12} {r['score']:>6.1f} {r['rsi']:>6.1f} {r['idx_rsi'] or '-':>7} {r['trend']:>12} {r['dev_ma20']:>+7.1f}% {r['chg_5d']:>+6.1f}% {r['fwd_5d'] or '-':>7} {r['fwd_10d'] or '-':>7}")

    # Show REDUCE signals
    if reduce:
        print(f"\n{'='*60}")
        print("REDUCE signals")
        print(f"{'='*60}")
        print(f"{'Date':<12} {'Score':>6} {'RSI':>6} {'IdxRSI':>7} {'Trend':>12} {'MA20dev':>8} {'5dChg':>7} {'Fwd5d':>7} {'Fwd10d':>7}")
        print("-" * 80)
        for r in reduce:
            print(f"{r['date']:<12} {r['score']:>6.1f} {r['rsi']:>6.1f} {r['idx_rsi'] or '-':>7} {r['trend']:>12} {r['dev_ma20']:>+7.1f}% {r['chg_5d']:>+6.1f}% {r['fwd_5d'] or '-':>7} {r['fwd_10d'] or '-':>7}")

    # Summary: does the scoring engine add value?
    print(f"\n{'='*60}")
    print("Verdict")
    print(f"{'='*60}")

    buy_fwd5 = [r["fwd_5d"] for r in buy_more if r["fwd_5d"] is not None]
    reduce_fwd5 = [r["fwd_5d"] for r in reduce if r["fwd_5d"] is not None]
    all_fwd5 = [r["fwd_5d"] for r in results if r["fwd_5d"] is not None]

    if buy_fwd5:
        print(f"BUY_MORE 5d avg: {sum(buy_fwd5)/len(buy_fwd5):+.2f}% vs all avg: {sum(all_fwd5)/len(all_fwd5):+.2f}%")
    if reduce_fwd5:
        print(f"REDUCE  5d avg: {sum(reduce_fwd5)/len(reduce_fwd5):+.2f}% vs all avg: {sum(all_fwd5)/len(all_fwd5):+.2f}%")
        # Did we avoid pain?
        avoided = sum(1 for v in reduce_fwd5 if v < 0)
        print(f"REDUCE correctly called ahead of losses: {avoided}/{len(reduce_fwd5)} ({100*avoided/len(reduce_fwd5):.0f}%)")


if __name__ == "__main__":
    main()

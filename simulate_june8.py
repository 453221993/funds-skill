"""
Simulate fund_advisor analysis as if today were 2026-06-08 (Monday).
Key adjustments:
- NAV: filter AKShare time series to <= 2026-06-08
- Stocks (mootdx): skip 4 latest bars (6/12→6/8)
- Index: same offset approach
- News: CANNOT go back in time — skipped (noted in report)
"""
import os, sys, json
from datetime import datetime

# Clear proxy
for key in list(os.environ.keys()):
    if key.lower().endswith('_proxy'):
        os.environ.pop(key, None)

import akshare as ak
import pandas as pd
import numpy as np
from mootdx.quotes import Quotes

SIM_DATE = "2026-06-08"
SIM_DATE_DT = pd.Timestamp(SIM_DATE)
# Trading days from 6/8 to 6/12: Mon(8), Tue(9), Wed(10), Thu(11), Fri(12)
# mootdx returns latest as index 0, so skip 4 to land on 6/8
TDX_SKIP_BARS = 4

# ── Copy from fund_advisor.py ──────────────────────────────────────────
MA_SHORT, MA_MEDIUM, MA_LONG = 5, 20, 60
RSI_PERIOD = 14
LOOKBACK_BARS = 100
NAV_LOOKBACK = 60
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

FUNDS = {
    "012700": {
        "name": "易方达证券公司ETF联接C", "type": "index",
        "tracked_index": "399975", "tracked_index_name": "CSI Securities Index",
        "amount": 8414.60, "cost_nav": 1.0951,
        "sector_keywords": ["券商", "证券", "投行", "两融", "IPO", "成交量", "印花税", "券商板块", "证券板块"],
        "sector_label": "证券",
    },
    "026211": {
        "name": "平安科技精选混合C", "type": "active", "amount": 6675.72,
        "cost_nav": 1.8869,
        "sector_keywords": ["光模块", "CPO", "光通信", "芯片", "半导体", "AI算力", "英伟达", "GPU", "算力", "PCB", "人工智能", "大模型", "数据中心"],
        "sector_label": "光通信/半导体",
    },
    "017994": {
        "name": "方正富邦远见成长混合C", "type": "active", "amount": 13851.16,
        "cost_nav": 1.5230,
        "sector_keywords": ["人形机器人", "具身智能", "轴承", "减速器", "汽配", "汽车零部件", "传感器", "精密传动", "特斯拉机器人", "智能制造", "工业母机", "机器人"],
        "sector_label": "机器人/汽零",
    },
}

_tdx_client = None
def get_tdx():
    global _tdx_client
    if _tdx_client is None:
        _tdx_client = Quotes.factory(market='std', timeout=15)
    return _tdx_client

def compute_rsi(series: pd.Series, period: int = RSI_PERIOD) -> float:
    series = pd.to_numeric(series, errors='coerce').dropna()
    if len(series) < period + 1:
        return 50.0
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    last_avg_loss = avg_loss.iloc[-1]
    if pd.isna(last_avg_loss) or last_avg_loss == 0:
        last_avg_gain = avg_gain.iloc[-1]
        return 100.0 if not pd.isna(last_avg_gain) and last_avg_gain > 0 else 50.0
    rs = avg_gain.iloc[-1] / last_avg_loss
    return float(100 - (100 / (1 + rs)))

def compute_ma_dev(price: float, ma: float) -> float:
    return (price / ma - 1) * 100 if ma != 0 else 0.0

def classify_rsi(rsi: float) -> str:
    if rsi > 70: return "overbought"
    elif rsi < 30: return "oversold"
    return "neutral"

def classify_trend(price: float, ma5: float, ma20: float, ma60: float) -> str:
    if price > ma5 > ma20 > ma60: return "strong_up"
    elif price > ma5 > ma20: return "up"
    elif price < ma5 < ma20 < ma60: return "strong_down"
    elif price < ma5 < ma20: return "down"
    return "sideways"

def safe_fetch(fn, *args, **kwargs):
    try: return fn(*args, **kwargs)
    except Exception as e:
        print(f"  [WARN] {fn.__name__}: {e}")
        return None

# ── SIMULATED: NAV Analysis ──────────────────────────────────────────────
def analyze_fund_nav_sim(code: str) -> dict:
    print(f"  Fetching NAV (sim: <= {SIM_DATE})...")
    df = safe_fetch(ak.fund_open_fund_info_em, symbol=code, indicator="单位净值走势")
    if df is None or df.empty:
        return None

    cols = df.columns.tolist()
    date_col, nav_col = cols[0], cols[1]
    df[date_col] = pd.to_datetime(df[date_col])
    # ── KEY: filter to simulation date ──
    df = df[df[date_col] <= SIM_DATE_DT]
    df = df.sort_values(date_col).tail(NAV_LOOKBACK)
    df[nav_col] = pd.to_numeric(df[nav_col], errors='coerce')
    df = df.dropna(subset=[nav_col])

    if len(df) < 20:
        return {"nav_current": float(df[nav_col].iloc[-1]), "error": "insufficient_data"}

    nav = df[nav_col]
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

# ── SIMULATED: Index Analysis ────────────────────────────────────────────
def analyze_index_sim(index_code: str) -> dict:
    print(f"  Fetching index: {index_code} (sim: skip {TDX_SKIP_BARS} bars)...")
    close = None
    # Try TDX with offset
    try:
        client = get_tdx()
        raw = client.bars(symbol=index_code, frequency=9, start=TDX_SKIP_BARS, offset=LOOKBACK_BARS)
        if raw is not None and not raw.empty:
            if 'year' in raw.columns:
                raw = raw[raw['year'].astype(int) > 2000]
            if len(raw) >= 20:
                close = pd.to_numeric(raw['close'], errors='coerce').dropna()
    except Exception as e:
        print(f"  [WARN] TDX index: {e}")

    # Fallback to AKShare (filter by date)
    if close is None or len(close) < 20:
        print(f"  Trying AKShare fallback (filtered to {SIM_DATE})...")
        try:
            prefix = 'sz' if index_code.startswith(('399', '000', '002', '300')) else 'sh'
            df_ak = ak.stock_zh_index_daily(symbol=f'{prefix}{index_code}')
            if df_ak is not None and not df_ak.empty:
                df_ak['date'] = pd.to_datetime(df_ak['date'])
                df_ak = df_ak[df_ak['date'] <= SIM_DATE_DT]
                df_ak = df_ak.sort_values('date').tail(LOOKBACK_BARS)
                close = pd.to_numeric(df_ak['close'], errors='coerce').dropna()
        except Exception as e:
            print(f"  [WARN] AKShare index: {e}")
            return None

    if close is None or len(close) < 20:
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

# ── SIMULATED: Holdings ──────────────────────────────────────────────────
def analyze_fund_holdings_sim(code: str) -> dict:
    print(f"  Fetching holdings...")
    df = safe_fetch(ak.fund_portfolio_hold_em, symbol=code, date="2025")
    if df is None or df.empty:
        return None
    cols = df.columns.tolist()
    holdings = []
    for _, row in df.head(10).iterrows():
        code_val = str(row[cols[1]]).strip()
        name_val = str(row[cols[2]]).strip()
        weight = float(row[cols[3]]) if pd.notna(row[cols[3]]) else 0.0
        holdings.append({"code": code_val, "name": name_val, "weight_pct": weight})
    return {"top_holdings": holdings, "total": len(df)}

# ── SIMULATED: Stock Analysis ────────────────────────────────────────────
def analyze_stocks_sim(stock_codes: list[str]) -> dict:
    results = {}
    client = get_tdx()
    for code in stock_codes:
        print(f"  Analyzing stock: {code} (skip {TDX_SKIP_BARS} bars)...")
        try:
            df = client.bars(symbol=code, frequency=9, start=TDX_SKIP_BARS, offset=LOOKBACK_BARS)
            if df is None or df.empty:
                print(f"    [WARN] No data for {code}")
                continue
            df = df[df['year'] > 0].copy()
            close = pd.to_numeric(df['close'], errors='coerce').dropna()
            if len(close) < 20:
                print(f"    [WARN] Insufficient data: {len(close)} rows")
                continue

            cur = float(close.iloc[-1])
            prev = float(close.iloc[-2]) if len(close) >= 2 else cur
            ma5 = float(close.rolling(MA_SHORT).mean().iloc[-1])
            ma20 = float(close.rolling(MA_MEDIUM).mean().iloc[-1])
            ma60 = float(close.rolling(MA_LONG).mean().iloc[-1]) if len(close) >= MA_LONG else ma20
            rsi = compute_rsi(close)
            dev20 = compute_ma_dev(cur, ma20)
            chg_1d = round((cur/prev - 1)*100, 2)
            chg_5d = round((cur/float(close.iloc[-6]) - 1)*100, 2) if len(close) >= 6 else 0
            chg_20d = round((cur/float(close.iloc[-21]) - 1)*100, 2) if len(close) >= 21 else 0
            trend = classify_trend(cur, ma5, ma20, ma60)

            results[code] = {
                "price": round(cur, 2),
                "change_1d": chg_1d, "change_5d": chg_5d, "change_20d": chg_20d,
                "ma5": round(ma5, 2), "ma20": round(ma20, 2), "ma60": round(ma60, 2),
                "dev_ma20_pct": round(dev20, 2),
                "rsi": round(rsi, 1), "rsi_status": classify_rsi(rsi),
                "trend": trend,
            }
        except Exception as e:
            print(f"    [WARN] {code}: {e}")
    return results

# ── Scoring engine (imported from fund_advisor v3) ────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fund_advisor import generate_suggestion, calculate_position_sizing, TOTAL_CAPITAL, generate_report, predict_next_day

# ── Main ──────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print(f"[Simulation] Date: {SIM_DATE} (Monday)")
    print("=" * 60)

    results = []

    for code, cfg in FUNDS.items():
        print(f"\n{'='*60}")
        print(f"[Analyzing] {code} {cfg['name']}")
        print(f"{'='*60}")

        result = {"code": code, "name": cfg["name"], "type": cfg["type"], "amount": cfg["amount"]}

        # 1) NAV (simulated)
        nav = analyze_fund_nav_sim(code)
        result["nav_analysis"] = nav

        # 2) Index (simulated)
        idx = None
        if cfg["type"] == "index":
            idx = analyze_index_sim(cfg["tracked_index"])
            result["index_analysis"] = idx
            result["tracked_index"] = cfg["tracked_index"]
            result["tracked_index_name"] = cfg["tracked_index_name"]

        # 3) Holdings + stocks (simulated)
        hld = None; stock_results = None
        if cfg["type"] == "active":
            hld = analyze_fund_holdings_sim(code)
            result["holdings_analysis"] = hld
            if hld and hld.get("top_holdings"):
                codes = [h["code"] for h in hld["top_holdings"]]
                stock_results = analyze_stocks_sim(codes)
                result["stock_analysis"] = stock_results

        # 4) News — CANNOT simulate, pass empty
        news = {"matched_count": 0, "positive_count": 0, "negative_count": 0,
                "score_adjust": 0, "signals": ["[SIM] 消息面无法回溯至6月8日"],
                "sector_label": cfg.get("sector_label", "")}
        result["news_analysis"] = news

        # 5) Suggestion
        sug = generate_suggestion(nav, idx, stock_results, news, cfg["type"])
        result["suggestion"] = sug

        # 6) Position sizing
        total_holdings = sum(f["amount"] for f in FUNDS.values())
        nav_cur = nav.get("nav_current") if nav else None
        sizing = calculate_position_sizing(
            score=sug["score"],
            action=sug["action"],
            current_amount=cfg["amount"],
            cost_nav=cfg.get("cost_nav"),
            nav_current=nav_cur if nav_cur else 0,
            total_capital=TOTAL_CAPITAL,
            total_holdings=total_holdings,
        )
        result["sizing"] = sizing

        # 7) Next-day prediction
        pred = predict_next_day(nav, idx, stock_results, news)
        result["prediction"] = pred

        print(f"\n  >>> {sug['action_cn']} (score: {sug['score']}/100)")
        if sug["reasons_bull"]:
            print(f"  >>> Bull: {' | '.join(sug['reasons_bull'])}")
        if sug["reasons_bear"]:
            print(f"  >>> Bear: {' | '.join(sug['reasons_bear'])}")
        if sizing.get("sizing_logic"):
            print(f"  >>> Sizing: {' | '.join(sizing['sizing_logic'])}")

        results.append(result)

    # Report (use fund_advisor's generate_report, with simulated header)
    print(f"\n{'='*60}\n[Report] Generating...")
    report = generate_report(results)
    # Replace title line with simulated version
    sim_title = (
        f"# 📊 今日基金体检报告（模拟回测）\n\n"
        f"**回测日期：** {SIM_DATE}（周一）\n\n"
        f"> ⚠️ 消息面无法回溯，仅技术面数据。\n"
    )
    report = report.replace(
        "# 📊 今日基金体检报告\n\n",
        sim_title + "\n",
        1,
    )

    md_path = os.path.join(OUTPUT_DIR, "daily_report_june8.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[Report] {md_path}")

    json_path = os.path.join(OUTPUT_DIR, "daily_report_june8.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"[JSON] {json_path}")

    print(f"\n{report}")
    return results

if __name__ == "__main__":
    main()

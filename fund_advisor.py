"""
Fund Advisor v2 - Daily A-share fund holdings analysis and operation suggestions.
Fixed: uses mootdx (TCP/TDX protocol) for stock data to bypass VPN HTTP filter.
"""
import os
import json
from datetime import datetime, timedelta
from typing import Optional

# Clear system proxy before importing HTTP libraries
for key in list(os.environ.keys()):
    if key.lower().endswith('_proxy'):
        os.environ.pop(key, None)

import akshare as ak
import pandas as pd
import numpy as np
from mootdx.quotes import Quotes

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Total investable capital (RMB). Used for position-sizing recommendations.
# Set to 0 or leave blank to skip position-sizing output.
TOTAL_CAPITAL = 33000

FUNDS = {
    "012700": {
        "name": "易方达证券公司ETF联接C",
        "type": "index",
        "tracked_index": "399975",
        "tracked_index_name": "CSI Securities Index",
        "amount": 8414.60,
        # cost_nav: calculated from P&L -1%: 1.0841 / 0.99 = 1.0951
        "cost_nav": 1.0951,
        "sector_keywords": [
            "券商", "证券", "投行", "两融", "IPO",
            "成交量", "印花税", "券商板块", "证券板块",
        ],
        "sector_label": "证券",
    },
    "026211": {
        "name": "平安科技精选混合C",
        "type": "active",
        "amount": 6675.72,
        # cost_nav: calculated from P&L +2.7%: 1.9378 / 1.027 = 1.8869
        "cost_nav": 1.8869,
        "sector_keywords": [
            "光模块", "CPO", "光通信", "芯片", "半导体",
            "AI算力", "英伟达", "GPU", "算力", "PCB",
            "人工智能", "大模型", "数据中心",
        ],
        "sector_label": "光通信/半导体",
    },
    "017994": {
        "name": "方正富邦远见成长混合C",
        "type": "active",
        "amount": 13851.16,
        # cost_nav: calculated from P&L -6.38%: 1.4258 / 0.9362 = 1.5230
        "cost_nav": 1.5230,
        "sector_keywords": [
            "人形机器人", "具身智能", "轴承", "减速器",
            "汽配", "汽车零部件", "传感器", "精密传动",
            "特斯拉机器人", "智能制造", "工业母机", "机器人",
        ],
        "sector_label": "机器人/汽零",
    },
}

# News sentiment lexicon
MA_SHORT, MA_MEDIUM, MA_LONG = 5, 20, 60
RSI_PERIOD = 14
LOOKBACK_BARS = 100
NAV_LOOKBACK = 60

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# TDX client (lazy init)
_tdx_client = None

def get_tdx():
    global _tdx_client
    if _tdx_client is None:
        _tdx_client = Quotes.factory(market='std', timeout=15)
    return _tdx_client

# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------

def compute_rsi(series: pd.Series, period: int = RSI_PERIOD) -> float:
    """Wilder's smoothed RSI."""
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
        # No downward movement -> strong uptrend
        last_avg_gain = avg_gain.iloc[-1]
        return 100.0 if not pd.isna(last_avg_gain) and last_avg_gain > 0 else 50.0
    rs = avg_gain.iloc[-1] / last_avg_loss
    return float(100 - (100 / (1 + rs)))

def compute_ma_dev(price: float, ma: float) -> float:
    if ma == 0:
        return 0.0
    return (price / ma - 1) * 100

def classify_rsi(rsi: float) -> str:
    if rsi > 70:
        return "overbought"
    elif rsi < 30:
        return "oversold"
    return "neutral"

def classify_trend(price: float, ma5: float, ma20: float, ma60: float) -> str:
    if price > ma5 > ma20 > ma60:
        return "strong_up"
    elif price > ma5 > ma20:
        return "up"
    elif price < ma5 < ma20 < ma60:
        return "strong_down"
    elif price < ma5 < ma20:
        return "down"
    return "sideways"

def safe_fetch(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        print(f"  [WARN] {fn.__name__}: {e}")
        return None

# ---------------------------------------------------------------------------
# Sector News Analysis
# ---------------------------------------------------------------------------
# Cache news data within a single run (fetched once, shared across funds)
_news_cache: Optional[pd.DataFrame] = None

def _fetch_financial_news() -> pd.DataFrame:
    """Fetch latest financial news from Eastmoney (cached per run)."""
    global _news_cache
    if _news_cache is not None:
        return _news_cache
    try:
        df = ak.stock_info_global_em()
        if df is not None and not df.empty:
            # Columns: [标题, 摘要, 发布时间, 链接]
            _news_cache = df
        else:
            _news_cache = pd.DataFrame()
    except Exception as e:
        print(f"  [WARN] News fetch: {e}")
        _news_cache = pd.DataFrame()
    return _news_cache


# ── Stock notices cache ──
_notices_cache: Optional[pd.DataFrame] = None

# Notice types worth flagging (inclusion-based — only flag genuinely important ones)
_NOTICE_IMPORTANT = [
    "业绩预告", "业绩快报", "盈利", "亏损", "净利润", "经营情况",
    "增减持", "减持", "增持", "股份变动",
    "重大合同", "中标", "订单", "战略合作",
    "监管", "处罚", "问询", "立案", "调查", "警示函",
    "停牌", "复牌", "终止上市", "退市",
    "回购", "股权激励",
    "资产重组", "并购", "收购", "重大资产",
    "分红", "送转", "权益分派",
    "异常波动", "风险提示",
    "诉讼", "仲裁",
    "控制权", "实际控制人", "变更",
    "非公开发行", "定增",
    "对外投资", "项目投资",
    "新产品", "技术突破", "获批",
]


def fetch_holdings_notices(stock_codes: list[str]) -> dict[str, list[dict]]:
    """Fetch important stock notices for given codes. Cached per run.

    Only returns notices matching _NOTICE_IMPORTANT keywords (earnings,
    regulatory, M&A, major contracts, etc.). Routine meetings/IR are ignored.
    Returns {code: [{title, type, stock_name}]}. Empty dict if nothing important.
    """
    global _notices_cache
    if _notices_cache is None:
        try:
            # Try today, fall back 1-3 days for non-trading days
            df = None
            for offset in range(4):
                d = (datetime.now() - timedelta(days=offset)).strftime("%Y%m%d")
                df = safe_fetch(ak.stock_notice_report, date=d)
                if df is not None and not df.empty:
                    break
            _notices_cache = df if (df is not None and not df.empty) else pd.DataFrame()
        except Exception as e:
            print(f"  [WARN] Notices fetch: {e}")
            _notices_cache = pd.DataFrame()

    if _notices_cache.empty:
        return {}

    code_col = _notices_cache.columns[0]
    name_col = _notices_cache.columns[1]
    title_col = _notices_cache.columns[2]
    type_col = _notices_cache.columns[3]

    result = {}
    for code in stock_codes:
        mask = _notices_cache[code_col].astype(str).str.strip() == str(code).strip()
        matched = _notices_cache[mask]
        if matched.empty:
            continue

        important = []
        for _, row in matched.iterrows():
            ntype, ntitle = str(row[type_col]), str(row[title_col])
            if any(s in ntype or s in ntitle for s in _NOTICE_IMPORTANT):
                important.append({"title": ntitle[:100], "type": ntype, "stock_name": str(row[name_col])})

        if important:
            result[code] = important[:5]

    return result


def analyze_sector_news(keywords: list[str], sector_label: str) -> dict:
    """
    Fetch financial news, filter by sector keywords, assess sentiment via LLM.
    Falls back to score_adjust=0 if LLM unavailable.
    """
    df = _fetch_financial_news()
    if df is None or df.empty:
        return {"matched_count": 0, "headlines": [],
                "score_adjust": 0, "signals": [], "error": "no_news_data"}

    # Columns: 标题(0), 摘要(1), 发布时间(2), 链接(3)
    title_col = df.columns[0]
    summary_col = df.columns[1]
    time_col = df.columns[2]

    # ── Date filter: keep only the latest trading day's news ──
    df[time_col] = pd.to_datetime(df[time_col])
    latest_date = df[time_col].dt.date.max()
    df = df[df[time_col].dt.date == latest_date]
    print(f"  News date: {latest_date}")

    # Filter by keywords
    pattern = '|'.join(keywords)
    mask = df[title_col].str.contains(pattern, na=False) | df[summary_col].str.contains(pattern, na=False)
    matched = df[mask]

    if matched.empty:
        return {"matched_count": 0, "headlines": [],
                "score_adjust": 0, "signals": [f"今日无{sector_label}相关新闻"],
                "sector_label": sector_label}

    # Extract headlines (up to 15, keep full text for LLM assessment)
    headlines = []
    for _, row in matched.head(15).iterrows():
        title = str(row[title_col])[:100]
        summary = str(row[summary_col])[:120] if pd.notna(row[summary_col]) else ""
        headlines.append({"title": title, "summary": summary})

    total = len(matched)

    # Build signals (score_adjust filled later by batch_news_sentiment)
    signals = [f"{sector_label}相关新闻 {total} 条"]
    for h in headlines[:5]:
        signals.append(h["title"])

    return {
        "matched_count": total,
        "headlines": headlines,
        "score_adjust": 0,  # filled by batch_news_sentiment
        "signals": signals,
        "sector_label": sector_label,
    }


def _score_one_sector(headlines: list[dict], sector_label: str,
                     notices: Optional[dict[str, list[dict]]] = None) -> float:
    """Score news sentiment for a single sector via DeepSeek API.

    Returns -10 to +10. Falls back to 0 on any error.
    Each sector is evaluated independently to avoid cross-contamination.
    If stock_notices provided (for active funds), they are weighted more heavily.
    """
    if not headlines and not notices:
        return 0.0

    try:
        from openai import OpenAI
    except ImportError:
        return 0.0

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        return 0.0

    # Build sector news section
    lines = "\n".join(
        f"  - {h['title'][:100]}"
        for h in headlines[:15]
    ) if headlines else "（无板块新闻）"

    # Build stock notices section (if any)
    notice_lines = ""
    if notices:
        parts = []
        for code, ns in notices.items():
            for n in ns:
                parts.append(f"  - [{n['stock_name']} {code}] {n['type']}: {n['title'][:80]}")
        if parts:
            notice_lines = "\n".join(parts)
            notice_lines = f"\n\n## 重仓股今日公告（影响比板块新闻更大）：\n{notice_lines}"

    prompt = f"""评估"{sector_label}"板块的今日情绪。请输出一个数字。

## 板块新闻
{lines}{notice_lines}

规则：
- 范围 -10（强烈利空）到 +10（强烈利好）
- 个股公告 > 板块新闻：业绩亏损、监管问询、减持 → 直接往负向拉；重大订单、回购、业绩大增 → 直接往正向拉
- PR宣传稿、常规合作新闻、路演不算利好
- 只输出一个数字，不要其他内容"""

    try:
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        response = client.chat.completions.create(
            model="deepseek-v4-pro",
            messages=[
                {"role": "system", "content": "你是一个A股新闻情绪分析助手。只输出一个-10到10的数字。"},
                {"role": "user", "content": prompt},
            ],
            stream=False,
        )
        text = response.choices[0].message.content.strip()

        import re
        match = re.search(r'[-]?\d+(?:\.\d+)?', text)
        if match:
            return max(-10.0, min(10.0, float(match.group())))
        return 0.0
    except Exception as e:
        print(f"  [WARN] News sentiment for {sector_label} failed: {e}")
        return 0.0


def batch_news_sentiment(fund_news: list[dict]) -> None:
    """Score news sentiment for all funds — one API call per sector, run in PARALLEL.

    Each entry in fund_news is a dict with keys: code, sector_label, headlines.
    score_adjust and signals are written back in-place.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    active = [n for n in fund_news if n.get("headlines")]
    if not active:
        return

    # Submit all in parallel
    with ThreadPoolExecutor(max_workers=len(active)) as executor:
        futures = {
            executor.submit(
                _score_one_sector,
                n["headlines"], n["sector_label"], n.get("notices")
            ): n
            for n in active
        }
        for future in as_completed(futures):
            n = futures[future]
            try:
                adj = future.result()
            except Exception:
                adj = 0.0

            n["score_adjust"] = adj

            # Update signals
            sector = n["sector_label"]
            total = n["matched_count"]
            if adj >= 5:
                n["signals"][0] = f"{sector}新闻明显偏暖 (+{adj})"
            elif adj >= 2:
                n["signals"][0] = f"{sector}新闻略偏暖 (+{adj})"
            elif adj <= -5:
                n["signals"][0] = f"{sector}新闻明显偏冷 ({adj})"
            elif adj <= -2:
                n["signals"][0] = f"{sector}新闻略偏冷 ({adj})"

# ---------------------------------------------------------------------------
# Fund NAV Analysis
# ---------------------------------------------------------------------------

def analyze_fund_nav(code: str) -> Optional[dict]:
    print(f"  Fetching NAV...")
    df = safe_fetch(ak.fund_open_fund_info_em, symbol=code, indicator="单位净值走势")
    if df is None or df.empty:
        return None

    cols = df.columns.tolist()
    date_col, nav_col = cols[0], cols[1]
    df[date_col] = pd.to_datetime(df[date_col])
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

# ---------------------------------------------------------------------------
# Index Analysis (via mootdx TCP)
# ---------------------------------------------------------------------------

def analyze_index(index_code: str) -> Optional[dict]:
    print(f"  Fetching index: {index_code}...")
    close = None
    # Try TDX first, fall back to AKShare
    try:
        client = get_tdx()
        raw = client.bars(symbol=index_code, frequency=9, start=0, offset=LOOKBACK_BARS)
        if raw is not None and not raw.empty:
            if 'year' in raw.columns:
                raw = raw[raw['year'].astype(int) > 2000]
            if len(raw) >= 20:
                close = pd.to_numeric(raw['close'], errors='coerce').dropna()
    except Exception as e:
        print(f"  [WARN] TDX index: {e}")

    # Fallback to AKShare
    if close is None or len(close) < 20:
        print(f"  Trying AKShare fallback for index...")
        try:
            prefix = 'sz' if index_code.startswith(('399', '000', '002', '300')) else 'sh'
            df_ak = ak.stock_zh_index_daily(symbol=f'{prefix}{index_code}')
            if df_ak is not None and not df_ak.empty:
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
        "trend": classify_trend(cur, ma20, ma20, ma60),  # simplified
    }

# ---------------------------------------------------------------------------
# Fund Holdings
# ---------------------------------------------------------------------------

def analyze_fund_holdings(code: str) -> Optional[dict]:
    print(f"  Fetching holdings...")
    df = safe_fetch(ak.fund_portfolio_hold_em, symbol=code, date="2025")
    if df is None or df.empty:
        return None

    cols = df.columns.tolist()
    # AKShare returns: [序号, 股票代码, 股票名称, 占净值比例, 持股数, 持仓市值, 季度]
    holdings = []
    for _, row in df.head(10).iterrows():
        code_val = str(row[cols[1]]).strip()
        name_val = str(row[cols[2]]).strip()
        weight = float(row[cols[3]]) if pd.notna(row[cols[3]]) else 0.0
        holdings.append({"code": code_val, "name": name_val, "weight_pct": weight})
    return {"top_holdings": holdings, "total": len(df)}

# ---------------------------------------------------------------------------
# Stock Analysis (via mootdx TCP)
# ---------------------------------------------------------------------------

def analyze_stocks(stock_codes: list[str]) -> dict[str, dict]:
    results = {}
    client = get_tdx()
    for code in stock_codes:
        print(f"  Analyzing stock: {code}...")
        try:
            df = client.bars(symbol=code, frequency=9, start=0, offset=LOOKBACK_BARS)
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

# ---------------------------------------------------------------------------
# Suggestion Engine
# ---------------------------------------------------------------------------

def generate_suggestion(nav: Optional[dict], idx: Optional[dict],
                        stocks: Optional[dict], news: Optional[dict],
                        fund_type: str) -> dict:
    """Continuous sliding-scale scoring engine (v3).

    Key changes from v2 binary thresholds:
    - RSI: (50-RSI)*0.5  continuous, capped ±15  (was: <30→+10, >70→-10)
    - MA20 dev: -dev*0.8  continuous, capped ±12  (was: <-5→+8, >10→-8)
    - 5d momentum: piecewise linear  (was: <-8→+5, >10→-5)
    - Trend: graduated levels  (was: ±10 flat)
    - NEW: trend-momentum divergence signal (uptrend+surging = chasing risk)
    - Stocks: internal RSI/trend also continuous
    """
    score = 50.0
    bull, bear = [], []

    if nav and "error" not in nav:
        t = nav["trend"]
        rsi = nav["rsi"]
        dev = nav["dev_ma20_pct"]
        chg5 = nav["change_5d"]

        # ── RSI: continuous, centered at 50 ──────────────────────
        # RSI 30 → +10, RSI 25 → +12.5, RSI 70 → -10, RSI 80 → -15
        rsi_adj = round((50 - rsi) * 0.5, 1)
        rsi_adj = max(-15, min(15, rsi_adj))
        score += rsi_adj
        if rsi_adj >= 5:
            bull.append(f"RSI={rsi:.0f} 超卖 (+{rsi_adj:.0f})")
        elif rsi_adj >= 2:
            bull.append(f"RSI={rsi:.0f} 偏低 (+{rsi_adj:.0f})")
        elif rsi_adj <= -5:
            bear.append(f"RSI={rsi:.0f} 超买 ({rsi_adj:.0f})")
        elif rsi_adj <= -2:
            bear.append(f"RSI={rsi:.0f} 偏高 ({rsi_adj:.0f})")

        # ── MA20 deviation: continuous ───────────────────────────
        # dev -5% → +4, dev -10% → +8, dev 10% → -8, dev 20% → -12(截断)
        ma_adj = round(-dev * 0.8, 1)
        ma_adj = max(-12, min(12, ma_adj))
        score += ma_adj
        if ma_adj >= 4:
            bull.append(f"低于MA20 {dev:.1f}% (+{ma_adj:.0f})")
        elif ma_adj >= 1.5:
            bull.append(f"略低于MA20 {dev:.1f}% (+{ma_adj:.0f})")
        elif ma_adj <= -6:
            bear.append(f"高于MA20 {dev:.1f}% ({ma_adj:.0f})")
        elif ma_adj <= -2:
            bear.append(f"略高于MA20 {dev:.1f}% ({ma_adj:.0f})")

        # ── Trend: graduated levels ──────────────────────────────
        if t == "strong_up":
            score += 12; bull.append("NAV 强上升趋势")
        elif t == "up":
            score += 8; bull.append("NAV 上升趋势")
        elif t == "strong_down":
            score -= 12; bear.append("NAV 强下降趋势")
        elif t == "down":
            score -= 8; bear.append("NAV 下降趋势")

        # ── Short-term momentum: piecewise linear ────────────────
        chg_adj = 0.0
        if chg5 > 15:
            chg_adj = -10
            bear.append(f"5日涨{chg5:.1f}% 极端过热 ({chg_adj:.0f})")
        elif chg5 > 10:
            chg_adj = round(-5 - (chg5 - 10) * 1.0, 1)
            bear.append(f"5日涨{chg5:.1f}% 短期过热 ({chg_adj:.0f})")
        elif chg5 > 5:
            chg_adj = round(-(chg5 - 5) * 1.0, 1)
            if chg_adj <= -2:
                bear.append(f"5日涨{chg5:.1f}% 偏快 ({chg_adj:.0f})")
        elif chg5 < -12:
            chg_adj = 8
            bull.append(f"5日跌{abs(chg5):.1f}% 极端恐慌 (+8)")
        elif chg5 < -8:
            chg_adj = round(5 + (abs(chg5) - 8) * 0.75, 1)
            bull.append(f"5日跌{abs(chg5):.1f}% 短期超跌 (+{chg_adj:.0f})")
        elif chg5 < -3:
            chg_adj = round((abs(chg5) - 3) * 1.0, 1)
            if chg_adj >= 2:
                bull.append(f"5日跌{abs(chg5):.1f}% 小幅回调 (+{chg_adj:.0f})")
        # -3% to +5%: neutral, no adjustment
        score += chg_adj

        # ── Trend-momentum divergence (NEW) ──────────────────────
        # "Uptrend + short-term surging" = chasing risk
        if t in ("strong_up", "up") and chg5 > 8:
            div_pen = round(-(chg5 - 5) * 0.6, 1)
            div_pen = max(-6, div_pen)
            score += div_pen
            bear.append(f"趋势上行+急涨 追高风险 ({div_pen:.0f})")
        # "Downtrend + short-term plunging" = capitulation opportunity
        elif t in ("strong_down", "down") and chg5 < -8:
            div_bonus = round((abs(chg5) - 5) * 0.6, 1)
            div_bonus = min(6, div_bonus)
            score += div_bonus
            bull.append(f"趋势下行+急跌 恐慌机会 (+{div_bonus:.0f})")

    # ── Index signals (continuous) ────────────────────────────────────
    if idx:
        idx_rsi = idx.get("rsi", 50)
        idx_rsi_adj = round((50 - idx_rsi) * 0.3, 1)
        idx_rsi_adj = max(-8, min(8, idx_rsi_adj))
        score += idx_rsi_adj
        if idx_rsi_adj >= 3:
            bull.append(f"指数RSI={idx_rsi:.0f} 偏低 (+{idx_rsi_adj:.0f})")
        elif idx_rsi_adj <= -3:
            bear.append(f"指数RSI={idx_rsi:.0f} 偏高 ({idx_rsi_adj:.0f})")

        idx_trend = idx.get("trend", "sideways")
        if idx_trend in ("strong_up", "up"):
            score += 5; bull.append("指数上行趋势")
        elif idx_trend in ("strong_down", "down"):
            score -= 5; bear.append("指数下行趋势")

    # ── Stock signals (continuous) ────────────────────────────────────
    if stocks:
        ss = []
        for sr in stocks.values():
            s = 50.0
            # Continuous RSI
            s += (50 - sr["rsi"]) * 0.5
            # Graduated trend
            t_s = sr["trend"]
            if t_s == "strong_up": s += 12
            elif t_s == "up": s += 8
            elif t_s == "strong_down": s -= 12
            elif t_s == "down": s -= 8
            ss.append(s)
        if ss:
            avg = sum(ss) / len(ss)
            score += (avg - 50) * 0.5
            up_n = sum(1 for s in ss if s > 55)
            down_n = sum(1 for s in ss if s < 45)
            if up_n > down_n:
                bull.append(f"{up_n}/{len(ss)} stocks bullish")
            elif down_n > up_n:
                bear.append(f"{down_n}/{len(ss)} stocks bearish")

    # ── News / sentiment (unchanged continuous input) ──────────────────
    if news and news.get("matched_count", 0) > 0:
        adj = news.get("score_adjust", 0)
        if adj != 0:
            score += adj
            if adj > 0:
                bull.append(f"消息面 +{adj}")
            else:
                bear.append(f"消息面 {adj}")
        for sig in news.get("signals", [])[1:4]:  # skip first (summary), take headlines
            if sig.startswith("[+]"):
                bull.append(sig[4:])
            elif sig.startswith("[-]"):
                bear.append(sig[4:])

    score = max(0, min(100, round(score, 1)))

    if score >= 65:
        action, action_cn = "BUY_MORE", "加仓"
    elif score >= 40:
        action, action_cn = "HOLD", "持有"
    else:
        action, action_cn = "REDUCE", "减仓"

    return {
        "score": score,
        "action": action, "action_cn": action_cn,
        "reasons_bull": bull, "reasons_bear": bear,
        "news_signals": news.get("signals", []) if news else [],
    }

# ---------------------------------------------------------------------------
# Position Sizing
# ---------------------------------------------------------------------------

def calculate_position_sizing(
    score: float, action: str, current_amount: float,
    cost_nav: Optional[float], nav_current: float,
    total_capital: float, total_holdings: float,
) -> dict:
    """
    Calculate suggested add/reduce amount based on score intensity,
    profit/loss status, and available capital.

    Returns:
        { suggested_amount (positive=add, negative=reduce),
          pnl_pct, pnl_label, sizing_note, sizing_logic: [str] }
    """
    if total_capital <= 0:
        return {
            "suggested_amount": 0,
            "pnl_pct": None, "pnl_label": "未知",
            "sizing_note": "需配置 TOTAL_CAPITAL 和 cost_nav 后显示仓位建议",
            "sizing_logic": [],
        }

    available_cash = max(0, total_capital - total_holdings)

    # ── Profit/Loss ──
    if cost_nav and cost_nav > 0:
        pnl_pct = round((nav_current / cost_nav - 1) * 100, 1)
        if pnl_pct <= -15: pnl_label = "深度亏损"
        elif pnl_pct < -5: pnl_label = "亏损"
        elif pnl_pct < 0: pnl_label = "小幅亏损"
        elif pnl_pct < 5: pnl_label = "小幅盈利"
        elif pnl_pct < 15: pnl_label = "盈利"
        elif pnl_pct < 30: pnl_label = "大幅盈利"
        else: pnl_label = "暴利"
    else:
        pnl_pct = None
        pnl_label = "未知"

    suggested_amount = 0
    sizing_logic = []

    if action == "BUY_MORE":
        # ── Score intensity ──
        if score >= 85:
            intensity = 1.0; level = "满仓"
        elif score >= 78:
            intensity = 0.7; level = "积极"
        elif score >= 70:
            intensity = 0.5; level = "中等"
        else:  # 65-70
            intensity = 0.3; level = "试探"

        sizing_logic.append(f"评分{score:.0f} → {level}加仓 (×{intensity})")

        # ── P&L correction ──
        if pnl_pct is not None:
            if pnl_pct < -15:     pl_mult = 1.3; pl_reason = "深度亏损 积极摊薄"
            elif pnl_pct < -5:    pl_mult = 1.1; pl_reason = "亏损 适度摊薄"
            elif pnl_pct < 0:     pl_mult = 1.0; pl_reason = "小幅亏损 正常"
            elif pnl_pct < 15:    pl_mult = 1.0; pl_reason = "盈利中 正常"
            elif pnl_pct < 30:    pl_mult = 0.7; pl_reason = "大幅盈利 追高降仓"
            else:                 pl_mult = 0.4; pl_reason = "暴利 谨慎追高"

            if pl_mult != 1.0:
                sizing_logic.append(f"盈亏{pnl_pct:+.1f}% → {pl_reason} (×{pl_mult})")
        else:
            pl_mult = 1.0

        # Simple: available_cash × intensity × P&L correction, capped at available_cash
        base_amount = available_cash * intensity * pl_mult
        base_amount = min(base_amount, available_cash * 0.8)  # never use >80% cash on one fund

        # Round to reasonable increments
        if base_amount >= 5000:
            suggested_amount = round(base_amount / 500) * 500
        elif base_amount >= 1000:
            suggested_amount = round(base_amount / 200) * 200
        else:
            suggested_amount = round(base_amount / 100) * 100

        suggested_amount = max(suggested_amount, 0)

        if suggested_amount < 100:
            sizing_note = "加仓金额过小，暂不加仓"
        else:
            sizing_note = f"建议加仓 ¥{suggested_amount:,.0f}"

    elif action == "REDUCE":
        # ── Score intensity for reduction ──
        if score < 25:
            reduce_pct = 0.50; level = "清仓式"
        elif score < 35:
            reduce_pct = 0.35; level = "大幅"
        else:
            reduce_pct = 0.20; level = "小幅"

        sizing_logic.append(f"评分{score:.0f} → {level}减仓 ({reduce_pct*100:.0f}%)")

        # ── P&L correction ──
        if pnl_pct is not None:
            if pnl_pct > 20:      pl_mult = 1.3; pl_reason = "高盈利 锁定利润"
            elif pnl_pct > 10:    pl_mult = 1.1; pl_reason = "有盈利 适度止盈"
            elif pnl_pct > 0:     pl_mult = 1.0; pl_reason = "微利 正常减仓"
            elif pnl_pct > -10:   pl_mult = 0.8; pl_reason = "小幅亏损 轻减"
            else:                 pl_mult = 0.5; pl_reason = "深度亏损 减少割肉"

            if pl_mult != 1.0:
                sizing_logic.append(f"盈亏{pnl_pct:+.1f}% → {pl_reason} (×{pl_mult})")
        else:
            pl_mult = 1.0

        reduce_ratio = min(reduce_pct * pl_mult, 0.70)  # cap at 70%
        suggested_amount = -round(current_amount * reduce_ratio / 100) * 100
        sizing_note = f"建议减仓 ¥{abs(suggested_amount):,.0f}（{reduce_ratio*100:.0f}%）"

    else:  # HOLD
        sizing_note = "无需操作"

    return {
        "suggested_amount": suggested_amount,
        "pnl_pct": pnl_pct, "pnl_label": pnl_label,
        "sizing_note": sizing_note,
        "sizing_logic": sizing_logic,
    }

# ---------------------------------------------------------------------------
# Next-Day Direction Prediction
# ---------------------------------------------------------------------------

def predict_next_day(
    nav: Optional[dict], idx: Optional[dict],
    stocks: Optional[dict], news: Optional[dict],
) -> dict:
    """
    Short-term directional bias for the next trading day.
    Synthesizes RSI, trend, momentum, breadth, and news into a simple signal.
    Returns: { direction, confidence, reasons_bull, reasons_bear }
    """
    score = 0.0
    bull, bear = [], []

    if nav and "error" not in nav:
        rsi = nav["rsi"]
        trend = nav["trend"]
        chg1 = nav["change_1d"]
        chg5 = nav["change_5d"]
        dev20 = nav["dev_ma20_pct"]

        # ── RSI zone ──
        if rsi < 30:
            score += 3; bull.append(f"RSI={rsi:.0f} 超卖，技术反弹需求强")
        elif rsi < 40:
            score += 1.5; bull.append(f"RSI={rsi:.0f} 偏低位，反弹概率增加")
        elif rsi > 70:
            score -= 3; bear.append(f"RSI={rsi:.0f} 超买，回调压力较大")
        elif rsi > 60:
            score -= 1.5; bear.append(f"RSI={rsi:.0f} 偏高位，上行空间收窄")

        # ── Trend reversal ──
        if trend in ("strong_down", "down") and chg1 > 0:
            score += 2; bull.append("跌势中出现阳线，短期或企稳")
        elif trend in ("strong_up", "up") and chg1 < 0:
            score -= 2; bear.append("涨势中出现阴线，短期或回调")

        # ── Mean reversion ──
        if dev20 < -5:
            score += 2; bull.append(f"低于MA20 {dev20:.1f}%，均值回归向上")
        elif dev20 < -2:
            score += 1; bull.append(f"略低于MA20 {dev20:.1f}%，有回归动力")
        elif dev20 > 10:
            score -= 2; bear.append(f"高于MA20 {dev20:.1f}%，均值回归向下")
        elif dev20 > 5:
            score -= 1; bear.append(f"略高于MA20 {dev20:.1f}%，偏离偏大")

        # ── Momentum continuation ──
        if chg1 > 2:
            score += 1; bull.append(f"今日涨{chg1:.1f}%，短期动能向上")
        elif chg1 < -2:
            score -= 1; bear.append(f"今日跌{abs(chg1):.1f}%，短期动能向下")

        if chg5 > 5:
            score += 0.5
        elif chg5 < -5:
            score -= 0.5

    # ── Index signal ──
    if idx:
        idx_rsi = idx.get("rsi", 50)
        if idx_rsi < 30:
            score += 1; bull.append("指数超卖，板块环境偏暖")
        elif idx_rsi > 70:
            score -= 1; bear.append("指数超买，板块环境偏冷")

    # ── Stock breadth ──
    if stocks and len(stocks) >= 3:
        up_n = sum(1 for s in stocks.values() if s.get("change_1d", 0) > 0)
        if up_n >= len(stocks) * 0.7:
            score += 1; bull.append(f"{up_n}/{len(stocks)} 重仓股今日上涨")
        elif up_n <= len(stocks) * 0.3:
            score -= 1; bear.append(f"{len(stocks) - up_n}/{len(stocks)} 重仓股今日下跌")

    # ── News ──
    if news and news.get("matched_count", 0) >= 3:
        adj = news.get("score_adjust", 0)
        if adj >= 3:
            score += 1; bull.append("板块消息面偏暖")
        elif adj <= -3:
            score -= 1; bear.append("板块消息面偏冷")

    # ── Classify ──
    if score >= 3:
        direction = "偏多 ↑"
        confidence = "较高" if score >= 5 else "中等"
    elif score <= -3:
        direction = "偏空 ↓"
        confidence = "较高" if score <= -5 else "中等"
    else:
        direction = "震荡 →"
        confidence = "方向不明"

    return {
        "direction": direction,
        "confidence": confidence,
        "direction_score": round(score, 1),
        "reasons_bull": bull,
        "reasons_bear": bear,
    }

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def generate_report(results: list[dict]) -> str:
    """Generate a clean, structured report (template-based).

    For a more conversational "大白话" report, use generate_ai_report()
    which sends the structured data to an LLM with a plain-language prompt.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    total_holdings = sum(r["amount"] for r in results)
    add_n = sum(1 for r in results if r["suggestion"]["action"] == "BUY_MORE")
    reduce_n = sum(1 for r in results if r["suggestion"]["action"] == "REDUCE")
    hold_n = len(results) - add_n - reduce_n
    available = max(0, TOTAL_CAPITAL - total_holdings)

    signal_parts = []
    if add_n: signal_parts.append(f"{add_n}加仓")
    if hold_n: signal_parts.append(f"{hold_n}持有")
    if reduce_n: signal_parts.append(f"{reduce_n}减仓")
    signal_str = " · ".join(signal_parts)

    lines = [
        "# 📊 今日基金体检报告",
        "",
        f"**{now}**",
        "",
        f"💰 总资产 **¥{TOTAL_CAPITAL:,}** ｜ 持仓 **¥{total_holdings:,.0f}** ｜ 可加仓 **¥{available:,.0f}**",
        f"📋 信号：**{signal_str}**",
        "",
        "---",
        "",
    ]

    for r in results:
        sug = r["suggestion"]
        nav = r.get("nav_analysis") or {}
        idx = r.get("index_analysis")
        hld = r.get("holdings_analysis")
        stocks = r.get("stock_analysis") or {}
        sizing = r.get("sizing", {})
        pred = r.get("prediction", {})

        score = sug["score"]
        action = sug["action"]
        action_cn = sug["action_cn"]
        name = r["name"]
        code = r["code"]
        amount = r["amount"]
        nav_cur = nav.get("nav_current", None)
        pnl_pct = sizing.get("pnl_pct")

        # Emoji indicator
        icon = {"BUY_MORE": "🟢", "HOLD": "🟡", "REDUCE": "🔴"}[action]

        lines.append(f"## {icon} {code} {name} — **{action_cn}**（评分 {score}/100）")
        lines.append("")

        # Key numbers
        lines.append(f"净值 **{nav_cur or '-'}** ｜ 持仓 **¥{amount:,.0f}** ｜ "
                     f"今日 {nav.get('change_1d', 0):+.1f}% ｜ "
                     f"盈亏 {pnl_pct:+.1f}%" if pnl_pct is not None else "")
        lines.append("")

        # Reasons
        if sug.get("reasons_bull"):
            lines.append("✅ **看多理由：**")
            for reason in sug["reasons_bull"]:
                lines.append(f"- {reason}")
            lines.append("")
        if sug.get("reasons_bear"):
            lines.append("⚠️ **看空理由：**")
            for reason in sug["reasons_bear"]:
                lines.append(f"- {reason}")
            lines.append("")

        # Position sizing
        sizing_note = sizing.get("sizing_note", "")
        if sizing_note and "无需" not in sizing_note:
            lines.append(f"💡 **操作：**{sizing_note}")
            lines.append("")

        # Tomorrow
        if pred:
            lines.append(f"🔮 **明天：**{pred.get('direction', '-')}（置信度 {pred.get('confidence', '-')}）")
            for reason in pred.get("reasons_bull", [])[:2]:
                lines.append(f"  - {reason}")
            for reason in pred.get("reasons_bear", [])[:2]:
                lines.append(f"  - {reason}")
            lines.append("")

        # Index (for index funds)
        if idx:
            lines.append(f"📊 跟踪指数 **{r.get('tracked_index_name', '')}**（{r.get('tracked_index', '')}）"
                         f"：{idx.get('index_current', '-')}，今日 {idx.get('change_1d', 0):+.1f}%")
            lines.append("")

        # Top holdings (for active funds)
        if hld and hld.get("top_holdings"):
            lines.append("**📦 重仓股：**")
            lines.append("")
            lines.append("| 股票 | 占比 | 价格 | 5日涨跌 | RSI | 趋势 |")
            lines.append("|------|------|------|---------|-----|------|")
            for h in hld["top_holdings"][:5]:
                sc = h["code"]
                sr = stocks.get(sc, {})
                lines.append(
                    f"| {h['name']} | {h['weight_pct']}% | "
                    f"{sr.get('price', '-')} | {sr.get('change_5d', '-')}% | "
                    f"{sr.get('rsi', '-')} | {sr.get('trend', '-')} |"
                )
            lines.append("")

        # Sector news
        news = r.get("news_analysis", {})
        if news and news.get("signals") and "无法回溯" not in str(news["signals"][0]):
            lines.append(f"📰 **板块消息（{news.get('sector_label', '')}）：**")
            for sig in news["signals"][:6]:
                lines.append(f"- {sig}")
            lines.append("")

        lines.append("---")
        lines.append("")

    lines += [
        "",
        "> ⚠️ 算法自动分析，仅供参考，不构成投资建议。",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(json_only: bool = False):
    """Run analysis. If json_only=True, skip report generation and only save JSON."""
    print("=" * 60)
    print(f"[Fund Advisor v2] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if json_only:
        print("[Mode] JSON only (for skill / API consumption)")
    print("=" * 60)

    results = []

    for code, cfg in FUNDS.items():
        print(f"\n{'='*60}")
        print(f"[Analyzing] {code} {cfg['name']}")
        print(f"{'='*60}")

        result = {"code": code, "name": cfg["name"], "type": cfg["type"], "amount": cfg["amount"]}

        # 1) NAV analysis
        nav = analyze_fund_nav(code)
        result["nav_analysis"] = nav

        # 2) Index (for index funds)
        idx = None
        if cfg["type"] == "index":
            idx = analyze_index(cfg["tracked_index"])
            result["index_analysis"] = idx
            result["tracked_index"] = cfg["tracked_index"]
            result["tracked_index_name"] = cfg["tracked_index_name"]

        # 3) Holdings + stocks + notices (for active funds)
        hld = None; stock_results = None; stock_notices = None
        if cfg["type"] == "active":
            hld = analyze_fund_holdings(code)
            result["holdings_analysis"] = hld
            if hld and hld.get("top_holdings"):
                codes = [h["code"] for h in hld["top_holdings"]]
                stock_results = analyze_stocks(codes)
                result["stock_analysis"] = stock_results
                # Fetch individual stock notices
                stock_notices = fetch_holdings_notices(codes)
                if stock_notices:
                    print(f"  Notices: {sum(len(v) for v in stock_notices.values())} important notices for {len(stock_notices)} stocks")

        # 4) Sector news (fetch only, sentiment scored in batch below)
        news = analyze_sector_news(
            keywords=cfg.get("sector_keywords", []),
            sector_label=cfg.get("sector_label", ""),
        )
        result["news_analysis"] = news
        result["stock_notices"] = stock_notices  # attach to result for batch
        print(f"  News: {news.get('matched_count', 0)} matched")

        results.append(result)

    # ── Batch news sentiment (single API call for all funds) ──
    batch_news = [
        {
            "code": r["code"],
            "sector_label": r["news_analysis"].get("sector_label", ""),
            "headlines": r["news_analysis"].get("headlines", []),
            "matched_count": r["news_analysis"].get("matched_count", 0),
            "signals": r["news_analysis"].get("signals", []),
            "notices": r.get("stock_notices"),  # individual stock announcements
        }
        for r in results
    ]
    batch_news_sentiment(batch_news)

    # Apply results back to each fund and run scoring
    for r, bn in zip(results, batch_news):
        r["news_analysis"]["score_adjust"] = bn.get("score_adjust", 0)
        r["news_analysis"]["signals"] = bn.get("signals", [])
        news = r["news_analysis"]
        code = r["code"]
        cfg = FUNDS[code]
        nav = r.get("nav_analysis")
        idx = r.get("index_analysis")
        stock_results = r.get("stock_analysis")

        print(f"  {code} news sentiment: {news.get('score_adjust', 0):+.1f}")

        # 5) Suggestion
        sug = generate_suggestion(nav, idx, stock_results, news, cfg["type"])
        r["suggestion"] = sug

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
        r["sizing"] = sizing

        # 7) Next-day prediction
        pred = predict_next_day(nav, idx, stock_results, news)
        r["prediction"] = pred

        print(f"  >>> {sug['action_cn']} (score: {sug['score']}/100)")
        if sug["reasons_bull"]:
            print(f"  >>> Bull: {' | '.join(sug['reasons_bull'])}")
        if sug["reasons_bear"]:
            print(f"  >>> Bear: {' | '.join(sug['reasons_bear'])}")

    # 5) Save JSON (always). Generate report only if not --json mode.
    json_path = os.path.join(OUTPUT_DIR, "daily_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"[JSON] {json_path}")

    if not json_only:
        print(f"\n{'='*60}\n[Report] Generating...")
        report = generate_report(results)
        md_path = os.path.join(OUTPUT_DIR, "daily_report.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"[Report] {md_path}")

    return results

if __name__ == "__main__":
    import sys
    json_only = "--json" in sys.argv
    main(json_only=json_only)

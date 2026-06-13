---
name: check-funds
description: Run the fund advisor analysis and present a plain-language daily report for 3 A-share funds.
---

# /check-funds — Daily Fund Portfolio Check

## How it works

1. Run `python fund_advisor.py --json` — fetches NAV, index, stock, news, notices data and scores all 3 funds
2. Read `daily_report.json`
3. Present a conversational report in plain Chinese

```bash
cd "d:\claudecode\project\投资助手" && python fund_advisor.py --json
```

The Python script handles all data collection and scoring. **Do not** read or modify `fund_advisor.py` unless explicitly asked — the scoring engine is deliberately "vibe-trading" (structured intuition, not backtested).

## The 3 funds

| Code | Name | Type | Amount | Cost NAV | P&L |
|------|------|------|--------|----------|-----|
| 012700 | 易方达证券公司ETF联接C | index | ¥8,414.60 | 1.0951 | -1.0% |
| 026211 | 平安科技精选混合C | active | ¥6,675.72 | 1.8869 | +2.7% |
| 017994 | 方正富邦远见成长混合C | active | ¥13,851.16 | 1.5230 | -6.4% |

Total capital: ¥33,000. Available cash = 33000 - total_holdings.

## Scoring engine (v3 continuous)

- **≥65 → BUY_MORE (加仓)**. 65-69 试探, 70-77 中等, 78-84 积极, ≥85 满仓
- **40-64 → HOLD (持有)**. Closer to 65 = closer to buy signal.
- **<40 → REDUCE (减仓)**. <25 清仓式, 25-34 大幅, 35-39 小幅

The score is built from continuous signals (no binary thresholds):
- RSI: `(50-RSI)*0.5` capped ±15
- MA20 deviation: `-dev*0.8` capped ±12
- Trend: graduated levels (strong_up +12, up +8, down -8, strong_down -12)
- 5d momentum: piecewise linear (overbought penalty, oversold bonus)
- Trend-momentum divergence: uptrend+surging = chasing risk penalty; downtrend+plunging = capitulation bonus
- Index signals (for index funds), stock breadth (for active funds)
- **News sentiment**: DeepSeek-assessed per sector, -10 to +10 (see below)
- **Stock notices**: important announcements for top holdings (see below)

Position sizing: BUY → `available_cash × intensity × pl_multiplier`. REDUCE → `current_amount × reduce_pct × pl_multiplier` capped at 70%.

## News & sentiment pipeline

The Python script handles a 3-layer news system:

1. **Sector news** — `ak.stock_info_global_em()` → filter by sector keywords → up to 15 headlines per fund
2. **Stock notices** — `ak.stock_notice_report()` → filter by top holdings codes → only important types (earnings, regulatory, M&A, contracts, etc. — NOT routine meetings/IR)
3. **LLM assessment** — each sector's headlines + notices are sent to DeepSeek in parallel (one call per sector, no cross-contamination) → returns -10~+10 sentiment score → feeds into scoring engine

The `score_adjust` field in the JSON is the DeepSeek-assessed sentiment. Report it as-is ("消息面偏暖 +X" or "消息面偏冷 -X"). If stock notices are present, they carry more weight than sector news in DeepSeek's assessment.

On days with no stock notices (95% of days), the notices layer is silently absent. When they appear, mention them prominently.

## Report format

```
# 📊 今日基金体检报告
日期
💰 总资产 ¥X | 持仓 ¥X | 可加仓 ¥X | 信号：X加仓 X持有 X减仓

---
🟢/🟡/🔴 基金名 — 操作建议（评分 X）
净值 X | 持仓 X元 | 盈亏 X%
一句话：[核心状态一句说完]
- 好信号：[白话解释，RSI=25说"严重超跌跌太狠"，别光列数字]
- 坏信号：[同上]
- 明天：[一句话]
- 重仓股：[主动基金写，每只一句，结合RSI和趋势翻译成大白话]
---
```

If score is within 5 points of BUY_MORE (65) or REDUCE (40), mention the gap explicitly: "离加仓线差X分".

If one fund has REDUCE, show the suggested amount and explain the sizing logic.

News headlines from `signals` can be referenced for context, but the sentiment assessment from DeepSeek (`score_adjust`) is the authoritative signal.

## Style rules

- **Tone**: 懂投资的朋友在聊天。自然、有洞察、说人话。
- **Explain numbers**: RSI=25 → "严重超跌，跌太狠了"；trend=strong_down → "持续大跌"；dev_ma20=-4.8 → "比20日均线便宜4.8%，有回归动力"
- **Don't just list**: 不要光写"RSI=51.9"或"消息面 +7.0"，要写"RSI 52，不贵也不便宜"或"具身智能新闻明显偏暖"
- **Stock notices take priority**: 如果 JSON 里有 `stock_notices`，报告中要重点提及。比如震裕科技发了业绩预亏，这比十条板块新闻都重要。
- **Be concise**: 每句话都有信息量，不要废话
- **No cringey slang**: 禁止"嘿兄弟""肝儿颤""挺支棱""吃面""出溜"
- **No disclaimer**: 别写"仅供参考""不构成投资建议"

## Data interpretation

JSON fields:
- `nav_current` → 净值
- `change_1d/5d/20d` → 涨跌幅
- `rsi`: <30超卖, 30-40偏低, 40-60正常, 60-70偏高, >70超买
- `trend`: strong_up→涨得很猛, up→在涨, sideways→横盘, down→在跌, strong_down→跌得狠
- `dev_ma20_pct`: 负值=比均线便宜, 正值=比均线贵
- `pnl_pct`: 持仓盈亏
- `score`: 综合评分 0-100
- `score_adjust`: DeepSeek 新闻情绪评分 (-10~+10)
- `bull_reasons/bear_reasons`: 评分引擎的看多/看空理由（翻译成大白话）
- `top_holdings[].rsi/trend/chg5`: 单只股票健康度
- `tomorrow_direction/bull/bear`: 次日预测
- `news.signals`: 新闻标题列表（signal[0]是总结，后面是标题）
- `stock_notices`: 重仓股重要公告（仅主动基金，无公告时为空）

## Follow-up questions

- "012700 细说一下" → expand analysis for that fund
- "跟上周比呢？" → check previous JSON snapshots
- "如果我加仓X元会怎样？" → calculate new cost basis
- "明天再跌2%会触发什么？" → simulate the scoring engine (参考评分公式手动推演)
- Any other investment question about these 3 funds

"""
Daily day-trading briefing with AI-powered economic analysis.
Posts rich Discord embeds every weekday at 08:30 KST.

Required GitHub Secrets:
  DISCORD_WEBHOOK_URL
  ANTHROPIC_API_KEY
"""

import os
import sys
import json
import pytz
import requests
import feedparser
import yfinance as yf
import anthropic
from datetime import datetime

WEBHOOK_URL = os.environ["DISCORD_WEBHOOK_URL"]
KST = pytz.timezone("Asia/Seoul")
CAPITAL = 1_000_000  # ₩1,000,000

# ─── Symbols ──────────────────────────────────────────────────────────────────

MARKET_SYMBOLS = {
    # 주요 지수
    "S&P500":       "^GSPC",
    "NASDAQ":       "^IXIC",
    "KOSPI":        "^KS11",
    # 공포·금리 지표
    "VIX(공포지수)": "^VIX",
    "미국10년금리":  "^TNX",
    "USD/KRW":      "USDKRW=X",
    # 섹터 ETF
    "기술(XLK)":    "XLK",
    "에너지(XLE)":  "XLE",
    "금융(XLF)":    "XLF",
    # 핵심 종목
    "NVDA":         "NVDA",
    "AMD":          "AMD",
    "MU":           "MU",
    "SOXL":         "SOXL",
    "SK하이닉스":   "000660.KS",
    "삼성전자":     "005930.KS",
    "KODEX레버리지":"122630.KS",
}

NEWS_FEEDS = [
    "https://feeds.reuters.com/reuters/businessNews",
    "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258",
    "https://finance.yahoo.com/news/rssindex",
]

# ─── Data Collection ──────────────────────────────────────────────────────────

def fetch_market_data() -> dict:
    result = {}
    for name, symbol in MARKET_SYMBOLS.items():
        try:
            hist = yf.Ticker(symbol).history(period="5d")
            if len(hist) < 2:
                continue
            cur, prev = hist.iloc[-1], hist.iloc[-2]
            chg = (cur["Close"] - prev["Close"]) / prev["Close"] * 100
            avg_vol = hist["Volume"].mean()
            result[name] = {
                "close":      round(float(cur["Close"]), 2),
                "change_pct": round(float(chg), 2),
                "high":       round(float(cur["High"]), 2),
                "low":        round(float(cur["Low"]), 2),
                "vol_ratio":  round(float(cur["Volume"] / avg_vol), 1) if avg_vol else 1.0,
            }
        except Exception as e:
            print(f"  [warn] {name}: {e}")
    return result


def fetch_news(max_items: int = 12) -> list[dict]:
    headlines = []
    for url in NEWS_FEEDS:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:5]:
                headlines.append({
                    "title":   entry.get("title", "").strip(),
                    "summary": entry.get("summary", "")[:300].strip(),
                    "source":  feed.feed.get("title", ""),
                })
        except Exception as e:
            print(f"  [warn] RSS {url}: {e}")
    return headlines[:max_items]


# ─── AI Analysis (Claude) ─────────────────────────────────────────────────────

SYSTEM_PROMPT = """당신은 경험 많은 주식 시장 분석가이자 데이트레이딩 전문가입니다.
한국의 소액 데이트레이더(시작 자본 ₩100만)를 위해 일일 브리핑을 작성합니다.
규칙:
- 전문 용어는 반드시 괄호 안에 쉬운 말로 풀어서 설명하세요
- 수치와 근거를 들어 구체적으로 분석하세요
- 응답은 반드시 유효한 JSON만 반환하세요 (마크다운 코드블록 없이)"""


def analyze_with_claude(market_data: dict, news: list[dict]) -> dict:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    prompt = f"""아래 실시간 시장 데이터와 뉴스를 분석하여 한국어 데이트레이딩 브리핑 JSON을 반환하세요.

## 시장 데이터
{json.dumps(market_data, ensure_ascii=False, indent=2)}

## 최신 뉴스 헤드라인
{json.dumps(news, ensure_ascii=False, indent=2)}

## 반환 형식 (이 JSON 스키마를 정확히 따르세요)
{{
  "market_mood": "공포 😨 또는 중립 😐 또는 탐욕 🤑 — 한 단어+이모지만",
  "risk_level": "낮음 🟢 또는 보통 🟡 또는 높음 🔴",
  "one_line_strategy": "오늘 핵심 전략을 한 문장으로 (30자 이내)",
  "macro_summary": "글로벌 거시경제 흐름 분석. 금리·인플레이션·달러 강약·경기 사이클 관점에서 3-4문장. 초보자도 이해할 수 있게 쉽게.",
  "key_news": [
    {{
      "headline": "뉴스 제목 (한국어, 30자 이내)",
      "why_it_matters": "이 뉴스가 주식 시장·데이트레이딩에 미치는 구체적 영향. 왜 중요한지 2문장으로 쉽게 설명.",
      "impact": "긍정적 📈 또는 부정적 📉 또는 중립 ➡️"
    }}
  ],
  "sector_analysis": "주요 섹터 동향 (기술·에너지·금융 등). 어느 섹터에 돈이 몰리고 빠지는지 3-4문장.",
  "ai_chip_focus": "AI·반도체 섹터 심층 분석. 현재 과매수/과매도 여부, 단기 촉매, 주의사항 포함. 3-4문장.",
  "trading_picks": [
    {{
      "name": "종목명",
      "symbol": "티커심볼",
      "market": "KR 또는 US",
      "reason": "오늘 이 종목에 주목해야 하는 이유. 뉴스·수급·기술적 근거 포함. 2-3문장.",
      "entry_hint": "진입 타이밍 힌트 (예: 갭업 후 눌림목, 지지선 터치 후 반등 등)",
      "stop_loss_pct": 숫자만,
      "target_pct": 숫자만,
      "risk": "낮음 또는 보통 또는 높음"
    }}
  ]
}}

trading_picks 정확히 5개, key_news 정확히 3개 작성."""

    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return json.loads(resp.content[0].text)


# ─── Discord Embed Builder ────────────────────────────────────────────────────

def _chg(pct: float) -> str:
    arrow = "▲" if pct >= 0 else "▼"
    icon  = "📈" if pct >= 1 else ("📉" if pct <= -1 else "➡️")
    return f"{icon} {arrow}{abs(pct):.2f}%"


def _krw(pct: float) -> str:
    return f"₩{int(CAPITAL * pct / 100):,}"


def _risk_icon(risk: str) -> str:
    return {"낮음": "🟢", "보통": "🟡", "높음": "🔴"}.get(risk, "🟡")


def build_embeds(analysis: dict, market: dict, now_kst: datetime) -> list[dict]:
    date_str = now_kst.strftime("%Y년 %m월 %d일 (%a)")
    mood = analysis["market_mood"]
    embed_color = (
        0x2ECC71 if "탐욕" in mood else
        0xE74C3C if "공포" in mood else
        0x3498DB
    )

    # ── Embed 1: 헤더 + 거시경제 ──────────────────────────────────────────
    indices   = ["S&P500", "NASDAQ", "KOSPI"]
    indicators = ["VIX(공포지수)", "미국10년금리", "USD/KRW"]

    def snapshot(keys):
        lines = []
        for k in keys:
            if k in market:
                d = market[k]
                lines.append(f"`{k}` {d['close']:,}  {_chg(d['change_pct'])}")
        return "\n".join(lines) or "—"

    embed1 = {
        "title": f"📊 데이트레이딩 브리핑 — {date_str}",
        "description": (
            f"시장 심리 **{mood}**  |  리스크 **{analysis['risk_level']}**\n\n"
            f"💡 **오늘의 전략**\n> {analysis['one_line_strategy']}"
        ),
        "color": embed_color,
        "fields": [
            {"name": "📈 주요 지수",       "value": snapshot(indices),    "inline": True},
            {"name": "🌡️ 공포·금리 지표", "value": snapshot(indicators), "inline": True},
            {"name": "​",             "value": "​",             "inline": False},
            {
                "name":  "🌍 거시경제 분석",
                "value": analysis["macro_summary"],
                "inline": False,
            },
        ],
        "timestamp": now_kst.astimezone(pytz.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
    }

    # ── Embed 2: 뉴스 & 시장 영향 ─────────────────────────────────────────
    news_fields = [
        {
            "name":  f"{item['impact']} {item['headline']}",
            "value": item["why_it_matters"],
            "inline": False,
        }
        for item in analysis.get("key_news", [])[:3]
    ]
    embed2 = {
        "title":  "📰 오늘의 핵심 뉴스 & 시장 영향",
        "color":  0xF39C12,
        "fields": news_fields,
    }

    # ── Embed 3: 섹터 & AI 반도체 ─────────────────────────────────────────
    sector_etfs   = ["기술(XLK)", "에너지(XLE)", "금융(XLF)"]
    chip_stocks   = ["NVDA", "AMD", "MU", "SK하이닉스", "삼성전자"]

    def mini_table(keys):
        lines = []
        for k in keys:
            if k in market:
                d = market[k]
                vol = f" ×{d['vol_ratio']:.1f}vol" if d["vol_ratio"] > 1.5 else ""
                lines.append(f"`{k}` {_chg(d['change_pct'])}{vol}")
        return "\n".join(lines) or "—"

    embed3 = {
        "title":  "🏭 섹터 동향 & AI·반도체 심층 분석",
        "color":  0x9B59B6,
        "fields": [
            {"name": "📊 섹터 ETF",        "value": mini_table(sector_etfs), "inline": True},
            {"name": "🤖 AI·반도체 종목",  "value": mini_table(chip_stocks), "inline": True},
            {"name": "​",             "value": "​",               "inline": False},
            {"name": "📉 섹터 흐름",       "value": analysis["sector_analysis"],  "inline": False},
            {"name": "🔬 AI·반도체 분석",  "value": analysis["ai_chip_focus"],    "inline": False},
        ],
    }

    # ── Embed 4: 트레이딩 픽 ──────────────────────────────────────────────
    pick_fields = []
    for pick in analysis.get("trading_picks", [])[:5]:
        flag = "🇰🇷" if pick.get("market") == "KR" else "🇺🇸"
        ri   = _risk_icon(pick["risk"])
        value = (
            f"{pick['reason']}\n"
            f"▶ 진입: {pick['entry_hint']}\n"
            f"✅ 익절 `+{pick['target_pct']}%` ({_krw(pick['target_pct'])})  "
            f"❌ 손절 `-{pick['stop_loss_pct']}%` ({_krw(pick['stop_loss_pct'])})\n"
            f"리스크 {ri} {pick['risk']}"
        )
        pick_fields.append({
            "name":   f"{flag} 🎯 {pick['name']} ({pick['symbol']})",
            "value":  value,
            "inline": False,
        })

    pick_fields.append({
        "name": "⚠️ 리스크 수칙",
        "value": (
            "□ 진입 전 **손절가 주문** 반드시 설정\n"
            "□ 레버리지·SOXL 계좌의 **20% 이하**만\n"
            "□ 1일 손실 **-3% (₩3만)** 시 당일 거래 즉시 중단\n"
            "□ 미국 주식 PDT: 3거래일 **3회 이하**\n"
            "⛔ **이 브리핑은 투자 권유가 아닙니다**"
        ),
        "inline": False,
    })

    embed4 = {
        "title":  "🎯 오늘의 데이트레이딩 픽 (₩100만 기준)",
        "color":  0xE74C3C,
        "fields": pick_fields,
        "footer": {"text": "시작 자본 ₩100만 | 목표 ₩1만~₩5만/일 | Powered by Claude Sonnet"},
    }

    return [embed1, embed2, embed3, embed4]


# ─── Post to Discord ──────────────────────────────────────────────────────────

def post_to_discord(embeds: list[dict]) -> None:
    # Send in batches of 4 (Discord limit is 10, but 4 keeps payload size safe)
    for i in range(0, len(embeds), 4):
        resp = requests.post(
            WEBHOOK_URL,
            json={"embeds": embeds[i:i+4]},
            headers={"Content-Type": "application/json"},
            timeout=15,
        )
        resp.raise_for_status()
    print(f"✅ Posted {len(embeds)} embeds to Discord.")


# ─── Entrypoint ───────────────────────────────────────────────────────────────

def main() -> None:
    now_kst = datetime.now(KST)

    if now_kst.weekday() >= 5:
        print(f"⏸️  Markets closed ({now_kst.strftime('%A')}) — skipping.")
        sys.exit(0)

    print("📡 Fetching market data...")
    market_data = fetch_market_data()
    print(f"   Got {len(market_data)} symbols.")

    print("📰 Fetching news...")
    news = fetch_news()
    print(f"   Got {len(news)} headlines.")

    print("🤖 Analyzing with Claude...")
    analysis = analyze_with_claude(market_data, news)

    print("🎨 Building Discord embeds...")
    embeds = build_embeds(analysis, market_data, datetime.now(KST))

    print("📨 Posting to Discord...")
    post_to_discord(embeds)


if __name__ == "__main__":
    main()

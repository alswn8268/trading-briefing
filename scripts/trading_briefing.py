"""
Daily day-trading briefing with AI-powered economic analysis.
Posts rich Discord embeds every weekday at 08:30 KST.

Required GitHub Secrets:
  DISCORD_WEBHOOK_URL
  ANTHROPIC_API_KEY
"""

import os
import re
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
    "S&P500":        "^GSPC",
    "NASDAQ":        "^IXIC",
    "KOSPI":         "^KS11",
    "VIX(공포지수)":  "^VIX",
    "미국10년금리":   "^TNX",
    "USD/KRW":       "USDKRW=X",
    "기술(XLK)":     "XLK",
    "에너지(XLE)":   "XLE",
    "금융(XLF)":     "XLF",
    "NVDA":          "NVDA",
    "AMD":           "AMD",
    "MU":            "MU",
    "SOXL":          "SOXL",
    "SK하이닉스":    "000660.KS",
    "삼성전자":      "005930.KS",
    "KODEX레버리지": "122630.KS",
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
                "vol_ratio":  round(float(cur["Volume"] / avg_vol), 1) if avg_vol else 1.0,
            }
        except Exception as e:
            print(f"  [warn] {name}: {e}")
    return result


def fetch_news(max_items: int = 8) -> list[dict]:
    headlines = []
    for url in NEWS_FEEDS:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:4]:
                title = entry.get("title", "").strip()
                if title:
                    headlines.append({
                        "title":  title[:120],
                        "source": feed.feed.get("title", ""),
                    })
        except Exception as e:
            print(f"  [warn] RSS {url}: {e}")
    return headlines[:max_items]


# ─── AI Analysis (Claude) ─────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "당신은 주식 시장 분석가입니다. "
    "한국 소액 데이트레이더(₩100만)를 위한 브리핑을 작성합니다. "
    "전문 용어는 괄호 안에 쉽게 풀어쓰고, 수치 근거를 포함하세요. "
    "반드시 완전한 유효한 JSON만 반환하세요."
)

PROMPT_TEMPLATE = """\
아래 데이터를 분석해 JSON을 반환하세요. 모든 문자열은 80자 이내로 작성하세요.

시장 데이터: {market_json}

뉴스: {news_json}

반환 형식:
{{
  "market_mood": "공포 😨 또는 중립 😐 또는 탐욕 🤑",
  "risk_level": "낮음 🟢 또는 보통 🟡 또는 높음 🔴",
  "one_line_strategy": "오늘 핵심 전략 (30자 이내)",
  "macro_summary": "거시경제 흐름 3문장. 금리·달러·경기 관점, 쉬운 말로.",
  "key_news": [
    {{
      "headline": "뉴스 제목 한국어 (25자 이내)",
      "why_it_matters": "주식 영향 2문장.",
      "impact": "긍정적 📈 또는 부정적 📉 또는 중립 ➡️"
    }}
  ],
  "sector_analysis": "섹터 자금 흐름 2-3문장.",
  "ai_chip_focus": "AI·반도체 현황 2-3문장.",
  "trading_picks": [
    {{
      "name": "종목명",
      "symbol": "티커",
      "market": "KR 또는 US",
      "reason": "주목 이유 2문장.",
      "entry_hint": "진입 힌트 (30자 이내)",
      "stop_loss_pct": 1.5,
      "target_pct": 2.5,
      "risk": "낮음 또는 보통 또는 높음"
    }}
  ]
}}

trading_picks 정확히 5개, key_news 정확히 3개."""


def _extract_json(raw: str) -> dict:
    """JSON 블록을 추출하고 파싱합니다."""
    # 마크다운 코드펜스 제거
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    # 가장 바깥쪽 { } 블록 추출
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        raw = m.group(0)
    return json.loads(raw)


def analyze_with_claude(market_data: dict, news: list[dict]) -> dict:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    prompt = PROMPT_TEMPLATE.format(
        market_json=json.dumps(market_data, ensure_ascii=False),
        news_json=json.dumps(news, ensure_ascii=False),
    )

    for attempt in range(3):
        try:
            resp = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                # prefill: { 로 시작을 강제해 반드시 JSON이 나오게 함
                messages=[
                    {"role": "user",      "content": prompt},
                    {"role": "assistant", "content": "{"},
                ],
            )

            if not resp.content or not resp.content[0].text.strip():
                raise ValueError(f"Empty Claude response (stop_reason={resp.stop_reason})")

            # prefill 한 글자 { 를 앞에 다시 붙여서 완전한 JSON 복원
            raw = "{" + resp.content[0].text
            return _extract_json(raw)

        except (json.JSONDecodeError, ValueError) as e:
            print(f"  [warn] attempt {attempt + 1}/3 failed: {e}")
            if attempt == 2:
                raise RuntimeError(f"Claude 분석 실패 (3회 시도): {e}") from e

    raise RuntimeError("unreachable")


# ─── Discord Embed Builder ────────────────────────────────────────────────────

def _chg(pct: float) -> str:
    arrow = "▲" if pct >= 0 else "▼"
    icon  = "📈" if pct >= 1 else ("📉" if pct <= -1 else "➡️")
    return f"{icon} {arrow}{abs(pct):.2f}%"


def _krw(pct: float) -> str:
    return f"₩{int(CAPITAL * pct / 100):,}"


def _risk_icon(risk: str) -> str:
    return {"낮음": "🟢", "보통": "🟡", "높음": "🔴"}.get(risk, "🟡")


def _trim(text: str, limit: int = 1020) -> str:
    """Discord 필드 값 1024자 제한 대응."""
    return text if len(text) <= limit else text[:limit - 3] + "..."


def build_embeds(analysis: dict, market: dict, now_kst: datetime) -> list[dict]:
    date_str = now_kst.strftime("%Y년 %m월 %d일 (%a)")
    mood = analysis.get("market_mood", "중립 😐")
    embed_color = (
        0x2ECC71 if "탐욕" in mood else
        0xE74C3C if "공포" in mood else
        0x3498DB
    )

    # ── Embed 1: 헤더 + 거시경제 ──────────────────────────────────────────
    def snapshot(keys: list[str]) -> str:
        lines = [
            f"`{k}` {market[k]['close']:,}  {_chg(market[k]['change_pct'])}"
            for k in keys if k in market
        ]
        return "\n".join(lines) or "—"

    embed1 = {
        "title": f"📊 데이트레이딩 브리핑 — {date_str}",
        "description": _trim(
            f"시장 심리 **{mood}**  |  리스크 **{analysis.get('risk_level', '—')}**\n\n"
            f"💡 **오늘의 전략**\n> {analysis.get('one_line_strategy', '—')}",
            4090,
        ),
        "color": embed_color,
        "fields": [
            {
                "name":   "📈 주요 지수",
                "value":  _trim(snapshot(["S&P500", "NASDAQ", "KOSPI"])),
                "inline": True,
            },
            {
                "name":   "🌡️ 공포·금리",
                "value":  _trim(snapshot(["VIX(공포지수)", "미국10년금리", "USD/KRW"])),
                "inline": True,
            },
            {
                "name":   "🌍 거시경제 분석",
                "value":  _trim(analysis.get("macro_summary", "—")),
                "inline": False,
            },
        ],
        "timestamp": now_kst.astimezone(pytz.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
    }

    # ── Embed 2: 핵심 뉴스 ────────────────────────────────────────────────
    news_fields = [
        {
            "name":   _trim(f"{item.get('impact','➡️')} {item.get('headline','뉴스')}", 255),
            "value":  _trim(item.get("why_it_matters", "—")),
            "inline": False,
        }
        for item in analysis.get("key_news", [])[:3]
    ]
    embed2 = {
        "title":  "📰 오늘의 핵심 뉴스 & 시장 영향",
        "color":  0xF39C12,
        "fields": news_fields or [{"name": "뉴스 없음", "value": "—", "inline": False}],
    }

    # ── Embed 3: 섹터 & AI 반도체 ─────────────────────────────────────────
    embed3 = {
        "title":  "🏭 섹터 동향 & AI·반도체 분석",
        "color":  0x9B59B6,
        "fields": [
            {
                "name":   "📊 섹터 ETF",
                "value":  _trim(snapshot(["기술(XLK)", "에너지(XLE)", "금융(XLF)"])),
                "inline": True,
            },
            {
                "name":   "🤖 AI·반도체",
                "value":  _trim(snapshot(["NVDA", "AMD", "MU", "SK하이닉스", "삼성전자"])),
                "inline": True,
            },
            {
                "name":   "📉 섹터 흐름",
                "value":  _trim(analysis.get("sector_analysis", "—")),
                "inline": False,
            },
            {
                "name":   "🔬 AI·반도체 심층",
                "value":  _trim(analysis.get("ai_chip_focus", "—")),
                "inline": False,
            },
        ],
    }

    # ── Embed 4: 트레이딩 픽 ──────────────────────────────────────────────
    pick_fields = []
    for pick in analysis.get("trading_picks", [])[:5]:
        flag = "🇰🇷" if pick.get("market") == "KR" else "🇺🇸"
        ri   = _risk_icon(pick.get("risk", "보통"))
        sl   = pick.get("stop_loss_pct", 1.5)
        tp   = pick.get("target_pct", 2.0)
        value = _trim(
            f"{pick.get('reason', '—')}\n"
            f"▶ 진입: {pick.get('entry_hint', '—')}\n"
            f"✅ 익절 `+{tp}%` ({_krw(tp)})  "
            f"❌ 손절 `-{sl}%` ({_krw(sl)})\n"
            f"리스크 {ri} {pick.get('risk', '보통')}"
        )
        pick_fields.append({
            "name":   _trim(f"{flag} 🎯 {pick.get('name','?')} ({pick.get('symbol','?')})", 255),
            "value":  value,
            "inline": False,
        })

    pick_fields.append({
        "name":  "⚠️ 리스크 수칙",
        "value": (
            "□ 진입 전 **손절가 주문** 반드시 설정\n"
            "□ 레버리지·SOXL 계좌의 **20% 이하**만\n"
            "□ 1일 손실 **-3% (₩3만)** 시 즉시 거래 중단\n"
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
    for i in range(0, len(embeds), 4):
        resp = requests.post(
            WEBHOOK_URL,
            json={"embeds": embeds[i:i+4]},
            headers={"Content-Type": "application/json"},
            timeout=15,
        )
        if not resp.ok:
            print(f"  [error] Discord HTTP {resp.status_code}: {resp.text[:300]}")
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

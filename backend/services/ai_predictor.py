"""
AI-powered match prediction using Gemini Flash.
Combines our Poisson model output + live market data into the prediction framework.
"""
import httpx
from config import settings

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-1.5-flash:generateContent"
)


def _build_prompt(
    home_team: str,
    away_team: str,
    stage: str,
    kickoff: str,
    prediction: dict,
    odds_summary: dict,
    market_trends: dict,
) -> str:
    p = prediction or {}
    home_win  = round((p.get("home_win_prob", 0)) * 100, 1)
    draw      = round((p.get("draw_prob", 0)) * 100, 1)
    away_win  = round((p.get("away_win_prob", 0)) * 100, 1)
    home_xg   = p.get("expected_home_goals", "?")
    away_xg   = p.get("expected_away_goals", "?")
    score     = p.get("predicted_score", "?")
    over25    = round((p.get("over25_prob", 0)) * 100, 1)
    under25   = round((p.get("under25_prob", 0)) * 100, 1)
    btts_yes  = round((p.get("btts_yes_prob", 0)) * 100, 1)
    btts_no   = round((p.get("btts_no_prob", 0)) * 100, 1)

    # Format odds block
    odds_block = ""
    for m in odds_summary.get("markets", []):
        bt = m.get("bet_type", "")
        line = m.get("line", "")
        c = m.get("consensus", {})
        bo = m.get("best_odds", {})
        money = m.get("money_on", "unclear")
        sharp = m.get("sharp_signal")
        sharp_txt = ""
        if sharp and sharp.get("direction") != "neutral":
            sharp_txt = f" | Pinnacle sharp signal: {sharp['direction']} ({sharp['strength']})"

        if bt == "1X2":
            odds_block += (
                f"\n  Match Odds: Home {bo.get('home',{}).get('odds','?')} "
                f"/ Draw {bo.get('draw',{}).get('odds','?') if bo.get('draw') else '–'} "
                f"/ Away {bo.get('away',{}).get('odds','?')}"
                f"\n  Market implied: Home {round((c.get('home') or 0)*100,1)}% "
                f"Draw {round((c.get('draw') or 0)*100,1)}% "
                f"Away {round((c.get('away') or 0)*100,1)}%"
                f"\n  Money flow: {money}{sharp_txt}"
            )
        elif bt == "O/U":
            odds_block += (
                f"\n  Over/Under {line}: Over {bo.get('home',{}).get('odds','?')} "
                f"/ Under {bo.get('away',{}).get('odds','?')}"
                f" | Money flow: {money}{sharp_txt}"
            )
        elif bt == "AH":
            odds_block += (
                f"\n  Asian Handicap {line:+.1f}: "
                f"Home {bo.get('home',{}).get('odds','?')} "
                f"/ Away {bo.get('away',{}).get('odds','?')}"
                f" | Money flow: {money}{sharp_txt}"
            )

    # Movement summary
    move_lines = []
    for m in odds_summary.get("markets", []):
        mv = m.get("movement")
        if mv and mv.get("snapshots", 0) >= 2:
            move_lines.append(
                f"{m['bet_type']}: home {mv['home_trend']}, away {mv['away_trend']}"
            )
    movement_block = ", ".join(move_lines) if move_lines else "No movement data yet (need 2+ syncs)"

    return f"""You are a football match predictor. For {home_team} vs {away_team} \
(WC2026, {stage}, {kickoff}), produce a prediction using this framework:

=== OUR STATISTICAL MODEL OUTPUT ===
Win probability: {home_team} {home_win}% | Draw {draw}% | {away_team} {away_win}%
Expected goals: {home_team} {home_xg} xG — {away_team} {away_xg} xG
Model predicted score: {score}
Over 2.5: {over25}% | Under 2.5: {under25}%
BTTS Yes: {btts_yes}% | BTTS No: {btts_no}%

=== LIVE BETTING MARKET DATA ===
{odds_block if odds_block else "  No odds data available yet."}

Line movement since first sync: {movement_block}

=== PREDICTION FRAMEWORK ===

1. CONTEXT: stakes for each side (must-win / content-to-draw?), \
home edge (WC2026 is neutral venues), what a win/draw means for the group.

2. THREE PILLARS — use the model output above plus your knowledge:
   - Statistical models (use the xG and win % from our model above)
   - Betting market (use the odds and money flow data above)
   - Expert consensus (your knowledge of tipster views, supercomputer outputs)

3. FORM / TACTICS / PERSONNEL: recent form + xG trend, \
press-vs-low-block matchup, key injuries/returns, GK form. Treat H2H as secondary.

4. SITUATIONAL OVERRIDE (decisive for the goals line):
   - Favorite vs MUST-WIN underdog with LEAKY defense → lean OVER / handicap
   - Favorite vs DISCIPLINED content-to-draw underdog → lean UNDER / BTTS No
   - If situational and historical reads agree → high conviction.

5. VALUE: compare our model % vs market implied %. Flag the biggest gap. \
Note "win-but-concede" favorites (back the handicap, not the CS).

OUTPUT FORMAT (be direct, no hedging):
- One committed scoreline
- Goals line + BTTS read with reasoning
- Value flag (our model vs market — biggest gap)
- Best market(s) ranked by confidence
- Key lineup variable to watch before kickoff"""


async def generate_prediction(
    home_team: str,
    away_team: str,
    stage: str,
    kickoff: str,
    prediction: dict,
    odds_summary: dict,
    market_trends: dict,
) -> str:
    if not settings.gemini_api_key:
        return "Gemini API key not configured. Add GEMINI_API_KEY to backend/.env."

    prompt = _build_prompt(
        home_team, away_team, stage, kickoff,
        prediction, odds_summary, market_trends,
    )

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            f"{GEMINI_URL}?key={settings.gemini_api_key}",
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.4,
                    "maxOutputTokens": 1200,
                },
            },
        )
        data = r.json()

    if r.status_code != 200:
        error = data.get("error", {}).get("message", str(data))
        return f"Gemini API error: {error}"

    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        return f"Unexpected Gemini response: {data}"

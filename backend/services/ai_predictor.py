import httpx
from config import settings
from datetime import date

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = (
    "You are an elite football betting analyst specialising in tournament football. "
    "You give direct, committed predictions — no hedging, no 'could go either way'. "
    "You prioritise actionable betting markets over vague match previews. "
    "When the statistical model and market agree, you say so and raise conviction. "
    "When they diverge, you explain why and flag the value side."
)


def _fmt_line(line) -> str:
    if line is None:
        return ""
    try:
        v = float(line)
        return f"{v:+.1f}"
    except (TypeError, ValueError):
        return str(line)


def _build_prompt(
    home_team: str,
    away_team: str,
    stage: str,
    kickoff: str,
    prediction: dict,
    odds_summary: dict,
    value_bets: list,
) -> str:
    p = prediction or {}
    home_win = round(p.get("home_win_prob", 0) * 100, 1)
    draw     = round(p.get("draw_prob", 0) * 100, 1)
    away_win = round(p.get("away_win_prob", 0) * 100, 1)
    home_xg  = p.get("expected_home_goals", "?")
    away_xg  = p.get("expected_away_goals", "?")
    score    = p.get("predicted_score", "?")
    over25   = round(p.get("over25_prob", 0) * 100, 1)
    under25  = round(p.get("under25_prob", 0) * 100, 1)
    btts_yes = round(p.get("btts_yes_prob", 0) * 100, 1)
    btts_no  = round(p.get("btts_no_prob", 0) * 100, 1)

    # --- Odds block ---
    odds_lines = []
    for m in odds_summary.get("markets", []):
        bt    = m.get("bet_type", "")
        line  = m.get("line")
        c     = m.get("consensus", {})
        bo    = m.get("best_odds", {})
        money = m.get("money_on") or "unclear"
        sharp = m.get("sharp_signal")
        mv    = m.get("movement")

        sharp_txt = ""
        if sharp and sharp.get("direction") != "neutral":
            sharp_txt = f" | Pinnacle: {sharp['direction']} ({sharp['strength']})"

        move_txt = ""
        if mv and mv.get("snapshots", 0) >= 2:
            move_txt = f" | Move: home {mv['home_trend']}, away {mv['away_trend']}"

        if bt == "1X2":
            h_imp = round((c.get("home") or 0) * 100, 1)
            d_imp = round((c.get("draw") or 0) * 100, 1)
            a_imp = round((c.get("away") or 0) * 100, 1)
            h_odds = bo.get("home", {}).get("odds", "?")
            d_odds = bo.get("draw", {}).get("odds", "?") if bo.get("draw") else "–"
            a_odds = bo.get("away", {}).get("odds", "?")
            odds_lines.append(
                f"  1X2 best odds: {home_team} {h_odds} / Draw {d_odds} / {away_team} {a_odds}\n"
                f"  Market implied: {home_team} {h_imp}% | Draw {d_imp}% | {away_team} {a_imp}%\n"
                f"  Money flow: {money}{sharp_txt}{move_txt}"
            )
        elif bt == "O/U":
            over_odds  = bo.get("home", {}).get("odds", "?")
            under_odds = bo.get("away", {}).get("odds", "?")
            odds_lines.append(
                f"  O/U {line}: Over {over_odds} / Under {under_odds}"
                f" | Money: {money}{sharp_txt}{move_txt}"
            )
        elif bt == "AH":
            h_odds = bo.get("home", {}).get("odds", "?")
            a_odds = bo.get("away", {}).get("odds", "?")
            odds_lines.append(
                f"  AH {_fmt_line(line)}: {home_team} {h_odds} / {away_team} {a_odds}"
                f" | Money: {money}{sharp_txt}{move_txt}"
            )

    odds_block = "\n".join(odds_lines) if odds_lines else "  No odds data available yet."

    # --- Value bets block (our model's best edges) ---
    value_lines = []
    for vb in sorted(value_bets or [], key=lambda x: x.get("edge", 0), reverse=True)[:5]:
        sel      = vb.get("selection", "")
        bt       = vb.get("bet_type", "")
        our_prob = round(vb.get("our_probability", 0) * 100, 1)
        mkt_prob = round(vb.get("implied_probability", 0) * 100, 1)
        edge     = round(vb.get("edge", 0) * 100, 1)
        bk_odds  = vb.get("bookmaker_odds", "?")
        bk       = vb.get("bookmaker", "")
        kelly    = round(vb.get("kelly_fraction", 0) * 100, 1)
        value_lines.append(
            f"  {bt} — {sel}: our {our_prob}% vs market {mkt_prob}% "
            f"| edge +{edge}% | @ {bk_odds} ({bk}) | Kelly {kelly}%"
        )
    value_block = (
        "\n".join(value_lines)
        if value_lines
        else "  No value bets detected (model and market are aligned)."
    )

    today = date.today().strftime("%d %b %Y")

    return f"""You are a football match predictor. For {home_team} vs {away_team} \
(FIFA World Cup 2026, {stage}, neutral venue, {kickoff}), produce a prediction \
using this framework:

Analysis date: {today} — WC2026 is currently in progress. Use your knowledge of \
how these teams have performed in the tournament so far.

=== DATA WE HAVE ===

STATISTICAL MODEL (our Poisson sim — treat as the supercomputer input):
  Win probability: {home_team} {home_win}% | Draw {draw}% | {away_team} {away_win}%
  Expected goals:  {home_team} {home_xg} xG | {away_team} {away_xg} xG
  Sim correct score: {score}
  Over 2.5: {over25}% | Under 2.5: {under25}%
  BTTS Yes: {btts_yes}% | BTTS No: {btts_no}%

BETTING MARKET:
{odds_block}

MODEL EDGE vs MARKET (our value flags):
{value_block}

=== PREDICTION FRAMEWORK ===

1. CONTEXT: stakes for each side (must-win / content-to-draw?), \
no home edge (neutral venue), what a win/draw means for group progression.

2. THREE PILLARS — gather and state each:
   - Statistical models: use our sim output above (win %, xG, correct score)
   - Betting market: use the moneyline, implied %, O/U line, money flow and \
any movement from the market data above
   - Expert consensus: draw on your training knowledge — where do analysts, \
tipsters and supercomputers typically cluster for these two sides? \
Note where they agree or split.

3. FORM / TACTICS / PERSONNEL: use your knowledge of these teams at WC2026 — \
recent form + xG trend (finishing or just dominating?), press-vs-low-block \
matchup, key injuries/returns, GK form. Treat H2H as secondary.

4. SITUATIONAL OVERRIDE (decisive for the goals line):
   - Favourite vs MUST-WIN underdog with a LEAKY defence → weight \
early-goal/chase-and-counter risk ABOVE clean-sheet history → \
lean OVER / handicap; fade the Under.
   - Favourite vs DISCIPLINED, content-to-draw underdog with little \
attack → underdog sits deep → lean UNDER / BTTS No.
   - If situational and historical reads agree → high conviction. \
If they conflict → trust the live setup.

5. VALUE: compare our model % vs market implied % from the value flags above. \
Flag the biggest gap. Note "win-but-concede" favourites \
(back the handicap, not the clean sheet).

OUTPUT (be direct, no hedging):
- One committed scoreline
- Goals line + BTTS read with reasoning
- Value flag (our model vs market — biggest gap)
- Best market(s) ranked by risk + the one lineup variable to confirm before kickoff"""


async def generate_prediction(
    home_team: str,
    away_team: str,
    stage: str,
    kickoff: str,
    prediction: dict,
    odds_summary: dict,
    value_bets: list,
) -> str:
    if not settings.groq_api_key:
        return "Groq API key not configured. Add GROQ_API_KEY to backend/.env."

    prompt = _build_prompt(
        home_team, away_team, stage, kickoff,
        prediction, odds_summary, value_bets,
    )

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {settings.groq_api_key}"},
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.4,
                "max_tokens": 1400,
            },
        )
        data = r.json()

    if r.status_code != 200:
        error = data.get("error", {}).get("message", str(data))
        return f"Groq API error: {error}"

    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        return f"Unexpected Groq response: {data}"

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from database import get_db
from db.crud import get_upcoming_fixtures, get_fixture_by_id, get_value_bets_for_fixture
from models.schemas import FixtureSchema, FixtureWithValueBets, ValueBetSchema
from models.db_models import OddsSnapshot, Fixture
from config import settings
from services import betfair_client
from services.ai_predictor import generate_prediction

router = APIRouter(prefix="/fixtures", tags=["fixtures"])


@router.get("/upcoming", response_model=list[FixtureSchema])
async def upcoming_fixtures(days: int = 7, db: AsyncSession = Depends(get_db)):
    return await get_upcoming_fixtures(db, days)


@router.get("/{fixture_id}/odds-movement")
async def odds_movement(fixture_id: int, db: AsyncSession = Depends(get_db)):
    """
    Return the historical odds snapshots for a fixture grouped by (bookmaker, bet_type, line).
    Each group contains ordered snapshots showing how odds drifted over time.
    """
    result = await db.execute(
        select(OddsSnapshot)
        .where(OddsSnapshot.fixture_id == fixture_id)
        .order_by(OddsSnapshot.bookmaker, OddsSnapshot.bet_type, OddsSnapshot.line, OddsSnapshot.captured_at)
    )
    snapshots = result.scalars().all()

    # Group into series
    groups: dict[tuple, list] = {}
    for s in snapshots:
        key = (s.bookmaker, s.bet_type, s.line)
        if key not in groups:
            groups[key] = []
        groups[key].append({
            "captured_at": s.captured_at.isoformat(),
            "home_odds": s.home_odds,
            "draw_odds": s.draw_odds,
            "away_odds": s.away_odds,
        })

    # Only return series that have at least one data point
    series = [
        {
            "bookmaker": k[0],
            "bet_type": k[1],
            "line": k[2],
            "history": v,
        }
        for k, v in groups.items()
        if v
    ]
    # Sort: most data points first (most interesting)
    series.sort(key=lambda x: len(x["history"]), reverse=True)
    return series


@router.get("/{fixture_id}/market-trends")
async def market_trends(fixture_id: int, db: AsyncSession = Depends(get_db)):
    """
    Compute betting trend signals from multi-bookmaker odds.
    - Consensus probability: average implied prob across all books (overround removed)
    - Sharp signal: Pinnacle implied prob vs soft-book average
    - Line movement: how odds shifted between syncs (shortening = money coming in)
    - Money direction: combined signal from sharp gap + movement
    """
    from sqlalchemy.orm import selectinload
    from collections import defaultdict

    result = await db.execute(
        select(Fixture).where(Fixture.id == fixture_id).options(selectinload(Fixture.odds))
    )
    fixture = result.scalar_one_or_none()
    if not fixture:
        raise HTTPException(status_code=404, detail="Fixture not found")

    if not fixture.odds:
        return {"markets": [], "message": "No odds data — run Sync Odds first."}

    # Load snapshots for line movement history
    snaps_result = await db.execute(
        select(OddsSnapshot)
        .where(OddsSnapshot.fixture_id == fixture_id)
        .order_by(OddsSnapshot.captured_at)
    )
    snapshots = snaps_result.scalars().all()

    # Group current odds by (bet_type, line)
    by_market: dict[tuple, list] = defaultdict(list)
    for o in fixture.odds:
        by_market[(o.bet_type, o.line)].append(o)

    # Group snapshots by (bet_type, line)
    snap_by_market: dict[tuple, list] = defaultdict(list)
    for s in snapshots:
        snap_by_market[(s.bet_type, s.line)].append(s)

    def implied(val):
        return (1 / val) if val and val > 1.01 else None

    def trend_label(move):
        if move is None: return "stable"
        if move < -0.03: return "shortening"
        if move > 0.03:  return "drifting"
        return "stable"

    output = []
    for (bet_type, line), odds_list in by_market.items():
        if len(odds_list) < 2:
            continue

        pinnacle = next((o for o in odds_list if "pinnacle" in o.bookmaker.lower()), None)
        soft = [o for o in odds_list if "pinnacle" not in o.bookmaker.lower()]

        # Consensus: average implied prob across ALL books, then remove overround
        def avg_implied(attr):
            vals = [implied(getattr(o, attr)) for o in odds_list if implied(getattr(o, attr))]
            return sum(vals) / len(vals) if vals else None

        raw_home = avg_implied("home_odds")
        raw_draw = avg_implied("draw_odds")
        raw_away = avg_implied("away_odds")
        total = (raw_home or 0) + (raw_draw or 0) + (raw_away or 0)
        if total > 0:
            consensus = {
                "home": round(raw_home / total, 4) if raw_home else None,
                "draw": round(raw_draw / total, 4) if raw_draw else None,
                "away": round(raw_away / total, 4) if raw_away else None,
            }
        else:
            consensus = {"home": None, "draw": None, "away": None}

        # Best available odds across all books
        def best(attr, bk_attr="bookmaker"):
            vals = [(getattr(o, attr), o.bookmaker) for o in odds_list if getattr(o, attr)]
            return max(vals, key=lambda x: x[0]) if vals else (None, None)

        bh, bh_bk = best("home_odds")
        bd, bd_bk = best("draw_odds")
        ba, ba_bk = best("away_odds")

        # Sharp signal: Pinnacle implied vs soft-book average
        sharp_signal = None
        if pinnacle and soft:
            soft_home = [implied(o.home_odds) for o in soft if implied(o.home_odds)]
            soft_away = [implied(o.away_odds) for o in soft if implied(o.away_odds)]
            pin_home = implied(pinnacle.home_odds)
            pin_away = implied(pinnacle.away_odds)
            if soft_home and pin_home:
                gap_home = pin_home - (sum(soft_home) / len(soft_home))
                gap_away = (pin_away - (sum(soft_away) / len(soft_away))) if (soft_away and pin_away) else 0
                if abs(gap_home) >= 0.02:
                    direction = "home" if gap_home > 0 else "away"
                    strength = "strong" if abs(gap_home) >= 0.05 else "mild"
                else:
                    direction = "neutral"
                    strength = "neutral"
                sharp_signal = {
                    "direction": direction,
                    "strength": strength,
                    "home_gap": round(gap_home, 4),
                    "away_gap": round(gap_away, 4),
                }

        # Line movement from snapshots
        movement = None
        msnaps = snap_by_market.get((bet_type, line), [])
        if len(msnaps) >= 2:
            first, last = msnaps[0], msnaps[-1]
            hm = (last.home_odds - first.home_odds) if (last.home_odds and first.home_odds) else None
            am = (last.away_odds - first.away_odds) if (last.away_odds and first.away_odds) else None
            movement = {
                "home_trend": trend_label(hm),
                "away_trend": trend_label(am),
                "home_move": round(hm, 3) if hm else None,
                "away_move": round(am, 3) if am else None,
                "snapshots": len(msnaps),
            }

        # Combined money direction signal
        money_on = None
        if sharp_signal and sharp_signal["direction"] != "neutral":
            money_on = sharp_signal["direction"]
        elif movement:
            h_short = movement["home_trend"] == "shortening"
            a_short = movement["away_trend"] == "shortening"
            if h_short and not a_short: money_on = "home"
            elif a_short and not h_short: money_on = "away"

        output.append({
            "bet_type": bet_type,
            "line": line,
            "bookmaker_count": len(odds_list),
            "has_pinnacle": pinnacle is not None,
            "consensus": consensus,
            "best_odds": {
                "home": {"odds": bh, "bookmaker": bh_bk},
                "draw": {"odds": bd, "bookmaker": bd_bk} if bd else None,
                "away": {"odds": ba, "bookmaker": ba_bk},
            },
            "sharp_signal": sharp_signal,
            "movement": movement,
            "money_on": money_on,
        })

    order = {"1X2": 0, "O/U": 1, "BTTS": 2, "AH": 3}
    output.sort(key=lambda x: order.get(x["bet_type"], 99))
    return {"markets": output}


@router.get("/{fixture_id}/calculator-hints")
async def calculator_hints(fixture_id: int, db: AsyncSession = Depends(get_db)):
    """
    Auto-fill hints for the /calculator page.
    Returns suggested band width, trap flags, and reasons for each.
    """
    from sqlalchemy.orm import selectinload
    from sqlalchemy import or_

    result = await db.execute(
        select(Fixture)
        .where(Fixture.id == fixture_id)
        .options(
            selectinload(Fixture.home_team),
            selectinload(Fixture.away_team),
            selectinload(Fixture.odds),
        )
    )
    fixture = result.scalar_one_or_none()
    if not fixture:
        raise HTTPException(status_code=404, detail="Fixture not found")

    home_id = fixture.home_team_id
    away_id = fixture.away_team_id
    home_group = fixture.home_team.group
    away_group = fixture.away_team.group

    # ── 1. Games played per team ─────────────────────────────────────────
    from models.db_models import BettingOdds, OddsSnapshot, ValueBet

    async def games_played(team_id: int) -> int:
        r = await db.execute(
            select(Fixture).where(
                Fixture.status == "FT",
                or_(Fixture.home_team_id == team_id, Fixture.away_team_id == team_id),
            )
        )
        return len(r.scalars().all())

    home_games = await games_played(home_id)
    away_games = await games_played(away_id)
    min_games  = min(home_games, away_games)

    # ── 2. Suggested band ────────────────────────────────────────────────
    if min_games <= 1:   band = 0.30
    elif min_games <= 2: band = 0.25
    elif min_games <= 3: band = 0.20
    else:                band = 0.15

    # ── 3. Recency trap ──────────────────────────────────────────────────
    recency = min_games <= 2
    recency_reason = (
        f"{fixture.home_team.name} has {home_games} WC game(s), "
        f"{fixture.away_team.name} has {away_games} — "
        "small sample makes one scoreline very distorting"
        if recency else "Both teams have enough WC data"
    )

    # ── 4. Steam trap — first vs latest 1X2 snapshot ────────────────────
    snaps_result = await db.execute(
        select(OddsSnapshot)
        .where(OddsSnapshot.fixture_id == fixture_id, OddsSnapshot.bet_type == "1X2")
        .order_by(OddsSnapshot.captured_at)
    )
    snaps = snaps_result.scalars().all()

    steam = False
    steam_reason = "No odds movement data yet (only one sync run)"
    if len(snaps) >= 2:
        first, last = snaps[0], snaps[-1]
        hm = (last.home_odds - first.home_odds) if (last.home_odds and first.home_odds) else 0
        am = (last.away_odds - first.away_odds) if (last.away_odds and first.away_odds) else 0
        max_move = max(abs(hm), abs(am))
        if max_move >= 0.08:
            steam = True
            side   = fixture.home_team.name if abs(hm) > abs(am) else fixture.away_team.name
            mover  = hm if abs(hm) > abs(am) else am
            direction = "shortened" if mover < 0 else "drifted out"
            steam_reason = (
                f"{side} {direction} {abs(mover):.2f} pts since first sync "
                f"({first.captured_at.strftime('%d %b %H:%M')} → "
                f"{last.captured_at.strftime('%d %b %H:%M')})"
            )
        else:
            steam_reason = f"Odds stable — max move {max_move:.2f} pts across {len(snaps)} syncs"

    # ── 5. Consensus trap — no-vig favourite probability ─────────────────
    odds_1x2 = [o for o in fixture.odds if o.bet_type == "1X2"]
    consensus = False
    consensus_reason = "Odds not synced yet"
    if odds_1x2:
        def avg_no_vig(attr: str) -> float:
            vals = [1 / getattr(o, attr) for o in odds_1x2 if getattr(o, attr) and getattr(o, attr) > 1]
            if not vals: return 0.0
            raw = sum(vals) / len(vals)
            total = sum(
                (1 / getattr(o, a) for o in odds_1x2 for a in ("home_odds", "draw_odds", "away_odds")
                 if getattr(o, a) and getattr(o, a) > 1),
                0.0,
            ) / len(odds_1x2)
            overround = total
            return raw / overround if overround else raw

        nv_home = avg_no_vig("home_odds")
        nv_away = avg_no_vig("away_odds")
        fav_prob = max(nv_home, nv_away)
        fav_name = fixture.home_team.name if nv_home > nv_away else fixture.away_team.name
        if fav_prob >= 0.65:
            consensus = True
            consensus_reason = (
                f"{fav_name} at {round(fav_prob * 100)}% no-vig — heavy public favourite, "
                "price is already shaded by the crowd"
            )
        else:
            consensus_reason = f"Favourite at {round(fav_prob * 100)}% no-vig — not extreme, no consensus flag"

    # ── 6. Gamestate trap — group standings ──────────────────────────────
    gamestate = False
    gamestate_reason = "Check group standings manually"
    try:
        if home_group and home_group == away_group:
            # Both teams in same group — compute standings
            group_fixtures_r = await db.execute(
                select(Fixture)
                .where(Fixture.stage == "Group Stage", Fixture.status == "FT")
                .options(selectinload(Fixture.home_team), selectinload(Fixture.away_team))
            )
            group_fixtures = [
                f for f in group_fixtures_r.scalars().all()
                if f.home_team.group == home_group or f.away_team.group == home_group
            ]

            points: dict[int, int] = {}
            for gf in group_fixtures:
                for tid in (gf.home_team_id, gf.away_team_id):
                    points.setdefault(tid, 0)
                hg, ag = gf.home_goals or 0, gf.away_goals or 0
                if hg > ag:
                    points[gf.home_team_id] += 3
                elif hg == ag:
                    points[gf.home_team_id] += 1
                    points[gf.away_team_id] += 1
                else:
                    points[gf.away_team_id] += 3

            pts_home = points.get(home_id, 0)
            pts_away = points.get(away_id, 0)

            # Simulate draw: would both teams be safe (top 2 of 3-team group)?
            group_teams = set()
            for gf in group_fixtures:
                group_teams.add(gf.home_team_id)
                group_teams.add(gf.away_team_id)
            group_teams.update([home_id, away_id])

            others = [tid for tid in group_teams if tid not in (home_id, away_id)]
            other_pts = [points.get(tid, 0) for tid in others]
            max_other = max(other_pts) if other_pts else 0

            draw_home = pts_home + 1
            draw_away = pts_away + 1
            both_safe_on_draw = draw_home > max_other and draw_away > max_other

            if both_safe_on_draw:
                gamestate = True
                gamestate_reason = (
                    f"Both teams qualify if they draw: "
                    f"{fixture.home_team.name} would have {draw_home} pts, "
                    f"{fixture.away_team.name} {draw_away} pts — "
                    f"rival(s) stuck on {max_other} pts"
                )
            else:
                gamestate_reason = (
                    f"Draw does NOT guarantee both teams — "
                    f"{fixture.home_team.name} {pts_home} pts, "
                    f"{fixture.away_team.name} {pts_away} pts currently"
                )
        else:
            gamestate_reason = "Teams in different groups — no shared draw incentive"
    except Exception:
        gamestate_reason = "Could not compute standings — check manually"

    return {
        "games_played": {"home": home_games, "away": away_games},
        "suggested_band": band,
        "traps": {
            "gamestate":  gamestate,
            "consensus":  consensus,
            "recency":    recency,
            "steam":      steam,
        },
        "reasons": {
            "gamestate":  gamestate_reason,
            "consensus":  consensus_reason,
            "recency":    recency_reason,
            "steam":      steam_reason,
        },
    }


@router.get("/{fixture_id}/ai-prediction")
async def ai_prediction(fixture_id: int, db: AsyncSession = Depends(get_db)):
    """
    Generate a full AI match analysis using the prediction framework.
    Feeds our Poisson output + live market data into Gemini Flash.
    """
    from sqlalchemy.orm import selectinload
    from services.ai_predictor import generate_prediction
    from datetime import timezone

    result = await db.execute(
        select(Fixture)
        .where(Fixture.id == fixture_id)
        .options(
            selectinload(Fixture.home_team),
            selectinload(Fixture.away_team),
            selectinload(Fixture.prediction),
            selectinload(Fixture.odds),
        )
    )
    fixture = result.scalar_one_or_none()
    if not fixture:
        raise HTTPException(status_code=404, detail="Fixture not found")

    # Build prediction dict
    pred = fixture.prediction
    pred_dict = {
        "home_win_prob": pred.home_win_prob,
        "draw_prob": pred.draw_prob,
        "away_win_prob": pred.away_win_prob,
        "expected_home_goals": pred.expected_home_goals,
        "expected_away_goals": pred.expected_away_goals,
        "over25_prob": pred.over25_prob,
        "under25_prob": pred.under25_prob,
        "btts_yes_prob": pred.btts_yes_prob,
        "btts_no_prob": pred.btts_no_prob,
        "predicted_score": pred.predicted_score,
    } if pred else {}

    # Build odds summary (reuse market_trends logic inline)
    from collections import defaultdict
    snaps_result = await db.execute(
        select(OddsSnapshot)
        .where(OddsSnapshot.fixture_id == fixture_id)
        .order_by(OddsSnapshot.captured_at)
    )
    snapshots = snaps_result.scalars().all()
    snap_by_market: dict = defaultdict(list)
    for s in snapshots:
        snap_by_market[(s.bet_type, s.line)].append(s)

    def implied(val):
        return (1 / val) if val and val > 1.01 else None

    def trend_label(move):
        if move is None: return "stable"
        if move < -0.03: return "shortening"
        if move > 0.03: return "drifting"
        return "stable"

    by_market: dict = defaultdict(list)
    for o in fixture.odds:
        by_market[(o.bet_type, o.line)].append(o)

    markets_out = []
    for (bet_type, line), odds_list in by_market.items():
        if len(odds_list) < 2:
            continue
        pinnacle = next((o for o in odds_list if "pinnacle" in o.bookmaker.lower()), None)
        soft = [o for o in odds_list if "pinnacle" not in o.bookmaker.lower()]

        def avg_imp(attr):
            vals = [implied(getattr(o, attr)) for o in odds_list if implied(getattr(o, attr))]
            return sum(vals) / len(vals) if vals else None

        rh, rd, ra = avg_imp("home_odds"), avg_imp("draw_odds"), avg_imp("away_odds")
        total = (rh or 0) + (rd or 0) + (ra or 0)
        consensus = {
            "home": round(rh / total, 4) if rh and total else None,
            "draw": round(rd / total, 4) if rd and total else None,
            "away": round(ra / total, 4) if ra and total else None,
        }

        def best(attr):
            vals = [(getattr(o, attr), o.bookmaker) for o in odds_list if getattr(o, attr)]
            return max(vals, key=lambda x: x[0]) if vals else (None, None)

        bh, bh_bk = best("home_odds")
        bd, bd_bk = best("draw_odds")
        ba, ba_bk = best("away_odds")

        sharp_signal = None
        if pinnacle and soft:
            soft_home = [implied(o.home_odds) for o in soft if implied(o.home_odds)]
            pin_home = implied(pinnacle.home_odds)
            if soft_home and pin_home:
                gap = pin_home - sum(soft_home) / len(soft_home)
                direction = "home" if gap > 0.02 else "away" if gap < -0.02 else "neutral"
                strength = "strong" if abs(gap) >= 0.05 else "mild"
                sharp_signal = {"direction": direction, "strength": strength, "home_gap": round(gap, 4)}

        msnaps = snap_by_market.get((bet_type, line), [])
        movement = None
        if len(msnaps) >= 2:
            hm = (msnaps[-1].home_odds - msnaps[0].home_odds) if (msnaps[-1].home_odds and msnaps[0].home_odds) else None
            am = (msnaps[-1].away_odds - msnaps[0].away_odds) if (msnaps[-1].away_odds and msnaps[0].away_odds) else None
            movement = {"home_trend": trend_label(hm), "away_trend": trend_label(am), "snapshots": len(msnaps)}

        money_on = None
        if sharp_signal and sharp_signal["direction"] != "neutral":
            money_on = sharp_signal["direction"]
        elif movement:
            if movement["home_trend"] == "shortening" and movement["away_trend"] != "shortening":
                money_on = "home"
            elif movement["away_trend"] == "shortening" and movement["home_trend"] != "shortening":
                money_on = "away"

        markets_out.append({
            "bet_type": bet_type, "line": line,
            "consensus": consensus,
            "best_odds": {
                "home": {"odds": bh, "bookmaker": bh_bk},
                "draw": {"odds": bd, "bookmaker": bd_bk} if bd else None,
                "away": {"odds": ba, "bookmaker": ba_bk},
            },
            "sharp_signal": sharp_signal,
            "movement": movement,
            "money_on": money_on,
        })

    # Build value bets list for the prompt
    from db.crud import get_value_bets_for_fixture
    vb_rows = await get_value_bets_for_fixture(db, fixture_id)
    value_bets_list = [
        {
            "bet_type": vb.bet_type,
            "selection": vb.selection,
            "our_probability": vb.our_probability,
            "implied_probability": vb.implied_probability,
            "edge": vb.edge,
            "bookmaker_odds": vb.bookmaker_odds,
            "bookmaker": vb.bookmaker,
            "kelly_fraction": vb.kelly_fraction,
        }
        for vb in (vb_rows or [])
        if vb.is_value
    ]

    kickoff_str = fixture.kickoff.strftime("%d %b %Y %H:%M UTC")
    text = await generate_prediction(
        home_team=fixture.home_team.name,
        away_team=fixture.away_team.name,
        stage=fixture.stage,
        kickoff=kickoff_str,
        prediction=pred_dict,
        odds_summary={"markets": markets_out},
        value_bets=value_bets_list,
    )
    return {"analysis": text}


@router.get("/{fixture_id}/betfair")
async def betfair_volume(fixture_id: int, db: AsyncSession = Depends(get_db)):
    """
    Return live Betfair Exchange volume data for a fixture.
    Shows how much money is matched on each market and outcome.
    Requires BETFAIR_USERNAME / BETFAIR_PASSWORD / BETFAIR_APP_KEY in .env.
    """
    if not settings.betfair_username or not settings.betfair_password or not settings.betfair_app_key:
        return {"markets": []}

    result = await db.execute(
        select(Fixture).where(Fixture.id == fixture_id)
    )
    fixture = result.scalar_one_or_none()
    if not fixture:
        raise HTTPException(status_code=404, detail="Fixture not found")

    from sqlalchemy.orm import selectinload
    result = await db.execute(
        select(Fixture)
        .where(Fixture.id == fixture_id)
        .options(selectinload(Fixture.home_team), selectinload(Fixture.away_team))
    )
    fixture = result.scalar_one_or_none()

    token = await betfair_client.login(
        settings.betfair_username,
        settings.betfair_password,
        settings.betfair_app_key,
    )
    if not token:
        return {"error": "Betfair login failed — check credentials", "markets": []}

    markets = await betfair_client.get_market_volumes(
        fixture.home_team.name,
        fixture.away_team.name,
        fixture.kickoff,
        settings.betfair_app_key,
        token,
    )
    return {"markets": markets}


@router.get("/{fixture_id}", response_model=FixtureWithValueBets)
async def fixture_detail(fixture_id: int, db: AsyncSession = Depends(get_db)):
    fixture = await get_fixture_by_id(db, fixture_id)
    if not fixture:
        raise HTTPException(status_code=404, detail="Fixture not found")

    value_bets = await get_value_bets_for_fixture(db, fixture_id)
    result = FixtureWithValueBets.model_validate(fixture)
    result.value_bets = [ValueBetSchema.model_validate(b) for b in value_bets]
    return result

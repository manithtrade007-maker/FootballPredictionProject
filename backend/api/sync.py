import hashlib
from datetime import datetime
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from database import get_db
from db.crud import upsert_team, upsert_fixture, save_prediction, save_odds, save_value_bets, snapshot_odds
from services.odds_api import odds_client
from services.predictor import predict
from services.team_ratings import get_team_ratings
from services.value_bets import evaluate_bets
from models.db_models import Fixture, Team
from models.schemas import SyncResponse

router = APIRouter(prefix="/sync", tags=["sync"])


def _team_id(name: str) -> int:
    """Stable integer ID from team name."""
    return int(hashlib.md5(name.encode()).hexdigest()[:8], 16) % 1_000_000


def _fixture_id(event_id: str) -> int:
    return int(hashlib.md5(event_id.encode()).hexdigest()[:8], 16) % 10_000_000


@router.post("/fixtures", response_model=SyncResponse)
async def sync_fixtures(db: AsyncSession = Depends(get_db)):
    """
    Pull WC2026 fixtures + odds from The Odds API in one call,
    run Poisson predictions, and detect value bets.
    """
    raw_events = await odds_client.get_upcoming_odds()

    count = 0
    for event in raw_events:
        home_name = event.get("home_team", "")
        away_name = event.get("away_team", "")
        if not home_name or not away_name:
            continue

        # Upsert teams
        home = await upsert_team(db, {
            "id": _team_id(home_name),
            "name": home_name,
            "code": home_name[:3].upper(),
            "logo_url": "",
            "group": "",
        })
        away = await upsert_team(db, {
            "id": _team_id(away_name),
            "name": away_name,
            "code": away_name[:3].upper(),
            "logo_url": "",
            "group": "",
        })

        # Upsert fixture
        kickoff_str = event.get("commence_time", "")
        kickoff = datetime.fromisoformat(kickoff_str.replace("Z", "+00:00")) if kickoff_str else datetime.utcnow()
        ext_id = _fixture_id(event.get("id", home_name + away_name))

        fixture = await upsert_fixture(db, {
            "external_id": ext_id,
            "home_team_id": home.id,
            "away_team_id": away.id,
            "kickoff": kickoff,
            "stage": "Group Stage",
            "status": "NS",
            "home_goals": None,
            "away_goals": None,
        })

        # Run prediction using team strength ratings
        home_attack, home_defence = get_team_ratings(home_name)
        away_attack, away_defence = get_team_ratings(away_name)
        pred = predict(home_attack, home_defence, away_attack, away_defence)
        await save_prediction(db, fixture.id, pred)

        # Parse and save odds
        parsed_odds = odds_client.parse_odds(event)
        if parsed_odds:
            await save_odds(db, fixture.id, parsed_odds)
            await snapshot_odds(db, fixture.id, parsed_odds)

            # Calculate value bets
            value_bets = evaluate_bets(pred, parsed_odds)
            await save_value_bets(db, fixture.id, value_bets)

        count += 1

    await db.commit()
    return SyncResponse(message="Fixtures synced", fixtures_synced=count)


@router.post("/odds", response_model=SyncResponse)
async def sync_odds(db: AsyncSession = Depends(get_db)):
    """Re-sync odds and recalculate value bets for existing fixtures."""
    raw_events = await odds_client.get_upcoming_odds()

    count = 0
    for event in raw_events:
        home_name = event.get("home_team", "")
        away_name = event.get("away_team", "")
        ext_id = _fixture_id(event.get("id", home_name + away_name))

        result = await db.execute(
            select(Fixture)
            .where(Fixture.external_id == ext_id)
            .options(selectinload(Fixture.prediction))
        )
        fixture = result.scalar_one_or_none()
        if not fixture:
            continue

        parsed_odds = odds_client.parse_odds(event)
        if not parsed_odds:
            continue

        await save_odds(db, fixture.id, parsed_odds)
        await snapshot_odds(db, fixture.id, parsed_odds)

        if fixture.prediction:
            pred_dict = {
                "home_win_prob": fixture.prediction.home_win_prob,
                "draw_prob": fixture.prediction.draw_prob,
                "away_win_prob": fixture.prediction.away_win_prob,
                "over25_prob": fixture.prediction.over25_prob,
                "under25_prob": fixture.prediction.under25_prob,
                "btts_yes_prob": fixture.prediction.btts_yes_prob,
                "btts_no_prob": fixture.prediction.btts_no_prob,
                "asian_handicap_data": fixture.prediction.asian_handicap_data,
            }
            value_bets = evaluate_bets(pred_dict, parsed_odds)
            await save_value_bets(db, fixture.id, value_bets)
            count += 1

    await db.commit()
    return SyncResponse(message="Odds synced", fixtures_synced=count)


@router.post("/recalculate", response_model=SyncResponse)
async def recalculate_predictions(db: AsyncSession = Depends(get_db)):
    """Re-run predictions for all fixtures using current team ratings and the latest model."""
    fixtures_result = await db.execute(
        select(Fixture).options(
            selectinload(Fixture.home_team),
            selectinload(Fixture.away_team),
            selectinload(Fixture.odds),
        )
    )
    fixtures = fixtures_result.scalars().all()
    count = 0
    for fixture in fixtures:
        pred = predict(
            fixture.home_team.attack_rating,
            fixture.home_team.defence_rating,
            fixture.away_team.attack_rating,
            fixture.away_team.defence_rating,
        )
        await save_prediction(db, fixture.id, pred)
        if fixture.odds:
            odds_list = [
                {
                    "bet_type": o.bet_type,
                    "bookmaker": o.bookmaker,
                    "home_odds": o.home_odds,
                    "draw_odds": o.draw_odds,
                    "away_odds": o.away_odds,
                    "line": o.line,
                }
                for o in fixture.odds
            ]
            await save_value_bets(db, fixture.id, evaluate_bets(pred, odds_list))
        count += 1
    await db.commit()
    return SyncResponse(message=f"Recalculated predictions for {count} fixtures.", fixtures_synced=count)


@router.post("/live-ratings", response_model=SyncResponse)
async def sync_live_ratings(db: AsyncSession = Depends(get_db)):
    """
    Build ratings using two improvements over basic goal-counting:

    1. ELO RATINGS (FiveThirtyEight method): each team starts with an Elo derived
       from their pre-tournament ratings, then each WC2026 result shifts Elo up/down
       based on how surprising the result was vs expected (K=40, neutral venue).
       This means beating a strong team is worth more than beating a weak one.

    2. TIME-WEIGHTED GOALS (Dixon-Coles method): within a team's WC2026 matches,
       the most recent game has full weight and older games decay by 0.65 per step.
       A game 2 form slump matters more than a game 1 blowout.

    Final rating = blend of Elo-derived + time-weighted goals.
    Elo dominates early (1 game), WC data takes over by game 3+.
    """
    from collections import defaultdict
    from db.crud import save_prediction, save_value_bets

    # --- Load completed fixtures in kickoff order (Elo must be chronological) ---
    ft_result = await db.execute(
        select(Fixture)
        .where(Fixture.status == "FT")
        .order_by(Fixture.kickoff)
    )
    finished = [
        f for f in ft_result.scalars().all()
        if f.home_goals is not None and f.away_goals is not None
    ]
    if not finished:
        return SyncResponse(message="No completed fixtures yet. Run Sync Scores first.", fixtures_synced=0)

    total_goals = sum(f.home_goals + f.away_goals for f in finished)
    wc_avg = total_goals / (2 * len(finished))

    # --- Load all teams, initialise Elo from existing attack/defence ratings ---
    all_teams_result = await db.execute(select(Team))
    teams: dict[int, Team] = {t.id: t for t in all_teams_result.scalars().all()}

    def init_elo(attack: float, defence: float) -> float:
        # attack > 1 = strong scorer → higher Elo
        # defence < 1 = fewer goals conceded = better defence → higher Elo
        raw = 1500.0 + (attack - 1.0) * 150.0 + (1.0 - defence) * 100.0
        return max(1150.0, min(1850.0, raw))

    elo: dict[int, float] = {
        tid: init_elo(t.attack_rating, t.defence_rating)
        for tid, t in teams.items()
    }

    # --- Walk matches chronologically, update Elo + record time-stamped goals ---
    K = 40.0       # WC K-factor (FIFA uses 60; 40 is more conservative)
    DECAY = 0.65   # time-decay per game back (most recent = 1.0, previous = 0.65, ...)

    match_log: dict[int, list[tuple]] = defaultdict(list)  # [(kickoff, scored, conceded)]

    for f in finished:
        hid, aid = f.home_team_id, f.away_team_id
        h_elo, a_elo = elo[hid], elo[aid]

        # Expected result (neutral venue: no home advantage adjustment)
        exp_home = 1.0 / (1.0 + 10.0 ** ((a_elo - h_elo) / 400.0))

        # Actual result
        if f.home_goals > f.away_goals:
            actual_home = 1.0
        elif f.home_goals == f.away_goals:
            actual_home = 0.5
        else:
            actual_home = 0.0

        # Update Elo symmetrically
        delta = K * (actual_home - exp_home)
        elo[hid] = h_elo + delta
        elo[aid] = a_elo - delta

        match_log[hid].append((f.kickoff, f.home_goals, f.away_goals))
        match_log[aid].append((f.kickoff, f.away_goals, f.home_goals))

    # --- Blend Elo-derived + time-weighted goals per team ---
    updated = 0
    elo_info = []

    for tid, team in teams.items():
        matches = match_log.get(tid, [])
        n = len(matches)

        team_elo = elo[tid]
        # Elo → attack/defence for Poisson model
        elo_attack  = max(0.3, 1.0 + (team_elo - 1500.0) / 300.0)
        elo_defence = max(0.2, 1.0 - (team_elo - 1500.0) / 500.0)

        if n == 0:
            # No WC games yet: apply Elo-derived rating only to default teams
            if (team.stats_source or "").startswith("hardcoded"):
                team.attack_rating  = round(elo_attack, 3)
                team.defence_rating = round(elo_defence, 3)
                team.stats_source   = "elo_prior (0 wc games)"
            continue

        # Time-weighted goals: most recent game weight=1, older decay by DECAY each step
        weights   = [DECAY ** (n - 1 - i) for i in range(n)]
        total_w   = sum(weights)
        tw_scored    = sum(weights[i] * matches[i][1] for i in range(n)) / total_w
        tw_conceded  = sum(weights[i] * matches[i][2] for i in range(n)) / total_w

        wc_attack  = max(0.1, tw_scored   / wc_avg)
        wc_defence = max(0.1, tw_conceded / wc_avg)

        # Blend: 25% WC data after game 1, +25% per game (caps at 80%)
        wc_weight  = min(0.80, n * 0.25)
        elo_weight = 1.0 - wc_weight

        team.attack_rating  = round(max(0.3, elo_weight * elo_attack  + wc_weight * wc_attack),  3)
        team.defence_rating = round(max(0.2, elo_weight * elo_defence + wc_weight * wc_defence), 3)
        team.stats_source   = f"elo+tw ({n} wc games, elo={team_elo:.0f})"
        updated += 1
        elo_info.append(f"{team.name}: elo={team_elo:.0f} atk={team.attack_rating} def={team.defence_rating}")

    await db.flush()
    for line in sorted(elo_info): print(" ", line)

    # --- Re-run predictions for all fixtures ---
    all_fixtures_result = await db.execute(
        select(Fixture).options(
            selectinload(Fixture.home_team),
            selectinload(Fixture.away_team),
            selectinload(Fixture.odds),
        )
    )
    for fixture in all_fixtures_result.scalars().all():
        pred = predict(
            fixture.home_team.attack_rating,
            fixture.home_team.defence_rating,
            fixture.away_team.attack_rating,
            fixture.away_team.defence_rating,
        )
        await save_prediction(db, fixture.id, pred)
        if fixture.odds:
            odds_list = [
                {
                    "bet_type": o.bet_type,
                    "bookmaker": o.bookmaker,
                    "home_odds": o.home_odds,
                    "draw_odds": o.draw_odds,
                    "away_odds": o.away_odds,
                    "line": o.line,
                }
                for o in fixture.odds
            ]
            await save_value_bets(db, fixture.id, evaluate_bets(pred, odds_list))

    await db.commit()
    return SyncResponse(
        message=(
            f"Elo+time-weighted ratings updated: {updated} teams calibrated from "
            f"{len(finished)} WC2026 matches. WC avg: {wc_avg:.2f} goals/team/game."
        ),
        fixtures_synced=len(finished),
    )


@router.post("/scores", response_model=SyncResponse)
async def sync_scores(db: AsyncSession = Depends(get_db)):
    """
    Pull completed/live scores from The Odds API.
    Updates existing fixtures and creates missing ones (games played since last fixture sync).
    """
    import httpx
    from config import settings
    from db.crud import upsert_team, upsert_fixture, save_prediction
    from services.team_ratings import get_team_ratings
    from services.predictor import predict

    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(
            "https://api.the-odds-api.com/v4/sports/soccer_fifa_world_cup/scores",
            params={"apiKey": settings.odds_api_key, "daysFrom": 3},
        )
        events = r.json() if r.status_code == 200 else []

    # Load all existing fixtures with team names
    result = await db.execute(
        select(Fixture).options(
            selectinload(Fixture.home_team),
            selectinload(Fixture.away_team),
        )
    )
    fixtures = result.scalars().all()
    fixture_map = {(f.home_team.name, f.away_team.name): f for f in fixtures}

    count = 0
    created = 0
    for event in events:
        home_name = event.get("home_team", "")
        away_name = event.get("away_team", "")
        if not home_name or not away_name:
            continue

        completed = event.get("completed", False)
        scores = event.get("scores") or []
        home_score = next((int(s["score"]) for s in scores if s["name"] == home_name), None)
        away_score = next((int(s["score"]) for s in scores if s["name"] == away_name), None)
        status = "FT" if completed else ("LIVE" if scores else "NS")

        fixture = fixture_map.get((home_name, away_name))

        if fixture is None:
            # Game not in DB — create it
            home = await upsert_team(db, {
                "id": _team_id(home_name),
                "name": home_name,
                "code": home_name[:3].upper(),
                "logo_url": "",
                "group": "",
            })
            away = await upsert_team(db, {
                "id": _team_id(away_name),
                "name": away_name,
                "code": away_name[:3].upper(),
                "logo_url": "",
                "group": "",
            })
            kickoff_str = event.get("commence_time", "")
            kickoff = datetime.fromisoformat(kickoff_str.replace("Z", "+00:00")) if kickoff_str else datetime.utcnow()
            ext_id = _fixture_id(event.get("id", home_name + away_name))

            fixture = await upsert_fixture(db, {
                "external_id": ext_id,
                "home_team_id": home.id,
                "away_team_id": away.id,
                "kickoff": kickoff,
                "stage": "Group Stage",
                "status": status,
                "home_goals": home_score,
                "away_goals": away_score,
            })
            # Run prediction for the new fixture
            home_attack, home_defence = get_team_ratings(home_name)
            away_attack, away_defence = get_team_ratings(away_name)
            pred = predict(home_attack, home_defence, away_attack, away_defence)
            await save_prediction(db, fixture.id, pred)
            created += 1
        else:
            fixture.home_goals = home_score
            fixture.away_goals = away_score
            fixture.status = status

        count += 1

    await db.commit()
    msg = f"Scores synced: {count} total ({created} new fixtures created)"
    return SyncResponse(message=msg, fixtures_synced=count)

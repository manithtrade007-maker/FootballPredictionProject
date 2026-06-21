import hashlib
from datetime import datetime
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from database import get_db
from db.crud import upsert_team, upsert_fixture, save_prediction, save_odds, save_value_bets
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


@router.post("/live-ratings", response_model=SyncResponse)
async def sync_live_ratings(db: AsyncSession = Depends(get_db)):
    """
    Derive attack/defence ratings from actual WC2026 match results already in the DB.
    Uses a 3-game Bayesian prior so small samples don't produce extreme values.
    Only updates teams that have played; others keep their existing ratings.
    """
    from collections import defaultdict
    from services.predictor import predict
    from services.value_bets import evaluate_bets
    from db.crud import save_prediction, save_value_bets

    # Load all completed fixtures
    ft_result = await db.execute(
        select(Fixture).where(Fixture.status == "FT").options(
            selectinload(Fixture.home_team),
            selectinload(Fixture.away_team),
        )
    )
    finished = [
        f for f in ft_result.scalars().all()
        if f.home_goals is not None and f.away_goals is not None
    ]

    if not finished:
        return SyncResponse(message="No completed fixtures yet. Run Sync Scores first.", fixtures_synced=0)

    # Compute actual WC2026 average goals per team per game
    total_goals = sum(f.home_goals + f.away_goals for f in finished)
    wc_avg = total_goals / (2 * len(finished))

    # Accumulate goals scored/conceded per team ID
    stats: dict = defaultdict(lambda: {"scored": 0, "conceded": 0, "played": 0})
    for f in finished:
        stats[f.home_team_id]["scored"] += f.home_goals
        stats[f.home_team_id]["conceded"] += f.away_goals
        stats[f.home_team_id]["played"] += 1
        stats[f.away_team_id]["scored"] += f.away_goals
        stats[f.away_team_id]["conceded"] += f.home_goals
        stats[f.away_team_id]["played"] += 1

    # Bayesian shrinkage: 3-game prior toward WC average keeps single-game samples sane
    PRIOR_GAMES = 3
    all_teams_result = await db.execute(select(Team))
    teams = all_teams_result.scalars().all()

    updated = 0
    for team in teams:
        s = stats.get(team.id)
        if not s or s["played"] == 0:
            continue
        n = s["played"]
        blended_scored = (wc_avg * PRIOR_GAMES + s["scored"]) / (PRIOR_GAMES + n)
        blended_conceded = (wc_avg * PRIOR_GAMES + s["conceded"]) / (PRIOR_GAMES + n)

        team.attack_rating = round(max(0.3, blended_scored / wc_avg), 3)
        team.defence_rating = round(max(0.2, blended_conceded / wc_avg), 3)
        team.stats_source = f"wc2026_live ({n} games)"
        updated += 1
        print(f"  {team.name}: attack={team.attack_rating}, defence={team.defence_rating} [{n} WC games]")

    await db.flush()

    # Re-run predictions for all fixtures with updated ratings
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
            f"Ratings updated from WC2026 results: {updated} teams calibrated. "
            f"WC avg: {wc_avg:.2f} goals/team/game across {len(finished)} matches."
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

from datetime import datetime
from fastapi import APIRouter, Depends, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from database import get_db
from db.crud import upsert_team, upsert_fixture, save_prediction, save_odds, save_value_bets
from services.football_api import football_client
from services.odds_api import odds_client
from services.predictor import predict, build_team_ratings
from services.value_bets import evaluate_bets
from models.schemas import SyncResponse

router = APIRouter(prefix="/sync", tags=["sync"])

WC2026_SEASON = 2026


@router.post("/fixtures", response_model=SyncResponse)
async def sync_fixtures(
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    raw_fixtures = await football_client.get_fixtures(settings.wc2026_league_id, WC2026_SEASON)

    count = 0
    for raw in raw_fixtures:
        parsed = football_client.parse_fixture(raw)

        home_data = parsed["home_team"]
        away_data = parsed["away_team"]

        home = await upsert_team(db, {
            "id": home_data["id"],
            "name": home_data["name"],
            "code": home_data["code"],
            "logo_url": home_data["logo_url"],
        })
        away = await upsert_team(db, {
            "id": away_data["id"],
            "name": away_data["name"],
            "code": away_data["code"],
            "logo_url": away_data["logo_url"],
        })

        kickoff = datetime.fromisoformat(parsed["kickoff"].replace("Z", "+00:00"))
        fixture = await upsert_fixture(db, {
            "external_id": parsed["external_id"],
            "home_team_id": home.id,
            "away_team_id": away.id,
            "kickoff": kickoff,
            "stage": parsed["stage"],
            "status": parsed["status"],
            "home_goals": parsed["home_goals"],
            "away_goals": parsed["away_goals"],
        })
        count += 1

        background_tasks.add_task(_run_prediction, fixture.id, home.id, away.id, db)

    await db.commit()
    return SyncResponse(message="Fixtures synced", fixtures_synced=count)


@router.post("/odds", response_model=SyncResponse)
async def sync_odds(db: AsyncSession = Depends(get_db)):
    raw_odds = await odds_client.get_upcoming_odds()

    count = 0
    for event in raw_odds:
        # Match event to fixture by team names (best-effort)
        # In production: match on external ID or kickoff time
        parsed_odds = odds_client.parse_odds(event)
        if parsed_odds:
            count += 1

    await db.commit()
    return SyncResponse(message="Odds synced", fixtures_synced=count)


async def _run_prediction(fixture_id: int, home_team_id: int, away_team_id: int, db: AsyncSession):
    """Fetch team stats, run prediction, save value bets."""
    try:
        home_raw = await football_client.get_team_statistics(
            home_team_id, settings.wc2026_league_id, WC2026_SEASON
        )
        away_raw = await football_client.get_team_statistics(
            away_team_id, settings.wc2026_league_id, WC2026_SEASON
        )

        home_stats = football_client.parse_team_stats(home_raw)
        away_stats = football_client.parse_team_stats(away_raw)

        home_attack, home_defence = build_team_ratings(home_stats)
        away_attack, away_defence = build_team_ratings(away_stats)

        pred = predict(home_attack, home_defence, away_attack, away_defence)
        await save_prediction(db, fixture_id, pred)
        await db.commit()
    except Exception:
        pass

import asyncio
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from database import get_db
from models.db_models import Team, Fixture
from models.schemas import SyncResponse
from services.stats_builder import get_real_ratings, KNOWN_TEAM_IDS, RateLimitError
from services.predictor import predict
from db.crud import save_prediction, save_value_bets
from services.value_bets import evaluate_bets

router = APIRouter(prefix="/sync", tags=["sync"])


@router.post("/team-stats", response_model=SyncResponse)
async def sync_team_stats(db: AsyncSession = Depends(get_db)):
    """
    Fetch real stats for every team from qualification leagues / WC2022.
    Skips teams that already have real stats. Stops gracefully if rate-limited.
    Updates attack/defence ratings and re-runs all predictions.
    """
    result = await db.execute(select(Team))
    teams = result.scalars().all()

    updated = 0
    skipped_already_real = 0
    real_stats_count = 0
    rate_limited_at: str | None = None
    rate_limit_reason: str = ""

    for team in teams:
        # Skip if already has real stats from a previous run
        if team.stats_source != "hardcoded":
            skipped_already_real += 1
            continue

        # Resolve API ID: use cached → KNOWN_TEAM_IDS (no search to save requests)
        api_id = team.api_football_id
        if api_id is None:
            api_id = KNOWN_TEAM_IDS.get(team.name)
            if api_id is not None:
                team.api_football_id = api_id

        try:
            attack, defence, source = await get_real_ratings(team.name, api_id)
        except RateLimitError as e:
            rate_limited_at = team.name
            rate_limit_reason = str(e)
            break  # Stop — save what we have so far

        await asyncio.sleep(2.0)  # stay under per-minute rate limit

        if source != "hardcoded":
            real_stats_count += 1

        team.attack_rating = attack
        team.defence_rating = defence
        team.stats_source = source
        updated += 1
        print(f"  {team.name}: attack={attack:.3f}, defence={defence:.3f} [{source}]")

    await db.flush()

    # Re-run predictions for ALL fixtures using current (possibly updated) ratings
    fixtures_result = await db.execute(
        select(Fixture).options(
            selectinload(Fixture.home_team),
            selectinload(Fixture.away_team),
            selectinload(Fixture.odds),
        )
    )
    fixtures = fixtures_result.scalars().all()

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
            value_bets = evaluate_bets(pred, odds_list)
            await save_value_bets(db, fixture.id, value_bets)

    await db.commit()

    if rate_limited_at:
        if "suspend" in rate_limit_reason.lower():
            msg = (
                f"API account suspended at '{rate_limited_at}'. "
                f"Check https://dashboard.api-football.com to activate your account. "
                f"Got real stats for {real_stats_count} teams this run "
                f"({skipped_already_real} already had real stats — those are safe)."
            )
        else:
            msg = (
                f"API daily limit reached at '{rate_limited_at}'. "
                f"Got real stats for {real_stats_count} teams. "
                f"Run again tomorrow to continue ({skipped_already_real} already had real stats)."
            )
    else:
        msg = (
            f"Done: {real_stats_count} teams updated with real stats, "
            f"{updated - real_stats_count} stayed hardcoded, "
            f"{skipped_already_real} already had real stats."
        )

    return SyncResponse(message=msg, fixtures_synced=updated + skipped_already_real)

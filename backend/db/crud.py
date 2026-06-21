from datetime import datetime, timedelta
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from models.db_models import Team, Fixture, Prediction, BettingOdds, ValueBet, OddsSnapshot


async def upsert_team(db: AsyncSession, team_data: dict) -> Team:
    result = await db.execute(select(Team).where(Team.id == team_data["id"]))
    team = result.scalar_one_or_none()
    if not team:
        team = Team(**team_data)
        db.add(team)
    else:
        for k, v in team_data.items():
            setattr(team, k, v)
    await db.flush()
    return team


async def upsert_fixture(db: AsyncSession, fixture_data: dict) -> Fixture:
    result = await db.execute(
        select(Fixture).where(Fixture.external_id == fixture_data["external_id"])
    )
    fixture = result.scalar_one_or_none()
    if not fixture:
        fixture = Fixture(**fixture_data)
        db.add(fixture)
    else:
        for k, v in fixture_data.items():
            setattr(fixture, k, v)
    await db.flush()
    return fixture


async def save_prediction(db: AsyncSession, fixture_id: int, pred_data: dict) -> Prediction:
    result = await db.execute(
        select(Prediction).where(Prediction.fixture_id == fixture_id)
    )
    pred = result.scalar_one_or_none()
    if not pred:
        pred = Prediction(fixture_id=fixture_id, **pred_data)
        db.add(pred)
    else:
        for k, v in pred_data.items():
            setattr(pred, k, v)
        pred.created_at = datetime.utcnow()
    await db.flush()
    return pred


async def save_odds(db: AsyncSession, fixture_id: int, odds_list: list[dict]):
    await db.execute(delete(BettingOdds).where(BettingOdds.fixture_id == fixture_id))
    for o in odds_list:
        db.add(BettingOdds(fixture_id=fixture_id, **o))
    await db.flush()


async def snapshot_odds(db: AsyncSession, fixture_id: int, odds_list: list[dict]):
    """Append a timestamped copy of the current odds (never overwrites — historical record)."""
    now = datetime.utcnow()
    for o in odds_list:
        db.add(OddsSnapshot(fixture_id=fixture_id, captured_at=now, **o))
    await db.flush()


async def save_value_bets(db: AsyncSession, fixture_id: int, bets: list[dict]):
    await db.execute(delete(ValueBet).where(ValueBet.fixture_id == fixture_id))
    for b in bets:
        db.add(ValueBet(fixture_id=fixture_id, **b))
    await db.flush()


async def get_upcoming_fixtures(db: AsyncSession, days: int = 7) -> list[Fixture]:
    now = datetime.utcnow()
    past = now - timedelta(days=7)   # include results from last 7 days
    cutoff = now + timedelta(days=days)
    result = await db.execute(
        select(Fixture)
        .where(Fixture.kickoff >= past, Fixture.kickoff <= cutoff)
        .options(
            selectinload(Fixture.home_team),
            selectinload(Fixture.away_team),
            selectinload(Fixture.prediction),
            selectinload(Fixture.odds),
        )
        .order_by(Fixture.kickoff)
    )
    return result.scalars().all()


async def get_fixture_by_id(db: AsyncSession, fixture_id: int) -> Fixture | None:
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
    return result.scalar_one_or_none()


async def get_value_bets_for_fixture(db: AsyncSession, fixture_id: int) -> list[ValueBet]:
    result = await db.execute(
        select(ValueBet)
        .where(ValueBet.fixture_id == fixture_id, ValueBet.is_value == True)
        .order_by(ValueBet.edge.desc())
    )
    return result.scalars().all()


async def get_all_value_bets(db: AsyncSession) -> list[dict]:
    result = await db.execute(
        select(ValueBet)
        .where(ValueBet.is_value == True)
        .options(
            selectinload(ValueBet.fixture).selectinload(Fixture.home_team),
            selectinload(ValueBet.fixture).selectinload(Fixture.away_team),
        )
        .order_by(ValueBet.edge.desc())
    )
    rows = result.scalars().all()
    out = []
    for vb in rows:
        d = {
            "bet_type": vb.bet_type,
            "selection": vb.selection,
            "our_probability": vb.our_probability,
            "bookmaker_odds": vb.bookmaker_odds,
            "bookmaker": vb.bookmaker,
            "implied_probability": vb.implied_probability,
            "edge": vb.edge,
            "kelly_fraction": vb.kelly_fraction,
            "is_value": vb.is_value,
            "fixture_id": vb.fixture_id,
            "home_team": vb.fixture.home_team.name if vb.fixture else None,
            "away_team": vb.fixture.away_team.name if vb.fixture else None,
            "kickoff": vb.fixture.kickoff.isoformat() if vb.fixture else None,
            "stage": vb.fixture.stage if vb.fixture else None,
        }
        out.append(d)
    return out

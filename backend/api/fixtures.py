from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from db.crud import get_upcoming_fixtures, get_fixture_by_id, get_value_bets_for_fixture
from models.schemas import FixtureSchema, FixtureWithValueBets, ValueBetSchema

router = APIRouter(prefix="/fixtures", tags=["fixtures"])


@router.get("/upcoming", response_model=list[FixtureSchema])
async def upcoming_fixtures(days: int = 7, db: AsyncSession = Depends(get_db)):
    return await get_upcoming_fixtures(db, days)


@router.get("/{fixture_id}", response_model=FixtureWithValueBets)
async def fixture_detail(fixture_id: int, db: AsyncSession = Depends(get_db)):
    fixture = await get_fixture_by_id(db, fixture_id)
    if not fixture:
        raise HTTPException(status_code=404, detail="Fixture not found")

    value_bets = await get_value_bets_for_fixture(db, fixture_id)
    result = FixtureWithValueBets.model_validate(fixture)
    result.value_bets = [ValueBetSchema.model_validate(b) for b in value_bets]
    return result

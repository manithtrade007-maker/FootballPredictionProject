from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from db.crud import get_all_value_bets
from models.schemas import ValueBetSchema

router = APIRouter(prefix="/value-bets", tags=["value-bets"])


@router.get("/", response_model=list[ValueBetSchema])
async def all_value_bets(db: AsyncSession = Depends(get_db)):
    bets = await get_all_value_bets(db)
    return bets

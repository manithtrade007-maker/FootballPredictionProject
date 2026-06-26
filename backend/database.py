from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import text
from config import settings

engine = create_async_engine(settings.database_url, echo=False)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def get_db() -> AsyncSession:
    async with SessionLocal() as session:
        yield session


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # Safe migrations — silently skip if column already exists
        migrations = [
            "ALTER TABLE teams ADD COLUMN api_football_id INTEGER DEFAULT NULL",
            "ALTER TABLE value_bets ADD COLUMN verdict TEXT",
            "ALTER TABLE value_bets ADD COLUMN action TEXT",
            "ALTER TABLE value_bets ADD COLUMN min_edge REAL",
            "ALTER TABLE value_bets ADD COLUMN max_edge REAL",
            "ALTER TABLE value_bets ADD COLUMN why_line TEXT",
        ]
        for sql in migrations:
            try:
                await conn.execute(text(sql))
            except Exception:
                pass

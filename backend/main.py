from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from database import init_db
from api.fixtures import router as fixtures_router
from api.sync import router as sync_router
from api.value_bets import router as value_bets_router
from api.team_stats import router as team_stats_router

app = FastAPI(title="WC2026 Betting Assistant", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(fixtures_router)
app.include_router(sync_router)
app.include_router(value_bets_router)
app.include_router(team_stats_router)


@app.on_event("startup")
async def startup():
    await init_db()


@app.get("/health")
async def health():
    return {"status": "ok"}

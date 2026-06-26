from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    api_football_key: str = ""
    api_football_base_url: str = "https://v3.football.api-sports.io"
    odds_api_key: str = ""
    odds_api_base_url: str = "https://api.the-odds-api.com/v4"
    # Override with DATABASE_URL env var on Railway (point to /data/football.db volume)
    database_url: str = "sqlite+aiosqlite:///./football.db"
    wc2026_league_id: int = 1
    betfair_username: str = ""
    betfair_password: str = ""
    betfair_app_key: str = ""
    gemini_api_key: str = ""
    groq_api_key: str = ""
    # Comma-separated allowed origins — set ALLOWED_ORIGINS in Railway env vars
    allowed_origins: str = "http://localhost:3000"

    class Config:
        env_file = ".env"


settings = Settings()

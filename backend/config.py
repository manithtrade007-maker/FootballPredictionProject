from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    api_football_key: str = ""
    api_football_base_url: str = "https://v3.football.api-sports.io"
    odds_api_key: str = ""
    odds_api_base_url: str = "https://api.the-odds-api.com/v4"
    database_url: str = "sqlite+aiosqlite:///./football.db"
    wc2026_league_id: int = 1  # API-Football World Cup league ID

    class Config:
        env_file = ".env"


settings = Settings()

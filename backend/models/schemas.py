from datetime import datetime
from pydantic import BaseModel


class TeamSchema(BaseModel):
    id: int
    name: str
    code: str
    logo_url: str
    group: str

    class Config:
        from_attributes = True


class PredictionSchema(BaseModel):
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    expected_home_goals: float
    expected_away_goals: float
    over25_prob: float
    under25_prob: float
    btts_yes_prob: float
    btts_no_prob: float
    asian_handicap_data: dict
    predicted_score: str
    confidence: float

    class Config:
        from_attributes = True


class OddsSchema(BaseModel):
    bookmaker: str
    bet_type: str
    home_odds: float | None
    draw_odds: float | None
    away_odds: float | None
    line: float | None

    class Config:
        from_attributes = True


class ValueBetSchema(BaseModel):
    bet_type: str
    selection: str
    our_probability: float
    bookmaker_odds: float
    bookmaker: str
    implied_probability: float
    edge: float
    kelly_fraction: float
    is_value: bool
    fixture_id: int | None = None
    home_team: str | None = None
    away_team: str | None = None
    kickoff: str | None = None
    stage: str | None = None

    class Config:
        from_attributes = True


class FixtureSchema(BaseModel):
    id: int
    external_id: int
    home_team: TeamSchema
    away_team: TeamSchema
    kickoff: datetime
    stage: str
    status: str
    home_goals: int | None
    away_goals: int | None
    prediction: PredictionSchema | None
    odds: list[OddsSchema]

    class Config:
        from_attributes = True


class FixtureWithValueBets(FixtureSchema):
    value_bets: list[ValueBetSchema] = []


class SyncResponse(BaseModel):
    message: str
    fixtures_synced: int

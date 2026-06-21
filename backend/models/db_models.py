from datetime import datetime
from sqlalchemy import String, Integer, Float, DateTime, Boolean, ForeignKey, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from database import Base


class Team(Base):
    __tablename__ = "teams"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    code: Mapped[str] = mapped_column(String(10))
    logo_url: Mapped[str] = mapped_column(String(255), default="")
    group: Mapped[str] = mapped_column(String(10), default="")

    home_fixtures: Mapped[list["Fixture"]] = relationship("Fixture", foreign_keys="Fixture.home_team_id", back_populates="home_team")
    away_fixtures: Mapped[list["Fixture"]] = relationship("Fixture", foreign_keys="Fixture.away_team_id", back_populates="away_team")


class Fixture(Base):
    __tablename__ = "fixtures"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    external_id: Mapped[int] = mapped_column(Integer, unique=True)
    home_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    away_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    kickoff: Mapped[datetime] = mapped_column(DateTime)
    stage: Mapped[str] = mapped_column(String(50), default="Group Stage")
    status: Mapped[str] = mapped_column(String(20), default="NS")  # NS, LIVE, FT
    home_goals: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_goals: Mapped[int | None] = mapped_column(Integer, nullable=True)

    home_team: Mapped["Team"] = relationship("Team", foreign_keys=[home_team_id], back_populates="home_fixtures")
    away_team: Mapped["Team"] = relationship("Team", foreign_keys=[away_team_id], back_populates="away_fixtures")
    prediction: Mapped["Prediction"] = relationship("Prediction", back_populates="fixture", uselist=False)
    odds: Mapped[list["BettingOdds"]] = relationship("BettingOdds", back_populates="fixture")


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    fixture_id: Mapped[int] = mapped_column(ForeignKey("fixtures.id"), unique=True)
    home_win_prob: Mapped[float] = mapped_column(Float)
    draw_prob: Mapped[float] = mapped_column(Float)
    away_win_prob: Mapped[float] = mapped_column(Float)
    expected_home_goals: Mapped[float] = mapped_column(Float)
    expected_away_goals: Mapped[float] = mapped_column(Float)
    over25_prob: Mapped[float] = mapped_column(Float)
    under25_prob: Mapped[float] = mapped_column(Float)
    btts_yes_prob: Mapped[float] = mapped_column(Float)
    btts_no_prob: Mapped[float] = mapped_column(Float)
    asian_handicap_data: Mapped[dict] = mapped_column(JSON, default=dict)
    predicted_score: Mapped[str] = mapped_column(String(10))
    confidence: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    fixture: Mapped["Fixture"] = relationship("Fixture", back_populates="prediction")


class BettingOdds(Base):
    __tablename__ = "betting_odds"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    fixture_id: Mapped[int] = mapped_column(ForeignKey("fixtures.id"))
    bookmaker: Mapped[str] = mapped_column(String(50))
    bet_type: Mapped[str] = mapped_column(String(30))  # 1X2, O/U, BTTS, AH
    home_odds: Mapped[float | None] = mapped_column(Float, nullable=True)
    draw_odds: Mapped[float | None] = mapped_column(Float, nullable=True)
    away_odds: Mapped[float | None] = mapped_column(Float, nullable=True)
    line: Mapped[float | None] = mapped_column(Float, nullable=True)  # for O/U and AH
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    fixture: Mapped["Fixture"] = relationship("Fixture", back_populates="odds")


class ValueBet(Base):
    __tablename__ = "value_bets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    fixture_id: Mapped[int] = mapped_column(ForeignKey("fixtures.id"))
    bet_type: Mapped[str] = mapped_column(String(30))
    selection: Mapped[str] = mapped_column(String(50))
    our_probability: Mapped[float] = mapped_column(Float)
    bookmaker_odds: Mapped[float] = mapped_column(Float)
    bookmaker: Mapped[str] = mapped_column(String(50))
    implied_probability: Mapped[float] = mapped_column(Float)
    edge: Mapped[float] = mapped_column(Float)  # our_prob - implied_prob
    kelly_fraction: Mapped[float] = mapped_column(Float)
    is_value: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

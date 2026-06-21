import httpx
from config import settings


class OddsAPIClient:
    SPORT_KEY = "soccer_fifa_world_cup"
    REGIONS = "eu"
    MARKETS = "h2h,totals,spreads"

    def __init__(self):
        self.base_url = settings.odds_api_base_url
        self.api_key = settings.odds_api_key

    async def _get(self, endpoint: str, params: dict = None) -> dict | list:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(
                f"{self.base_url}/{endpoint}",
                params={"apiKey": self.api_key, **(params or {})},
            )
            response.raise_for_status()
            return response.json()

    async def get_upcoming_odds(self) -> list[dict]:
        data = await self._get(
            f"sports/{self.SPORT_KEY}/odds",
            {
                "regions": self.REGIONS,
                "markets": self.MARKETS,
                "oddsFormat": "decimal",
                "dateFormat": "iso",
            },
        )
        return data if isinstance(data, list) else []

    def parse_odds(self, raw: dict) -> list[dict]:
        """Convert Odds API response into our flat odds format."""
        results = []
        home_team = raw.get("home_team", "")
        away_team = raw.get("away_team", "")

        for bookmaker in raw.get("bookmakers", [])[:3]:  # top 3 bookmakers
            name = bookmaker.get("title", "")

            for market in bookmaker.get("markets", []):
                key = market.get("key")
                outcomes = {o["name"]: o["price"] for o in market.get("outcomes", [])}

                if key == "h2h":
                    results.append({
                        "bookmaker": name,
                        "bet_type": "1X2",
                        "home_odds": outcomes.get(home_team),
                        "draw_odds": outcomes.get("Draw"),
                        "away_odds": outcomes.get(away_team),
                        "line": None,
                    })
                elif key == "totals":
                    for outcome in market.get("outcomes", []):
                        point = outcome.get("point", 2.5)
                        if point == 2.5:
                            over = outcomes.get("Over")
                            under = outcomes.get("Under")
                            results.append({
                                "bookmaker": name,
                                "bet_type": "O/U",
                                "home_odds": over,
                                "draw_odds": None,
                                "away_odds": under,
                                "line": 2.5,
                            })
                            break
                elif key == "btts":
                    results.append({
                        "bookmaker": name,
                        "bet_type": "BTTS",
                        "home_odds": outcomes.get("Yes"),
                        "draw_odds": None,
                        "away_odds": outcomes.get("No"),
                        "line": None,
                    })
                elif key == "spreads":
                    # AH/spreads: outcomes have a "point" field
                    # point for home team is negative if home is favorite
                    home_outcome = next((o for o in market.get("outcomes", []) if o["name"] == home_team), None)
                    away_outcome = next((o for o in market.get("outcomes", []) if o["name"] == away_team), None)
                    if home_outcome and away_outcome:
                        # Store line from HOME team's perspective (negative = home gives goals)
                        home_line = home_outcome.get("point", 0)
                        results.append({
                            "bookmaker": name,
                            "bet_type": "AH",
                            "home_odds": home_outcome["price"],
                            "draw_odds": None,
                            "away_odds": away_outcome["price"],
                            "line": home_line,
                        })

        return results


odds_client = OddsAPIClient()

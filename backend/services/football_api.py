import httpx
from config import settings


class FootballAPIClient:
    def __init__(self):
        self.base_url = settings.api_football_base_url
        self.headers = {
            "x-apisports-key": settings.api_football_key,
        }

    async def _get(self, endpoint: str, params: dict = None) -> dict:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(
                f"{self.base_url}/{endpoint}",
                headers=self.headers,
                params=params or {},
            )
            response.raise_for_status()
            return response.json()

    async def get_fixtures(self, league_id: int, season: int) -> list[dict]:
        data = await self._get("fixtures", {"league": league_id, "season": season})
        return data.get("response", [])

    async def get_team_statistics(self, team_id: int, league_id: int, season: int) -> dict:
        data = await self._get("teams/statistics", {
            "team": team_id,
            "league": league_id,
            "season": season,
        })
        return data.get("response", {})

    async def get_head_to_head(self, team1_id: int, team2_id: int, last: int = 10) -> list[dict]:
        data = await self._get("fixtures/headtohead", {
            "h2h": f"{team1_id}-{team2_id}",
            "last": last,
        })
        return data.get("response", [])

    async def get_standings(self, league_id: int, season: int) -> list[dict]:
        data = await self._get("standings", {"league": league_id, "season": season})
        return data.get("response", [])

    def parse_fixture(self, raw: dict) -> dict:
        fixture = raw.get("fixture", {})
        teams = raw.get("teams", {})
        goals = raw.get("goals", {})
        league = raw.get("league", {})

        return {
            "external_id": fixture.get("id"),
            "kickoff": fixture.get("date"),
            "status": fixture.get("status", {}).get("short", "NS"),
            "stage": league.get("round", "Group Stage"),
            "home_team": {
                "id": teams.get("home", {}).get("id"),
                "name": teams.get("home", {}).get("name", ""),
                "code": teams.get("home", {}).get("name", "")[:3].upper(),
                "logo_url": teams.get("home", {}).get("logo", ""),
            },
            "away_team": {
                "id": teams.get("away", {}).get("id"),
                "name": teams.get("away", {}).get("name", ""),
                "code": teams.get("away", {}).get("name", "")[:3].upper(),
                "logo_url": teams.get("away", {}).get("logo", ""),
            },
            "home_goals": goals.get("home"),
            "away_goals": goals.get("away"),
        }

    def parse_team_stats(self, raw: dict) -> dict:
        goals = raw.get("goals", {})
        fixtures = raw.get("fixtures", {})
        played = fixtures.get("played", {}).get("total", 1) or 1

        return {
            "matches_played": played,
            "goals_scored": goals.get("for", {}).get("total", {}).get("total", 0),
            "goals_conceded": goals.get("against", {}).get("total", {}).get("total", 0),
        }


football_client = FootballAPIClient()

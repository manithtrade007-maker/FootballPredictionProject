"""
Fetches real team stats from API-Football qualification leagues and WC2022.
Uses the free plan (seasons 2022-2024 are accessible).
"""
import httpx
from config import settings

# WC average goals (baseline for rating calculation)
WC_AVG_HOME = 1.36
WC_AVG_AWAY = 1.10

# Pre-confirmed API-Football national team IDs (avoids spending search requests)
KNOWN_TEAM_IDS: dict[str, int] = {
    # Confirmed via API
    "France": 2,
    "Germany": 25,
    "Spain": 9,
    "Brazil": 6,
    "USA": 657,
    "South Korea": 17,
    "Ivory Coast": 1501,
    "DR Congo": 27739,
    # High-confidence from API-Football documentation
    "Argentina": 26,
    "Portugal": 27,
    "England": 47,
    "Italy": 768,
    "Netherlands": 1118,
    "Belgium": 1,
    "Croatia": 3,
    "Switzerland": 42,
    "Austria": 29,
    "Poland": 24,
    "Turkey": 21,
    "Romania": 30,
    "Uruguay": 7,
    "Colombia": 8,
    "Ecuador": 4,
    "Chile": 12,
    "Peru": 14,
    "Venezuela": 35,
    "Bolivia": 22,
    "Paraguay": 20,
    "Mexico": 16,
    "Canada": 108,
    "Costa Rica": 18,
    "Panama": 83,
    "Honduras": 33,
    "Jamaica": 1517,
    "El Salvador": 247,
    "Japan": 40,
    "Iran": 67,
    "Saudi Arabia": 86,
    "Qatar": 72,
    "Australia": 13,
    "Iraq": 119,
    "Morocco": 32,
    "Senegal": 15,
    "Nigeria": 34,
    "Egypt": 45,
    "Cameroon": 23,
    "Ghana": 39,
    "Algeria": 28,
    "Tunisia": 36,
    "New Zealand": 66,
    "Sweden": 5,
    "Norway": 107,
    "Denmark": 1613,
    "Scotland": 1131,
    "Serbia": 1149,
    "Ukraine": 772,
    "Wales": 773,
    "Czech Republic": 775,
    "Greece": 770,
    "Hungary": 769,
    "Slovakia": 776,
    "South Africa": 186,
    "Haiti": 488,
    "Cape Verde": 2562,
    "Jordan": 7869,
    "Uzbekistan": 5217,
    "Mali": 3419,
}

# Map team name → (league_id, season) for their qualification campaign
TEAM_QUALIFICATION: dict[str, tuple[int, int]] = {
    # UEFA (League 32, season 2024 = WC2026 European qualifiers)
    "France":       (32, 2024), "England":     (32, 2024), "Germany":     (32, 2024),
    "Spain":        (32, 2024), "Portugal":    (32, 2024), "Netherlands": (32, 2024),
    "Belgium":      (32, 2024), "Italy":       (32, 2024), "Croatia":     (32, 2024),
    "Denmark":      (32, 2024), "Switzerland": (32, 2024), "Austria":     (32, 2024),
    "Poland":       (32, 2024), "Serbia":      (32, 2024), "Turkey":      (32, 2024),
    "Ukraine":      (32, 2024), "Sweden":      (32, 2024), "Scotland":    (32, 2024),
    "Wales":        (32, 2024), "Hungary":     (32, 2024), "Slovakia":    (32, 2024),
    "Czech Republic": (32, 2024), "Romania":   (32, 2024), "Greece":      (32, 2024),
    "Norway":       (32, 2024),

    # CONMEBOL (League 34, season 2026 = South American qualifiers)
    "Brazil":    (34, 2026), "Argentina": (34, 2026), "Uruguay":    (34, 2026),
    "Colombia":  (34, 2026), "Ecuador":   (34, 2026), "Peru":       (34, 2026),
    "Chile":     (34, 2026), "Venezuela": (34, 2026), "Bolivia":    (34, 2026),
    "Paraguay":  (34, 2026),

    # CONCACAF (League 31, season 2026)
    "USA":         (31, 2026), "Mexico":      (31, 2026), "Canada":      (31, 2026),
    "Costa Rica":  (31, 2026), "Panama":      (31, 2026), "Honduras":    (31, 2026),
    "Jamaica":     (31, 2026), "El Salvador": (31, 2026),

    # AFC (League 30, season 2026 = Asian qualifiers)
    "Japan":        (30, 2026), "South Korea": (30, 2026), "Iran":        (30, 2026),
    "Saudi Arabia": (30, 2026), "Qatar":       (30, 2026), "Australia":   (30, 2026),
    "Iraq":         (30, 2026), "Jordan":      (30, 2026), "Uzbekistan":  (30, 2026),

    # CAF (League 29, season 2026 = African qualifiers)
    "Morocco":     (29, 2026), "Senegal":    (29, 2026), "Nigeria":     (29, 2026),
    "Egypt":       (29, 2026), "Cameroon":   (29, 2026), "Ghana":       (29, 2026),
    "Algeria":     (29, 2026), "Tunisia":    (29, 2026), "Mali":        (29, 2026),
    "Ivory Coast": (29, 2026), "Cape Verde": (29, 2026),

    # OFC (League 33, season 2026 = Oceania qualifiers)
    "New Zealand": (33, 2026),
}

# Fallback: use WC2022 for teams not found in qualification
WC2022_LEAGUE = 1
WC2022_SEASON = 2022


class RateLimitError(Exception):
    """Raised when the API daily request limit is hit (unrecoverable today)."""
    pass


async def _api_get(params: dict, retries: int = 3) -> dict:
    """Make an API-Football GET request with per-minute rate limit retry."""
    import asyncio
    await asyncio.sleep(6)  # free plan: 10 req/min → 6s gap keeps us safely under
    for attempt in range(retries):
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(
                f"{settings.api_football_base_url}/teams/statistics",
                headers={"x-apisports-key": settings.api_football_key},
                params=params,
            )
            data = r.json()
            errors = data.get("errors", {})
            if not errors:
                return data

            if isinstance(errors, dict):
                msg = errors.get("rateLimit", errors.get("access", ""))
            else:
                msg = str(errors)
            if "suspend" in msg.lower():
                raise RateLimitError(f"Account suspended: {msg}")
            if "daily" in msg.lower() or "day" in msg.lower():
                raise RateLimitError(f"Daily limit reached: {msg}")
            if "minute" in msg.lower() or "per-minute" in msg.lower():
                wait = 15 * (attempt + 1)
                print(f"    Per-minute limit hit, waiting {wait}s...")
                await asyncio.sleep(wait)
                continue
            # Other error (e.g. season not on free plan) — return as-is
            return data
    return {}


async def fetch_team_stats(team_id: int, league_id: int, season: int) -> dict | None:
    """Fetch team stats from API-Football for a given league/season.
    Raises RateLimitError if the daily limit is reached.
    Retries automatically on per-minute rate limit.
    """
    try:
        data = await _api_get({"team": team_id, "league": league_id, "season": season})

        if data.get("errors"):
            return None  # season blocked or other non-fatal error

        response = data.get("response", {})
        if not response:
            return None

        goals = response.get("goals", {})
        fixtures = response.get("fixtures", {})
        played = fixtures.get("played", {}).get("total", 0)
        if not played:
            return None

        scored = goals.get("for", {}).get("total", {}).get("total", 0)
        conceded = goals.get("against", {}).get("total", {}).get("total", 0)
        return {"played": played, "scored": scored, "conceded": conceded}
    except RateLimitError:
        raise
    except Exception as e:
        print(f"    Exception in fetch_team_stats({team_id}, {league_id}, {season}): {type(e).__name__}: {e}")
        return None


async def search_team_id(team_name: str) -> int | None:
    """Find API-Football team ID — checks hardcoded map first to save API requests."""
    if team_name in KNOWN_TEAM_IDS:
        return KNOWN_TEAM_IDS[team_name]

    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(
            f"{settings.api_football_base_url}/teams",
            headers={"x-apisports-key": settings.api_football_key},
            params={"search": team_name},
        )
        data = r.json()
        errors = data.get("errors", {})
        if errors:
            err_str = str(errors)
            if "suspend" in err_str.lower():
                raise RateLimitError(f"Account suspended: {errors}")
            print(f"    API error searching '{team_name}': {errors}")
            return None
        results = data.get("response", [])
        if results:
            return results[0]["team"]["id"]
    return None


def calculate_ratings(stats: dict, is_home_context: bool = True) -> tuple[float, float]:
    """
    Convert goals stats into attack/defence ratings relative to WC average.
    attack  = goals_scored_per_game / WC_avg
    defence = goals_conceded_per_game / WC_avg  (lower = better defending)
    """
    played = max(stats["played"], 1)
    avg_scored = stats["scored"] / played
    avg_conceded = stats["conceded"] / played

    attack = round(avg_scored / WC_AVG_HOME, 3)
    defence = round(avg_conceded / WC_AVG_AWAY, 3)

    # Clamp to sensible range
    attack = max(0.4, min(attack, 2.5))
    defence = max(0.3, min(defence, 2.0))

    return attack, defence


async def get_real_ratings(team_name: str, api_team_id: int | None = None) -> tuple[float, float, str]:
    """
    Try to get real ratings for a team in priority order:
    1. Qualification campaign stats
    2. WC2022 stats
    3. Hardcoded fallback
    Returns (attack, defence, source_description)
    Raises RateLimitError if the daily API limit is hit.
    """
    from services.team_ratings import get_team_ratings as get_hardcoded

    # Resolve API team ID — use hardcoded map first, then search
    if api_team_id is None:
        api_team_id = await search_team_id(team_name)

    if api_team_id is None:
        a, d = get_hardcoded(team_name)
        return a, d, "hardcoded"

    # Try qualification league first (RateLimitError propagates to caller)
    if team_name in TEAM_QUALIFICATION:
        league_id, season = TEAM_QUALIFICATION[team_name]
        stats = await fetch_team_stats(api_team_id, league_id, season)
        if stats and stats["played"] >= 3:
            a, d = calculate_ratings(stats)
            return a, d, f"qualification (league {league_id}, {season})"

    # Fallback to WC2022
    stats = await fetch_team_stats(api_team_id, WC2022_LEAGUE, WC2022_SEASON)
    if stats and stats["played"] >= 2:
        a, d = calculate_ratings(stats)
        return a, d, "WC2022"

    # Final fallback: hardcoded
    a, d = get_hardcoded(team_name)
    return a, d, "hardcoded"

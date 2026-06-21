"""
Betfair Exchange API client.
Fetches real market volume data — how much money is matched on each outcome.
Requires a Betfair account + app key from developer.betfair.com (free).
"""
import httpx
from datetime import datetime, timedelta

LOGIN_URL = "https://identitysso.betfair.com/api/login"
API_URL = "https://api.betfair.com/exchange/betting/rest/v1.0"

# Betfair market type codes → human-readable labels
MARKET_TYPES = {
    "MATCH_ODDS": "Match Odds (1X2)",
    "ASIAN_HANDICAP": "Asian Handicap",
    "OVER_UNDER_25": "Over/Under 2.5",
    "BOTH_TEAMS_TO_SCORE": "Both Teams to Score",
}


async def login(username: str, password: str, app_key: str) -> str | None:
    """Get a Betfair session token. Returns None if credentials are wrong."""
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.post(
            LOGIN_URL,
            data={"username": username, "password": password},
            headers={
                "X-Application": app_key,
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
        data = r.json()
        if data.get("status") == "SUCCESS":
            return data.get("token")
    return None


async def get_market_volumes(
    home_team: str,
    away_team: str,
    kickoff: datetime,
    app_key: str,
    token: str,
) -> list[dict]:
    """
    Find WC2026 event on Betfair matching these teams, then return volume data
    for Match Odds, Asian Handicap, Over/Under 2.5, and BTTS markets.
    """
    headers = {
        "X-Authentication": token,
        "X-Application": app_key,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    # Search for the event in a ±3h window around kickoff
    date_from = (kickoff - timedelta(hours=3)).strftime("%Y-%m-%dT%H:%M:%SZ")
    date_to = (kickoff + timedelta(hours=3)).strftime("%Y-%m-%dT%H:%M:%SZ")

    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(
            f"{API_URL}/listEvents/",
            json={
                "filter": {
                    "eventTypeIds": ["1"],  # 1 = Soccer
                    "marketStartTime": {"from": date_from, "to": date_to},
                }
            },
            headers=headers,
        )
        events = r.json() if r.status_code == 200 else []

    # Match event by team names (Betfair format: "Team A v Team B")
    event_id = None
    home_lower = home_team.lower()
    away_lower = away_team.lower()
    for ev in events:
        name = ev.get("event", {}).get("name", "").lower()
        if (home_lower in name or away_lower in name):
            event_id = ev["event"]["id"]
            break

    if not event_id:
        return []

    # Get market catalogue for the event
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(
            f"{API_URL}/listMarketCatalogue/",
            json={
                "filter": {
                    "eventIds": [event_id],
                    "marketTypeCodes": list(MARKET_TYPES.keys()),
                },
                "marketProjection": ["RUNNER_DESCRIPTION"],
                "maxResults": 10,
            },
            headers=headers,
        )
        markets = r.json() if r.status_code == 200 else []

    if not markets:
        return []

    market_ids = [m["marketId"] for m in markets]
    market_label = {m["marketId"]: MARKET_TYPES.get(m.get("marketType", ""), m.get("marketName", "")) for m in markets}
    runner_names = {
        m["marketId"]: {
            str(rn["selectionId"]): rn.get("runnerName", "Unknown")
            for rn in m.get("runners", [])
        }
        for m in markets
    }

    # Get live market books (prices + total matched)
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(
            f"{API_URL}/listMarketBook/",
            json={
                "marketIds": market_ids,
                "priceProjection": {"priceData": ["EX_BEST_OFFERS"]},
            },
            headers=headers,
        )
        books = r.json() if r.status_code == 200 else []

    results = []
    for book in books:
        mid = book.get("marketId", "")
        runners_out = []
        for runner in book.get("runners", []):
            sel_id = str(runner.get("selectionId", ""))
            name = runner_names.get(mid, {}).get(sel_id, sel_id)
            best_back = None
            backs = runner.get("ex", {}).get("availableToBack", [])
            if backs:
                best_back = backs[0].get("price")
            runners_out.append({
                "name": name,
                "matched": round(runner.get("totalMatched", 0)),
                "best_back": best_back,
            })
        runners_out.sort(key=lambda x: x["matched"], reverse=True)
        results.append({
            "market": market_label.get(mid, mid),
            "total_matched": round(book.get("totalMatched", 0)),
            "status": book.get("status", ""),
            "runners": runners_out,
        })

    results.sort(key=lambda x: x["total_matched"], reverse=True)
    return results

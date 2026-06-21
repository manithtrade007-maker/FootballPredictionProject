def implied_probability(decimal_odds: float) -> float:
    """Convert decimal odds to implied probability (removes bookmaker margin)."""
    if decimal_odds <= 1.0:
        return 1.0
    return 1.0 / decimal_odds


def kelly_fraction(our_prob: float, decimal_odds: float) -> float:
    """
    Full Kelly Criterion stake fraction.
    f = (bp - q) / b  where b = decimal_odds - 1, p = our_prob, q = 1 - p
    """
    b = decimal_odds - 1.0
    q = 1.0 - our_prob
    f = (b * our_prob - q) / b
    return max(0.0, round(f, 4))


def half_kelly(our_prob: float, decimal_odds: float) -> float:
    """Half Kelly — more conservative, recommended for betting."""
    return kelly_fraction(our_prob, decimal_odds) / 2.0


MIN_EDGE = 0.03   # minimum edge to flag as a value bet (3%)
MIN_ODDS = 1.30   # ignore very short odds


def evaluate_bets(prediction: dict, odds_list: list[dict]) -> list[dict]:
    """
    Compare our predicted probabilities against bookmaker odds.
    Returns a list of value bet assessments sorted by edge descending.
    """
    results = []

    for odds in odds_list:
        bet_type = odds.get("bet_type")
        bookmaker = odds.get("bookmaker", "Unknown")

        checks = _get_checks(prediction, odds, bet_type)

        for selection, our_prob, decimal_odds in checks:
            if decimal_odds is None or decimal_odds < MIN_ODDS or our_prob is None:
                continue

            implied_prob = implied_probability(decimal_odds)
            edge = our_prob - implied_prob
            is_value = edge >= MIN_EDGE
            kelly = half_kelly(our_prob, decimal_odds)

            results.append({
                "bet_type": bet_type,
                "selection": selection,
                "our_probability": round(our_prob, 4),
                "bookmaker_odds": decimal_odds,
                "bookmaker": bookmaker,
                "implied_probability": round(implied_prob, 4),
                "edge": round(edge, 4),
                "kelly_fraction": kelly,
                "is_value": is_value,
            })

    results.sort(key=lambda x: x["edge"], reverse=True)
    return results


def _get_checks(prediction: dict, odds: dict, bet_type: str) -> list[tuple]:
    """Map bet type to (selection, our_prob, bookmaker_odds) tuples."""
    if bet_type == "1X2":
        return [
            ("Home Win", prediction.get("home_win_prob"), odds.get("home_odds")),
            ("Draw", prediction.get("draw_prob"), odds.get("draw_odds")),
            ("Away Win", prediction.get("away_win_prob"), odds.get("away_odds")),
        ]
    elif bet_type == "O/U":
        return [
            ("Over 2.5", prediction.get("over25_prob"), odds.get("home_odds")),
            ("Under 2.5", prediction.get("under25_prob"), odds.get("away_odds")),
        ]
    elif bet_type == "BTTS":
        return [
            ("BTTS Yes", prediction.get("btts_yes_prob"), odds.get("home_odds")),
            ("BTTS No", prediction.get("btts_no_prob"), odds.get("away_odds")),
        ]
    elif bet_type == "AH":
        line = odds.get("line")
        if line is None:
            return []
        ah_data = prediction.get("asian_handicap_data", {}).get(str(line), {})
        return [
            (f"Home {line:+}", ah_data.get("home_cover"), odds.get("home_odds")),
            (f"Away {-line:+}", ah_data.get("away_cover"), odds.get("away_odds")),
        ]
    return []

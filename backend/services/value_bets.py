from .predictor import predict_from_xg
from .verdict import apply_traps, lambda_band_sweep, compute_verdict, build_why_line


def implied_probability(decimal_odds: float) -> float:
    if decimal_odds <= 1.0:
        return 1.0
    return 1.0 / decimal_odds


def kelly_fraction(our_prob: float, decimal_odds: float) -> float:
    b = decimal_odds - 1.0
    q = 1.0 - our_prob
    f = (b * our_prob - q) / b
    return max(0.0, round(f, 4))


def half_kelly(our_prob: float, decimal_odds: float) -> float:
    return kelly_fraction(our_prob, decimal_odds) / 2.0


MIN_EDGE = 0.03
MIN_ODDS = 1.30


def evaluate_bets(prediction: dict, odds_list: list[dict]) -> list[dict]:
    """
    Compare predicted probabilities against bookmaker odds.
    Enriches each result with lambda-band robustness verdict (ROBUST/FRAGILE/MARGINAL/NO_EDGE).
    No trap flags are active in automated mode (traps are for the manual /calculator page).
    """
    lambda_a = prediction.get("expected_home_goals", 1.2)
    lambda_b = prediction.get("expected_away_goals", 1.2)

    results = []
    for odds in odds_list:
        bet_type = odds.get("bet_type")
        bookmaker = odds.get("bookmaker", "Unknown")

        checks = _get_checks(prediction, odds, bet_type)

        for selection, our_prob, decimal_odds, ah_home_line in checks:
            if decimal_odds is None or decimal_odds < MIN_ODDS or our_prob is None:
                continue

            implied_prob = implied_probability(decimal_odds)
            raw_edge = our_prob - implied_prob
            is_value = raw_edge >= MIN_EDGE
            kelly = half_kelly(our_prob, decimal_odds)

            # Traps: empty dict in auto mode (no situational flags available)
            adj_edge, trap_mult = apply_traps(raw_edge, {})

            # Lambda band robustness sweep
            focus_fn = _make_focus_fn(selection, ah_home_line)
            if focus_fn is not None:
                min_edge, max_edge = lambda_band_sweep(
                    focus_fn, lambda_a, lambda_b, implied_prob
                )
            else:
                min_edge = max_edge = raw_edge

            verdict, action, why = compute_verdict(adj_edge, min_edge, raw_edge)
            why_ln = build_why_line(our_prob, implied_prob, adj_edge, verdict, action, why)

            results.append({
                "bet_type":          bet_type,
                "selection":         selection,
                "our_probability":   round(our_prob, 4),
                "bookmaker_odds":    decimal_odds,
                "bookmaker":         bookmaker,
                "implied_probability": round(implied_prob, 4),
                "edge":              round(raw_edge, 4),
                "kelly_fraction":    kelly,
                "is_value":          is_value,
                "verdict":           verdict,
                "action":            action,
                "min_edge":          round(min_edge, 4),
                "max_edge":          round(max_edge, 4),
                "why_line":          why_ln,
            })

    results.sort(key=lambda x: x["edge"], reverse=True)
    return results


def _get_checks(prediction: dict, odds: dict, bet_type: str) -> list[tuple]:
    """Return (selection, our_prob, bookmaker_odds, ah_home_line) tuples."""
    if bet_type == "1X2":
        return [
            ("Home Win", prediction.get("home_win_prob"), odds.get("home_odds"), None),
            ("Draw",     prediction.get("draw_prob"),     odds.get("draw_odds"), None),
            ("Away Win", prediction.get("away_win_prob"), odds.get("away_odds"), None),
        ]
    elif bet_type == "O/U":
        return [
            ("Over 2.5",  prediction.get("over25_prob"),  odds.get("home_odds"), None),
            ("Under 2.5", prediction.get("under25_prob"), odds.get("away_odds"), None),
        ]
    elif bet_type == "BTTS":
        return [
            ("BTTS Yes", prediction.get("btts_yes_prob"), odds.get("home_odds"), None),
            ("BTTS No",  prediction.get("btts_no_prob"),  odds.get("away_odds"), None),
        ]
    elif bet_type == "AH":
        line = odds.get("line")
        if line is None:
            return []
        ah_data = prediction.get("asian_handicap_data", {}).get(str(float(line)), {})
        if not ah_data:
            rounded = round(line * 4) / 4
            ah_data = prediction.get("asian_handicap_data", {}).get(str(rounded), {})
        if not ah_data:
            return []

        home_cover = ah_data.get("home_cover", 0)
        away_cover = ah_data.get("away_cover", 0)
        push       = ah_data.get("push", 0)
        if push > 0:
            eff_home = home_cover + push * 0.5
            eff_away = away_cover + push * 0.5
        else:
            eff_home = home_cover
            eff_away = away_cover

        return [
            (f"Home AH {line:+.1f}", eff_home, odds.get("home_odds"), line),
            (f"Away AH {-line:+.1f}", eff_away, odds.get("away_odds"), line),
        ]
    return []


def _make_focus_fn(selection: str, ah_home_line: float | None):
    """Return (la, lb) -> probability callable for the given selection."""
    s = selection.lower()
    if "home win" in s:
        return lambda la, lb: predict_from_xg(la, lb)["home_win"]
    if s == "draw":
        return lambda la, lb: predict_from_xg(la, lb)["draw"]
    if "away win" in s:
        return lambda la, lb: predict_from_xg(la, lb)["away_win"]
    if s == "over 2.5":
        return lambda la, lb: predict_from_xg(la, lb)["over25"]
    if s == "under 2.5":
        return lambda la, lb: predict_from_xg(la, lb)["under25"]
    if s == "btts yes":
        return lambda la, lb: predict_from_xg(la, lb)["btts_yes"]
    if s == "btts no":
        return lambda la, lb: predict_from_xg(la, lb)["btts_no"]
    if "home ah" in s and ah_home_line is not None:
        def _home_ah_fn(la, lb, _ln=ah_home_line):
            ah = predict_from_xg(la, lb)["ah_data"]
            d = ah.get(str(float(_ln))) or ah.get(str(round(_ln * 4) / 4)) or {}
            return d.get("home_cover", 0) + d.get("push", 0) * 0.5
        return _home_ah_fn
    if "away ah" in s and ah_home_line is not None:
        def _away_ah_fn(la, lb, _ln=ah_home_line):
            ah = predict_from_xg(la, lb)["ah_data"]
            d = ah.get(str(float(_ln))) or ah.get(str(round(_ln * 4) / 4)) or {}
            return d.get("away_cover", 0) + d.get("push", 0) * 0.5
        return _away_ah_fn
    return None

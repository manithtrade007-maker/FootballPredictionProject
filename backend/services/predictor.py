import math
import numpy as np


MAX_GOALS = 10

# WC2026 neutral-venue average — no home advantage at USA/Canada/Mexico venues.
# Both teams use the same baseline (historical WC average ≈ 1.33 goals/team/game).
WC_NEUTRAL_AVG = 1.23

# Dixon-Coles rho: corrects Poisson underestimation of 0-0, 1-0, 0-1, 1-1 scorelines.
# Empirically fitted for football; 0.13 is the standard value from the original 1997 paper.
DC_RHO = 0.13


def _poisson_pmf(k: int, lam: float) -> float:
    return (lam ** k) * math.exp(-lam) / math.factorial(k)


def _goal_matrix(home_xg: float, away_xg: float) -> np.ndarray:
    """Build (MAX_GOALS x MAX_GOALS) joint probability matrix with Dixon-Coles correction."""
    home_probs = np.array([_poisson_pmf(i, home_xg) for i in range(MAX_GOALS)])
    away_probs = np.array([_poisson_pmf(i, away_xg) for i in range(MAX_GOALS)])
    matrix = np.outer(home_probs, away_probs)

    # Dixon-Coles correction: fix underestimation of low-scoring scorelines
    lam, mu, rho = home_xg, away_xg, DC_RHO
    matrix[0, 0] *= (1 - lam * mu * rho)
    matrix[1, 0] *= (1 + mu * rho)
    matrix[0, 1] *= (1 + lam * rho)
    matrix[1, 1] *= (1 - rho)

    # Renormalise so probabilities still sum to 1
    matrix /= matrix.sum()
    return matrix


def predict(
    home_attack: float,
    home_defence: float,
    away_attack: float,
    away_defence: float,
) -> dict:
    """
    Dixon-Coles corrected Poisson match prediction for neutral WC venues.
    No home advantage applied — WC2026 matches are at neutral USA/Canada/Mexico venues.
    """
    # Neutral venue: both teams use the same WC baseline (no home boost)
    home_xg = home_attack * away_defence * WC_NEUTRAL_AVG
    away_xg = away_attack * home_defence * WC_NEUTRAL_AVG

    home_xg = max(0.3, min(home_xg, 5.0))
    away_xg = max(0.3, min(away_xg, 5.0))

    matrix = _goal_matrix(home_xg, away_xg)

    home_win = float(np.sum(np.tril(matrix, -1)))
    draw = float(np.sum(np.diag(matrix)))
    away_win = float(np.sum(np.triu(matrix, 1)))

    over25 = float(1 - sum(
        matrix[h][a]
        for h in range(MAX_GOALS)
        for a in range(MAX_GOALS)
        if h + a <= 2
    ))
    under25 = 1.0 - over25

    btts_yes = float(1 - sum(
        matrix[h][a]
        for h in range(MAX_GOALS)
        for a in range(MAX_GOALS)
        if h == 0 or a == 0
    ))
    btts_no = 1.0 - btts_yes

    # Modal scoreline: the single most likely exact result from the matrix
    modal_idx = np.unravel_index(np.argmax(matrix), matrix.shape)
    predicted_score = f"{modal_idx[0]}-{modal_idx[1]}"

    max_prob = max(home_win, draw, away_win)
    confidence = round(max_prob * 100, 1)

    asian_handicap = _asian_handicap(matrix, home_xg, away_xg)

    return {
        "home_win_prob": round(home_win, 4),
        "draw_prob": round(draw, 4),
        "away_win_prob": round(away_win, 4),
        "expected_home_goals": round(home_xg, 2),
        "expected_away_goals": round(away_xg, 2),
        "over25_prob": round(over25, 4),
        "under25_prob": round(under25, 4),
        "btts_yes_prob": round(btts_yes, 4),
        "btts_no_prob": round(btts_no, 4),
        "asian_handicap_data": asian_handicap,
        "predicted_score": predicted_score,
        "confidence": confidence,
    }


def _asian_handicap(matrix: np.ndarray, home_xg: float, away_xg: float) -> dict:
    """
    Calculate Asian Handicap probabilities for a wide range of lines.
    Line is expressed as the HOME team's handicap (negative = gives goals, positive = receives goals).

    Half-line (±0.5 step): no push possible — clean win/loss split.
    Whole-line (±1.0 step): push when margin equals handicap exactly.
    Quarter-line (±0.25 step / split bet): half stake on each adjacent half-line.
    """
    # Full-step and half-step lines (-3.5 to +3.5)
    half_lines = [-3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    result = {}

    # Pre-compute raw probabilities for every half-step line
    raw: dict[float, dict] = {}
    for line in half_lines:
        home_cover = 0.0
        push = 0.0
        away_cover = 0.0
        for h in range(MAX_GOALS):
            for a in range(MAX_GOALS):
                diff = h - a + line
                prob = matrix[h][a]
                if diff > 0:
                    home_cover += prob
                elif diff == 0:
                    push += prob
                else:
                    away_cover += prob
        raw[line] = {
            "home_cover": round(home_cover, 4),
            "push": round(push, 4),
            "away_cover": round(away_cover, 4),
        }
        result[str(line)] = raw[line]

    # Quarter-line (split) handicaps: -0.75, -0.25, +0.25, +0.75 and larger
    quarter_lines = [-2.75, -2.25, -1.75, -1.25, -0.75, -0.25, 0.25, 0.75, 1.25, 1.75, 2.25, 2.75]
    for qline in quarter_lines:
        # Split bet: half stake on floor half-line, half stake on ceil half-line
        low = round(qline - 0.25, 2)   # e.g. -0.75 → low=-1.0
        high = round(qline + 0.25, 2)  # e.g. -0.75 → high=-0.5
        low_data = raw.get(low, {"home_cover": 0, "push": 0, "away_cover": 0})
        high_data = raw.get(high, {"home_cover": 0, "push": 0, "away_cover": 0})

        # For split bets, there is no push — push on one half becomes a half-win/half-loss
        # effective_home = 0.5*home_cover(low) + 0.5*home_cover(high) + 0.5*push(low or high if applicable)
        # Because: if low pushes, you get half refunded → counts as 0 EV on that half
        # Represent as effective probabilities (push is absorbed proportionally):
        eff_home = round(0.5 * (low_data["home_cover"] + low_data["push"]) +
                         0.5 * high_data["home_cover"], 4)
        eff_away = round(0.5 * low_data["away_cover"] +
                         0.5 * (high_data["away_cover"] + high_data["push"]), 4)

        result[str(qline)] = {
            "home_cover": eff_home,
            "push": 0.0,  # no push on split bets
            "away_cover": eff_away,
        }

    return result


def predict_from_xg(home_xg: float, away_xg: float) -> dict:
    """Compute outcome probabilities directly from expected-goal values (for band sweep)."""
    home_xg = max(0.3, min(home_xg, 5.0))
    away_xg = max(0.3, min(away_xg, 5.0))
    matrix = _goal_matrix(home_xg, away_xg)

    home_win = float(np.sum(np.tril(matrix, -1)))
    draw     = float(np.sum(np.diag(matrix)))
    away_win = float(np.sum(np.triu(matrix, 1)))

    over25 = float(1 - sum(
        matrix[h][a]
        for h in range(MAX_GOALS)
        for a in range(MAX_GOALS)
        if h + a <= 2
    ))
    btts_yes = float(1 - sum(
        matrix[h][a]
        for h in range(MAX_GOALS)
        for a in range(MAX_GOALS)
        if h == 0 or a == 0
    ))

    return {
        "home_win":  home_win,
        "draw":      draw,
        "away_win":  away_win,
        "over25":    over25,
        "under25":   1.0 - over25,
        "btts_yes":  btts_yes,
        "btts_no":   1.0 - btts_yes,
        "ah_data":   _asian_handicap(matrix, home_xg, away_xg),
    }


def build_team_ratings(stats: dict) -> tuple[float, float]:
    """
    Convert raw team stats into attack/defence ratings relative to WC average.
    stats should contain: goals_scored, goals_conceded, matches_played
    Returns (attack_rating, defence_rating)
    """
    played = max(stats.get("matches_played", 1), 1)
    scored = stats.get("goals_scored", WC_AVG_GOALS_HOME * played)
    conceded = stats.get("goals_conceded", WC_AVG_GOALS_AWAY * played)

    attack = (scored / played) / WC_NEUTRAL_AVG
    defence = (conceded / played) / WC_NEUTRAL_AVG

    return round(attack, 3), round(defence, 3)

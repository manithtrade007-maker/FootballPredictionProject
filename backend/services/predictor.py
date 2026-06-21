from scipy.stats import poisson
import numpy as np


MAX_GOALS = 10

# Average goals per match in World Cup history (used as league baseline)
WC_AVG_GOALS_HOME = 1.36
WC_AVG_GOALS_AWAY = 1.10


def _goal_matrix(home_xg: float, away_xg: float) -> np.ndarray:
    """Build a (MAX_GOALS x MAX_GOALS) joint probability matrix."""
    home_probs = np.array([poisson.pmf(i, home_xg) for i in range(MAX_GOALS)])
    away_probs = np.array([poisson.pmf(i, away_xg) for i in range(MAX_GOALS)])
    return np.outer(home_probs, away_probs)


def predict(
    home_attack: float,
    home_defence: float,
    away_attack: float,
    away_defence: float,
) -> dict:
    """
    Poisson-based match prediction.

    Attack/defence ratings are relative to WC average (1.0 = average).
    Expected goals = attack_rating * opponent_defence_rating * league_average
    """
    home_xg = home_attack * away_defence * WC_AVG_GOALS_HOME
    away_xg = away_attack * home_defence * WC_AVG_GOALS_AWAY

    # Clamp to sensible range
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

    predicted_home = int(np.argmax(matrix) // MAX_GOALS)
    predicted_away = int(np.argmax(matrix) % MAX_GOALS)
    predicted_score = f"{predicted_home}-{predicted_away}"

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
    """Calculate Asian Handicap probabilities for common lines."""
    lines = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
    result = {}

    for line in lines:
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

        result[str(line)] = {
            "home_cover": round(home_cover, 4),
            "push": round(push, 4),
            "away_cover": round(away_cover, 4),
        }

    return result


def build_team_ratings(stats: dict) -> tuple[float, float]:
    """
    Convert raw team stats into attack/defence ratings relative to WC average.
    stats should contain: goals_scored, goals_conceded, matches_played
    Returns (attack_rating, defence_rating)
    """
    played = max(stats.get("matches_played", 1), 1)
    scored = stats.get("goals_scored", WC_AVG_GOALS_HOME * played)
    conceded = stats.get("goals_conceded", WC_AVG_GOALS_AWAY * played)

    attack = (scored / played) / WC_AVG_GOALS_HOME
    defence = (conceded / played) / WC_AVG_GOALS_AWAY

    return round(attack, 3), round(defence, 3)

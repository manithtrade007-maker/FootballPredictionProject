"""
WC2026 team strength ratings based on FIFA rankings + WC2022 performance.
Scale: attack/defence relative to WC average (1.0 = average team).
Higher attack = scores more. Lower defence = concedes less (better defending).
"""

TEAM_RATINGS: dict[str, tuple[float, float]] = {
    # (attack_rating, defence_rating)
    # Elite
    "France":       (1.45, 0.70),
    "England":      (1.35, 0.75),
    "Brazil":       (1.40, 0.72),
    "Argentina":    (1.42, 0.68),
    "Spain":        (1.38, 0.72),
    "Germany":      (1.32, 0.78),
    "Portugal":     (1.35, 0.80),
    "Netherlands":  (1.28, 0.80),
    "Belgium":      (1.20, 0.85),
    "Italy":        (1.15, 0.82),
    # Strong
    "Morocco":      (1.10, 0.82),
    "Senegal":      (1.08, 0.88),
    "Japan":        (1.10, 0.85),
    "South Korea":  (1.05, 0.90),
    "USA":          (1.08, 0.88),
    "Mexico":       (1.05, 0.92),
    "Croatia":      (1.12, 0.83),
    "Denmark":      (1.10, 0.85),
    "Switzerland":  (1.08, 0.86),
    "Uruguay":      (1.12, 0.85),
    "Colombia":     (1.10, 0.88),
    "Ecuador":      (1.05, 0.90),
    "Peru":         (1.00, 0.92),
    "Chile":        (1.02, 0.92),
    "Canada":       (1.00, 0.92),
    "Australia":    (1.00, 0.93),
    "Cameroon":     (0.98, 0.95),
    "Ghana":        (0.95, 0.98),
    "Tunisia":      (0.95, 0.97),
    "Nigeria":      (1.00, 0.95),
    # Moderate
    "Poland":       (1.05, 0.90),
    "Serbia":       (1.05, 0.92),
    "Austria":      (1.00, 0.92),
    "Turkey":       (1.02, 0.93),
    "Ukraine":      (1.00, 0.93),
    "Sweden":       (1.00, 0.92),
    "Wales":        (0.95, 0.95),
    "Scotland":     (0.95, 0.95),
    "Iran":         (0.90, 1.00),
    "Saudi Arabia": (0.88, 1.02),
    "Qatar":        (0.80, 1.08),
    "Ivory Coast":  (0.98, 0.95),
    "Algeria":      (0.92, 1.00),
    "Egypt":        (0.90, 1.00),
    "Mali":         (0.88, 1.02),
    "Cape Verde":   (0.85, 1.05),
    "New Zealand":  (0.80, 1.08),
    "Panama":       (0.82, 1.05),
    "Costa Rica":   (0.85, 1.05),
    "Honduras":     (0.82, 1.08),
    "Jamaica":      (0.80, 1.10),
    "Venezuela":    (0.88, 1.02),
    "Bolivia":      (0.80, 1.08),
    "Paraguay":     (0.90, 1.00),
    "El Salvador":  (0.78, 1.12),
}

DEFAULT_RATINGS = (1.0, 1.0)


def get_team_ratings(team_name: str) -> tuple[float, float]:
    """Return (attack, defence) for a team, with fuzzy matching."""
    if team_name in TEAM_RATINGS:
        return TEAM_RATINGS[team_name]
    # Fuzzy: check if any key is a substring
    for key, ratings in TEAM_RATINGS.items():
        if key.lower() in team_name.lower() or team_name.lower() in key.lower():
            return ratings
    return DEFAULT_RATINGS

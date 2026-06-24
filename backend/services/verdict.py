"""
Verdict engine — implements the MODEL_SPEC.md pipeline stages 4-6.

Stage 4: Trap filters  — haircut the raw edge for known situational biases
Stage 5: Lambda band   — sweep ±band on a 5×5 grid; check if edge holds everywhere
Stage 6: Verdict       — ROBUST / FRAGILE / MARGINAL / NO_EDGE → BET / LEAN / PASS
"""

from __future__ import annotations
import math
from typing import Callable

# ── Trap definitions ────────────────────────────────────────────────────────
TRAPS: dict[str, float] = {
    "gamestate":  0.50,   # both teams happy with a draw → suppresses scoring
    "consensus":  0.30,   # everyone on same side → price already shaded
    "recency":    0.25,   # one blowout skews the team's true level
    "steam":      0.30,   # shortening favourite = public money inflating price
}

VERDICT_THRESHOLD = 0.03  # minimum adj edge to be considered playable


def apply_traps(raw_edge: float, active_traps: dict[str, bool]) -> tuple[float, float]:
    """
    Multiply the magnitude of a positive edge by (1 - cut) for each active trap.
    Never apply haircuts to a negative edge.
    Returns (adj_edge, trap_multiplier).
    """
    if raw_edge <= 0:
        return raw_edge, 1.0

    mult = 1.0
    for trap_id, cut in TRAPS.items():
        if active_traps.get(trap_id, False):
            mult *= (1.0 - cut)

    return round(raw_edge * mult, 6), round(mult, 6)


def lambda_band_sweep(
    focus_prob_fn: Callable[[float, float], float],
    lambda_a: float,
    lambda_b: float,
    breakeven: float,
    band: float = 0.20,
    grid: int = 5,
) -> tuple[float, float]:
    """
    Sweep both lambdas across [central - band, central + band] on a grid×grid grid.
    At each point compute raw edge = focusProb - breakeven.
    Returns (min_edge, max_edge) across all grid points.
    """
    step = (2 * band) / (grid - 1) if grid > 1 else 0
    edges: list[float] = []

    for i in range(grid):
        la = max(0.05, lambda_a - band + i * step)
        for j in range(grid):
            lb = max(0.05, lambda_b - band + j * step)
            prob = focus_prob_fn(la, lb)
            edges.append(prob - breakeven)

    return round(min(edges), 6), round(max(edges), 6)


def compute_verdict(
    adj_edge: float,
    min_edge: float,
    raw_edge: float,
    threshold: float = VERDICT_THRESHOLD,
) -> tuple[str, str, str]:
    """
    Returns (verdict, action, why).

    ROBUST   → adj_edge >= threshold AND min_edge > 0
    FRAGILE  → adj_edge > 0 AND min_edge <= 0
    MARGINAL → raw_edge > 0 AND adj_edge < threshold
    NO_EDGE  → negative value
    """
    if adj_edge >= threshold and min_edge > 0:
        verdict = "ROBUST"
        action  = "BET"
        why     = "edge holds across the whole λ band"
    elif adj_edge > 0 and min_edge <= 0:
        verdict = "FRAGILE"
        action  = "LEAN"
        why     = "edge flips negative inside your own λ range — not real"
    elif raw_edge > 0 and adj_edge < threshold:
        verdict = "MARGINAL"
        action  = "LEAN"
        why     = "edge below threshold after trap haircut"
    else:
        verdict = "NO_EDGE"
        action  = "PASS"
        why     = "price is below model probability"

    return verdict, action, why


def recommend_instrument(
    focus_market: str,
    verdict: str,
    action: str,
    gamestate_active: bool,
) -> str:
    """Return a short string describing the best instrument given traps + verdict."""
    if action == "PASS":
        return "No bet — sit it out."
    if focus_market in ("A", "B") and gamestate_active:
        side = "A" if focus_market == "A" else "B"
        return f"Consider DNB_{side} (refunds the draw the gamestate trap points to)."
    if focus_market in ("DNB_A", "DNB_B"):
        return "DNB endorsed — draw refunds rather than sinks the bet."
    if focus_market == "Over25" and gamestate_active:
        return "Caution: gamestate suppresses goals. Draw or Under fits better."
    if focus_market == "Under25":
        return "Under aligns with the draw-suppresses-goals read."
    if focus_market == "Draw":
        return "Draw directly backs the game-state both teams are pulled toward."
    return f"{focus_market} — {action.lower()} (small, entertainment stake)."


def build_why_line(
    focus_prob: float,
    breakeven: float,
    adj_edge: float,
    verdict: str,
    action: str,
    why: str,
) -> str:
    fp  = round(focus_prob * 100, 1)
    be  = round(breakeven  * 100, 1)
    ae  = round(adj_edge   * 100, 1)
    sign = "+" if ae >= 0 else ""
    line = f"Model {fp}% vs break-even {be}% → {sign}{ae}% after traps · {why}"
    if action != "PASS":
        line += " Entertainment stake only."
    return line

"""
EV-weighted action selection with opponent-adaptive thresholds and controlled
randomness injection.
"""

import random

FOLD, RAISE, CHECK, CALL, DISCARD = 0, 1, 2, 3, 4


# ---------------------------------------------------------------------------
# Pre-flop action (street 0)
# ---------------------------------------------------------------------------

def preflop_action(observation: dict, strength: float, detail: dict,
                   categorizer, profile) -> tuple:
    """
    Decide a pre-flop action using the O(1) preflop-table lookup and
    opponent category awareness.

    Returns (action_type, raise_amount, 0, 0).
    """
    valid = observation["valid_actions"]
    min_raise = observation["min_raise"]
    max_raise = observation["max_raise"]
    pot = observation["pot_size"]
    facing_raise = observation["opp_bet"] > observation["my_bet"]

    dom = categorizer.dominant_category()
    fold_prob = categorizer.get_weighted_fold_prob(0, profile)
    hands = profile.hands_played

    # Adaptive thresholds (calibrated to actual strength distribution:
    # p10=0.49, p25=0.55, p50=0.60, p75=0.65, p90=0.72)
    is_sb = observation.get("blind_position", 0) == 0

    raise_thresh = 0.70 if is_sb else 0.72
    call_thresh = 0.50
    fold_thresh = 0.35

    if hands >= 30:
        if dom in ("nit", "rock"):
            raise_thresh = 0.62
            fold_thresh = 0.50 if facing_raise else 0.35
        elif dom == "calling_station":
            raise_thresh = 0.78
        elif dom == "maniac":
            call_thresh = 0.40

    def _raise_size(base_frac: float) -> int:
        amt = max(int(pot * base_frac), min_raise)
        return max(min(amt, max_raise), min_raise)

    # Strong hand: raise
    if strength >= raise_thresh and valid[RAISE]:
        return (RAISE, _raise_size(0.75), 0, 0)

    # Facing a raise: call if hand is decent, fold trash
    if facing_raise:
        if strength >= call_thresh and valid[CALL]:
            return (CALL, 0, 0, 0)
        if strength < fold_thresh and valid[FOLD]:
            return (FOLD, 0, 0, 0)
        if valid[CALL]:
            return (CALL, 0, 0, 0)
        return (FOLD, 0, 0, 0)

    # Not facing a raise: check or complete SB
    if valid[CHECK]:
        return (CHECK, 0, 0, 0)
    if valid[CALL]:
        return (CALL, 0, 0, 0)
    return (FOLD, 0, 0, 0)


# ---------------------------------------------------------------------------
# Post-discard betting (streets 1-3)
# ---------------------------------------------------------------------------

def choose_action(observation: dict, equity: float,
                  categorizer, profile) -> tuple:
    """
    Core post-discard betting strategy.  Foundation is pot-odds play (proven
    profitable), with opponent-adaptive adjustments layered on top.

    Returns (action_type, raise_amount, 0, 0).
    """
    valid = observation["valid_actions"]
    street = observation["street"]
    my_bet = observation["my_bet"]
    opp_bet = observation["opp_bet"]
    pot = observation["pot_size"]
    min_raise = observation["min_raise"]
    max_raise = observation["max_raise"]

    facing_raise = opp_bet > my_bet
    continue_cost = opp_bet - my_bet
    pot_odds = continue_cost / (continue_cost + pot) if continue_cost > 0 else 0

    fold_prob = categorizer.get_weighted_fold_prob(street, profile)
    dom = categorizer.dominant_category()
    hands_played = profile.hands_played

    def _raise_size(base_frac: float) -> int:
        amt = max(int(pot * base_frac), min_raise)
        return max(min(amt, max_raise), min_raise)

    # --- Raise threshold: varies by opponent type ---
    raise_thresh = 0.75
    if hands_played >= 30:
        if dom in ("nit", "rock"):
            raise_thresh = 0.65
        elif dom == "calling_station":
            raise_thresh = 0.70
        elif dom == "maniac":
            raise_thresh = 0.60

    # --- Bet sizing: adapt to opponent ---
    if dom == "calling_station" and hands_played >= 30:
        raise_frac = 0.85
    elif dom in ("nit", "rock") and hands_played >= 30:
        raise_frac = 0.50
    else:
        raise_frac = 0.75

    # --- Core decision ---

    # Raise with strong hands
    if equity > raise_thresh and valid[RAISE]:
        return (RAISE, _raise_size(raise_frac), 0, 0)

    # Facing a raise: call with pot odds
    if facing_raise:
        if equity >= pot_odds and valid[CALL]:
            return (CALL, 0, 0, 0)
        if valid[CHECK]:
            return (CHECK, 0, 0, 0)
        return (FOLD, 0, 0, 0)

    # Not facing a raise: bet medium-strong hands to extract value
    if equity > 0.65 and valid[RAISE]:
        return (RAISE, _raise_size(0.50), 0, 0)

    if valid[CHECK]:
        return (CHECK, 0, 0, 0)
    if valid[CALL]:
        return (CALL, 0, 0, 0)
    return (FOLD, 0, 0, 0)

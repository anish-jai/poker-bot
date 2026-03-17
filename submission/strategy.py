"""
EV-weighted action selection with opponent-adaptive thresholds.

Key design principles learned from real tournament play:
- Bet big when you bet (deny cheap re-raises)
- Fold more when facing large bets (pot odds alone isn't enough)
- Don't escalate raise wars without a monster
- Steal blinds from nits who fold everything
"""

import random

FOLD, RAISE, CHECK, CALL, DISCARD = 0, 1, 2, 3, 4


# ---------------------------------------------------------------------------
# Pre-flop action (street 0)
# ---------------------------------------------------------------------------

def preflop_action(observation: dict, strength: float, detail: dict,
                   categorizer, profile) -> tuple:
    valid = observation["valid_actions"]
    min_raise = observation["min_raise"]
    max_raise = observation["max_raise"]
    pot = observation["pot_size"]
    my_bet = observation["my_bet"]
    opp_bet = observation["opp_bet"]
    facing_raise = opp_bet > my_bet

    dom = categorizer.dominant_category()
    hands = profile.hands_played

    is_sb = observation.get("blind_position", 0) == 0

    raise_thresh = 0.68 if is_sb else 0.70
    call_thresh = 0.50
    fold_thresh = 0.38

    if hands >= 20:
        if dom in ("nit", "rock"):
            raise_thresh = 0.55
            fold_thresh = 0.55 if facing_raise else 0.35
        elif dom == "calling_station":
            raise_thresh = 0.75
            call_thresh = 0.45
        elif dom == "maniac":
            call_thresh = 0.42
            fold_thresh = 0.30

    # Steal detection: if opponent folds > 80% of hands, raise almost everything
    if hands >= 30:
        opp_fold_rate = profile.fold_to_raise_rate()
        if opp_fold_rate > 0.70:
            raise_thresh = 0.45
        elif opp_fold_rate > 0.50:
            raise_thresh = min(raise_thresh, 0.55)

    def _raise_size(base_frac: float) -> int:
        amt = max(int(pot * base_frac), min_raise)
        return max(min(amt, max_raise), min_raise)

    if strength >= raise_thresh and valid[RAISE]:
        return (RAISE, _raise_size(0.75), 0, 0)

    if facing_raise:
        cost = opp_bet - my_bet
        raise_size_ratio = cost / max(pot, 1)
        if raise_size_ratio > 3.0 and strength < 0.72:
            return (FOLD, 0, 0, 0)
        if strength >= call_thresh and valid[CALL]:
            return (CALL, 0, 0, 0)
        if strength < fold_thresh and valid[FOLD]:
            return (FOLD, 0, 0, 0)
        if valid[CALL]:
            return (CALL, 0, 0, 0)
        return (FOLD, 0, 0, 0)

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

    dom = categorizer.dominant_category()
    hands_played = profile.hands_played

    def _raise_size(base_frac: float) -> int:
        amt = max(int(pot * base_frac), min_raise)
        return max(min(amt, max_raise), min_raise)

    # Detect how much of our stack is already committed
    committed_frac = my_bet / 100.0

    # --- Raise threshold: varies by opponent type ---
    raise_thresh = 0.75
    if hands_played >= 20:
        if dom in ("nit", "rock"):
            raise_thresh = 0.62
        elif dom == "calling_station":
            raise_thresh = 0.68
        elif dom == "maniac":
            raise_thresh = 0.72

    # --- Bet sizing: BIG to deny cheap re-raises ---
    if dom == "calling_station" and hands_played >= 20:
        raise_frac = 0.85
    elif dom in ("nit", "rock") and hands_played >= 20:
        raise_frac = 0.60
    else:
        raise_frac = 0.75

    # --------------- FACING A RAISE ---------------
    if facing_raise:
        bet_to_pot = continue_cost / max(pot, 1)

        # Risk premium: require equity above pot odds by a margin that
        # scales with bet size. A min-raise needs ~pot_odds equity,
        # but a pot-sized bet or bigger needs meaningfully more.
        risk_premium = 0.04 + 0.06 * min(bet_to_pot, 2.0)
        call_threshold = pot_odds + risk_premium

        # Against big bets (>60% of remaining stack), tighten further
        if continue_cost > 60 - my_bet:
            call_threshold = max(call_threshold, 0.55)

        # If we already raised this street and they re-raised,
        # we need a very strong hand to continue
        we_raised_this_street = _did_we_raise_this_street(observation)
        if we_raised_this_street:
            call_threshold = max(call_threshold, 0.60)
            if equity > 0.82 and valid[RAISE]:
                return (RAISE, _raise_size(0.75), 0, 0)
            if equity >= call_threshold and valid[CALL]:
                return (CALL, 0, 0, 0)
            return (FOLD, 0, 0, 0)

        # Strong hand facing a raise: re-raise
        if equity > 0.82 and valid[RAISE]:
            return (RAISE, _raise_size(raise_frac), 0, 0)

        # Call if equity justifies it
        if equity >= call_threshold and valid[CALL]:
            return (CALL, 0, 0, 0)

        if valid[CHECK]:
            return (CHECK, 0, 0, 0)
        return (FOLD, 0, 0, 0)

    # --------------- NOT FACING A RAISE ---------------

    # Strong hand: raise big for value
    if equity > raise_thresh and valid[RAISE]:
        return (RAISE, _raise_size(raise_frac), 0, 0)

    # Medium-strong: value bet (but size it properly)
    if equity > 0.62 and valid[RAISE]:
        return (RAISE, _raise_size(0.60), 0, 0)

    if valid[CHECK]:
        return (CHECK, 0, 0, 0)
    if valid[CALL]:
        return (CALL, 0, 0, 0)
    return (FOLD, 0, 0, 0)


def _did_we_raise_this_street(observation: dict) -> bool:
    """Heuristic: if our bet > opponent's previous bet on this street,
    we likely raised. We check if my_bet > big_blind equivalent and
    my_bet is substantial relative to pot."""
    my_bet = observation["my_bet"]
    opp_bet = observation["opp_bet"]
    pot = observation["pot_size"]
    # If we have a significant bet already and opponent has raised above it,
    # we likely raised earlier. Use the fact that if my_bet > 2 (big blind)
    # and opp_bet > my_bet, we probably bet first.
    return my_bet > 2 and opp_bet > my_bet

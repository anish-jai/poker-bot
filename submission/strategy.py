"""
EV-weighted action selection with opponent-adaptive thresholds.

Key design principles learned from real tournament play:
- Fold weak hands pre-flop instead of calling everything
- Bet big when you bet (deny cheap re-raises)
- Fold more when facing large bets (pot odds alone isn't enough)
- Don't escalate raise wars without a monster
- Steal blinds from opponents in fold-lockdown
- Don't value bet into tight opponents who only call with better
"""

FOLD, RAISE, CHECK, CALL, DISCARD = 0, 1, 2, 3, 4


# ---------------------------------------------------------------------------
# Pre-flop action (street 0)
# ---------------------------------------------------------------------------

def preflop_action(observation: dict, strength: float, detail: dict,
                   categorizer, profile, opp_in_lockdown: bool = False) -> tuple:
    valid = observation["valid_actions"]
    min_raise = observation["min_raise"]
    max_raise = observation["max_raise"]
    pot = observation["pot_size"]
    my_bet = observation["my_bet"]
    opp_bet = observation["opp_bet"]
    facing_raise = opp_bet > my_bet
    cost = opp_bet - my_bet if facing_raise else 0

    dom = categorizer.dominant_category()
    hands = profile.hands_played

    is_sb = observation.get("blind_position", 0) == 0

    # If opponent is in fold-lockdown, min-raise everything to steal blinds
    if opp_in_lockdown:
        if valid[RAISE]:
            return (RAISE, min_raise, 0, 0)
        if valid[CALL]:
            return (CALL, 0, 0, 0)
        if valid[CHECK]:
            return (CHECK, 0, 0, 0)
        return (FOLD, 0, 0, 0)

    raise_thresh = 0.68 if is_sb else 0.70
    call_thresh = 0.50
    fold_thresh = 0.48

    if hands >= 20:
        if dom in ("nit", "rock"):
            raise_thresh = 0.55
            fold_thresh = 0.55 if facing_raise else 0.45
        elif dom == "calling_station":
            raise_thresh = 0.75
            call_thresh = 0.45
        elif dom == "maniac":
            call_thresh = 0.42
            fold_thresh = 0.40

    # Steal detection: if opponent folds > 70% to raises, raise wide
    if hands >= 30:
        opp_fold_rate = profile.fold_to_raise_rate()
        if opp_fold_rate > 0.70:
            raise_thresh = 0.45
        elif opp_fold_rate > 0.50:
            raise_thresh = min(raise_thresh, 0.55)

    def _raise_size(base_frac: float) -> int:
        amt = max(int(pot * base_frac), min_raise)
        return max(min(amt, max_raise), min_raise)

    # Premium hand: raise for value
    if strength >= raise_thresh and valid[RAISE]:
        return (RAISE, _raise_size(0.75), 0, 0)

    if facing_raise:
        # Escalating fold threshold based on raise size
        if cost > 20:
            fold_thresh = max(fold_thresh, 0.65)
        elif cost > 5:
            fold_thresh = max(fold_thresh, 0.55)

        if strength < fold_thresh and valid[FOLD]:
            return (FOLD, 0, 0, 0)
        if strength >= call_thresh and valid[CALL]:
            return (CALL, 0, 0, 0)
        if valid[FOLD]:
            return (FOLD, 0, 0, 0)
        return (CALL, 0, 0, 0)

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
    is_tight = dom in ("nit", "rock", "tag") and hands_played >= 20

    def _raise_size(base_frac: float) -> int:
        amt = max(int(pot * base_frac), min_raise)
        return max(min(amt, max_raise), min_raise)

    committed_frac = my_bet / 100.0

    # --- Raise threshold: varies by opponent type ---
    raise_thresh = 0.78
    if hands_played >= 20:
        if dom in ("nit", "rock"):
            raise_thresh = 0.80
        elif dom == "calling_station":
            raise_thresh = 0.68
        elif dom == "maniac":
            raise_thresh = 0.72

    # --- Medium-strong value bet threshold ---
    # Against tight opponents, only bet with strong equity (they fold
    # weak and call strong, so thin value bets lose money)
    value_bet_thresh = 0.78 if is_tight else 0.70

    # --- Bet sizing ---
    if dom == "calling_station" and hands_played >= 20:
        raise_frac = 0.85
    elif dom in ("nit", "rock") and hands_played >= 20:
        raise_frac = 0.60
    else:
        raise_frac = 0.75

    # --------------- POT COMMITMENT GUARD ---------------
    # Don't commit heavy chips without strong equity
    if my_bet > 50 and equity < 0.70:
        if facing_raise:
            if valid[FOLD]:
                return (FOLD, 0, 0, 0)
        if valid[CHECK]:
            return (CHECK, 0, 0, 0)

    # --------------- FACING A RAISE ---------------
    if facing_raise:
        bet_to_pot = continue_cost / max(pot, 1)

        # Risk premium: raised from 0.04/0.06 to 0.08/0.10
        risk_premium = 0.08 + 0.10 * min(bet_to_pot, 2.0)
        call_threshold = pot_odds + risk_premium

        # Against big bets (>60% of remaining stack), tighten further
        if continue_cost > 60 - my_bet:
            call_threshold = max(call_threshold, 0.65)

        # Pot commitment: approaching all-in, need very strong hand
        if my_bet + continue_cost >= 90:
            call_threshold = max(call_threshold, 0.80)

        # If we already raised this street and they re-raised,
        # we need a very strong hand to continue
        we_raised_this_street = _did_we_raise_this_street(observation)
        if we_raised_this_street:
            call_threshold = max(call_threshold, 0.65)
            if equity > 0.85 and valid[RAISE]:
                return (RAISE, _raise_size(0.75), 0, 0)
            if equity >= call_threshold and valid[CALL]:
                return (CALL, 0, 0, 0)
            return (FOLD, 0, 0, 0)

        # Strong hand facing a raise: re-raise
        if equity > 0.85 and valid[RAISE]:
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

    # Medium-strong: value bet only if threshold met
    if equity > value_bet_thresh and valid[RAISE]:
        return (RAISE, _raise_size(0.60), 0, 0)

    if valid[CHECK]:
        return (CHECK, 0, 0, 0)
    if valid[CALL]:
        return (CALL, 0, 0, 0)
    return (FOLD, 0, 0, 0)


def _did_we_raise_this_street(observation: dict) -> bool:
    my_bet = observation["my_bet"]
    opp_bet = observation["opp_bet"]
    return my_bet > 2 and opp_bet > my_bet

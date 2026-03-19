from collections import deque

from agents.agent import Agent
from gym_env import PokerEnv

from submission.preflop_table import get_preflop_strength, get_preflop_detail
from submission.equity import compute_exact_equity, find_best_keep
from submission.opponent_model import (
    OpponentProfile, OpponentCategorizer, narrow_range_from_discards,
    FOLD, RAISE, CHECK, CALL, DISCARD,
)
from submission import strategy


class PlayerAgent(Agent):
    MAX_RAISES_PER_STREET = 3

    def __init__(self, stream: bool = True):
        super().__init__(stream)
        self.action_types = PokerEnv.ActionType

        self.opp_profile = OpponentProfile()
        self.opp_categorizer = OpponentCategorizer()

        self.hand_number = 0
        self._last_opp_discards: list[int] = []
        self._my_discards: list[int] = []
        self._prev_opp_bet = 0
        self._prev_my_bet = 0
        self._prev_street = -1

        self._bankroll = 0
        self._fold_lockdown = False
        self._total_hands = 1000

        # Raise cap per street (prevents timeout from raise wars)
        self._my_raises_this_street = 0
        self._current_street = -1

        # Opponent lockdown detection: rolling window of recent hand outcomes
        self._opp_recent_folds: deque[bool] = deque(maxlen=30)
        self._opp_in_lockdown = False

    def __name__(self):
        return "PlayerAgent"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _cards(self, raw) -> list[int]:
        """Filter -1 sentinels from a card tuple."""
        return [c for c in raw if c != -1]

    def _dead_cards(self, obs: dict) -> set[int]:
        """Collect all known dead cards (discards from both sides)."""
        dead = set()
        for c in obs.get("my_discarded_cards", []):
            if c != -1:
                dead.add(c)
        for c in obs.get("opp_discarded_cards", []):
            if c != -1:
                dead.add(c)
        return dead

    def _opp_discards(self, obs: dict) -> list[int]:
        return [c for c in obs.get("opp_discarded_cards", []) if c != -1]

    def _range_weights(self, obs: dict) -> dict | None:
        opp_disc = self._opp_discards(obs)
        if len(opp_disc) != 3:
            return None
        my_cards = self._cards(obs["my_cards"])
        community = self._cards(obs["community_cards"])
        dead = self._dead_cards(obs)
        known = set(my_cards) | set(community) | dead
        remaining = [c for c in range(27) if c not in known]
        if len(remaining) < 2:
            return None
        return narrow_range_from_discards(opp_disc, remaining)

    def _enforce_raise_cap(self, action: tuple, valid) -> tuple:
        """Hard-cap raises per street to prevent timeout from raise wars."""
        if action[0] == RAISE:
            if self._my_raises_this_street >= self.MAX_RAISES_PER_STREET:
                if valid[CALL]:
                    return (CALL, 0, 0, 0)
                if valid[CHECK]:
                    return (CHECK, 0, 0, 0)
                return (FOLD, 0, 0, 0)
            self._my_raises_this_street += 1
        return action

    # ------------------------------------------------------------------
    # act()  –  main decision pipeline
    # ------------------------------------------------------------------

    def act(self, observation, reward, terminated, truncated, info):
        valid = observation["valid_actions"]
        street = observation["street"]
        my_cards = self._cards(observation["my_cards"])
        community = self._cards(observation["community_cards"])

        # Track opponent action that happened before our turn
        self._track_opp_action(observation)

        # Reset raise counter when street advances
        if street != self._current_street:
            self._my_raises_this_street = 0
            self._current_street = street

        # --- Fold lockdown: re-check every action in case observe() missed it ---
        if not self._fold_lockdown:
            self._check_fold_lockdown(observation)

        if self._fold_lockdown:
            if valid[DISCARD]:
                return (DISCARD, 0, 0, 1)
            if valid[CHECK]:
                return (CHECK, 0, 0, 0)
            if valid[FOLD]:
                return (FOLD, 0, 0, 0)
            return (CALL, 0, 0, 0)

        # --- Pre-flop (street 0): O(1) lookup ---
        if street == 0 and not valid[DISCARD]:
            strength = get_preflop_strength(tuple(my_cards))
            detail = get_preflop_detail(tuple(my_cards))
            action = strategy.preflop_action(
                observation, strength, detail,
                self.opp_categorizer, self.opp_profile,
                opp_in_lockdown=self._opp_in_lockdown,
            )
            return self._enforce_raise_cap(action, valid)

        # --- Discard phase (street 1, mandatory) ---
        if valid[DISCARD]:
            # If opponent is in lockdown, skip expensive equity computation
            if self._opp_in_lockdown:
                return (DISCARD, 0, 0, 1)
            dead = self._dead_cards(observation)
            rw = self._range_weights(observation)
            i, j, eq = find_best_keep(my_cards, community, dead, rw)
            self._my_discards = [my_cards[k] for k in range(len(my_cards))
                                 if k != i and k != j]
            return (DISCARD, 0, i, j)

        # --- Post-discard betting (streets 1-3): exact equity ---
        # If opponent is in lockdown, just min-raise to pressure them
        if self._opp_in_lockdown:
            if valid[RAISE]:
                action = (RAISE, observation["min_raise"], 0, 0)
                return self._enforce_raise_cap(action, valid)
            if valid[CHECK]:
                return (CHECK, 0, 0, 0)
            if valid[CALL]:
                return (CALL, 0, 0, 0)
            return (FOLD, 0, 0, 0)

        dead = self._dead_cards(observation)
        rw = self._range_weights(observation)
        eq = compute_exact_equity(my_cards, community, dead, rw)
        action = strategy.choose_action(
            observation, eq,
            self.opp_categorizer, self.opp_profile,
        )
        return self._enforce_raise_cap(action, valid)

    # ------------------------------------------------------------------
    # observe()  –  learning pipeline
    # ------------------------------------------------------------------

    def observe(self, observation, reward, terminated, truncated, info):
        self._track_opp_action(observation)

        # Record opponent discards when they appear
        opp_disc = self._opp_discards(observation)
        if len(opp_disc) == 3 and tuple(opp_disc) not in set(
            self.opp_profile.discard_history
        ):
            self.opp_profile.record_discard(tuple(opp_disc))

        if terminated:
            self._bankroll += reward

            # Track whether opponent folded pre-flop this hand
            opp_last = observation.get("opp_last_action")
            opp_folded = (opp_last == "FOLD") or (reward > 0 and reward <= 2)
            self._opp_recent_folds.append(opp_folded)

            # Detect opponent lockdown: >90% fold rate over last 30 hands
            if len(self._opp_recent_folds) >= 20:
                fold_rate = sum(self._opp_recent_folds) / len(self._opp_recent_folds)
                self._opp_in_lockdown = fold_rate > 0.85
            else:
                self._opp_in_lockdown = False

            # Check fold lockdown: can we win by folding every remaining hand?
            if not self._fold_lockdown:
                self._check_fold_lockdown(observation)

            # Record showdown info if available
            if "player_0_cards" in info:
                blind_pos = observation.get("blind_position", 0)
                if blind_pos == 0:
                    opp_cards_key = "player_1_cards"
                else:
                    opp_cards_key = "player_0_cards"
                opp_cards = info.get(opp_cards_key, [])
                board = info.get("community_cards", [])
                self.opp_profile.record_showdown(
                    opp_cards, board, opp_disc, reward,
                )

            self.opp_profile.record_hand_end()
            self.hand_number += 1

            # Update Bayesian beliefs periodically
            if self.hand_number % 10 == 0:
                self.opp_categorizer.update_beliefs(self.opp_profile)

            # Reset per-hand state
            self._last_opp_discards = []
            self._my_discards = []
            self._prev_opp_bet = 0
            self._prev_my_bet = 0
            self._prev_street = -1
            self._my_raises_this_street = 0
            self._current_street = -1

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_fold_lockdown(self, observation):
        """Check if we can win the match by folding every hand from now on.

        Blinds alternate: SB costs 1, BB costs 2.  We compute the EXACT
        total cost of folding every remaining hand and lock down the
        instant our bankroll exceeds that cost by at least 1 chip.
        """
        hands_left = self._total_hands - (self.hand_number + 1)
        if hands_left <= 0 or self._bankroll <= 0:
            return

        # Determine what blind we'll be NEXT hand.
        # blind_position: 0 = we are SB this hand, 1 = we are BB this hand.
        # Positions swap every hand, so next hand is the opposite.
        cur_blind = observation.get("blind_position", 0)
        next_is_sb = (cur_blind == 1)

        # Walk through every remaining hand and sum the blind cost.
        cost = int(1.5 * hands_left) + 1

        # Lock down if we'd still be ahead by at least 1 chip after paying
        # all remaining blinds.  bankroll - cost >= 1  ⟺  bankroll > cost
        if self._bankroll > cost:
            self._fold_lockdown = True

    _ACTION_MAP = {
        "FOLD": FOLD, "RAISE": RAISE, "CHECK": CHECK,
        "CALL": CALL, "DISCARD": DISCARD,
    }

    def _track_opp_action(self, observation):
        """Record opponent's last action using the engine-provided field."""
        opp_last = observation.get("opp_last_action")
        if opp_last is None or opp_last == "None":
            self._prev_opp_bet = observation["opp_bet"]
            self._prev_my_bet = observation["my_bet"]
            self._prev_street = observation["street"]
            return

        action_type = self._ACTION_MAP.get(opp_last)
        if action_type is None or action_type == DISCARD:
            self._prev_opp_bet = observation["opp_bet"]
            self._prev_my_bet = observation["my_bet"]
            self._prev_street = observation["street"]
            return

        street = observation["street"]
        if street > 3:
            return

        opp_bet = observation["opp_bet"]
        my_bet = observation["my_bet"]
        pot = observation.get("pot_size", my_bet + opp_bet)
        was_facing = self._prev_my_bet > self._prev_opp_bet
        raise_amt = max(opp_bet - self._prev_opp_bet, 0)

        self.opp_profile.record_action(
            street=street, action_type=action_type,
            raise_amount=raise_amt, pot_size=pot,
            was_facing_raise=was_facing,
        )

        self._prev_opp_bet = opp_bet
        self._prev_my_bet = my_bet
        self._prev_street = street

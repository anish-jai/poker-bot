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

        # --- Pre-flop (street 0): O(1) lookup ---
        if street == 0 and not valid[DISCARD]:
            strength = get_preflop_strength(tuple(my_cards))
            detail = get_preflop_detail(tuple(my_cards))
            return strategy.preflop_action(
                observation, strength, detail,
                self.opp_categorizer, self.opp_profile,
            )

        # --- Discard phase (street 1, mandatory) ---
        if valid[DISCARD]:
            dead = self._dead_cards(observation)
            rw = self._range_weights(observation)
            i, j, eq = find_best_keep(my_cards, community, dead, rw)
            # Remember what we discarded for later dead-card tracking
            self._my_discards = [my_cards[k] for k in range(len(my_cards))
                                 if k != i and k != j]
            return (DISCARD, 0, i, j)

        # --- Post-discard betting (streets 1-3): exact equity ---
        dead = self._dead_cards(observation)
        rw = self._range_weights(observation)
        eq = compute_exact_equity(my_cards, community, dead, rw)
        return strategy.choose_action(
            observation, eq,
            self.opp_categorizer, self.opp_profile,
        )

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

    # ------------------------------------------------------------------
    # Internal opponent action tracking
    # ------------------------------------------------------------------

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

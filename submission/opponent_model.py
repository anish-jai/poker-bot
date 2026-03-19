"""
Opponent profiling, Bayesian categorization, and discard-based range narrowing.
"""

import math
import itertools

DECK_SIZE = 27
FOLD, RAISE, CHECK, CALL, DISCARD = 0, 1, 2, 3, 4


# ---------------------------------------------------------------------------
# OpponentProfile  –  raw stat accumulator
# ---------------------------------------------------------------------------

class OpponentProfile:
    def __init__(self):
        self.hands_played = 0
        self.vpip_count = 0
        self.pfr_count = 0

        # Per-street action counts: {street: {"fold":n, "raise":n, "call":n, "check":n}}
        self.action_counts = {s: {"fold": 0, "raise": 0, "call": 0, "check": 0}
                              for s in range(4)}
        # Fold-to-raise tracking: {street: [times_folded_to_raise, times_faced_raise]}
        self.fold_to_raise = {s: [0, 0] for s in range(4)}

        self.check_raise_count = 0
        self.check_raise_opps = 0
        self.bet_sizes: list[float] = []

        self.showdowns: list[dict] = []
        self.discard_history: list[tuple] = []

        # Per-hand scratch state (reset each hand)
        self._hand_opp_actions: list[tuple] = []
        self._hand_checked_streets: set[int] = set()

    # ---- per-action recording ----

    def record_action(self, street: int, action_type: int,
                      raise_amount: int = 0, pot_size: int = 0,
                      was_facing_raise: bool = False):
        if action_type == FOLD:
            self.action_counts[street]["fold"] += 1
            if was_facing_raise:
                self.fold_to_raise[street][0] += 1
        elif action_type == RAISE:
            self.action_counts[street]["raise"] += 1
            if pot_size > 0:
                self.bet_sizes.append(raise_amount / pot_size)
            if street in self._hand_checked_streets:
                self.check_raise_count += 1
        elif action_type == CALL:
            self.action_counts[street]["call"] += 1
        elif action_type == CHECK:
            self.action_counts[street]["check"] += 1
            self._hand_checked_streets.add(street)
            self.check_raise_opps += 1

        if was_facing_raise and action_type != FOLD:
            pass  # they didn't fold
        if was_facing_raise:
            self.fold_to_raise[street][1] += 1

        # Pre-flop specific
        if street == 0:
            if action_type in (RAISE, CALL):
                self.vpip_count += 1
            if action_type == RAISE:
                self.pfr_count += 1

        self._hand_opp_actions.append((street, action_type, raise_amount))

    def record_discard(self, discarded_cards: tuple):
        self.discard_history.append(discarded_cards)

    def record_showdown(self, opp_cards, board, opp_discards, reward):
        self.showdowns.append({
            "opp_cards": tuple(opp_cards),
            "board": tuple(board),
            "opp_discards": tuple(opp_discards),
            "opp_actions": list(self._hand_opp_actions),
            "reward": reward,
        })

    def record_hand_end(self):
        self.hands_played += 1
        self._hand_opp_actions = []
        self._hand_checked_streets = set()

    # ---- derived metrics ----

    def vpip(self) -> float:
        return self.vpip_count / max(self.hands_played, 1)

    def pfr(self) -> float:
        return self.pfr_count / max(self.hands_played, 1)

    def fold_to_raise_rate(self, street: int | None = None) -> float:
        if street is not None:
            faced = self.fold_to_raise[street][1]
            return self.fold_to_raise[street][0] / max(faced, 1)
        total_folded = sum(v[0] for v in self.fold_to_raise.values())
        total_faced = sum(v[1] for v in self.fold_to_raise.values())
        return total_folded / max(total_faced, 1)

    def aggression_factor(self, street: int | None = None) -> float:
        if street is not None:
            r = self.action_counts[street]["raise"]
            c = self.action_counts[street]["call"]
            return r / max(c, 1)
        total_r = sum(a["raise"] for a in self.action_counts.values())
        total_c = sum(a["call"] for a in self.action_counts.values())
        return total_r / max(total_c, 1)

    def check_raise_freq(self) -> float:
        return self.check_raise_count / max(self.check_raise_opps, 1)

    def avg_bet_size(self) -> float:
        return sum(self.bet_sizes) / max(len(self.bet_sizes), 1)

    def bluff_frequency(self) -> float:
        if not self.showdowns:
            return 0.15  # default assumption
        bluffs = 0
        aggressive_showdowns = 0
        for sd in self.showdowns:
            raised_this_hand = any(a[1] == RAISE for a in sd["opp_actions"])
            if raised_this_hand:
                aggressive_showdowns += 1
                if sd["reward"] > 0:
                    bluffs += 1
        return bluffs / max(aggressive_showdowns, 1)


# ---------------------------------------------------------------------------
# OpponentCategorizer  –  Bayesian archetype classification
# ---------------------------------------------------------------------------

# Each archetype: {metric_name: (mu, sigma)}
ARCHETYPE_PROFILES = {
    "nit": {
        "vpip": (0.15, 0.08), "pfr": (0.08, 0.06),
        "fold_to_raise": (0.70, 0.12), "aggression": (0.8, 0.5),
    },
    "rock": {
        "vpip": (0.28, 0.08), "pfr": (0.18, 0.08),
        "fold_to_raise": (0.55, 0.12), "aggression": (1.2, 0.5),
    },
    "tag": {
        "vpip": (0.35, 0.08), "pfr": (0.28, 0.08),
        "fold_to_raise": (0.40, 0.12), "aggression": (2.0, 0.7),
    },
    "calling_station": {
        "vpip": (0.75, 0.10), "pfr": (0.08, 0.06),
        "fold_to_raise": (0.12, 0.08), "aggression": (0.4, 0.3),
    },
    "maniac": {
        "vpip": (0.80, 0.10), "pfr": (0.55, 0.12),
        "fold_to_raise": (0.15, 0.08), "aggression": (3.5, 1.0),
    },
    "tricky": {
        "vpip": (0.45, 0.10), "pfr": (0.30, 0.10),
        "fold_to_raise": (0.30, 0.10), "aggression": (1.8, 0.7),
    },
}

# Expected fold-to-raise probability per street for each archetype
ARCHETYPE_FOLD_PROBS = {
    "nit":            {0: 0.65, 1: 0.60, 2: 0.55, 3: 0.50},
    "rock":           {0: 0.50, 1: 0.45, 2: 0.40, 3: 0.35},
    "tag":            {0: 0.35, 1: 0.30, 2: 0.28, 3: 0.25},
    "calling_station": {0: 0.10, 1: 0.08, 2: 0.07, 3: 0.05},
    "maniac":         {0: 0.12, 1: 0.10, 2: 0.10, 3: 0.08},
    "tricky":         {0: 0.28, 1: 0.25, 2: 0.22, 3: 0.20},
}

# Expected aggression (bet/raise probability when they have the lead)
ARCHETYPE_AGG_PROBS = {
    "nit":            {0: 0.10, 1: 0.15, 2: 0.15, 3: 0.12},
    "rock":           {0: 0.20, 1: 0.25, 2: 0.25, 3: 0.20},
    "tag":            {0: 0.35, 1: 0.40, 2: 0.38, 3: 0.35},
    "calling_station": {0: 0.08, 1: 0.10, 2: 0.10, 3: 0.08},
    "maniac":         {0: 0.55, 1: 0.60, 2: 0.55, 3: 0.50},
    "tricky":         {0: 0.30, 1: 0.35, 2: 0.35, 3: 0.30},
}

CATEGORIES = list(ARCHETYPE_PROFILES.keys())


def _gauss_logpdf(x: float, mu: float, sigma: float) -> float:
    return -0.5 * ((x - mu) / sigma) ** 2 - math.log(sigma)


class OpponentCategorizer:
    def __init__(self):
        # Prior: slightly favour TAG
        self.beliefs = {
            "nit": 0.15, "rock": 0.15, "tag": 0.25,
            "calling_station": 0.15, "maniac": 0.15, "tricky": 0.15,
        }

    def update_beliefs(self, profile: OpponentProfile):
        if profile.hands_played < 5:
            return

        # Hard override: unmistakable maniac detected by raw stats
        if profile.hands_played >= 20:
            pfr = profile.pfr()
            agg = profile.aggression_factor()
            if pfr > 0.50 and agg > 3.0:
                for cat in CATEGORIES:
                    self.beliefs[cat] = 0.02
                self.beliefs["maniac"] = 0.88
                return

        observed = {
            "vpip": profile.vpip(),
            "pfr": profile.pfr(),
            "fold_to_raise": profile.fold_to_raise_rate(),
            "aggression": profile.aggression_factor(),
        }
        log_posteriors = {}
        for cat in CATEGORIES:
            lp = math.log(max(self.beliefs[cat], 1e-12))
            for metric, (mu, sigma) in ARCHETYPE_PROFILES[cat].items():
                lp += _gauss_logpdf(observed[metric], mu, sigma)
            log_posteriors[cat] = lp

        # Normalize in log space
        max_lp = max(log_posteriors.values())
        total = 0.0
        for cat in CATEGORIES:
            log_posteriors[cat] -= max_lp
            log_posteriors[cat] = math.exp(log_posteriors[cat])
            total += log_posteriors[cat]

        # Laplace smoothing: blend 98% posterior, 2% uniform
        for cat in CATEGORIES:
            self.beliefs[cat] = 0.98 * (log_posteriors[cat] / total) + \
                                0.02 * (1.0 / len(CATEGORIES))

    def dominant_category(self) -> str:
        return max(self.beliefs, key=self.beliefs.get)

    def get_weighted_fold_prob(self, street: int,
                               profile: OpponentProfile | None = None) -> float:
        """Expected fold-to-raise probability on this street, weighted by beliefs."""
        prior_est = sum(
            self.beliefs[cat] * ARCHETYPE_FOLD_PROBS[cat][street]
            for cat in CATEGORIES
        )
        if profile is None or profile.hands_played < 15:
            return prior_est
        obs = profile.fold_to_raise_rate(street)
        blend = min(profile.hands_played / 100, 0.7)
        return blend * obs + (1 - blend) * prior_est

    def get_weighted_agg_prob(self, street: int,
                              profile: OpponentProfile | None = None) -> float:
        """Expected aggression probability on this street, weighted by beliefs."""
        prior_est = sum(
            self.beliefs[cat] * ARCHETYPE_AGG_PROBS[cat][street]
            for cat in CATEGORIES
        )
        if profile is None or profile.hands_played < 15:
            return prior_est
        obs_agg = profile.aggression_factor(street)
        obs_normed = min(obs_agg / (obs_agg + 1), 0.9)
        blend = min(profile.hands_played / 100, 0.7)
        return blend * obs_normed + (1 - blend) * prior_est


# ---------------------------------------------------------------------------
# Range narrowing from revealed discards
# ---------------------------------------------------------------------------

def narrow_range_from_discards(
    opp_discards: list[int],
    remaining_cards: list[int],
) -> dict[tuple[int, int], float]:
    """
    Given opponent's 3 revealed discards, return a weight dict for all possible
    2-card opponent holdings from remaining_cards.

    Heuristics:
     - If opponent discarded all of one suit -> they are unlikely to hold that suit
     - If they discarded high cards -> upweight paired/connected holdings
     - If they discarded low cards -> upweight high-card holdings
    """
    discards = [c for c in opp_discards if c != -1]
    if len(discards) != 3:
        return {combo: 1.0 for combo in itertools.combinations(remaining_cards, 2)}

    disc_suits = [c // 9 for c in discards]
    disc_ranks = [c % 9 for c in discards]
    disc_suit_counts = {}
    for s in disc_suits:
        disc_suit_counts[s] = disc_suit_counts.get(s, 0) + 1

    # Suit they completely discarded (if any)
    flushed_out_suits = {s for s, cnt in disc_suit_counts.items() if cnt >= 2}

    avg_disc_rank = sum(disc_ranks) / 3
    high_card_discard = avg_disc_rank >= 6  # discarded 8s, 9s, Aces
    low_card_discard = avg_disc_rank <= 3   # discarded 2s, 3s, 4s, 5s

    weights: dict[tuple[int, int], float] = {}
    for combo in itertools.combinations(remaining_cards, 2):
        c1, c2 = combo
        w = 1.0

        s1, s2 = c1 // 9, c2 // 9
        r1, r2 = c1 % 9, c2 % 9

        # Downweight if they hold the suit they discarded heavily
        if s1 in flushed_out_suits:
            w *= 0.4
        if s2 in flushed_out_suits:
            w *= 0.4

        # If they threw away high cards, they probably kept something strong
        if high_card_discard:
            if r1 == r2:
                w *= 1.8
            if abs(r1 - r2) <= 1:
                w *= 1.3
        elif low_card_discard:
            avg_keep = (r1 + r2) / 2
            if avg_keep >= 5:
                w *= 1.5
            if r1 == 8 or r2 == 8:  # Aces (rank index 8)
                w *= 1.3

        # Suited cards in a non-discarded suit -> flush draw likely kept
        if s1 == s2 and s1 not in flushed_out_suits:
            w *= 1.2

        weights[combo] = w

    return weights

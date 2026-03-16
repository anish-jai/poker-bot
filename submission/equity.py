"""
Exact equity calculator and discard optimizer for the 27-card poker variant.

All evaluations use precomputed treys card arrays to avoid per-call string
parsing overhead.  The evaluate_hand() hot path runs ~1-2us per call.
"""

import itertools
import random
from treys import Card, Evaluator

RANKS = "23456789A"
SUITS = "dhs"
DECK_SIZE = 27
KEEP_COMBOS = list(itertools.combinations(range(5), 2))

_evaluator = Evaluator()

# Precompute treys card ints for the full 27-card deck.
# ALT maps Ace -> Ten so the evaluator handles 6-7-8-9-A straights.
TREYS: list[int] = []
ALT_TREYS: list[int] = []
for _i in range(DECK_SIZE):
    _s = RANKS[_i % 9] + SUITS[_i // 9]
    TREYS.append(Card.new(_s))
    ALT_TREYS.append(Card.new(_s.replace("A", "T")))


def evaluate_hand(hand_idx: list[int], board_idx: list[int]) -> int:
    """Evaluate a hand given card indices (0-26).  Lower rank = stronger."""
    h = [TREYS[c] for c in hand_idx]
    b = [TREYS[c] for c in board_idx]
    h_alt = [ALT_TREYS[c] for c in hand_idx]
    b_alt = [ALT_TREYS[c] for c in board_idx]
    return min(_evaluator.evaluate(h, b), _evaluator.evaluate(h_alt, b_alt))


def _eval_treys(hand_t, board_t, hand_a, board_a) -> int:
    """Hot-path: evaluate already-converted treys cards."""
    return min(_evaluator.evaluate(hand_t, board_t),
               _evaluator.evaluate(hand_a, board_a))


def compute_exact_equity(
    my_cards: list[int],
    community_cards: list[int],
    dead_cards: set[int],
    range_weights: dict | None = None,
) -> float:
    """
    Exact win probability by enumerating all opponent hands and board runouts.

    my_cards:        2 ints (our hole cards after discard)
    community_cards: 0-5 ints (visible board cards, no -1 sentinels)
    dead_cards:      set of ints (discards, etc.)
    range_weights:   optional {(c1,c2): weight} for opponent hands.
                     If None, uniform over all legal 2-card combos.
    Returns float in [0, 1].
    """
    known = set(my_cards) | set(community_cards) | dead_cards
    remaining = [c for c in range(DECK_SIZE) if c not in known]
    board_needed = 5 - len(community_cards)

    my_t = [TREYS[c] for c in my_cards]
    my_a = [ALT_TREYS[c] for c in my_cards]
    comm_t = [TREYS[c] for c in community_cards]
    comm_a = [ALT_TREYS[c] for c in community_cards]

    total_weight = 0.0
    win_weight = 0.0
    tie_weight = 0.0

    for opp in itertools.combinations(remaining, 2):
        opp_key = opp
        w = range_weights.get(opp_key, 0.0) if range_weights is not None else 1.0
        if w <= 0:
            continue

        opp_t = [TREYS[opp[0]], TREYS[opp[1]]]
        opp_a = [ALT_TREYS[opp[0]], ALT_TREYS[opp[1]]]

        if board_needed == 0:
            my_rank = _eval_treys(my_t, comm_t, my_a, comm_a)
            opp_rank = _eval_treys(opp_t, comm_t, opp_a, comm_a)
            if my_rank < opp_rank:
                win_weight += w
            elif my_rank == opp_rank:
                tie_weight += w
            total_weight += w
        elif board_needed == 1:
            # Turn: enumerate all single remaining cards (~14)
            for r in remaining:
                if r == opp[0] or r == opp[1]:
                    continue
                full_t = comm_t + [TREYS[r]]
                full_a = comm_a + [ALT_TREYS[r]]
                my_rank = _eval_treys(my_t, full_t, my_a, full_a)
                opp_rank = _eval_treys(opp_t, full_t, opp_a, full_a)
                if my_rank < opp_rank:
                    win_weight += w
                elif my_rank == opp_rank:
                    tie_weight += w
                total_weight += w
        else:
            # Flop (2 cards needed): exact enumeration (~287ms total)
            board_pool = [c for c in remaining if c not in opp]
            for runout in itertools.combinations(board_pool, board_needed):
                full_t = comm_t + [TREYS[r] for r in runout]
                full_a = comm_a + [ALT_TREYS[r] for r in runout]
                my_rank = _eval_treys(my_t, full_t, my_a, full_a)
                opp_rank = _eval_treys(opp_t, full_t, opp_a, full_a)
                if my_rank < opp_rank:
                    win_weight += w
                elif my_rank == opp_rank:
                    tie_weight += w
                total_weight += w

    if total_weight == 0:
        return 0.5
    return (win_weight + 0.5 * tie_weight) / total_weight


def find_best_keep(
    my_cards_5: list[int],
    community_cards_3: list[int],
    dead_cards: set[int],
    range_weights: dict | None = None,
) -> tuple[int, int, float]:
    """
    Find the best 2 cards to keep from 5 hole cards.

    Uses exact flop-only equity against all opponent hands (~3ms).
    In this 27-card variant, flop-only evaluation is a strong proxy
    because the small deck means current hand strength correlates
    highly with final equity.

    Returns (idx_i, idx_j, best_equity) where indices are into my_cards_5.
    """
    comm_t = [TREYS[c] for c in community_cards_3]
    comm_a = [ALT_TREYS[c] for c in community_cards_3]

    best_i, best_j, best_eq = 0, 1, -1.0

    for i, j in KEEP_COMBOS:
        keep = [my_cards_5[i], my_cards_5[j]]
        discarded = {my_cards_5[k] for k in range(5) if k != i and k != j}
        all_dead = set(keep) | set(community_cards_3) | dead_cards | discarded
        remaining = [c for c in range(DECK_SIZE) if c not in all_dead]

        keep_t = [TREYS[keep[0]], TREYS[keep[1]]]
        keep_a = [ALT_TREYS[keep[0]], ALT_TREYS[keep[1]]]

        wins = 0.0
        ties = 0.0
        total = 0.0

        for opp in itertools.combinations(remaining, 2):
            w = range_weights.get(opp, 1.0) if range_weights is not None else 1.0
            if w <= 0:
                continue
            opp_t = [TREYS[opp[0]], TREYS[opp[1]]]
            opp_a = [ALT_TREYS[opp[0]], ALT_TREYS[opp[1]]]
            my_rank = _eval_treys(keep_t, comm_t, keep_a, comm_a)
            opp_rank = _eval_treys(opp_t, comm_t, opp_a, comm_a)
            if my_rank < opp_rank:
                wins += w
            elif my_rank == opp_rank:
                ties += w
            total += w

        eq = (wins + 0.5 * ties) / total if total > 0 else 0.0
        if eq > best_eq:
            best_eq = eq
            best_i, best_j = i, j

    return best_i, best_j, best_eq

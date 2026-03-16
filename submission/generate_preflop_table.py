#!/usr/bin/env python3
"""
Offline script: precompute a pre-flop hand-strength lookup table.

Run from the project root:
    python submission/generate_preflop_table.py

Produces submission/data/preflop_table.pkl  (~14K entries, ~200-400 KB).
"""

import sys
import os
import pickle
import itertools
import multiprocessing
import time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_SCRIPT_DIR, "..")
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _SCRIPT_DIR)

from tqdm import tqdm
from treys import Card, Evaluator
from preflop_table import canonicalize

RANKS = "23456789A"
SUITS = "dhs"
DECK_SIZE = 27
KEEP_COMBOS = list(itertools.combinations(range(5), 2))


def _int_card_to_str(card_int: int) -> str:
    return RANKS[card_int % 9] + SUITS[card_int // 9]


# Precompute treys representations for the whole 27-card deck.
# ALT converts Ace -> Ten to handle the Ace-high straight (6-7-8-9-A).
TREYS_CARDS: list[int] = []
ALT_TREYS_CARDS: list[int] = []
for _i in range(DECK_SIZE):
    _s = _int_card_to_str(_i)
    TREYS_CARDS.append(Card.new(_s))
    ALT_TREYS_CARDS.append(Card.new(_s.replace("A", "T")))


def _evaluate_class(args: tuple) -> tuple:
    """Worker: compute best-keep rank distribution for one canonical hand class."""
    canonical_key, hand = args
    evaluator = Evaluator()

    remaining = [c for c in range(DECK_SIZE) if c not in hand]

    best_ranks = []

    for flop in itertools.combinations(remaining, 3):
        flop_treys = [TREYS_CARDS[f] for f in flop]
        flop_alt = [ALT_TREYS_CARDS[f] for f in flop]
        best_rank = 999_999

        for i, j in KEEP_COMBOS:
            keep = [TREYS_CARDS[hand[i]], TREYS_CARDS[hand[j]]]
            keep_alt = [ALT_TREYS_CARDS[hand[i]], ALT_TREYS_CARDS[hand[j]]]
            reg = evaluator.evaluate(keep, flop_treys)
            alt = evaluator.evaluate(keep_alt, flop_alt)
            rank = min(reg, alt)
            if rank < best_rank:
                best_rank = rank

        best_ranks.append(best_rank)

    best_ranks.sort()
    n = len(best_ranks)
    stats = {
        "mean": sum(best_ranks) / n,
        "p25": best_ranks[n // 4],
        "p50": best_ranks[n // 2],
        "p75": best_ranks[3 * n // 4],
    }
    return canonical_key, stats


def main():
    print("=== Pre-flop Lookup Table Generator ===\n")

    # 1. Canonicalize all C(27,5) = 80,730 hands into isomorphic classes
    print("Step 1: Generating and canonicalizing all C(27,5) = 80,730 hands...")
    t0 = time.time()
    classes: dict[tuple, tuple] = {}
    for hand in itertools.combinations(range(DECK_SIZE), 5):
        key = canonicalize(hand)
        if key not in classes:
            classes[key] = hand
    num_classes = len(classes)
    print(f"  {num_classes} isomorphic classes  ({time.time() - t0:.1f}s)")

    # 2. Evaluate each class in parallel
    work_items = list(classes.items())
    num_workers = max(1, multiprocessing.cpu_count() - 1)
    total_evals = num_classes * 1540 * 10
    print(f"\nStep 2: {num_classes} classes x 1,540 flops x 10 keeps = {total_evals:,} evals")
    print(f"  Workers: {num_workers}")

    t1 = time.time()
    table: dict[tuple, dict] = {}

    with multiprocessing.Pool(num_workers) as pool:
        results = pool.imap_unordered(_evaluate_class, work_items, chunksize=64)
        for key, stats in tqdm(results, total=num_classes, desc="  Computing", unit=" cls"):
            table[key] = stats

    elapsed = time.time() - t1
    print(f"  Done in {elapsed:.1f}s  ({total_evals / elapsed:,.0f} evals/s)")

    # 3. Save
    data_dir = os.path.join(_SCRIPT_DIR, "data")
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, "preflop_table.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(table, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_kb = os.path.getsize(out_path) / 1024
    print(f"\nStep 3: Saved {out_path}  ({len(table):,} entries, {size_kb:.1f} KB)")

    # 4. Sanity check
    means = [s["mean"] for s in table.values()]
    means.sort()
    print(f"\nSanity check (by mean):")
    print(f"  Min avg rank : {means[0]:.1f}   (best)")
    print(f"  Max avg rank : {means[-1]:.1f}  (worst)")
    print(f"  Mean avg rank: {sum(means) / len(means):.1f}")

    sorted_items = sorted(table.items(), key=lambda x: x[1]["mean"])
    print(f"\n  Top 5 strongest starting hands:")
    for key, s in sorted_items[:5]:
        cards = [_int_card_to_str(c) for c in key]
        print(f"    {cards}  mean={s['mean']:.1f}  p25={s['p25']}  p50={s['p50']}  p75={s['p75']}")
    print(f"\n  Top 5 weakest starting hands:")
    for key, s in sorted_items[-5:]:
        cards = [_int_card_to_str(c) for c in key]
        print(f"    {cards}  mean={s['mean']:.1f}  p25={s['p25']}  p50={s['p50']}  p75={s['p75']}")

    # Show a high-variance vs low-variance comparison
    by_spread = sorted(table.items(), key=lambda x: x[1]["p75"] - x[1]["p25"])
    print(f"\n  Tightest spread (most consistent):")
    for key, s in by_spread[:3]:
        cards = [_int_card_to_str(c) for c in key]
        spread = s["p75"] - s["p25"]
        print(f"    {cards}  mean={s['mean']:.1f}  p25={s['p25']}  p75={s['p75']}  spread={spread}")
    print(f"\n  Widest spread (most volatile):")
    for key, s in by_spread[-3:]:
        cards = [_int_card_to_str(c) for c in key]
        spread = s["p75"] - s["p25"]
        print(f"    {cards}  mean={s['mean']:.1f}  p25={s['p25']}  p75={s['p75']}  spread={spread}")


if __name__ == "__main__":
    main()

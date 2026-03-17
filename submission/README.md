# Poker Bot — CMU AI Poker Tournament 2026

## Overview

This bot plays a **non-standard heads-up poker variant** with a 27-card deck (ranks 2–9 + Ace, suits diamonds/hearts/spades). Each player receives 5 hole cards, must discard 3 after the flop (discards are revealed to the opponent), and plays the remaining 2 through turn and river betting streets. Matches are 1000 hands with a cumulative time limit.

The bot combines **exact equity computation**, a **precomputed hand-strength lookup table**, **Bayesian opponent modeling**, and an **adaptive betting strategy** to make decisions across all four phases of play.

## Architecture

```
submission/
├── player.py                # Entry point — orchestrates the full decision pipeline
├── equity.py                # Exact equity calculator + discard optimizer
├── opponent_model.py        # Opponent profiling + Bayesian archetype classification
├── strategy.py              # Betting logic (pre-flop + post-flop)
├── preflop_table.py         # Runtime loader for the precomputed lookup table
├── generate_preflop_table.py  # Offline script that builds the lookup table
└── data/
    └── preflop_table.pkl    # ~14K entries mapping canonical 5-card hands to strength stats
```

### Decision Pipeline (`player.py → act()`)

```
Observation
    │
    ├── Street 0 (pre-flop)
    │       preflop_table.get_preflop_strength()  →  strategy.preflop_action()
    │       O(1) lookup, ~0.1ms
    │
    ├── Street 1, discard phase
    │       equity.find_best_keep()               →  (DISCARD, 0, i, j)
    │       Exact enumeration of all C(5,2)=10 keeps × all opponent hands, ~3ms
    │
    └── Streets 1–3, betting
            equity.compute_exact_equity()          →  strategy.choose_action()
            Exact enumeration of all opponent hands + board completions
            River: ~1ms  |  Turn: ~5ms  |  Flop: ~287ms
```

### Learning Pipeline (`player.py → observe()`)

After every opponent action, the bot records it in `OpponentProfile`. Every 10 hands, `OpponentCategorizer` runs a Bayesian update over 6 archetypes (Nit, Rock, TAG, Calling Station, Maniac, Tricky), shifting the belief distribution based on observed VPIP, PFR, fold-to-raise rates, and aggression. These beliefs feed back into the strategy layer to adjust raise thresholds, bet sizing, and calling ranges.

## Module Details

### `equity.py` — Exact Equity Engine

The core math module. All evaluations use precomputed `treys` card arrays (one normal, one with Ace→Ten for handling 6-7-8-9-A straights) to avoid per-call string parsing.

- **`evaluate_hand(hand, board)`** — Single hand evaluation, ~1–2μs.
- **`compute_exact_equity(my_cards, community, dead, range_weights)`** — Enumerates all legal opponent 2-card hands from the remaining deck and all board completions needed. No sampling at any street; the small deck (27 cards) makes full enumeration tractable.
- **`find_best_keep(my_5_cards, community_3, dead, range_weights)`** — Evaluates all 10 possible 2-card keeps against all opponent hands using the current flop. Selects the keep with the highest win rate.

### `opponent_model.py` — Profiling & Bayesian Categorization

- **`OpponentProfile`** — Accumulates raw counts: VPIP, PFR, actions per street, fold-to-raise rates, check-raise frequency, bet sizing patterns, showdown history, and discard patterns.
- **`OpponentCategorizer`** — Maintains a probability distribution over 6 archetypes. Each archetype has a Gaussian signature for key metrics (e.g. TAG: VPIP~0.35, PFR~0.28, aggression~2.0). Bayesian updates multiply the prior by the Gaussian likelihood of observed metrics, with Laplace smoothing to prevent collapse.
- **`narrow_range_from_discards(opp_discards, remaining)`** — Uses the 3 revealed opponent discards to adjust weights on possible opponent holdings. Heuristics: downweight suits that were discarded heavily, upweight paired/connected hands if high cards were discarded, upweight high cards if low cards were discarded.

### `strategy.py` — Betting Logic

**Pre-flop (`preflop_action`):**
Uses the O(1) preflop table lookup. Thresholds are calibrated to the actual strength distribution (p10=0.49, p50=0.60, p90=0.72). Raises with top ~10–15%, calls most hands, folds trash when facing a raise. Thresholds shift after 30+ hands based on opponent archetype (e.g. steals more vs Nits, tightens vs Calling Stations).

**Post-flop (`choose_action`):**
Foundation is pot-odds play with layered adaptations:
1. **Raise** with equity above a threshold (default 0.75, varies by opponent type)
2. **Call** facing a raise when equity meets pot odds
3. **Value bet** medium-strong hands (equity > 0.65) when first to act
4. **Check/fold** otherwise

Bet sizing adapts: larger vs Calling Stations (they call anyway), smaller vs Nits/Rocks (to get called).

### `preflop_table.py` + `generate_preflop_table.py`

The generator (run once offline, ~45s) enumerates all C(27,5) = 80,730 starting hands, canonicalizes them under the S₃ suit permutation group into ~14,058 isomorphic classes, and for each class evaluates all C(22,3) = 1,540 possible flops × 10 keeps. It stores per-class statistics: mean, 25th, 50th, and 75th percentile of the best-keep rank distribution.

At runtime, `preflop_table.py` loads the pickle, normalizes per-field (mean uses min/max of means, p25 uses min/max of p25s, etc.) to [0, 1], and returns O(1) strength lookups.

## Time Budget

| Phase | Time | Method |
|---|---|---|
| Pre-flop | ~0.1ms | O(1) table lookup |
| Discard | ~3ms | 10 keeps × exact flop equity |
| Flop betting | ~287ms | 120 opponents × 91 runouts (exact) |
| Turn betting | ~5ms | 120 opponents × 14 runouts |
| River betting | ~1ms | 120 opponents, no runouts |
| Opponent model | ~0.1ms | Arithmetic |
| **Total per hand** | **~300ms avg** | **Well under 1.5s limit** |

## Running

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install tqdm treys gym numpy uvicorn fastapi pydantic requests

# Generate the preflop table (only needed once, ~45s)
python submission/generate_preflop_table.py

# Run the test suite (vs FoldAgent, CallingStation, AllIn, Random)
python agent_test.py

# Run a match vs ProbabilityAgent
python run.py
```

## Performance

Tested against the provided agents over 5-hand matches (test suite) and 200-hand matches (ProbabilityAgent):

| Opponent | Result |
|---|---|
| FoldAgent | Won every hand |
| CallingStationAgent | +135 in 5 hands |
| AllInAgent | +435 in 5 hands |
| RandomAgent | +579 in 5 hands |
| ProbabilityAgent | +468 avg over 3×200-hand trials (+2.34/hand) |

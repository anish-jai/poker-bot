<!-- # CMU AI Poker Tournament — Bot Development Context -->

## Competition Overview
- **Deadline:** March 21, 2026, 11:59 PM EDT
- **Format:** 1000-hand heads-up matches, 1 second per hand compute limit
- **Submission:** Python class `PlayerAgent` in `submission/player.py`, max 1GB
- **Engine repo:** `github.com/cmu-dsc/poker-engine-2026`

## Game Variant (NOT Standard Hold'em)
- **27-card deck:** Ranks 2–9 and Ace only, suits ♦♥♠ only (no face cards, no clubs)
- **Card encoding:** `card_int = suit_index * 9 + rank_index` (0–26), where RANKS="23456789A", SUITS="dhs"
- **5-card deal + mandatory discard:** Each player gets 5 hole cards, must discard 3 and keep 2 after the flop
- **Discards are revealed** to the opponent — major info source
- **Max bet:** 100 chips per hand, small blind 1, big blind 2

## Street Flow
1. **Pre-flop (street 0):** 5 cards dealt to each player → blinds → betting
2. **Flop (street 1):** 3 community cards → DISCARD ROUND (each player keeps 2, discards 3, revealed) → betting
3. **Turn (street 2):** 4th community card → betting
4. **River (street 3):** 5th community card → betting → showdown

## Action Space
```python
(action_type: int, raise_amount: int, keep_card_1: int, keep_card_2: int)
# action_type: FOLD=0, RAISE=1, CHECK=2, CALL=3, DISCARD=4, INVALID=5
# keep_card_1/2: indices 0–4 into your 5 hole cards (only used for DISCARD)
# CRITICAL: invalid actions are treated as FOLDS — always check valid_actions
```

## Observation Dict
```python
{
    "street": int,                    # 0-3
    "acting_agent": int,              # 0 or 1
    "my_cards": Tuple[int, ...],      # 5 slots, -1 for empty
    "community_cards": Tuple[int, ...], # 5 slots, -1 for undealt
    "my_bet": int,
    "opp_bet": int,
    "my_discarded_cards": Tuple[int, ...],   # 3 slots
    "opp_discarded_cards": Tuple[int, ...],  # 3 slots, visible after discard
    "min_raise": int,
    "max_raise": int,
    "valid_actions": List[bool]       # indexed by ActionType.value
}
```

## Hand Evaluation
```python
from gym import WrappedEval
evaluator = WrappedEval()
rank = evaluator.evaluate(player_cards, board_cards)  # lower = better
```

## Allowed Libraries
numpy, pandas, scipy, scikit-learn, torch, tensorflow, keras, stable-baselines3, gymnasium, treys, tqdm, joblib, pickle, standard library. **NO external API calls, NO network access, NO subprocess.**

## Compute Analysis — Enumeration is Feasible (No Monte Carlo Needed)
The 27-card deck makes exhaustive enumeration practical:

| Stage | Known Cards | Unknown | Opponent Holdings | Scenarios | Time Est |
|-------|------------|---------|-------------------|-----------|----------|
| Pre-flop | 5 (your hand) | 22 | C(22,5) = 26,334 | Too many with lookahead | Use heuristic |
| Discard decision | 8 (your 5 + 3 community) | 19 | Depends on timing | ~109K evals for all 10 keeps | 0.2–1.0s (tight) |
| Post-discard betting | 11 (2+3+3disc+3disc) | 16 | C(16,2) = 120 | 120 × ~14 remaining = 1,680 | <10ms |
| River | All community known | 16 | C(16,2) = 120 | 120 | <5ms |

## Recommended Architecture

### 1. Pre-flop Heuristic (street 0) — NO enumeration
Fast scoring function: count pairs, suited cards, high cards (aces, 8s, 9s), straight connectivity.
- Raise with strong pairs, suited aces
- Call/check with decent hands
- Fold garbage
- Target: <1ms

### 2. Discard Optimizer (street 1, discard phase) — BIGGEST COMPUTE INVESTMENT
Evaluate all C(5,2) = 10 possible keep combos:
- For each keep combo, estimate equity against current board (skip turn/river lookahead to stay fast)
- Pick the keep combo that maximizes equity
- Also consider info leakage: what do your discards reveal?
- Target: <0.3s

### 3. Post-Discard Equity Calculator (streets 1–3) — EXACT ENUMERATION
Full enumeration is cheap:
- Enumerate all C(16,2) = 120 possible opponent hands
- For turn: multiply by remaining community cards
- Compute exact win probability
- Target: <50ms

### 4. Opponent Modeling (across hands)
Track opponent action frequencies by street and situation:
- Fold/call/raise rates per street
- Aggression after discard (what discards correlate with aggression?)
- Adjust thresholds: bluff more vs folders, value-bet more vs callers
- Reference: DBBR algorithm (Ganzfried & Sandholm, AAMAS 2011) — observe action frequencies, blend with a baseline strategy using Dirichlet prior, compute best response

### 5. Revealed Discard Exploitation
When opponent discards 3 cards:
- Narrow their hand range (what suits/ranks did they throw away?)
- If they discarded all of one suit → not pursuing that flush
- If they kept cards that don't connect with the board → likely have a made hand (pair+)
- Use this to refine the 120 possible opponent hands to a much smaller weighted set

## Key Strategic Notes
- **Flushes are easier** with only 3 suits (9 cards per suit) — flush draws are common
- **Straights are harder** — only ranks 2-9,A means gaps in sequence
- **Pairs/trips more common** — fewer ranks concentrates card distribution
- **Discard info is the unique edge** — no other poker variant gives you this
- **Position matters** — acting second gives info advantage

## Bot Class Template
```python
from agents.agent import Agent
from gym_env import PokerEnv

class PlayerAgent(Agent):
    def __init__(self, stream: bool = True):
        super().__init__(stream)
        self.action_types = PokerEnv.ActionType
        # Initialize evaluator, precompute tables, load models here

    def __name__(self):
        return "PlayerAgent"

    def act(self, observation, reward, terminated, truncated, info):
        valid_actions = observation["valid_actions"]
        # Your logic here
        # ALWAYS check valid_actions before returning
        if valid_actions[self.action_types.DISCARD.value]:
            return self.action_types.DISCARD.value, 0, best_keep_1, best_keep_2
        # ... betting logic ...
        return action_type, raise_amount, 0, 0
```

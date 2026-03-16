import os
import pickle

SUIT_PERMUTATIONS = [
    (0, 1, 2), (0, 2, 1), (1, 0, 2),
    (1, 2, 0), (2, 0, 1), (2, 1, 0),
]


def canonicalize(hand: tuple[int, ...]) -> tuple[int, ...]:
    """
    Return the lexicographically smallest sorted tuple under all 6 suit
    permutations (S_3).  Card encoding: card_int = suit_index * 9 + rank_index.
    """
    best = None
    for perm in SUIT_PERMUTATIONS:
        remapped = tuple(sorted(perm[c // 9] * 9 + (c % 9) for c in hand))
        if best is None or remapped < best:
            best = remapped
    return best


_TABLE: dict[tuple[int, ...], dict] | None = None
_BOUNDS: dict[str, tuple[float, float]] | None = None


def _load_table():
    global _TABLE, _BOUNDS
    if _TABLE is not None:
        return
    path = os.path.join(os.path.dirname(__file__), "data", "preflop_table.pkl")
    with open(path, "rb") as f:
        _TABLE = pickle.load(f)
    _BOUNDS = {}
    for key in ("mean", "p25", "p50", "p75"):
        vals = [s[key] for s in _TABLE.values()]
        _BOUNDS[key] = (min(vals), max(vals))


def _normalize(raw: float, field: str = "mean") -> float:
    """Convert a raw treys rank (lower=better) to a 0-1 strength (higher=better)."""
    lo, hi = _BOUNDS[field]
    if hi == lo:
        return 0.5
    return 1.0 - (raw - lo) / (hi - lo)


def get_preflop_strength(hand: tuple[int, ...] | list[int]) -> float:
    """
    Return a strength score in [0, 1] for a 5-card starting hand.
    1.0 = best possible hand, 0.0 = worst.
    Uses the mean best-keep rank across all flops.
    """
    _load_table()
    key = canonicalize(tuple(hand))
    entry = _TABLE.get(key)
    if entry is None:
        return 0.5
    return _normalize(entry["mean"], "mean")


def get_preflop_detail(hand: tuple[int, ...] | list[int]) -> dict:
    """
    Return detailed pre-flop stats for a 5-card starting hand.
    All strength values are normalized to [0, 1] (higher = better).
    Keys: mean, p25, p50, p75, spread.
    spread = p25_strength - p75_strength (tight spread = consistent hand).
    """
    _load_table()
    key = canonicalize(tuple(hand))
    entry = _TABLE.get(key)
    if entry is None:
        return {"mean": 0.5, "p25": 0.5, "p50": 0.5, "p75": 0.5, "spread": 0.0}
    p25_str = _normalize(entry["p25"], "p25")
    p75_str = _normalize(entry["p75"], "p75")
    return {
        "mean": _normalize(entry["mean"], "mean"),
        "p25": p25_str,
        "p50": _normalize(entry["p50"], "p50"),
        "p75": p75_str,
        "spread": p25_str - p75_str,
    }

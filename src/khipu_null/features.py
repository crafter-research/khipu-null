"""Pattern features. Each feature scores one khipu between 0 and 1, or returns None when it does
not apply (for example, no group is large enough).

A khipu is a `Khipu`: arrays of its cords at the pattern's level, in position order, plus the
index arrays of its groups."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class Khipu:
    id: str
    values: np.ndarray            # float, NaN when the cord has no knots
    colors: np.ndarray            # object (str)
    attachment: np.ndarray | None
    groups: tuple[np.ndarray, ...]  # cord indices of each group, in position order
    slots: np.ndarray              # index of each cord inside its group (0 = first)

    def with_(self, **kw) -> "Khipu":
        return replace(self, **kw)


Feature = Callable[[Khipu, dict], float | None]


def _runs(idx: np.ndarray, size: int):
    for i in range(len(idx) // size):
        yield idx[i * size:(i + 1) * size]


def group_size(k: Khipu, spec: dict) -> float | None:
    """Share of cords that sit in groups whose size is a multiple of `size`."""
    size = int(spec["size"])
    n = sum(len(g) for g in k.groups)
    return sum(len(g) for g in k.groups if len(g) % size == 0) / n if n else None


def first_cord_range(k: Khipu, spec: dict) -> float | None:
    """Share of consecutive runs of `size` cords (within groups) whose first cord value is in [min, max]."""
    size, lo, hi = int(spec["size"]), float(spec["min"]), float(spec["max"])
    firsts = np.array([k.values[r[0]] for g in k.groups for r in _runs(g, size)], dtype=float)
    firsts = firsts[~np.isnan(firsts)]
    return float(((firsts >= lo) & (firsts <= hi)).mean()) if len(firsts) else None


def value_range(k: Khipu, spec: dict) -> float | None:
    """Share of valued cords whose value is in [min, max]."""
    v = k.values[~np.isnan(k.values)]
    return float(((v >= float(spec["min"])) & (v <= float(spec["max"]))).mean()) if len(v) else None


def group_sum(k: Khipu, spec: dict) -> float | None:
    """Share of groups (or runs of `size`, if given) where the cord at `sum_position` (first or last)
    equals the sum of the other cords. Units need at least two other valued cords and a non-zero total."""
    first = spec.get("sum_position", "first") == "first"
    size = spec.get("size")
    hits = []
    for g in k.groups:
        for u in (_runs(g, int(size)) if size else [g]):
            if len(u) < 3:
                continue
            v = k.values[u]
            s, rest = (v[0], v[1:]) if first else (v[-1], v[:-1])
            rest = rest[~np.isnan(rest)]
            if np.isnan(s) or len(rest) < 2 or rest.sum() == 0:
                continue
            hits.append(s == rest.sum())
    return float(np.mean(hits)) if hits else None


def color_repeat(k: Khipu, spec: dict) -> float | None:
    """Share of groups (with at least two full periods) whose colours repeat with `period` and that
    use at least two colours inside one period. Single-colour groups repeat with any period, so they
    are excluded rather than counted as matches."""
    p = int(spec["period"])
    hits = []
    for g in k.groups:
        c = k.colors[g]
        if len(c) < 2 * p or len(set(c[:p])) < 2:
            continue
        n = len(c) // p * p
        hits.append(bool(np.all(c[: n - p] == c[p:n])))
    return float(np.mean(hits)) if hits else None


def attachment_split(k: Khipu, spec: dict) -> float | None:
    """Share of runs of `size` whose first cord has attachment `code` (for example V or R)."""
    if k.attachment is None:
        return None
    size, code = int(spec["size"]), str(spec["code"])
    firsts = [k.attachment[r[0]] for g in k.groups for r in _runs(g, size)]
    return float(np.mean([a == code for a in firsts])) if firsts else None


FEATURES: dict[str, Feature] = {
    "group_size": group_size,
    "first_cord_range": first_cord_range,
    "value_range": value_range,
    "group_sum": group_sum,
    "color_repeat": color_repeat,
    "attachment_split": attachment_split,
}

# The null each feature is tested against: (field it moves, how). Features with no entry describe
# structure or a whole-khipu distribution, which no permutation of that khipu can change, so they
# filter and rank but cannot reach a low p-value. A feature spec can override with `null: <mode>`.
#   first_cord_range: are the first cords of runs more in range than cords picked at random?
#   group_sum:        given each slot's values, is the sum relation more frequent than chance?
#   color_repeat:     does the repeat survive rotating colours along the khipu (bands kept)?
#   attachment_split: are run-first cords tied with `code` more than cords at random places?
NULLS = {
    "first_cord_range": ("value", "within_khipu"),
    "group_sum": ("value", "within_slot"),
    "color_repeat": ("color", "rotate"),
    "attachment_split": ("attachment", "rotate"),
}

"""Scoring, structure-preserving nulls, and ranking with multiple-comparison control."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .features import FEATURES, NULLS, Khipu

NULL_DEFAULTS = {"draws": "auto", "seed": 0}


def to_khipus(corpus: pd.DataFrame, level: int, min_cords: int) -> list[Khipu]:
    out = []
    cords = corpus[corpus["level"] == level]
    for kid, k in cords.groupby("khipu_id", sort=True):
        k = k.sort_values("position").reset_index(drop=True)
        if len(k) < min_cords:
            continue
        groups = tuple(sorted((np.asarray(ix) for ix in k.groupby("group_id", sort=False).indices.values()),
                              key=lambda g: g[0]))
        slots = np.zeros(len(k), dtype=int)
        for g in groups:
            slots[g] = np.arange(len(g))
        att = k["attachment"].astype(str).to_numpy(dtype=object) if "attachment" in k else None
        out.append(Khipu(str(kid), k["value"].to_numpy(dtype=float), k["color"].astype(str).to_numpy(dtype=object),
                         att, groups, slots))
    return out


def null_of(f: dict) -> tuple[str, str] | None:
    base = NULLS.get(f["type"])
    if base is None:
        return None
    return (base[0], f.get("null", base[1]))


def move(k: Khipu, field: str, mode: str, rng: np.random.Generator) -> Khipu:
    """Return a copy of the khipu with one field permuted. Groups, positions and levels stay fixed.

    within_khipu: shuffle across the khipu. within_group: shuffle inside each group.
    within_slot: shuffle among cords in the same slot of their group (all first cords together).
    rotate: circular shift along the khipu, which keeps runs and bands intact."""
    arr = {"value": k.values, "color": k.colors, "attachment": k.attachment}[field]
    if arr is None:
        return k
    n = len(arr)
    if mode == "within_khipu":
        new = arr[rng.permutation(n)]
    elif mode == "within_group":
        new = arr.copy()
        for g in k.groups:
            new[g] = arr[rng.permutation(g)]
    elif mode == "within_slot":
        new = arr.copy()
        for sl in np.unique(k.slots):
            ix = np.flatnonzero(k.slots == sl)
            new[ix] = arr[ix[rng.permutation(len(ix))]]
    elif mode == "rotate":
        new = np.roll(arr, int(rng.integers(1, n))) if n > 1 else arr
    else:
        raise ValueError(f"unknown null mode: {mode}")
    key = {"value": "values", "color": "colors", "attachment": "attachment"}[field]
    return k.with_(**{key: new})


def combine(parts: list[tuple[float | None, float]]) -> float | None:
    num = sum(w * s for s, w in parts if s is not None)
    den = sum(w for s, w in parts if s is not None)
    return num / den if den else None


def score(k: Khipu, pattern: dict) -> tuple[float | None, dict]:
    parts = {f["type"]: FEATURES[f["type"]](k, f) for f in pattern["features"]}
    return combine([(parts[f["type"]], float(f.get("weight", 1.0))) for f in pattern["features"]]), parts


def null_score(k: Khipu, pattern: dict, rng: np.random.Generator) -> float | None:
    """Each feature is evaluated on its own null version of the khipu. Features without a null
    keep their observed value, so they shift both sides equally."""
    parts = []
    for f in pattern["features"]:
        nm = null_of(f)
        kk = move(k, *nm, rng) if nm else k
        parts.append((FEATURES[f["type"]](kk, f), float(f.get("weight", 1.0))))
    return combine(parts)


def bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values."""
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank_, i in enumerate(order[::-1]):
        prev = min(prev, p[i] * n / (n - rank_))
        q[i] = prev
    return q


def auto_draws(n_khipus: int, top: int, alpha: float = 0.05) -> int:
    """Enough draws that the top `top` khipus could reach q <= alpha if they beat every draw:
    q_min = n / (top * (draws + 1)), so draws >= n / (top * alpha). At least 200."""
    return max(200, math.ceil(n_khipus / (top * alpha)))


def rank(corpus: pd.DataFrame, pattern: dict, top: int = 5) -> dict:
    null = {**NULL_DEFAULTS, **pattern.get("null", {})}
    khipus = to_khipus(corpus, int(pattern.get("level", 1)), int(pattern.get("min_cords", 1)))
    scored = [(k, *score(k, pattern)) for k in khipus]
    scored = [(k, s, parts) for k, s, parts in scored if s is not None]
    if not scored:
        return {"table": pd.DataFrame(), "summary": {"khipus": 0}}
    draws = auto_draws(len(scored), top) if null["draws"] == "auto" else int(null["draws"])
    tested = [f for f in pattern["features"] if null_of(f)]
    rng = np.random.default_rng(int(null["seed"]))
    sims = np.full((len(scored), draws), np.nan)
    rows = []
    for i, (k, obs, parts) in enumerate(scored):
        for d in range(draws):
            s = null_score(k, pattern, rng)
            sims[i, d] = np.nan if s is None else s
        ok = sims[i][~np.isnan(sims[i])]
        p = (1 + np.sum(ok >= obs - 1e-12)) / (1 + len(ok)) if tested else 1.0
        rows.append({"khipu_id": k.id, "cords": len(k.values), "score": obs,
                     "null_mean": float(ok.mean()) if len(ok) else np.nan, "p": float(p),
                     **{f"f_{n}": v for n, v in parts.items()}})
    table = pd.DataFrame(rows)
    table["q"] = bh(table["p"].to_numpy())
    order = table.sort_values(["score", "p"], ascending=[False, True]).index.to_numpy()
    table = table.loc[order].reset_index(drop=True)
    sims = sims[order]
    table.insert(0, "rank", range(1, len(table) + 1))

    # Selection test: is our top-k better than the top-k picked from null versions of every khipu?
    filled = np.where(np.isnan(sims), -np.inf, sims)
    null_top = np.sort(filled, axis=0)[::-1][:top].mean(axis=0)
    obs_top = float(table["score"].head(top).mean())
    summary = {
        "khipus": len(table),
        "draws": draws,
        "min_attainable_p": 1 / (draws + 1),
        "tested_features": {f["type"]: "{}:{}".format(*null_of(f)) for f in tested},
        "untested_features": [f["type"] for f in pattern["features"] if not null_of(f)],
        "top": top,
        "top_mean_score": obs_top,
        "null_top_mean_95": [float(np.percentile(null_top, 2.5)), float(np.percentile(null_top, 97.5))],
        "top_p": float((1 + np.sum(null_top >= obs_top - 1e-12)) / (1 + len(null_top))),
        "significant_at_q05": int((table["q"] <= 0.05).sum()),
    }
    return {"table": table, "summary": summary}

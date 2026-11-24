"""Corpus loading. A corpus is one row per cord with these columns:

khipu_id   identifier of the khipu
cord_id    identifier of the cord, unique within the corpus
group_id   identifier of the cord group (cluster) the cord belongs to
position   order of the cord along the khipu (numeric, increasing from the start)
level      1 for pendants hanging from the primary cord, 2+ for subsidiaries
color      colour code as recorded (any string)
value      decoded knot value (number; empty means no knots)

Optional: attachment (recto/verso code), and any other columns, which are kept.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = ["khipu_id", "cord_id", "group_id", "position", "level", "color", "value"]

# OKR v2.1.0 and the Khipu Field Guide disagree on the decimal place of single knots on these
# khipus (OKR 100, Field Guide 10). The errata option applies the Field Guide reading. KH0326 is
# supported by Urton (2017, Appendix B); KH0472 only by the Field Guide. Not a confirmed OKR error.
OKR_TENS_AS_HUNDREDS = {"KH0326": "urton-appendix", "KH0472": "kfg-only"}

_SEG = re.compile(r"([^/()]+)\(([\d.]+)-([\d.]+)\)")


def main_color(c: str) -> str:
    """Colour covering the longest stretch of a multi-segment cord, as the Khipu Field Guide shows it."""
    segs = _SEG.findall(c or "")
    if not segs:
        return (c or "").strip()
    span: dict[str, float] = {}
    for col, a, b in segs:
        span[col] = span.get(col, 0.0) + float(b) - float(a)
    return max(span, key=span.get)


def load(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"corpus is missing columns: {', '.join(missing)}")
    df["color"] = df["color"].fillna("").astype(str)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["level"] = pd.to_numeric(df["level"], errors="coerce").fillna(1).astype(int)
    df["position"] = pd.to_numeric(df["position"], errors="coerce")
    if df["cord_id"].duplicated().any():
        raise ValueError("cord_id must be unique")
    return df.sort_values(["khipu_id", "position"]).reset_index(drop=True)


def _shift_tens(v: float) -> float:
    if np.isnan(v):
        return v
    v = int(v)
    h, t = v // 100 % 10, v // 10 % 10
    return float(v - h * 90) if t == 0 and h > 0 and v < 1000 else float(v)


def import_okr(db: str | Path, errata: str = "none") -> pd.DataFrame:
    """Build a corpus from the Open Khipu Repository SQLite file (v2.1.0).

    errata: "none" keeps OKR values; "confirmed" applies the Field Guide reading to KH0326 only;
    "all" also applies it to KH0472. Cords with two single-knot clusters (KH0326 p100, p106)
    are not fully corrected by this rule.
    """
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    cords = pd.read_sql(
        """
        select m.OKR_NUM khipu_id, c.CORD_ID cord_id, c.CLUSTER_ID group_id, c.CORD_ORDINAL position,
               c.CORD_LEVEL level, c.ATTACHMENT_TYPE attachment, k.value, k.n_knots
        from cord c join khipu_main m on m.KHIPU_ID = c.KHIPU_ID
        left join (
            select CORD_ID, sum(knot_value_type) value, count(*) n_knots from knot group by CORD_ID
        ) k on k.CORD_ID = c.CORD_ID
        """,
        con,
    )
    cords["n_knots"] = cords["n_knots"].fillna(0)
    colors = pd.read_sql(
        "select rowid rid, color_id, CORD_ID cord_id, FULL_COLOR color from ascher_cord_color", con
    )
    con.close()
    colors = colors.sort_values(["cord_id", "color_id", "rid"]).drop_duplicates("cord_id")
    df = cords.merge(colors[["cord_id", "color"]], on="cord_id", how="left")
    df["color"] = df["color"].fillna("").map(main_color)
    df.loc[df["n_knots"] == 0, "value"] = np.nan
    fix = {"none": set(), "confirmed": {"KH0326"}, "all": set(OKR_TENS_AS_HUNDREDS)}[errata]
    mask = df["khipu_id"].isin(fix)
    df.loc[mask, "value"] = df.loc[mask, "value"].map(_shift_tens)
    return df.drop(columns=["n_knots"]).sort_values(["khipu_id", "position"]).reset_index(drop=True)

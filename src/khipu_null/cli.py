"""khipu-null command line."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from . import __version__
from .corpus import import_okr, load
from .rank import rank

LIMITS = """## What this does not prove

- A low p-value says the match is unlikely under this null, given the structure of each khipu. It does not say what the khipu means.
- Each feature is tested against its own null, which keeps groups, positions and levels fixed and moves only the field it reads. Structural features and whole-khipu distributions cannot be tested this way; they rank but never reach a low p-value.
- Colours are rotated along the khipu by default, which keeps colour bands intact. Shuffling colours cord by cord breaks bands and makes chance matches look rare.
- Choosing the best 5 of many khipus inflates scores. Read the top-k test and the q-values, not the raw scores.
"""


def _report(res: dict, pattern: dict, out: Path, top: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    t, s = res["table"], res["summary"]
    t.to_csv(out / "ranking.csv", index=False)
    (out / "summary.json").write_text(json.dumps(s, indent=2))
    lines = [f"# khipu-null report: {pattern.get('name', 'pattern')}", "",
             f"khipu-null {__version__}. {s.get('khipus', 0)} khipus scored, {s.get('draws', 0)} null draws each.", ""]
    if s.get("khipus"):
        lines += [
            "Tested features (field:null): " + (", ".join(f"{k} ({v})" for k, v in s["tested_features"].items()) or "none") + ".",
            "Untested features (structure or whole-khipu distributions; they rank but cannot reach a low p): " + (", ".join(s["untested_features"]) or "none") + ".",
            "",
            f"Top {top} mean score {s['top_mean_score']:.3f}. Picking the top {top} from null versions of every khipu gives "
            f"[{s['null_top_mean_95'][0]:.3f}, {s['null_top_mean_95'][1]:.3f}] (95%), p = {s['top_p']:.3f}.",
            f"Khipus with q <= 0.05 (Benjamini-Hochberg): {s['significant_at_q05']}.",
            "",
            "| Rank | Khipu | Cords | Score | Null mean | p | q |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
        for r in t.head(max(top, 10)).itertuples():
            lines.append(f"| {r.rank} | {r.khipu_id} | {r.cords} | {r.score:.3f} | {r.null_mean:.3f} | {r.p:.3f} | {r.q:.3f} |")
        lines.append("")
    lines.append(LIMITS)
    lines.append("If you use khipu-null, please cite it (see CITATION.cff): Crafter Research (2026), khipu-null, https://github.com/crafter-research/khipu-null")
    (out / "report.md").write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="khipu-null", description=__doc__)
    ap.add_argument("--version", action="version", version=f"khipu-null {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    imp = sub.add_parser("import-okr", help="build a corpus CSV from the Open Khipu Repository SQLite file")
    imp.add_argument("db")
    imp.add_argument("-o", "--out", default="corpus.csv")
    imp.add_argument("--errata", choices=["none", "confirmed", "all"], default="none",
                     help="fix OKR tens-as-hundreds coding: confirmed = KH0326 only; all = also KH0472, KH0401")

    rk = sub.add_parser("rank", help="rank khipus against a pattern and test against the null")
    rk.add_argument("corpus")
    rk.add_argument("pattern")
    rk.add_argument("-o", "--out", default="out")
    rk.add_argument("--top", type=int, default=5)
    rk.add_argument("--draws", type=int, help="override null draws")
    rk.add_argument("--seed", type=int, help="override null seed")

    a = ap.parse_args(argv)
    if a.cmd == "import-okr":
        df = import_okr(a.db, a.errata)
        df.to_csv(a.out, index=False)
        print(f"{a.out}: {df['khipu_id'].nunique()} khipus, {len(df)} cords (errata: {a.errata})")
        return 0
    pattern = yaml.safe_load(Path(a.pattern).read_text())
    pattern.setdefault("null", {})
    if a.draws:
        pattern["null"]["draws"] = a.draws
    if a.seed is not None:
        pattern["null"]["seed"] = a.seed
    res = rank(load(a.corpus), pattern, top=a.top)
    _report(res, pattern, Path(a.out), a.top)
    print((Path(a.out) / "report.md").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())

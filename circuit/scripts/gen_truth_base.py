"""Generate a base-friendly truth dataset from cities.csv (no chat template).

Uses 2-shot in-context-learning so a non-instruct model has a strong format
prior for the {True, False} continuation. Output:
  data/specificity/truth_base.jsonl
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATASETS = REPO.parent / "datasets"
OUT = REPO / "data" / "specificity" / "truth_base.jsonl"

ICL_PREFIX = (
    "Statement: Paris is in France.\nAnswer: True\n\n"
    "Statement: Tokyo is in Brazil.\nAnswer: False\n\n"
)


def main(n_per_class: int = 500, seed: int = 42) -> None:
    csv_path = DATASETS / "cities.csv"
    if not csv_path.exists():
        raise SystemExit(f"missing {csv_path}")
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)
    true_rows = [r for r in all_rows if r["label"] == "1"]
    false_rows = [r for r in all_rows if r["label"] == "0"]
    rng = random.Random(seed)
    rng.shuffle(true_rows); rng.shuffle(false_rows)
    n = min(n_per_class, len(true_rows), len(false_rows))
    pool = true_rows[:n] + false_rows[:n]
    rng.shuffle(pool)

    rows = []
    for src in pool:
        stmt = src["statement"].strip()
        prompt = ICL_PREFIX + f"Statement: {stmt}\nAnswer:"
        rows.append({
            "id": f"truthbase_{len(rows):06d}",
            "statement": stmt,
            "prompt": prompt,
            "label": int(src["label"]),
            "high_word": "True",
            "low_word": "False",
            "space_prefix": True,
        })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {OUT}  n={len(rows)}  "
          f"pos={sum(r['label'] for r in rows)}")


if __name__ == "__main__":
    main()

"""Generate prompts for the Phase 2U specificity battery.

Emits jsonl rows in the v11-compatible format used by p2o_specific_cells.py:
  {"prompt": str, "label": float, "high_word": str, "low_word": str,
   "space_prefix": bool}

`space_prefix=True` mirrors the original v11 convention (the harness encodes
" {high_word}" and takes the last token).  For chat-template tasks (where the
model's first response token has no leading space), we set False and the
harness encodes "{high_word}" directly.

Tasks
  arithmetic  : "Is X greater than Y? Answer:" (no chat template; base prompt)
  truth       : "Statement: <s>. Is this true or false? Answer:"  (chat template)
  refusal     : harmful + harmless instruction (chat template, I/Sure logits)

Outputs:
  data/specificity/arithmetic.jsonl
  data/specificity/truth.jsonl
  data/specificity/refusal.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "data" / "specificity"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = REPO.parent / "datasets"
REFUSAL_SPLITS = REPO.parent / "refusal_direction" / "dataset" / "splits"

# Chat template for Gemma 2 IT (matches tokenizer.apply_chat_template
# with add_generation_prompt=True, but BOS is added by the tokenizer).
GEMMA2_IT_TEMPLATE = (
    "<start_of_turn>user\n{user}<end_of_turn>\n<start_of_turn>model\n"
)


def gen_arithmetic(n: int = 1000, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    seen = set()
    while len(rows) < n:
        x = rng.randint(1, 200)
        y = rng.randint(1, 200)
        if x == y:
            continue
        key = (x, y)
        if key in seen:
            continue
        seen.add(key)
        prompt = f"Is {x} greater than {y}? Answer:"
        label = 1 if x > y else 0
        rows.append({
            "id": f"arith_{len(rows):06d}",
            "x": x, "y": y,
            "prompt": prompt,
            "label": label,
            "high_word": "Yes",
            "low_word": "No",
            "space_prefix": True,
        })
    pos = sum(r["label"] for r in rows)
    print(f"  arithmetic: n={len(rows)}, pos={pos} ({pos/len(rows):.1%})")
    return rows


def gen_truth(n_per_class: int = 500, seed: int = 42) -> list[dict]:
    csv_path = DATASETS / "cities.csv"
    if not csv_path.exists():
        raise SystemExit(f"missing {csv_path}")
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)
    true_rows = [r for r in all_rows if r["label"] == "1"]
    false_rows = [r for r in all_rows if r["label"] == "0"]
    rng = random.Random(seed)
    rng.shuffle(true_rows)
    rng.shuffle(false_rows)
    n = min(n_per_class, len(true_rows), len(false_rows))
    pool = true_rows[:n] + false_rows[:n]
    rng.shuffle(pool)
    rows = []
    for src in pool:
        stmt = src["statement"].strip()
        user = (
            f"Statement: {stmt}\n"
            "Is this statement true or false? "
            "Answer with just 'True' or 'False'."
        )
        prompt = GEMMA2_IT_TEMPLATE.format(user=user)
        rows.append({
            "id": f"truth_{len(rows):06d}",
            "statement": stmt,
            "prompt": prompt,
            "label": int(src["label"]),
            "high_word": "True",
            "low_word": "False",
            "space_prefix": False,
        })
    print(f"  truth: n={len(rows)}, pos={sum(r['label'] for r in rows)}")
    return rows


def gen_refusal(n_per_class: int = 500, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    harmful = json.load((REFUSAL_SPLITS / "harmful_train.json").open())
    harmless = json.load((REFUSAL_SPLITS / "harmless_train.json").open())
    rng.shuffle(harmful)
    rng.shuffle(harmless)
    n = min(n_per_class, len(harmful), len(harmless))
    pool = [(h, 1) for h in harmful[:n]] + [(h, 0) for h in harmless[:n]]
    rng.shuffle(pool)
    rows = []
    for src, label in pool:
        instr = src["instruction"]
        prompt = GEMMA2_IT_TEMPLATE.format(user=instr)
        rows.append({
            "id": f"refusal_{len(rows):06d}",
            "instruction": instr,
            "prompt": prompt,
            "label": label,
            "high_word": "I",
            "low_word": "Sure",
            "space_prefix": False,
        })
    print(f"  refusal: n={len(rows)}, harmful={sum(r['label'] for r in rows)}")
    return rows


def gen_neutral_text(n: int = 500, seed: int = 42) -> list[dict]:
    """Varied natural-language prompts with no task structure and no
    chat-template wrapping. Drawn from harmless_train (which is full of
    natural-language imperatives like 'Compare and contrast...'), used
    as raw text. KL-only — no label, no high_word/low_word.
    """
    rng = random.Random(seed)
    src = json.load((REFUSAL_SPLITS / "harmless_train.json").open())
    rng.shuffle(src)
    rows = []
    for s in src[:n]:
        instr = s["instruction"].strip()
        # Strip very short and very long prompts so KL isn't dominated by
        # token-length artifacts.
        if len(instr) < 20 or len(instr) > 400:
            continue
        rows.append({
            "id": f"neutral_{len(rows):06d}",
            "prompt": instr,
            "label": None,
            "high_word": None,
            "low_word": None,
            "space_prefix": False,
        })
        if len(rows) >= n:
            break
    print(f"  neutral_text: n={len(rows)}")
    return rows


def write_jsonl(rows: list[dict], path: Path) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"    -> {path}  ({len(rows)} rows)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+",
                    default=["arithmetic", "truth", "refusal", "neutral_text"],
                    choices=["arithmetic", "truth", "refusal", "neutral_text"])
    ap.add_argument("--n", type=int, default=1000,
                    help="total prompts (per-class is n//2 for binary tasks)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if "arithmetic" in args.tasks:
        rows = gen_arithmetic(n=args.n, seed=args.seed)
        write_jsonl(rows, OUT_DIR / "arithmetic.jsonl")
    if "truth" in args.tasks:
        rows = gen_truth(n_per_class=args.n // 2, seed=args.seed)
        write_jsonl(rows, OUT_DIR / "truth.jsonl")
    if "refusal" in args.tasks:
        rows = gen_refusal(n_per_class=args.n // 2, seed=args.seed)
        write_jsonl(rows, OUT_DIR / "refusal.jsonl")
    if "neutral_text" in args.tasks:
        rows = gen_neutral_text(n=500, seed=args.seed)
        write_jsonl(rows, OUT_DIR / "neutral_text.jsonl")


if __name__ == "__main__":
    main()

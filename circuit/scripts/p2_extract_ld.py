"""Phase 2A — slim LD-only extractor for the shot-count behavioral sweep.

Reads a JSONL of trials (output of gen_p2_shot_sweep.py), runs a forward
pass on the model, and saves an NPZ with per-prompt last-token LD =
logit(high_word) - logit(low_word). No residuals, no attention captures.

Designed for cheap behavioral sweeps across multiple (pair, k) configurations.

Usage:
  python scripts/p2_extract_ld.py --model gemma2-2b \
                                  --pairs height weight speed \
                                  --k 0 1 2 4 8 15

Output:
  results/p2_ld/<model_short>/<pair>_k<k>.npz
    keys: id (object), x, z, z_eff, mu, mu_eff, sigma, sigma_eff,
          k, cell_seed, ld, low_token_id, high_token_id
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parent.parent
GOR = REPO.parent / "geometry-of-relativity"
sys.path.insert(0, str(GOR / "scripts" / "vast_remote"))
from _token_utils import tokens_of_word  # noqa: E402

MODEL_ID = {
    "gemma2-2b": "google/gemma-2-2b",
    "gemma2-2b-it": "google/gemma-2-2b-it",
    "gemma2-9b": "google/gemma-2-9b",
    "gemma2-9b-it": "google/gemma-2-9b-it",
}


def load_trials(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f]


def first_single_token(tok, word: str) -> tuple[int, list[int]]:
    """Return (token_id, all_ids). token_id is the single-token variant if any.
    Caller may want to warn on multi-token words."""
    ids = tokens_of_word(tok, word)
    return ids[0], ids


@torch.inference_mode()
def extract_ld_for_jsonl(
    model,
    tok,
    trials: list[dict],
    high_word: str,
    low_word: str,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, int, int, dict]:
    """Returns (ld_per_prompt, high_id, low_id, info)."""
    high_id, high_ids = first_single_token(tok, high_word)
    low_id, low_ids = first_single_token(tok, low_word)
    if len(high_ids) > 1 or len(low_ids) > 1:
        print(f"  WARNING: multi-token high/low words "
              f"high={high_word}->{high_ids} low={low_word}->{low_ids}; "
              f"using first token only.")

    # Left-pad for causal models so logits[:, -1, :] is always the real last
    # token. Some Gemma tokenizers default to right-pad.
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        # Use eos as pad (won't be attended to thanks to attention mask).
        tok.pad_token = tok.eos_token

    ld_out = np.zeros(len(trials), dtype=np.float64)

    t0 = time.time()
    n = len(trials)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_prompts = [t["prompt"] for t in trials[start:end]]
        enc = tok(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
            add_special_tokens=True,
        ).to(device)
        out = model(**enc)
        # last-token logits — left-padded, so always the final position.
        last_logits = out.logits[:, -1, :].to(torch.float32)
        ld = (last_logits[:, high_id] - last_logits[:, low_id]).cpu().numpy()
        ld_out[start:end] = ld
        if start == 0 or end == n or end % (batch_size * 10) == 0:
            elapsed = time.time() - t0
            rate = end / max(elapsed, 1e-9)
            eta = (n - end) / max(rate, 1e-9)
            print(f"    {end:>5d}/{n} prompts  {rate:6.1f} p/s  eta {eta:5.1f}s",
                  flush=True)

    info = {
        "high_word": high_word,
        "low_word": low_word,
        "high_token": tok.convert_ids_to_tokens([high_id])[0],
        "low_token": tok.convert_ids_to_tokens([low_id])[0],
        "high_multi_token": len(high_ids) > 1,
        "low_multi_token": len(low_ids) > 1,
    }
    return ld_out, high_id, low_id, info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gemma2-2b", choices=list(MODEL_ID))
    p.add_argument("--pairs", nargs="+", default=["height", "weight", "speed"])
    p.add_argument("--k", nargs="+", type=int, default=[0, 1, 2, 4, 8, 15])
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--data-dir", default=str(REPO / "data" / "p2_shot_sweep"))
    p.add_argument("--out-dir", default=str(REPO / "results" / "p2_ld"))
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {MODEL_ID[args.model]} (bf16, eager attention)...", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID[args.model])
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID[args.model],
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map={"": args.device},
    )
    model.eval()
    print(f"  loaded in {time.time() - t0:.1f}s", flush=True)

    overall = []
    for pair_name in args.pairs:
        for k in args.k:
            in_path = data_dir / f"{pair_name}_k{k}.jsonl"
            if not in_path.exists():
                print(f"[skip] {in_path} not found", flush=True)
                continue
            trials = load_trials(in_path)
            high_word = trials[0]["high_word"]
            low_word = trials[0]["low_word"]
            print(f"\n[{args.model}] {pair_name} k={k}: {len(trials)} prompts "
                  f"high='{high_word}' low='{low_word}'", flush=True)

            ld, high_id, low_id, info = extract_ld_for_jsonl(
                model, tok, trials, high_word, low_word,
                batch_size=args.batch_size, device=args.device,
            )

            # Pack metadata arrays.
            ids = np.array([t["id"] for t in trials], dtype=object)
            x = np.array([t["x"] for t in trials], dtype=np.float64)
            z = np.array([t["z"] for t in trials], dtype=np.float64)
            z_eff = np.array([t["z_eff"] for t in trials], dtype=np.float64)
            mu = np.array([t["mu"] for t in trials], dtype=np.float64)
            mu_eff = np.array([t["mu_eff"] for t in trials], dtype=np.float64)
            sigma = np.array([t["sigma"] for t in trials], dtype=np.float64)
            sigma_eff = np.array([t["sigma_eff"] for t in trials], dtype=np.float64)
            k_arr = np.array([t["k"] for t in trials], dtype=np.int32)
            cell_seed = np.array([t["cell_seed"] for t in trials], dtype=np.int32)

            out_path = out_dir / f"{pair_name}_k{k}.npz"
            np.savez(
                out_path,
                id=ids, x=x, z=z, z_eff=z_eff, mu=mu, mu_eff=mu_eff,
                sigma=sigma, sigma_eff=sigma_eff, k=k_arr, cell_seed=cell_seed,
                ld=ld,
                high_token_id=np.array([high_id], dtype=np.int64),
                low_token_id=np.array([low_id], dtype=np.int64),
                info_json=np.array([json.dumps(info)], dtype=object),
            )
            overall.append((pair_name, k, len(trials)))
            print(f"  -> {out_path}", flush=True)

    print(f"\nDone. {len(overall)} (pair, k) configs extracted.", flush=True)


if __name__ == "__main__":
    main()

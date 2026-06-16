"""Phase 1 — text ablation: three intervention modes at L20 of Gemma 2 2B.

For each pair (8 pairs) × mode (baseline + 4 interventions), hook L20,
re-run all held-out prompts (cell_seed != 0), record corr(LD, z),
corr(LD, x), and a small set of summary stats.

Direction:
  primal_z[pair] = mean(h_L20[z>+1]) − mean(h_L20[z<-1])
                   computed only on cell_seed=0 prompts (held-out clean).
  w_shared_proc  = unit(mean_i sign_i · primal_z[i])
                   sign_i flips i's primal to align with the unaligned mean.

Modes (applied at all token positions; matches Jaehoon's steering convention):
  baseline      — no hook
  add_neg α     — h ← h − α·d̂
  proj_out      — h ← h − (h·d̂)·d̂
  mean_ablate   — h ← h − (h·d̂)·d̂ + μ_proj·d̂   where μ_proj = E[h·d̂]
  random_proj   — same as proj_out but d̂ ← random unit vector
                  (3 different draws, summary mean reported)

Held-out evaluation set: per-pair, cell_seed != 0.

Perplexity control: WikiText-2 first 200 sentences, same hook applied. We
compare loss per token (and ratio to baseline).

Output:
  results/p1_text_ablation_gemma2-2b_L20.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parent.parent
GOR_ROOT = Path("/home/alexander/research_projects/geometry-of-relativity")
ALL_PAIRS = ["height", "age", "weight", "size", "speed", "wealth", "experience", "bmi_abs"]
MODEL_ID = "google/gemma-2-2b"
MODEL_SHORT = "gemma2-2b"
LATE_LAYER = 20
DEFAULT_ALPHA = 4.0


# ------------------------- direction construction -------------------------

def primal_z_from_npz(npz_path: Path, layer: int, cell_seed_filter: int | None
                      ) -> np.ndarray:
    """primal_z = mean(h | z>+1) − mean(h | z<-1) at the given layer.

    If cell_seed_filter is not None, restrict to prompts with that cell_seed
    (read from the trial JSONL, since the NPZ stores `seed` which is the
    pair-specific offset + cell_seed).
    """
    d = np.load(npz_path, allow_pickle=True)
    h = d["activations"][:, layer, :].astype(np.float64)
    z = d["z"]
    seed_full = d["seed"]   # pair_offset + cell_seed
    if cell_seed_filter is not None:
        # Reconstruct cell_seed: cell_seed = seed_full - pair_offset.
        # For each pair the offset is constant, so subtracting min(seed_full)
        # would give us 0..9 IF all 10 cell_seeds are present uniformly.
        offset = int(np.min(seed_full))
        cs = seed_full - offset
        mask = cs == cell_seed_filter
        h = h[mask]
        z = z[mask]
    return h[z > +1.0].mean(0) - h[z < -1.0].mean(0)


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def build_w_shared_proc(primals: list[np.ndarray]) -> np.ndarray:
    P = np.stack(primals)
    w_mean = unit(P.mean(0))
    P_aligned = np.array([p if (p @ w_mean) >= 0 else -p for p in P])
    return unit(P_aligned.mean(0))


# ------------------------- hook factory -------------------------

def make_layer_hook(direction_t: torch.Tensor, mode: str,
                    alpha: float = 0.0, mu_proj: float = 0.0):
    """direction_t is the unit direction in bf16 on the model device."""
    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output  # (B, T, d)
        if mode == "baseline":
            return output
        if mode == "add_neg":
            h = h - alpha * direction_t
        elif mode == "proj_out":
            proj = (h * direction_t).sum(-1, keepdim=True)
            h = h - proj * direction_t
        elif mode == "mean_ablate":
            proj = (h * direction_t).sum(-1, keepdim=True)
            h = h - proj * direction_t + mu_proj * direction_t
        else:
            raise ValueError(f"unknown mode {mode}")
        return (h,) + output[1:] if isinstance(output, tuple) else h
    return hook


# ------------------------- prompt + LD evaluation -------------------------

def load_holdout_prompts(pair: str) -> tuple[list[dict], np.ndarray]:
    """Return (prompt_dicts, indices) for cell_seed != 0."""
    p = GOR_ROOT / "data_gen" / f"v11_{pair}_trials.jsonl"
    out: list[dict] = []
    for line in p.open():
        t = json.loads(line)
        cs = t.get("cell_seed", t.get("seed"))
        if cs == 0:
            continue
        out.append(t)
    return out


def eval_LD_under_hook(model, tok, prompts: list[dict], hi_id: int, lo_id: int,
                       hook_fn, layer: int, batch_size: int, max_seq: int,
                       ) -> np.ndarray:
    """Forward over prompts with hook on L<layer>; return LD array (n,)."""
    decoder_layers = model.model.layers
    handle = decoder_layers[layer].register_forward_hook(hook_fn)
    out = np.zeros(len(prompts), dtype=np.float32)
    try:
        for b0 in range(0, len(prompts), batch_size):
            b = prompts[b0:b0 + batch_size]
            prompt_strs = [t["prompt"] for t in b]
            enc = tok(prompt_strs, return_tensors="pt", padding="max_length",
                      max_length=max_seq, truncation=True).to(model.device)
            with torch.no_grad():
                logits = model(input_ids=enc.input_ids,
                               attention_mask=enc.attention_mask,
                               use_cache=False).logits[:, -1, :].float()
            out[b0:b0 + len(b)] = (logits[:, hi_id] - logits[:, lo_id]).cpu().numpy()
    finally:
        handle.remove()
    return out


# ------------------------- WikiText perplexity -------------------------

def load_wikitext_chunk(tok, n_sentences: int = 200, min_tokens: int = 16
                        ) -> torch.Tensor:
    """Load WikiText-2 test, take first n_sentences with ≥min_tokens. Return
    a single concatenated input_ids tensor (1, total_tokens). Slow path: use
    the file in HF datasets cache if available, else download."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("install `datasets` for perplexity control")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    sentences = []
    for row in ds:
        s = row["text"].strip()
        if not s or s.startswith("="):
            continue
        ids = tok(s, return_tensors="pt", add_special_tokens=False).input_ids[0]
        if ids.shape[0] < min_tokens:
            continue
        sentences.append(ids)
        if len(sentences) >= n_sentences:
            break
    return sentences  # list of (T_i,) tensors


def wikitext_loss_under_hook(model, tok, sentences: list[torch.Tensor],
                              hook_fn, layer: int) -> float:
    """Mean per-token NLL on the WikiText sentences."""
    decoder_layers = model.model.layers
    handle = decoder_layers[layer].register_forward_hook(hook_fn)
    total_loss = 0.0
    total_tokens = 0
    try:
        for ids in sentences:
            ids = ids.to(model.device).unsqueeze(0)
            with torch.no_grad():
                out = model(input_ids=ids, labels=ids, use_cache=False)
            n_tok = ids.shape[1] - 1  # labels shift
            total_loss += float(out.loss) * n_tok
            total_tokens += n_tok
    finally:
        handle.remove()
    return total_loss / max(1, total_tokens)


# ------------------------- mean projection on data -------------------------

def mean_projection(npz_path: Path, layer: int, direction: np.ndarray) -> float:
    d = np.load(npz_path)
    h = d["activations"][:, layer, :].astype(np.float64)
    return float((h @ direction).mean())


# ------------------------- main -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", default=ALL_PAIRS,
                    choices=ALL_PAIRS + ["all"])
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    ap.add_argument("--max-seq", type=int, default=288)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-random", type=int, default=3,
                    help="# random unit-vector controls for proj_out")
    ap.add_argument("--n-wikitext", type=int, default=200)
    ap.add_argument("--out-name", default="p1_text_ablation_gemma2-2b_L20")
    args = ap.parse_args()

    pairs = ALL_PAIRS if args.pairs == ["all"] else list(args.pairs)
    L = LATE_LAYER

    # ---- 1. compute primal_z per pair (from cell_seed=0) and w_shared_proc ----
    print(f"[p1] computing primal_z @ L{L} for {len(pairs)} pairs (cell_seed=0)...")
    primals: dict[str, np.ndarray] = {}
    for p in pairs:
        npz = (GOR_ROOT / "results" / "v11" / MODEL_SHORT / p
               / f"{MODEL_SHORT}_{p}_v11_residuals.npz")
        primals[p] = primal_z_from_npz(npz, L, cell_seed_filter=0)
    P = np.stack([primals[p] for p in pairs])
    w_shared = build_w_shared_proc([primals[p] for p in pairs])
    pairwise_cos = []
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            pairwise_cos.append(float(unit(primals[pairs[i]]) @ unit(primals[pairs[j]])))
    print(f"[p1]   pairwise mean cos: {np.mean(pairwise_cos):+.3f}")
    cos_with_shared = {p: float(unit(primals[p]) @ w_shared) for p in pairs}
    print(f"[p1]   cos(w_shared, primal_z[pair]):")
    for p in pairs:
        print(f"          {p:<11} {cos_with_shared[p]:+.3f}")

    # μ_proj for mean_ablate: average projection on the FULL data per pair,
    # but we want a single scalar for the hook. Use weighted average across
    # pairs (heuristic; alternative is per-pair μ_proj).
    mu_projs = {}
    for p in pairs:
        npz = (GOR_ROOT / "results" / "v11" / MODEL_SHORT / p
               / f"{MODEL_SHORT}_{p}_v11_residuals.npz")
        mu_projs[p] = mean_projection(npz, L, w_shared)
    mu_proj_global = float(np.mean(list(mu_projs.values())))
    print(f"[p1]   μ_proj per pair: {mu_projs}")
    print(f"[p1]   μ_proj_global:   {mu_proj_global:+.4f}")

    # ---- 2. load model + tokenizer ----
    print(f"[p1] loading {MODEL_ID} (eager attn, bf16)...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, token=os.environ.get("HF_TOKEN"))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager",
        token=os.environ.get("HF_TOKEN"),
    ).eval()
    print(f"[p1]   loaded in {time.time() - t0:.1f}s")

    # ---- 3. WikiText sentences (one-time tokenization) ----
    print(f"[p1] loading {args.n_wikitext} WikiText-2 sentences...")
    wt = load_wikitext_chunk(tok, n_sentences=args.n_wikitext)
    print(f"[p1]   {len(wt)} sentences, total tokens: "
          f"{sum(int(s.shape[0]) for s in wt)}")

    # ---- 4. precompute pair-specific token ids ----
    pair_ids: dict[str, tuple[int, int]] = {}
    for p in pairs:
        # Read first trial to grab high/low words
        jl = GOR_ROOT / "data_gen" / f"v11_{p}_trials.jsonl"
        first = json.loads(next(jl.open()))
        hi = tok(first["high_word"], add_special_tokens=False).input_ids[0]
        lo = tok(first["low_word"], add_special_tokens=False).input_ids[0]
        pair_ids[p] = (hi, lo)

    # ---- 5. precompute the bf16 device direction ----
    d_device = torch.tensor(w_shared, dtype=torch.bfloat16, device=model.device)

    # ---- 6. random unit vectors (same magnitude as w_shared which is unit) ----
    rng = np.random.default_rng(42)
    rand_dirs = []
    for r in range(args.n_random):
        v = rng.standard_normal(w_shared.shape[0])
        rand_dirs.append(unit(v))
    rand_devices = [torch.tensor(rd, dtype=torch.bfloat16, device=model.device)
                    for rd in rand_dirs]

    # ---- 7. iterate: per pair × mode, run held-out prompts ----
    results = {
        "model_id": MODEL_ID,
        "layer": L,
        "alpha": args.alpha,
        "mu_proj_global": mu_proj_global,
        "pairwise_primal_cos_mean": float(np.mean(pairwise_cos)),
        "cos_w_shared_vs_primal": cos_with_shared,
        "per_pair": {},
        "wikitext_loss": {},
    }

    # Hooks defined per setting (we'll reuse per pair)
    SETTINGS = [
        ("baseline", lambda d: make_layer_hook(d, "baseline")),
        ("add_neg", lambda d: make_layer_hook(d, "add_neg", alpha=args.alpha)),
        ("add_pos", lambda d: make_layer_hook(d, "add_neg", alpha=-args.alpha)),
        ("proj_out", lambda d: make_layer_hook(d, "proj_out")),
        ("mean_ablate", lambda d: make_layer_hook(d, "mean_ablate",
                                                   mu_proj=mu_proj_global)),
    ]

    print("\n[p1] running held-out evals per pair × setting...")
    for p in pairs:
        prompts = load_holdout_prompts(p)
        hi_id, lo_id = pair_ids[p]
        zs = np.array([t["z"] for t in prompts], dtype=np.float64)
        xs = np.array([t["x"] for t in prompts], dtype=np.float64)

        results["per_pair"][p] = {"n_holdout": len(prompts)}
        for setting_name, hook_factory in SETTINGS:
            t1 = time.time()
            ld = eval_LD_under_hook(
                model, tok, prompts, hi_id, lo_id,
                hook_factory(d_device), L,
                batch_size=args.batch_size, max_seq=args.max_seq,
            )
            r_z = float(np.corrcoef(ld, zs)[0, 1])
            r_x = float(np.corrcoef(ld, xs)[0, 1])
            ld_mean = float(ld.mean())
            results["per_pair"][p][setting_name] = {
                "corr_LD_z": r_z,
                "corr_LD_x": r_x,
                "ld_mean": ld_mean,
                "elapsed_sec": round(time.time() - t1, 1),
            }
            print(f"[p1]   {p:<11} {setting_name:<12} "
                  f"r_z={r_z:+.3f}  r_x={r_x:+.3f}  "
                  f"<LD>={ld_mean:+.2f}  ({time.time() - t1:.0f}s)",
                  flush=True)

        # Random projection control: average over rand_dirs
        rand_results = []
        for r_idx, rd_t in enumerate(rand_devices):
            t1 = time.time()
            ld = eval_LD_under_hook(
                model, tok, prompts, hi_id, lo_id,
                make_layer_hook(rd_t, "proj_out"), L,
                batch_size=args.batch_size, max_seq=args.max_seq,
            )
            r_z = float(np.corrcoef(ld, zs)[0, 1])
            r_x = float(np.corrcoef(ld, xs)[0, 1])
            rand_results.append({"corr_LD_z": r_z, "corr_LD_x": r_x,
                                  "ld_mean": float(ld.mean())})
            print(f"[p1]   {p:<11} rand_proj_{r_idx} r_z={r_z:+.3f}  "
                  f"r_x={r_x:+.3f}  ({time.time() - t1:.0f}s)", flush=True)
        results["per_pair"][p]["random_proj_out"] = {
            "n_random": len(rand_results),
            "corr_LD_z_mean": float(np.mean([r["corr_LD_z"] for r in rand_results])),
            "corr_LD_z_std":  float(np.std([r["corr_LD_z"] for r in rand_results], ddof=1)),
            "corr_LD_x_mean": float(np.mean([r["corr_LD_x"] for r in rand_results])),
            "corr_LD_x_std":  float(np.std([r["corr_LD_x"] for r in rand_results], ddof=1)),
            "raw": rand_results,
        }

    # ---- 8. perplexity per setting (independent of pair) ----
    print("\n[p1] computing WikiText perplexity per setting...")
    for setting_name, hook_factory in SETTINGS:
        t1 = time.time()
        loss = wikitext_loss_under_hook(model, tok, wt,
                                         hook_factory(d_device), L)
        ppl = float(np.exp(loss))
        results["wikitext_loss"][setting_name] = {
            "loss": loss, "ppl": ppl,
            "elapsed_sec": round(time.time() - t1, 1),
        }
        print(f"[p1]   {setting_name:<12} loss={loss:.4f}  ppl={ppl:.2f}  "
              f"({time.time() - t1:.0f}s)", flush=True)

    # Random-proj WikiText (3 draws → mean)
    rand_losses = []
    for r_idx, rd_t in enumerate(rand_devices):
        loss = wikitext_loss_under_hook(model, tok, wt,
                                         make_layer_hook(rd_t, "proj_out"), L)
        rand_losses.append(loss)
        print(f"[p1]   random_proj_{r_idx} loss={loss:.4f}  ppl={np.exp(loss):.2f}",
              flush=True)
    results["wikitext_loss"]["random_proj_out_mean"] = {
        "loss_mean": float(np.mean(rand_losses)),
        "ppl_mean":  float(np.mean(np.exp(rand_losses))),
        "n_random": len(rand_losses),
    }

    # ---- 9. write results ----
    out_dir = REPO / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.out_name}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[p1] wrote {out_path}")


if __name__ == "__main__":
    main()

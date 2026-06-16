"""Phase 2O — N-sweep over top-N |cos|-ranked cell resample, with KL divergence.

For N ∈ {4, 8, 16, 32}, jointly resample the top-N cells from the Phase 2M
cosine grid and measure two metrics per N:
  1. Δr(LD, z): correlation drop on the {tall, short} readout axis
  2. mean KL(baseline || resampled) on the full last-token output distribution

KL is computed per prompt and averaged. KL is in nats. Picks up readout
shifts that the scalar r(LD, z) metric misses.

For comparison, also runs a same-N random control (5 trials each).

Output:
  results/p2o_n_sweep_<short>.json
  figures/p2o_n_sweep_<short>.png

Compute: 1 baseline + (~4 + 4×3 random) resample passes ≈ 4-5 min on 2B.

Usage:
  python p2o_n_sweep.py --short gemma2-2b --ns 4 8 16 32 --n-random 3
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parent.parent


def get_decoder_layers(model):
    for path in [("model", "layers"), ("model", "model", "layers")]:
        m = model
        ok = True
        for attr in path:
            if hasattr(m, attr):
                m = getattr(m, attr)
            else:
                ok = False; break
        if ok and hasattr(m, "__getitem__"):
            return m
    raise RuntimeError("could not locate decoder layers")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--short", required=True, choices=["gemma2-2b", "gemma2-9b"])
    ap.add_argument("--ns", type=int, nargs="+", default=[4, 8, 16, 32])
    ap.add_argument("--n-random", type=int, default=3,
                    help="random control trials per N")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sign", choices=["both", "positive", "negative"],
                    default="both",
                    help="filter cells by cosine sign before ranking")
    ap.add_argument("--score-source",
                    choices=["cos", "cos_sigma", "dla", "resid_cos"],
                    default="cos",
                    help="rank by |cos| (Phase 2M SAE-restricted), "
                         "|cos·σ| (Phase 2Q), |σ_DLA · ρ(DLA,z)| (Phase 2S "
                         "full DLA grid), or |cos(Δh_L, d_primal)| (Phase 2U "
                         "residual-cos full grid)")
    ap.add_argument("--out-suffix", default=None,
                    help="suffix on output JSON / PNG filename")
    ap.add_argument("--eval-feature", default=None,
                    help="if set, override the stimulus feature (head ranking "
                         "still comes from the cos/DLA grid). Use to evaluate "
                         "k=15-derived heads on a different feature/k.")
    ap.add_argument("--eval-k", type=int, default=None,
                    help="if set, override the stimulus k. Pairs with "
                         "--eval-feature; defaults to grid's feature/k.")
    args = ap.parse_args()

    cos_path_full = REPO / "results" / f"p2m_alignment_{args.short}_fullgrid.json"
    cos_path = (cos_path_full if cos_path_full.exists()
                else REPO / "results" / f"p2m_alignment_{args.short}.json")
    print(f"using Phase 2M cos: {cos_path.name}")
    cos_data = json.loads(cos_path.read_text())
    cos = np.array(cos_data["cos_grid"])
    layer_idxs = cos_data["layer_idxs"]
    head_idxs = cos_data["head_idxs"]
    feature = cos_data["feature"]
    k = cos_data["k"]
    model_id = cos_data["model"]

    # Optional σ from Phase 2Q for cos·σ ranking
    if args.score_source == "cos_sigma":
        zinfo_path = REPO / "results" / f"p2q_head_zinfo_{args.short}.json"
        if not zinfo_path.exists():
            raise SystemExit(f"--score-source cos_sigma requires {zinfo_path.name} "
                              f"(run scripts/p2q_head_zinfo.py first)")
        zinfo = json.loads(zinfo_path.read_text())
        sigma_grid_full = np.array(zinfo["sigma_grid"])  # (n_layers, n_heads)
        sigma = np.zeros_like(cos)
        for li, L in enumerate(layer_idxs):
            for hi, h in enumerate(head_idxs):
                sigma[li, hi] = sigma_grid_full[L, h]
        rank_value = cos * sigma  # signed cos·σ
        print(f"σ range over selected (L,h): "
              f"[{sigma.min():.3f}, {sigma.max():.3f}]")
    elif args.score_source == "dla":
        dla_path = REPO / "results" / f"p2s_dla_full_{args.short}.npz"
        if not dla_path.exists():
            raise SystemExit(f"--score-source dla requires {dla_path.name} "
                              f"(run scripts/p2s_dla_full_grid.py first)")
        dla = np.load(dla_path)
        std_dla_full = dla["std_dla"]   # (n_layers_full, n_heads_full)
        rho_full = dla["rho_grid"]
        n_layers_dla, n_heads_dla = std_dla_full.shape
        # Override grid to use full DLA range (Phase 2M may not cover all layers)
        layer_idxs = list(range(n_layers_dla))
        head_idxs = list(range(n_heads_dla))
        cos = rho_full              # 'cos' = ρ(DLA, z) for sign filter / annotation
        sigma = std_dla_full        # 'sigma' = σ_DLA for magnitude
        rank_value = cos * sigma    # signed σ_DLA · ρ(DLA, z)
        print(f"DLA mode: covers full grid {n_layers_dla}L x {n_heads_dla}H")
        print(f"  σ_DLA range: [{sigma.min():.3f}, {sigma.max():.3f}]")
        print(f"  ρ(DLA, z) range: [{cos.min():+.3f}, {cos.max():+.3f}]")
    elif args.score_source == "resid_cos":
        rc_path = REPO / "results" / f"p2u_residual_cos_{args.short}.npz"
        if not rc_path.exists():
            raise SystemExit(f"--score-source resid_cos requires {rc_path.name} "
                              f"(run scripts/p2u_residual_cos.py first)")
        rc = np.load(rc_path)
        cos_full = rc["cos_grid"]   # (n_layers_full, n_heads_full)
        n_layers_rc, n_heads_rc = cos_full.shape
        layer_idxs = list(range(n_layers_rc))
        head_idxs = list(range(n_heads_rc))
        cos = cos_full              # signed cos(Δh_L, d_primal)
        sigma = None
        rank_value = cos
        print(f"resid_cos mode: covers full grid {n_layers_rc}L x {n_heads_rc}H")
        print(f"  cos range: [{cos.min():+.3f}, {cos.max():+.3f}]")
    else:
        sigma = None
        rank_value = cos

    # Build score with sign mask
    if args.sign == "positive":
        mask = cos > 0
        score = np.where(mask, rank_value, -np.inf)
    elif args.sign == "negative":
        mask = cos < 0
        score = np.where(mask, -rank_value, -np.inf)
    else:
        score = np.abs(rank_value)
    max_n = max(args.ns)
    top_flat = np.argsort(score, axis=None)[::-1][:max_n]
    selected_cells: list[tuple[int, int, float, float, float]] = []
    for kf in top_flat:
        li, hi = np.unravel_index(kf, cos.shape)
        c = float(cos[li, hi])
        if not np.isfinite(score[li, hi]):
            continue
        sg = float(sigma[li, hi]) if sigma is not None else 0.0
        cs = c * sg
        selected_cells.append((layer_idxs[li], head_idxs[hi], c, sg, cs))
    if len(selected_cells) < max_n:
        print(f"WARN: only {len(selected_cells)} cells satisfy sign={args.sign}, "
              f"capping max_n")
        max_n = len(selected_cells)
        args.ns = [N for N in args.ns if N <= max_n]

    if args.score_source == "cos_sigma":
        rank_label = "cos·σ"
    elif args.score_source == "dla":
        rank_label = "σ_DLA · ρ(DLA,z)"
    elif args.score_source == "resid_cos":
        rank_label = "cos(Δh_L, d_primal)"
    else:
        rank_label = "|cos|"
    print(f"top-{max_n} cells by {rank_label}:")
    for L, h, c, sg, cs in selected_cells:
        if args.score_source == "cos_sigma":
            print(f"  L{L:>2d} H{h}: cos={c:+.3f}  σ={sg:.3f}  cos·σ={cs:+.4f}")
        elif args.score_source == "dla":
            print(f"  L{L:>2d} H{h}: ρ={c:+.3f}  σ_DLA={sg:.3f}  product={cs:+.4f}")
        else:
            print(f"  L{L:>2d} H{h}: cos={c:+.3f}")

    # Stimuli — by default use grid's (feature, k), but allow override so the
    # k=15-derived head ranking can be evaluated on k=1/5/50 prompts.
    grid_feature, grid_k = feature, k
    eval_feature = args.eval_feature or grid_feature
    eval_k = args.eval_k if args.eval_k is not None else grid_k
    stim_path = REPO / "data" / "p2_shot_sweep" / f"{eval_feature}_k{eval_k}.jsonl"
    if (eval_feature, eval_k) != (grid_feature, grid_k):
        print(f"\nstimulus override: grid=({grid_feature}, k={grid_k}) -> "
              f"eval=({eval_feature}, k={eval_k})")
    rows = [json.loads(l) for l in stim_path.open()]
    n = len(rows)
    z_arr = np.array([float(r.get("z_eff", r.get("z", 0))) for r in rows],
                       dtype=np.float32)
    x_arr = np.array([float(r.get("x", 0)) for r in rows], dtype=np.float32)
    print(f"loaded {n} prompts from {stim_path.name}")
    feature, k = eval_feature, eval_k

    # Model
    print(f"\nloading {model_id}...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_id, token=os.environ.get("HF_TOKEN"))
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="auto",
        token=os.environ.get("HF_TOKEN")).eval()
    print(f"  loaded in {time.time()-t0:.1f}s")
    layers = get_decoder_layers(model)
    n_layers = len(layers)
    n_heads = layers[0].self_attn.config.num_attention_heads
    head_dim = layers[0].self_attn.o_proj.in_features // n_heads
    high_w = rows[0].get("high_word", "tall")
    low_w = rows[0].get("low_word", "short")
    high_id = tok.encode(" " + high_w, add_special_tokens=False)[-1]
    low_id = tok.encode(" " + low_w, add_special_tokens=False)[-1]
    vocab = model.config.vocab_size
    print(f"  vocab={vocab}, n_layers={n_layers}, n_heads={n_heads}")

    # Identify all layers that might be needed (selected + random pool from any layer)
    pool_layers = list(range(n_layers))  # need all layers for random control
    print(f"\n[pass 1] baseline + capture pre-o_proj at all {n_layers} layers + log-probs")
    pool = np.zeros((n_layers, n, n_heads * head_dim), dtype=np.float32)
    captured = {}
    def make_capture(L):
        def hook(module, args_):
            captured[L] = args_[0].detach().float().cpu().numpy()
        return hook
    handles = [layers[L].self_attn.o_proj.register_forward_pre_hook(make_capture(L))
                for L in pool_layers]

    # Baseline log-probs stored on CPU as float16 for memory
    baseline_logp = np.zeros((n, vocab), dtype=np.float16)
    ld_baseline = np.zeros(n, dtype=np.float32)
    t1 = time.time()
    with torch.inference_mode():
        for i, row in enumerate(rows):
            inp = tok(row["prompt"], return_tensors="pt").to(model.device)
            out = model(**inp, use_cache=False)
            logits = out.logits[0, -1].float()
            log_p = F.log_softmax(logits, dim=-1)
            baseline_logp[i] = log_p.cpu().numpy().astype(np.float16)
            ld_baseline[i] = float(logits[high_id] - logits[low_id])
            for L in pool_layers:
                pool[L, i] = captured[L][0, -1]
            if (i + 1) % 200 == 0 or i == n - 1:
                print(f"  {i+1}/{n}  {(i+1)/max(1e-3, time.time()-t1):.1f} p/s",
                      flush=True)
    for h in handles:
        h.remove()
    base_r = float(np.corrcoef(ld_baseline, z_arr)[0, 1])
    base_r_x = float(np.corrcoef(ld_baseline, x_arr)[0, 1])
    print(f"  baseline r(LD,z) = {base_r:+.3f}  r(LD,x) = {base_r_x:+.3f}  "
          f"baseline_logp size={baseline_logp.nbytes/1e9:.2f} GB")

    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(n)

    def resample_run(cells: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
        """Runs forward with resample on cells, returns (LD per prompt, mean KL)."""
        by_layer: dict = {}
        for L, h in cells:
            by_layer.setdefault(L, []).append(h)
        handles = []
        state = {"i": 0}
        for L, hs in by_layer.items():
            heads_t = tuple(hs)
            def make_hook(L_=L, heads_=heads_t):
                def hook(module, args_):
                    x = args_[0].clone()
                    j = perm[state["i"]]
                    for h in heads_:
                        src = pool[L_, j, h * head_dim:(h + 1) * head_dim]
                        src_t = torch.tensor(src, dtype=x.dtype, device=x.device)
                        x[:, -1, h * head_dim:(h + 1) * head_dim] = src_t
                    return (x,) + args_[1:]
                return hook
            handles.append(layers[L].self_attn.o_proj
                            .register_forward_pre_hook(make_hook()))
        ld = np.zeros(n, dtype=np.float32)
        kls = np.zeros(n, dtype=np.float32)
        try:
            with torch.inference_mode():
                for i, row in enumerate(rows):
                    state["i"] = i
                    inp = tok(row["prompt"], return_tensors="pt").to(model.device)
                    out = model(**inp, use_cache=False)
                    logits = out.logits[0, -1].float()
                    log_p_r = F.log_softmax(logits, dim=-1)
                    ld[i] = float(logits[high_id] - logits[low_id])
                    # KL(baseline || resample) = sum_v p_b * (log p_b - log p_r)
                    log_p_b = torch.tensor(baseline_logp[i].astype(np.float32),
                                             device=model.device)
                    p_b = log_p_b.exp()
                    kl = (p_b * (log_p_b - log_p_r)).sum().item()
                    kls[i] = float(kl)
        finally:
            for h in handles:
                h.remove()
        return ld, kls

    # Selected runs per N
    print(f"\n[selected resample sweep over N ∈ {args.ns}]")
    selected_results = []
    for N in args.ns:
        cells_n = [(L, h) for L, h, _c, _sg, _cs in selected_cells[:N]]
        t1 = time.time()
        ld, kls = resample_run(cells_n)
        r = float(np.corrcoef(ld, z_arr)[0, 1])
        r_x = float(np.corrcoef(ld, x_arr)[0, 1])
        print(f"  N={N:>2d}: r(LD,z)={r:+.3f}  r(LD,x)={r_x:+.3f}  "
              f"Δr={r-base_r:+.3f}  KL_mean={kls.mean():.3f} nats  "
              f"({time.time()-t1:.0f}s)")
        selected_results.append({
            "N": N, "r": r, "r_x": r_x,
            "delta_r": r - base_r, "delta_r_x": r_x - base_r_x,
            "kl_mean": float(kls.mean()), "kl_std": float(kls.std()),
            "kl_p50": float(np.percentile(kls, 50)),
            "kl_p90": float(np.percentile(kls, 90)),
            "ld_mean": float(ld.mean()), "ld_std": float(ld.std(ddof=1)),
        })

    # Random controls per N
    print(f"\n[random resample sweep over N ∈ {args.ns}, {args.n_random} trials each]")
    all_cells = [(L, h) for L in range(n_layers) for h in range(n_heads)]
    random_results = []
    for N in args.ns:
        trials = []
        for trial in range(args.n_random):
            idx = rng.choice(len(all_cells), size=N, replace=False)
            cells_n = [all_cells[i] for i in idx]
            t1 = time.time()
            ld, kls = resample_run(cells_n)
            r = float(np.corrcoef(ld, z_arr)[0, 1])
            trials.append({
                "trial": trial, "r": r, "delta_r": r - base_r,
                "kl_mean": float(kls.mean()),
            })
            print(f"  N={N:>2d} trial {trial+1}/{args.n_random}: "
                  f"Δr={r-base_r:+.3f}  KL_mean={kls.mean():.3f}  "
                  f"({time.time()-t1:.0f}s)")
        rs = [t["delta_r"] for t in trials]
        kls_m = [t["kl_mean"] for t in trials]
        random_results.append({
            "N": N, "trials": trials,
            "delta_r_mean": float(np.mean(rs)),
            "delta_r_std": float(np.std(rs)),
            "kl_mean_mean": float(np.mean(kls_m)),
            "kl_mean_std": float(np.std(kls_m)),
        })

    out = {
        "model": model_id, "short": args.short,
        "feature": feature, "k": k, "n_prompts": int(n),
        "baseline_r_LD_z": base_r, "baseline_r_LD_x": base_r_x,
        "baseline_LD_mean": float(ld_baseline.mean()),
        "selected_cells": [{"layer": L, "head": h, "cos": c,
                             "sigma": sg, "cos_sigma": cs}
                            for L, h, c, sg, cs in selected_cells],
        "score_source": args.score_source,
        "selected_runs": selected_results,
        "random_runs": random_results,
    }
    suffix = args.out_suffix or args.sign
    base_name = f"p2o_n_sweep_{args.short}"
    if suffix and suffix != "both":
        base_name += f"_{suffix}"
    out_path = REPO / "results" / f"{base_name}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")

    # Plot — 2-panel: Δr vs N, KL_mean vs N
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    Ns = [r["N"] for r in selected_results]

    ax = axes[0]
    ax.axhline(0, color="black", lw=0.5, ls="--", alpha=0.5)
    sel_dr = [r["delta_r"] for r in selected_results]
    ax.plot(Ns, sel_dr, "-o", color="C0", lw=2, ms=10,
             label="|cos|-ranked top-N selected")
    rand_dr = [r["delta_r_mean"] for r in random_results]
    rand_dr_std = [r["delta_r_std"] for r in random_results]
    ax.errorbar(Ns, rand_dr, yerr=rand_dr_std, fmt="-s", color="gray",
                 alpha=0.7, label=f"random N-cell (mean±std, {args.n_random} trials)")
    ax.set_xscale("log")
    ax.set_xticks(Ns)
    ax.set_xticklabels([str(N) for N in Ns])
    ax.set_xlabel("N cells resampled")
    ax.set_ylabel("Δcorr(LD, z)")
    ax.set_title(f"{args.short} {feature} k={k}\nΔcorr(LD,z) vs N — selected vs random")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    ax = axes[1]
    sel_kl = [r["kl_mean"] for r in selected_results]
    ax.plot(Ns, sel_kl, "-o", color="C0", lw=2, ms=10,
             label="|cos|-ranked top-N")
    rand_kl = [r["kl_mean_mean"] for r in random_results]
    rand_kl_std = [r["kl_mean_std"] for r in random_results]
    ax.errorbar(Ns, rand_kl, yerr=rand_kl_std, fmt="-s", color="gray",
                 alpha=0.7, label=f"random N-cell")
    ax.set_xscale("log")
    ax.set_xticks(Ns)
    ax.set_xticklabels([str(N) for N in Ns])
    ax.set_xlabel("N cells resampled")
    ax.set_ylabel("mean KL(baseline || resample)  (nats)")
    ax.set_title("Output-distribution disruption (full vocab)")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out_png = REPO / "figures" / f"{base_name}.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()

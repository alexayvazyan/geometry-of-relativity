"""Phase 2M — SAE-alignment grid: head ablation × manifold steering cosine match.

Idea (Alex, 2026-05-01): the L1 single-feature / single-head ablations were
causally null vs r(LD,z), but that's a brittle metric. Replace with a
geometric question: does ablating head h at layer L' produce the *same kind*
of representational change at the readout that manifold steering produces?

Pipeline (per model):
  1. Forward 990 height k=15 prompts; capture residual at steering_layer
     (used to build manifold delta) and readout_layer (used to SAE-encode).
  2. Compute per-prompt manifold delta Δ_z(prompt) = mean(cell with z=0, same x)
     − mean(cell at this z, same x), in residual space at steering_layer.
  3. Three forward passes per condition:
       (a) baseline — capture readout SAE features
       (b) manifold — apply Δ at steering_layer, capture readout SAE features
       (c) per-(h, L') — zero head h's slice in z at layer L', capture readout SAE
     where L' ranges over layers 0 .. steering_layer (upstream only).
  4. Δ_manifold = mean[manifold] − mean[baseline]  (in SAE feature space)
     Δ_ablate(h, L') = mean[ablate(h, L')] − mean[baseline]
     cos_grid(h, L') = cos_sim(Δ_ablate(h, L'), Δ_manifold)

Output:
  results/p2m_alignment_<short>.json
  results/p2m_alignment_<short>_deltas.npz (per-cell Δ vectors)
  figures/p2m_alignment_<short>.png (n_layers × n_heads heatmap)

Compute (full grid, height k=15, n=990):
  - 2B (steer@L20, readout@L20, layers 0-19 × 8 heads = 160 cells): ~40 min
  - 9B (steer@L30, readout@L30, 30 × 16 = 480 cells): too long;
    use --layer-stride 2 --head-stride 2 to halve each dim → ~60 min.

Usage:
  python p2m_sae_alignment_grid.py --short gemma2-2b --steer-layer 20
  python p2m_sae_alignment_grid.py --short gemma2-9b --steer-layer 30 \\
      --layer-stride 2 --head-stride 2 --n-prompts 400
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
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parent.parent


def get_decoder_layers(model):
    for path in [("model", "layers"), ("model", "model", "layers"),
                  ("model", "language_model", "layers")]:
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


def load_res_sae(short: str, layer: int, width: str, l0: int) -> dict:
    repo = ("google/gemma-scope-2b-pt-res" if short.startswith("gemma2-2b")
             else "google/gemma-scope-9b-pt-res")
    fname = f"layer_{layer}/width_{width}/average_l0_{l0}/params.npz"
    print(f"  SAE: {repo}/{fname}", flush=True)
    path = hf_hub_download(repo, fname, token=os.environ.get("HF_TOKEN"))
    npz = np.load(path)
    return {k: npz[k].astype(np.float32) for k in npz.files}


def encode_jr(x: np.ndarray, sae: dict) -> np.ndarray:
    """JumpReLU encode: x (N,d) → post (N,F)."""
    pre = x @ sae["W_enc"] + sae["b_enc"]
    return pre * (pre > sae["threshold"]).astype(np.float32)


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                    help="HF id; auto-resolved from --short if omitted")
    ap.add_argument("--short", required=True,
                    choices=["gemma2-2b", "gemma2-9b"])
    ap.add_argument("--feature", default="height")
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--steer-layer", type=int, required=True,
                    help="layer where manifold delta is applied; readout SAE same layer")
    ap.add_argument("--sae-width", default="16k")
    ap.add_argument("--sae-l0", type=int, default=None,
                    help="default 71 for 2B/L20, 66 for 9B/L30")
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="manifold steering strength")
    ap.add_argument("--layer-stride", type=int, default=1,
                    help="subsample ablation layers (e.g. 2 for every other)")
    ap.add_argument("--head-stride", type=int, default=1,
                    help="subsample heads")
    ap.add_argument("--n-prompts", type=int, default=None,
                    help="cap n stimuli (default all 990)")
    ap.add_argument("--n-train-cells", type=int, default=3,
                    help="cell_seeds < n_train_cells used to build manifold lookup")
    ap.add_argument("--out-name", default=None)
    args = ap.parse_args()

    if args.model is None:
        args.model = ("google/gemma-2-2b" if args.short == "gemma2-2b"
                       else "google/gemma-2-9b")
    if args.sae_l0 is None:
        args.sae_l0 = 71 if args.short == "gemma2-2b" else 66

    out_name = args.out_name or f"p2m_alignment_{args.short}"
    out_json = REPO / "results" / f"{out_name}.json"
    out_npz = REPO / "results" / f"{out_name}_deltas.npz"
    out_png = REPO / "figures" / f"{out_name}.png"

    # 1. Stimuli
    stim_path = REPO / "data" / "p2_shot_sweep" / f"{args.feature}_k{args.k}.jsonl"
    rows = [json.loads(l) for l in stim_path.open()]
    if args.n_prompts:
        rows = rows[:args.n_prompts]
    n = len(rows)
    print(f"loaded {n} prompts from {stim_path.name}")

    # 2. Model + SAE
    print(f"\nloading {args.model}...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model, token=os.environ.get("HF_TOKEN"))
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto",
        token=os.environ.get("HF_TOKEN")).eval()
    print(f"  loaded in {time.time()-t0:.1f}s")
    layers = get_decoder_layers(model)
    n_layers = len(layers)
    L_steer = args.steer_layer
    if L_steer >= n_layers:
        raise SystemExit(f"steer_layer {L_steer} >= n_layers {n_layers}")
    n_heads = layers[L_steer].self_attn.config.num_attention_heads
    head_dim = layers[L_steer].self_attn.o_proj.in_features // n_heads
    print(f"  n_layers={n_layers}  n_heads={n_heads}  head_dim={head_dim}")
    high_w = rows[0].get("high_word", "tall")
    low_w = rows[0].get("low_word", "short")
    high_id = tok.encode(" " + high_w, add_special_tokens=False)[-1]
    low_id = tok.encode(" " + low_w, add_special_tokens=False)[-1]
    print(f"  LD readout: '{high_w}' − '{low_w}'  (ids {high_id}, {low_id})")

    sae = load_res_sae(args.short, L_steer, args.sae_width, args.sae_l0)
    n_feats = sae["W_enc"].shape[1]
    d_res = sae["W_enc"].shape[0]
    print(f"  SAE: W_enc={sae['W_enc'].shape}, threshold mean={sae['threshold'].mean():.3f}")

    # 3. Pass 1 — capture residual at L_steer (last token) for baseline
    captured_res = {"x": None}
    def cap_res(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        captured_res["x"] = h[0, -1].detach().float().cpu().numpy()
        return output
    h_cap = layers[L_steer].register_forward_hook(cap_res)

    res_baseline = np.zeros((n, d_res), dtype=np.float32)
    z_arr = np.zeros(n, dtype=np.float32)
    x_arr = np.zeros(n, dtype=np.float32)
    cell_seeds = np.zeros(n, dtype=np.int32)
    ld_baseline = np.zeros(n, dtype=np.float32)

    print(f"\n[pass 1] baseline forward × {n}, capture res@L{L_steer}...")
    t1 = time.time()
    with torch.inference_mode():
        for i, row in enumerate(rows):
            inp = tok(row["prompt"], return_tensors="pt").to(model.device)
            out = model(**inp, use_cache=False)
            res_baseline[i] = captured_res["x"]
            ld_baseline[i] = float(out.logits[0, -1, high_id].float()
                                    - out.logits[0, -1, low_id].float())
            z_arr[i] = float(row.get("z_eff", row.get("z", 0)))
            x_arr[i] = float(row["x"])
            cell_seeds[i] = int(row.get("cell_seed", 0))
            if (i + 1) % 200 == 0 or i == n - 1:
                rate = (i + 1) / max(1e-3, time.time() - t1)
                print(f"  {i+1}/{n}  {rate:.1f} p/s", flush=True)
    h_cap.remove()
    base_r = float(np.corrcoef(ld_baseline, z_arr)[0, 1])
    print(f"  baseline r(LD, z) = {base_r:+.3f}  <LD>={ld_baseline.mean():+.2f}")

    # 4. Build manifold delta from train fold (cell_seeds < n_train)
    train_mask = cell_seeds < args.n_train_cells
    res_tr = res_baseline[train_mask]
    z_tr = z_arr[train_mask]
    x_tr = x_arr[train_mask]
    cells: dict = defaultdict(list)
    for i in np.where(train_mask)[0]:
        key = (round(float(x_arr[i]), 2), round(float(z_arr[i]), 2))
        cells[key].append(res_baseline[i])
    lookup = {k: np.stack(v).mean(0) for k, v in cells.items()}
    by_x: dict = defaultdict(list)
    for (xk, zk), v in lookup.items():
        by_x[xk].append((zk, v))
    delta = np.zeros((n, d_res), dtype=np.float32)
    n_missing = 0
    for i in range(n):
        xi = round(float(x_arr[i]), 2)
        zi = round(float(z_arr[i]), 2)
        same = by_x.get(xi, [])
        if not same:
            n_missing += 1
            continue
        _, mu_target = min(same, key=lambda kv: abs(kv[0]))
        _, mu_source = min(same, key=lambda kv: abs(kv[0] - zi))
        delta[i] = mu_target - mu_source
    print(f"  manifold deltas: ||Δ|| mean={np.linalg.norm(delta, axis=1).mean():.2f}  "
          f"missing={n_missing}/{n}")

    # 5. SAE-encode baseline residuals → mean baseline feature vector
    sae_baseline = encode_jr(res_baseline, sae)
    base_mean = sae_baseline.mean(0)
    print(f"  baseline SAE: avg_l0={(sae_baseline > 0).sum(1).mean():.1f}, "
          f"n_active={(sae_baseline > 0).any(0).sum()}")

    # 6. Pass 2 — manifold-steered forward
    # IMPORTANT: register manifold_hook FIRST so it modifies output, then
    # cap_res fires next and captures the *modified* output. Hooks run in
    # registration order.
    captured_res["x"] = None
    state = {"i": 0, "delta_t": None}
    def manifold_hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        h = h.clone()
        h[:, -1, :] = h[:, -1, :] + state["delta_t"]
        if isinstance(output, tuple):
            return (h,) + output[1:]
        return h
    h_man = layers[L_steer].register_forward_hook(manifold_hook)
    h_cap = layers[L_steer].register_forward_hook(cap_res)

    res_manifold = np.zeros((n, d_res), dtype=np.float32)
    print(f"\n[pass 2] manifold forward × {n}, α={args.alpha}...")
    t1 = time.time()
    with torch.inference_mode():
        for i, row in enumerate(rows):
            state["delta_t"] = torch.tensor(args.alpha * delta[i],
                                              dtype=torch.bfloat16,
                                              device=model.device)
            inp = tok(row["prompt"], return_tensors="pt").to(model.device)
            _ = model(**inp, use_cache=False)
            res_manifold[i] = captured_res["x"]
            if (i + 1) % 200 == 0 or i == n - 1:
                rate = (i + 1) / max(1e-3, time.time() - t1)
                print(f"  {i+1}/{n}  {rate:.1f} p/s", flush=True)
    h_man.remove()
    h_cap.remove()
    sae_manifold = encode_jr(res_manifold, sae)
    manifold_mean = sae_manifold.mean(0)
    delta_manifold = manifold_mean - base_mean
    delta_manifold_norm = float(np.linalg.norm(delta_manifold))
    print(f"  Δ_manifold ||={delta_manifold_norm:.3f}  "
          f"top |Δ| feats: "
          f"{np.argsort(-np.abs(delta_manifold))[:5].tolist()}")

    # 7. Pass 3 — per-(h, L') ablation
    layer_idxs = list(range(0, L_steer + 1, args.layer_stride))
    head_idxs = list(range(0, n_heads, args.head_stride))
    n_cells = len(layer_idxs) * len(head_idxs)
    print(f"\n[pass 3] {n_cells} ablation cells "
          f"({len(layer_idxs)} layers × {len(head_idxs)} heads)")

    cos_grid = np.full((len(layer_idxs), len(head_idxs)), np.nan, dtype=np.float32)
    delta_norms = np.zeros_like(cos_grid)
    delta_per_cell = np.zeros((len(layer_idxs), len(head_idxs), n_feats), dtype=np.float32)

    captured_res["x"] = None
    h_cap = layers[L_steer].register_forward_hook(cap_res)

    abl_state = {"head": None, "layer": None}
    def make_ablate_hook(target_layer_idx):
        def hook(module, args_):
            x = args_[0]
            xm = x.clone()
            sl_lo = abl_state["head"] * head_dim
            sl_hi = (abl_state["head"] + 1) * head_dim
            xm[:, -1, sl_lo:sl_hi] = 0
            return (xm,) + args_[1:]
        return hook

    ablate_handles = []
    t_grid = time.time()
    for li, L_abl in enumerate(layer_idxs):
        # Register hook for THIS layer's o_proj only
        hook = make_ablate_hook(L_abl)
        h_abl = layers[L_abl].self_attn.o_proj.register_forward_pre_hook(hook)
        for hi, h_idx in enumerate(head_idxs):
            abl_state["head"] = h_idx
            abl_state["layer"] = L_abl
            res_abl = np.zeros((n, d_res), dtype=np.float32)
            t1 = time.time()
            with torch.inference_mode():
                for i, row in enumerate(rows):
                    inp = tok(row["prompt"], return_tensors="pt").to(model.device)
                    _ = model(**inp, use_cache=False)
                    res_abl[i] = captured_res["x"]
            sae_abl = encode_jr(res_abl, sae)
            d_a = sae_abl.mean(0) - base_mean
            cos_grid[li, hi] = cos_sim(d_a, delta_manifold)
            delta_norms[li, hi] = float(np.linalg.norm(d_a))
            delta_per_cell[li, hi] = d_a
            elapsed_total = time.time() - t_grid
            done = li * len(head_idxs) + hi + 1
            eta = elapsed_total / done * (n_cells - done)
            print(f"  L{L_abl} H{h_idx}: cos={cos_grid[li, hi]:+.3f}  "
                  f"||Δ_a||={delta_norms[li, hi]:.3f}  "
                  f"({time.time()-t1:.0f}s)  "
                  f"[{done}/{n_cells}, total elapsed {elapsed_total/60:.1f}m, eta {eta/60:.1f}m]",
                  flush=True)
        h_abl.remove()
    h_cap.remove()

    # 8. Save
    results = {
        "model": args.model, "short": args.short,
        "feature": args.feature, "k": args.k,
        "steer_layer": L_steer, "alpha": args.alpha,
        "sae": {"width": args.sae_width, "l0": args.sae_l0,
                 "n_features": int(n_feats)},
        "n_prompts": int(n), "baseline_r_LD_z": base_r,
        "delta_manifold_norm": delta_manifold_norm,
        "layer_idxs": [int(x) for x in layer_idxs],
        "head_idxs": [int(x) for x in head_idxs],
        "cos_grid": cos_grid.tolist(),
        "delta_norms": delta_norms.tolist(),
    }
    out_json.write_text(json.dumps(results, indent=2))
    np.savez(out_npz,
             cos_grid=cos_grid,
             delta_norms=delta_norms,
             delta_per_cell=delta_per_cell.astype(np.float16),
             delta_manifold=delta_manifold.astype(np.float32),
             base_mean=base_mean.astype(np.float32),
             layer_idxs=np.array(layer_idxs, dtype=np.int32),
             head_idxs=np.array(head_idxs, dtype=np.int32))
    print(f"\nwrote {out_json}")
    print(f"wrote {out_npz}")

    # 9. Plot
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.4 * len(layer_idxs) + 2))
    ax = axes[0]
    vmax = max(0.05, float(np.nanmax(np.abs(cos_grid))))
    im = ax.imshow(cos_grid, aspect="auto", cmap="RdBu_r",
                    vmin=-vmax, vmax=+vmax)
    ax.set_xticks(np.arange(len(head_idxs)))
    ax.set_xticklabels([f"H{h}" for h in head_idxs], fontsize=8)
    ax.set_yticks(np.arange(len(layer_idxs)))
    ax.set_yticklabels([f"L{l}" for l in layer_idxs], fontsize=8)
    ax.set_xlabel("head")
    ax.set_ylabel("ablated layer")
    ax.set_title(f"cos(Δ_ablate(h, L'), Δ_manifold) — readout SAE@L{L_steer}\n"
                  f"red = ablation removes manifold direction\n"
                  f"blue = ablation produces opposite direction")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04, label="cosine")

    ax = axes[1]
    im = ax.imshow(delta_norms, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(head_idxs)))
    ax.set_xticklabels([f"H{h}" for h in head_idxs], fontsize=8)
    ax.set_yticks(np.arange(len(layer_idxs)))
    ax.set_yticklabels([f"L{l}" for l in layer_idxs], fontsize=8)
    ax.set_xlabel("head")
    ax.set_ylabel("ablated layer")
    ax.set_title(f"||Δ_ablate(h, L')||  (effect size in SAE space)\n"
                  f"||Δ_manifold||={delta_manifold_norm:.2f} for reference")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04, label="L2 norm")

    fig.suptitle(f"{args.short} — Phase 2M SAE-alignment grid "
                  f"({args.feature} k={args.k}, α={args.alpha}, "
                  f"steer/readout L{L_steer}, n={n})", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()

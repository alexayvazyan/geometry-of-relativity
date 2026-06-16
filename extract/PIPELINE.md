# From-scratch data pipeline (GPU)

The committed `circuit/results/` and `results/` JSON/npz are the **cached output of this pipeline**.
`make figures` replots from them on CPU; this tier *regenerates* them with the model. It needs a GPU
(an H100, or an RTX 5090 — Gemma-2-9B fits in 32 GB at `--batch-size 8`) and the gated Gemma weights
(accept the license on the HF model page, or use a local cache).

**Verified.** On a 5090: `gen_p2_shot_sweep` reproduces the committed prompts byte-identical (990/990),
and `p2_extract_ld --model gemma2-9b` reproduces the committed per-prompt LD with **corr 0.999**
(small bf16 nondeterminism). `make extract` runs exactly those two steps.

The extraction scripts are self-contained here: `circuit/scripts/_token_utils.py` is vendored, and
`circuit/scripts/extract_v4_adjpairs.py` is a shim re-exporting the unified `data_gen/prompts.py`.

## DAG

All commands run from the repo root; outputs land in `circuit/{data,results}/`.

**0. Prompts** (CPU) — from `data_gen/prompts.py`:
```
python circuit/scripts/gen_p2_shot_sweep.py --pairs height weight speed --k 0 1 2 5 15 --n-seeds 3 --n-x 20 --n-z 20
```

**1. Logit differences** (GPU) → `circuit/results/p2_ld/<model>/<pair>_k<k>.npz` — feeds Fig p2a, p2d:
```
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python circuit/scripts/p2_extract_ld.py --model gemma2-9b --pairs height weight speed --k 0 1 2 5 15 --batch-size 8
```
Steps 0–1 are wrapped by `make extract`.

**2. Attention / residual caches** (GPU) — `p2_extract_attn.py`; feeds the DLA and ablation steps.

**3. Interventions** (GPU) → the circuit result JSON/npz the §5 figures consume:
- `p2c_ablate_heads.py`, `p2s_dla.py` / `p2s_dla_full_grid.py` → DLA scores, `p2s_dla_full_*` (ranking + fingerprint figures)
- `p2u_n_sweep_xfeat.py` → `p2u_n_sweep_xfeat_*.json` + `p2u_residual_cos_*.npz` (trio figure)
- `p2o_attention_modes.py` / `p2o_n_sweep.py` → `p2o_*` (intervention-modes figure)
- `p2u_specificity.py` + `p2u_aggregate.py` → `p2u_specificity_*` (specificity battery)
- `p2v_steering_with_arclength.py` and the `p2f_*alpha_sweep` / `p2v_xk_*` runners → steering trajectory + xk grids

Each takes `--short gemma2-9b` (and `--feature`/`--k` where relevant); see the script's `--help`.

**4. Behavioral tier (v9–v15).** The `order` / `distribution` / `fig5_layer` / `shared_direction` /
`x_transfer` JSON under `results/` were produced by the v9–v15 behavioral pipeline (J. Lee). Those
runner scripts are not consolidated here; the committed summaries are their output, and `make figures`
replots them. To regenerate from scratch, run that pipeline on the dense grid (`gen_v11_dense` →
forward pass → the v14/v15 analysis runners).

## Cost

The LD core (`make extract`, 3 pairs × 5 k) is ~minutes. The full intervention sweep across all
pairs, the specificity tasks, and the 5 cross-models is a multi-hour H100 session — which is why the
derived results (small JSON + npz) are committed so the figures reproduce on CPU.

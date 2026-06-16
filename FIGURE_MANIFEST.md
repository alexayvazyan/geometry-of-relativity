# Figure & table manifest

Every float in `paper/main.tex` mapped to the asset it embeds, the source artifact it is
built from, the code that generates it, and the reproduction tier. Source artifacts are taken
from the `% Source ...` comments in `paper/main.tex` (authoritative).

**Tiers**
- **STATIC** — committed asset, no computation (hand-made schematic or a spec table).
- **CACHED** — regenerated on CPU from a committed JSON results artifact (`make figures`).
- **GPU** — requires a model forward pass / intervention to (re)produce the results, then a plot
  (`make extract` + plot). Needs an H100 (or the local 5090 for Gemma-2-9B); cached intermediates
  are fetched from Hugging Face so the plot step still works without a GPU.

Origin root: **G** = `geometry-of-relativity/` (behavioral + geometry), **A** = `relativity_ablation/`
(attention circuit), now consolidated here.

**Drivers (run by `make figures`):** `figures/build_paper_result_figures.py` regenerates the
behavioral `fig_results_*`; `figures/build_circuit_figures.py` regenerates the section-5 `p2*` and
`xmodel` figures from `circuit/results/`. Committed-asset exceptions (not regenerated): the
`candidate_5` pipeline schematic (hand-made), `p2a_ld_vs_z_height_gemma2-9b_2x3.png` (phase
pipeline), and `fig_results_shared_direction_own_loo_abs_slopes_z_vs_x.png` (filename drifted from
its `run_v15` generator).

| Paper float | Asset (`paper/figures/…`) | Source artifact | Generating code | Tier | Root |
|---|---|---|---|---|---|
| Fig 1 `setup-pipeline` | `candidates_png/candidate_5_combined_pipeline_relcontrast.png` | hand-made schematic | — | STATIC | — |
| Tab 1 `domains` | — | domain specs | `config/models.py::DOMAINS` | STATIC | G |
| Fig 2 `dense-height-heatmap` | `fig_results_dense_height_heatmap_clean.png` | dense (x,z) grid LDs (height) | `analyze/` dense behavioral + clean replot | CACHED | G |
| Tab 2 `behavioral-z-corr` | — | dense grid corr(LD,z), corr(LD,x) ×8 | `analyze/` dense behavioral | CACHED | G |
| Fig 3 `relative-objective-phase` | `p2d_phase_k_sweep_9b.png` | `internal/kshot/phase/results/p2d_l0all_per_k_gemma2-9b_height.json`, `p2d_partial_l0_…_k1.json`, `p2e_residual_interventions_…_k15.json` | `plot_p2d_phase_k_sweep.py` | CACHED | G |
| Fig 4 `order-main` / `order-robustness-full` | `fig_results_order_ld_by_z_{main,clean}.png` | `results/v14_1/order/order_rows.jsonl` | order replot | CACHED | G |
| Fig 5 `distribution-shapes` / `distribution-robustness-full` | `fig_results_distribution_{bimodal_main,ld_by_z_clean}.png` | `results/v14/distribution/distribution_rows.jsonl` | distribution replot | CACHED | G |
| Fig 6 `layer-encode-use` | `fig_results_layer_z_x_encode_use_clean.png` | `results/v14_1/fig5/fig5_layer_x_z_metrics.json` | layer-metrics replot | CACHED | G |
| Fig 7 `shared-direction-steering` | `fig_results_shared_direction_own_loo_abs_slopes_z_vs_x.png` | `results/v15/shared_direction_loo_z_vs_x.json` | shared-direction replot | CACHED | G |
| Fig (app) `z-vs-x-transfer` | `fig_results_z_vs_x_transfer_clean.png` | `results/v13/x_transfer/cross_pair_transfer_x_8x8.json` + `…_summary.json` | transfer-matrix replot | CACHED | G |
| Fig `kshot-ld-evolution` | `p2a_ld_vs_z_height_gemma2-9b_2x3.png` | p2a shot-sweep LDs | `gen_p2_shot_sweep.py` + `analyze_p2a.py` | GPU | A |
| Fig `circuit-phase-trajectory` | `p2v_steering_with_arclength_gemma2-9b_height.png` | p2v steering trajectory | `p2v_steering_with_arclength.py` | GPU | A |
| Fig `circuit-trio` | `p2u_n_sweep_xfeat_gemma2-9b_dcorr.png` | p2u greedy N-sweep Δcorr | `p2u_n_sweep_xfeat.py` | GPU | A |
| Fig `circuit-spec` | `p2u_specificity_summary_gemma2-9b_dr.png` | p2u specificity battery | `p2u_specificity.py` + `…_summary.py` | GPU | A |
| Fig `kl` | `p2u_n_sweep_xfeat_gemma2-9b_kl.png` | p2u KL curves | `p2u_n_sweep_xfeat.py` | GPU | A |
| Fig `intervention-modes` | `p2o_attention_modes_comparison_gemma2-9b.png` | p2o zero/resample/query-zero | `p2o_attention_modes_comparison.py` | GPU | A |
| Fig `ranking-comparison` | `p2s_dla_vs_cossigma_sweep_gemma2-9b.png` | p2s DLA vs cos·σ ranking | `p2s_dla_vs_cossigma_plot_9b.py` | GPU | A |
| Fig `fingerprint` | `p2s_dla_xfeat_summary_gemma2-9b.png` | p2s cross-feature DLA heads | `p2s_dla_xfeat_summary.py` | GPU | A |
| Fig `xk-corr-grid-9b-height` | `p2v_xk_corr_grid_gemma2-9b_height.png` | p2v k∈{1,5,15} phase plane | `p2v_xk_corr_grid.py` | GPU | A |
| Fig `xk-deltar-overlay-9b-height` | `p2v_xk_deltar_overlay_gemma2-9b_height.png` | p2v Δcorr vs N across k | `p2v_xk_delta_r_overlay.py` | GPU | A |
| Fig `xmodel-geometry` / `xmodel-interventions` | `xmodel_{geometry,interventions}_grid.png` | 5-model PCA + intervention grids | `p2v_xmodel_appendix_composites.py` | GPU | A |

**Commented-out in current draft (not built):** `pca-montage` (`fig_results_pca_all_pairs_clean.png` ←
`figures/v11/pca/*_2d_L33.png`), `lexical-residual` (`fig_results_lexical_transfer_summary_clean.png` ←
`results/v12_2/residual_vs_lexical_transfer_summary.json`).

**Note on "clean" figures:** several behavioral figures are marked "cleaned and relabeled for paper
display" in `main.tex`. The CACHED plot scripts regenerate the *content* from the JSON artifacts; the
cosmetic relabel step is tracked separately so paper-identical output is reproducible.

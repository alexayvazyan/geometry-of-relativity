# The Geometry of Relativity — code & reproducibility

Code, data artifacts, and the paper source for **"The Geometry of Relativity: Context-Relative
Scalar Representations in LLMs."** When given in-context examples that define a comparison class,
LLMs judge gradable adjectives (*tall/short*, *heavy/light*, …) by a context-normalized standing
`z = (x − μ)/σ` rather than the raw value `x`. This `z` is linearly decodable, causally steerable,
and partly shared across concepts; a small set of attention heads is causally responsible for it.

> **Status: consolidation in progress.** This repo merges two research codebases
> (`geometry-of-relativity/` — behavioral + geometry; `relativity_ablation/` — attention circuit)
> into one reproducible package. The paper builds (`paper/`) and the bibliography is audited; code
> porting and the `make figures` pipeline are being wired up. See `FIGURE_MANIFEST.md` for the
> figure→code→artifact map and per-figure reproduction tier.

## Layout

```
config/      canonical model IDs, seeds, and the 8 adjective-domain specs (Table 1)
data_gen/    prompt construction (prompts.py) and trial generation
extract/     model forward passes + activation/attention caching (GPU)
analyze/     probes, behavioral readouts, DLA, resample/manifold ablations
figures/     one script per paper figure (reads results/ → writes paper/figures/)
results/     committed JSON artifacts (small); bulk caches fetched from Hugging Face
paper/       buildable LaTeX (main.tex, references.bib, icml2026 style, figures/)
```

## Reproduce

Three tiers (see `FIGURE_MANIFEST.md` for which figure is which):

1. **Paper PDF** — `cd paper && tectonic main.tex` (builds with the audited bibliography).
2. **`make figures` (CPU, no model)** — regenerates the behavioral/geometry figures from committed
   JSON results. This is the verifiable core.
3. **`make extract` (GPU)** — re-runs the Gemma-2-9B forward passes and interventions behind the
   circuit figures, then plots. Needs an H100 (or a local 5090 for the 9B model). Cached
   intermediates are pulled from Hugging Face so tier 2 works without a GPU.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .            # CPU analysis + plotting
pip install -e ".[gpu]"     # + torch/transformers for extraction
cp .env.example .env        # add HF_TOKEN (gated Gemma weights + cached activations)
```

## Models

Primary: **`google/gemma-2-9b`**. Cross-model replication: Qwen2.5-3B, Llama-3.2-3B, OLMo-2-7B,
Pythia-2.8B, Qwen3-14B. Canonical IDs in `config/models.py`.

## Provenance & license

Consolidated from joint work with Jaehoon Lee. Upstream behavioral/geometry development history lives
at `github.com/jaehoonlee0829/geometry-of-relativity`. MIT licensed (see `LICENSE`).

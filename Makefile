PY ?= python3

.PHONY: help figures extract paper fetch clean

help:
	@echo "make figures - regenerate cached-tier figures from results/ (CPU, no model)"
	@echo "make extract - FROM SCRATCH (GPU): regenerate prompts + LD arrays on gemma-2-9b"
	@echo "make paper   - build paper/main.pdf with tectonic"
	@echo "make clean   - remove LaTeX build artifacts"
	@echo "(full from-scratch pipeline incl. interventions: see extract/PIPELINE.md)"

# Runs every figure script. Each reads results/ and writes into paper/figures/.
figures:
	@for f in figures/*.py; do [ -e "$$f" ] || continue; echo ">> $$f"; $(PY) "$$f" || exit 1; done

# From-scratch DATA tier (GPU, gemma-2-9b). Regenerates the prompts and the per-prompt LD arrays
# behind the p2a / p2d figures, overwriting circuit/{data,results}/. Verified to reproduce the
# committed data (byte-identical prompts; LD corr 0.999). The full intervention pipeline that
# produces every circuit result JSON is documented in extract/PIPELINE.md.
EXTRACT_PAIRS ?= height weight speed
EXTRACT_KS ?= 0 1 2 5 15
extract:
	$(PY) circuit/scripts/gen_p2_shot_sweep.py --pairs $(EXTRACT_PAIRS) --k $(EXTRACT_KS) --n-seeds 3 --n-x 20 --n-z 20
	PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $(PY) circuit/scripts/p2_extract_ld.py \
		--model gemma2-9b --pairs $(EXTRACT_PAIRS) --k $(EXTRACT_KS) --batch-size 8

paper:
	cd paper && tectonic main.tex

clean:
	rm -f paper/main.pdf paper/main.aux paper/main.bbl paper/main.blg paper/main.log paper/main.out paper/build_run.txt

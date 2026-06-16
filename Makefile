PY ?= python

.PHONY: help figures extract paper fetch clean

help:
	@echo "make figures - regenerate cached-tier figures from results/ (CPU, no model)"
	@echo "make extract - run GPU forward passes + interventions (needs .[gpu] + a GPU)"
	@echo "make paper   - build paper/main.pdf with tectonic"
	@echo "make fetch   - pull bulk activation/attention caches from Hugging Face"
	@echo "make clean   - remove LaTeX build artifacts"

# Runs every figure script. Each reads results/ and writes into paper/figures/.
figures:
	@for f in figures/*.py; do [ -e "$$f" ] || continue; echo ">> $$f"; $(PY) "$$f" || exit 1; done

# GPU extraction + intervention scripts (produce the JSON artifacts the circuit figures need).
extract:
	@for f in extract/*.py; do [ -e "$$f" ] || continue; echo ">> $$f"; $(PY) "$$f" || exit 1; done

paper:
	cd paper && tectonic main.tex

fetch:
	$(PY) extract/fetch_from_hf.py

clean:
	rm -f paper/main.pdf paper/main.aux paper/main.bbl paper/main.blg paper/main.log paper/main.out paper/build_run.txt

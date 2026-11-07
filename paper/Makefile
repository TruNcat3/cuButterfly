SHELL := /bin/sh
.DEFAULT_GOAL := all

BUILD_DIR ?= build
TECTONIC ?=
PYTHON ?= python3
CAMPAIGN ?= ../results/full_current_20261025
FP16_ACCEPTANCE ?= ../results/a100_fp16_acceptance_20261105
REFERENCES := $(wildcard references.bib)
SECTIONS := $(wildcard sections/*.tex)
GENERATED := $(wildcard generated/*.tex generated/current_refresh/*.tex generated/fft_l_20261104/*.tex)
FIGURES := $(wildcard figures/*.pdf figures/appt/*.pdf figures/draft/*.pdf figures/drawio/exports/*.pdf)

.PHONY: all evidence drawio-figures refresh-data refresh-figures refresh

# Reproduce the committed plots without access to the research host's results.
refresh-figures:
	$(PYTHON) tools/render_current_refresh.py --figures-only --frozen-inputs

# Explicit refresh reads a live campaign once, then renders a stable snapshot.
# Ordinary PDF builds never resample a campaign running in the background.
refresh-data:
	$(PYTHON) tools/export_current_refresh.py --campaign "$(CAMPAIGN)" --output generated/current_refresh/snapshot.json
	$(PYTHON) tools/render_current_refresh.py $(if $(FP16_ACCEPTANCE),--fp16-acceptance "$(FP16_ACCEPTANCE)")

refresh: refresh-data
	$(MAKE) all evidence

drawio-figures:
	$(PYTHON) tools/export_drawio_figures.py

all: $(BUILD_DIR)/main.pdf
	@set -e; if ! cmp -s "$(BUILD_DIR)/main.pdf" main.pdf; then \
	  reader_copy=$$(mktemp .main.pdf.XXXXXX); \
	  cp "$(BUILD_DIR)/main.pdf" "$$reader_copy"; \
	  mv -f "$$reader_copy" main.pdf; \
	fi

$(BUILD_DIR)/main.pdf: main.tex IEEEtran.cls IEEEtran.bst build.sh $(REFERENCES) $(SECTIONS) $(GENERATED) $(FIGURES)
	@BUILD_DIR="$(BUILD_DIR)" TECTONIC="$(TECTONIC)" ./build.sh

evidence: $(BUILD_DIR)/main-evidence.pdf
	@set -e; if ! cmp -s "$(BUILD_DIR)/main-evidence.pdf" main-evidence.pdf; then \
	  reader_copy=$$(mktemp .main-evidence.pdf.XXXXXX); \
	  cp "$(BUILD_DIR)/main-evidence.pdf" "$$reader_copy"; \
	  mv -f "$$reader_copy" main-evidence.pdf; \
	fi

$(BUILD_DIR)/main-evidence.pdf: main-evidence.tex main.tex IEEEtran.cls IEEEtran.bst build.sh $(REFERENCES) $(SECTIONS) $(GENERATED) $(FIGURES)
	@SOURCE_TEX=main-evidence.tex BUILD_DIR="$(BUILD_DIR)" TECTONIC="$(TECTONIC)" ./build.sh

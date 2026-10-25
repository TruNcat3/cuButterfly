SHELL := /bin/sh
.DEFAULT_GOAL := all

BUILD_DIR ?= build
TECTONIC ?=
PYTHON ?= python3
REFERENCES := $(wildcard references.bib)
SECTIONS := $(wildcard sections/*.tex)
GENERATED := $(wildcard generated/*.tex)
FIGURES := $(wildcard figures/*.pdf figures/appt/*.pdf figures/draft/*.pdf figures/drawio/exports/*.pdf)

.PHONY: all evidence drawio-figures

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

TEX_FILES := $(sort $(wildcard [0-9][0-9]_*.tex))
PDF_FILES := $(TEX_FILES:.tex=.pdf)
BUILD_DIR := .build
LATEXMK ?= latexmk
SOURCE_DATE_EPOCH ?= 946684800
FORCE_SOURCE_DATE ?= 1
export SOURCE_DATE_EPOCH
export FORCE_SOURCE_DATE

.PHONY: slides check clean

slides: $(PDF_FILES)

%.pdf: %.tex
	@mkdir -p $(BUILD_DIR)
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error -outdir=$(BUILD_DIR) $<
	cp $(BUILD_DIR)/$@ $@

check: slides
	python ../test/verify_tutorial.py

clean:
	@for source in $(TEX_FILES); do \
		$(LATEXMK) -C -outdir=$(BUILD_DIR) "$$source" >/dev/null 2>&1 || true; \
	done
	rm -rf $(BUILD_DIR)

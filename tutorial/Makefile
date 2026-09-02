TEX_FILES := $(sort $(wildcard [0-9][0-9]_*.tex))
PDF_FILES := $(TEX_FILES:.tex=.pdf)
BUILD_DIR := .build
LATEXMK ?= latexmk
SOURCE_DATE_EPOCH ?= 946684800
FORCE_SOURCE_DATE ?= 1
export SOURCE_DATE_EPOCH
export FORCE_SOURCE_DATE

.PHONY: slides presentations presentations-html presentations-pdf \
        presentations-presenter check clean

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

# --- Quarto presentations (issue #322 pilot) -------------------------------
#
# One .qmd source, three outputs. Nothing here is committed: the rendered decks
# are build products, produced on demand and by CI. See presentations/README.md.
PRESENTATIONS_DIR := presentations
QUARTO ?= quarto
PILOT ?= 01_getting_started_with_vaft.qmd
SHOW_NOTES := _extensions/vaft/vaftslides/show-notes.tex

# Order matters twice over. `--output` renames the render rather than adding a
# second file, so the presenter build must run before the archival one or it
# would leave only the presenter PDF behind; and both Beamer targets write the
# same intermediate .tex in the same directory, so they must not run at once.
#
# Order-only prerequisites (|) enforce both under `make -j` as well as serially.
# Listing the targets in order is not enough: they are phony and independent, so
# a parallel make is free to interleave them.
presentations: presentations-html presentations-presenter presentations-pdf
presentations-presenter: | presentations-html
presentations-pdf: | presentations-presenter

presentations-html:
	cd $(PRESENTATIONS_DIR) && $(QUARTO) render $(PILOT) --to vaftslides-revealjs

presentations-pdf:
	cd $(PRESENTATIONS_DIR) && $(QUARTO) render $(PILOT) --to vaftslides-beamer

# Speaker notes are authored once, in ::: {.notes}. Beamer hides them by
# default, which is what the archival PDF wants; this build shows them.
presentations-presenter:
	cd $(PRESENTATIONS_DIR) && $(QUARTO) render $(PILOT) --to vaftslides-beamer \
		--include-in-header=$(SHOW_NOTES) \
		--output $(basename $(PILOT))-presenter.pdf

# Every session's slides are one Quarto source rendering to three outputs
# (issue #322). Nothing rendered is committed: the decks are build products,
# produced on demand and by CI. See presentations/README.md.
PRESENTATIONS_DIR := presentations
QUARTO ?= quarto
DECKS := $(sort $(notdir $(wildcard $(PRESENTATIONS_DIR)/[0-9][0-9]_*.qmd)))
SHOW_NOTES := _extensions/vaft/vaftslides/show-notes.tex

.PHONY: presentations presentations-html presentations-pdf \
        presentations-presenter check clean

# Order matters twice over. `--output` renames the render rather than adding a
# second file, so the presenter build must run before the archival one or it
# would leave only the presenter PDF behind; and both Beamer targets write the
# same intermediate .tex in the same directory, so they must not run at once.
#
# Order-only prerequisites (|) enforce both under `make -j` as well as serially,
# and each recipe loops the decks in sequence for the same reason: two decks
# rendering to Beamer at once would collide the same way.
presentations: presentations-html presentations-presenter presentations-pdf
presentations-presenter: | presentations-html
presentations-pdf: | presentations-presenter

presentations-html:
	@for deck in $(DECKS); do \
		echo "--- revealjs: $$deck"; \
		(cd $(PRESENTATIONS_DIR) && $(QUARTO) render "$$deck" --to vaftslides-revealjs) || exit 1; \
	done

presentations-pdf:
	@for deck in $(DECKS); do \
		echo "--- beamer: $$deck"; \
		(cd $(PRESENTATIONS_DIR) && $(QUARTO) render "$$deck" --to vaftslides-beamer) || exit 1; \
	done

# Speaker notes are authored once, in ::: {.notes}. Beamer hides them by
# default, which is what the archival PDF wants; this build shows them.
presentations-presenter:
	@for deck in $(DECKS); do \
		echo "--- presenter: $$deck"; \
		(cd $(PRESENTATIONS_DIR) && $(QUARTO) render "$$deck" --to vaftslides-beamer \
			--include-in-header=$(SHOW_NOTES) \
			--output "$${deck%.qmd}-presenter.pdf") || exit 1; \
	done

check:
	python ../test/verify_tutorial.py

clean:
	rm -rf $(PRESENTATIONS_DIR)/.quarto $(PRESENTATIONS_DIR)/_freeze
	rm -f $(PRESENTATIONS_DIR)/[0-9][0-9]_*.html $(PRESENTATIONS_DIR)/[0-9][0-9]_*.pdf
	rm -f $(PRESENTATIONS_DIR)/[0-9][0-9]_*.tex
	rm -rf $(PRESENTATIONS_DIR)/[0-9][0-9]_*_files

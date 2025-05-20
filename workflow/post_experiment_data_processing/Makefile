# Makefile
.PHONY: run clean openlogs

TIMESTAMP := $(shell date +%Y%m%d_%H%M%S)
LOG_DIR   := logs
LOG_FILE  := $(LOG_DIR)/snakemake_$(TIMESTAMP).log

CONFIG    := config.yaml

run:
	mkdir -p $(LOG_DIR)
	snakemake --cores 30 --configfile $(CONFIG) 2>&1 | tee $(LOG_FILE)
	
clean:
	snakemake --cores 1 --delete-all-output

openlogs:
	xdg-open $(LOG_DIR)
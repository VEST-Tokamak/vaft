# Makefile
.PHONY: run clean openlogs

TIMESTAMP := $(shell date +%Y%m%d_%H%M%S)
LOG_DIR   := logs
LOG_FILE  := $(LOG_DIR)/snakemake_$(TIMESTAMP).log
CORES     := 30


CONFIG    := config.yaml

run:
	mkdir -p $(LOG_DIR)
	snakemake --cores $(CORES) --configfile $(CONFIG) 2>&1 | tee $(LOG_FILE)
	rm -rf .snakemake

clean:
	snakemake --cores 1 --delete-all-output
	rm -rf .snakemake
	
openlogs:
	xdg-open $(LOG_DIR)
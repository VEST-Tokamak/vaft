#!/usr/bin/env python3
# -*- coding: utf-8 -*-
###############################################################################
#  snakemake_templates.py
#
#  This file contains templates to quickly write Snakemake pipelines:
#  1) "5-line script" template            (Snakemake → script:)
#  2) Rule template that calls the script (input / output / log / script)
#  3) Top-level rule all example
#
#  ❑ How to use
#    ① Copy needed blocks from this file → modify paths/function names for your project
#    ② Paste Rule blocks into Snakefile and define BASE_DIR·SHOTS variables
#    ③ Create 5-line script(.py) file according to script: path
###############################################################################

# ─────────────────────────────────────────────────────────────────────────────
# 1) "5-line script" template
#    - Used when executed by Snakemake's `script:` directive
#    - Log file is automatically connected to rule's log: entry
# ─────────────────────────────────────────────────────────────────────────────
"""
# snake_scripts/my_task.py  (5 lines excluding blank lines and comments)
"""
from vaft.code import init_snakemake_logger; 
from vaft.function import my_task_function

logger = init_snakemake_logger(snakemake)                   # ← log file + console
rc = my_task_function(int(snakemake.wildcards.shot),               # 입력 인자 ①
               snakemake.input[0],                          # 입력 인자 ②
               snakemake.output[0])                         # 출력 경로
if rc: logger.error("my_worker returned %s", rc); raise ValueError("my_worker failed")  # 실패 감지


# ─────────────────────────────────────────────────────────────────────────────
# 2) Snakemake Rule template
#    - Modify patterns like {base_dir}, {shot} according to your project
# ─────────────────────────────────────────────────────────────────────────────
"""
# ----- Snakefile -------------------------------------------------------------

rule my_task:
    # (1) Input file path
    input:
        lambda wc: f"{BASE_DIR}/{wc.shot}/raw/{wc.shot}.dat",
    # (2) Output file path
    output:
        lambda wc: f"{BASE_DIR}/{wc.shot}/results/{wc.shot}.json",
    # (3) Log file path (console logs are also redirected here)
    log:
        lambda wc: f"{BASE_DIR}/{wc.shot}/logs/my_task.log",
    # (4) Python script to be executed (5-line script)
    script:
        "snake_scripts/my_task.py"

# ---------------------------------------------------------------------------
"""


# ─────────────────────────────────────────────────────────────────────────────
# 3) Top-level rule all example
#    - Used when processing multiple shots (SHOTS) at once
# ─────────────────────────────────────────────────────────────────────────────
"""
# ----- Snakefile -------------------------------------------------------------

# Define final outputs: results/*.json must exist for all shots to complete workflow
rule all:
    input:
        expand("{base_dir}/{shot}/results/{shot}.json",
               base_dir=BASE_DIR,
               shot=SHOTS)

# ---------------------------------------------------------------------------
"""

###############################################################################
#  End. Copy and modify functions/paths as needed!
###############################################################################
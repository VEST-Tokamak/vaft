#!/bin/bash
# /home/user1/h5pyd/vaft/workflow/running_linear_stability/snakemake_worker.sh

# 스크립트 실행 중 오류 발생 시 즉시 중단
set -e

# snakemake 실행
snakemake --cores 28
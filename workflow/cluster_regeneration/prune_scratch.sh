#!/bin/bash
# prune_scratch.sh <lane> <chunk> <chunk shots file> <lane filedb>   (after_chunk hook)
# Solver scratch: DCON's euler.bin and the vacuum vmat.bin are ~98 % of the GPEC tree (~5 GB per
# MHD shot) and only feed later solves of the same chunk, all finished now. They are not
# Snakemake outputs and no stage product reads them, but a later rerun of an ideal-GPEC cell
# alone needs its DCON cell rerun first (euler.bin). Size, mtime and path stay in pruned.tsv
# (PRUNE_HASH=1 adds sha256). PRUNE_SOLVER_SCRATCH=2 also drops the rest of the solver work
# trees once the stage products are built; 0 keeps everything.
set -uo pipefail
LANE=${1:?lane}; C=${2:?chunk}; SHOTS=${3:?chunk shots file}; FILEDB=${4:?lane filedb}
[ "${PRUNE_SOLVER_SCRATCH:-1}" = 0 ] && exit 0
while read -r s; do
  [ -z "$s" ] && continue
  if [ "${PRUNE_SOLVER_SCRATCH:-1}" = 2 ]; then
    find "$FILEDB"/gpec/magnetic/chease/*/"$s"/n=*/work "$FILEDB/chease/magnetic/$s/output/work" \
      -type f ! -name '*.log' -printf "%s\t%TY-%Tm-%Td %TH:%TM\t%p\n" 2>/dev/null
  else
    find "$FILEDB"/gpec/magnetic/chease/*/"$s"/n=*/work -type f \( -name euler.bin -o -name vmat.bin \) \
      -printf "%s\t%TY-%Tm-%Td %TH:%TM\t%p\n" 2>/dev/null
  fi | while IFS=$'\t' read -r size mtime path; do
      sha=-; [ "${PRUNE_HASH:-0}" = 1 ] && sha=$(sha256sum < "$path" | cut -d' ' -f1)
      printf "%s\t%s\t%s\t%s\t%s\n" "$C" "$size" "$mtime" "$sha" "${path#"$FILEDB"/}" >> "$LANE/pruned.tsv"
      rm -f "$path"
    done
done < "$SHOTS"
exit 0

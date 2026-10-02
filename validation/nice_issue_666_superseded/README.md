# NICE issue #666 validation

NICE revision `7ad1ea8f3da4fee25a61a7c2c01b1773db5f4906` was built from the
local clone at `/Users/yun/git/nice` with AppleClang 21, Eigen 3 and
SuiteSparse. Its upstream `test_recon` CTest passed. The build force-included
`vaft/code/nice/upstream_compat.h` and `<cassert>`; no NICE source file was
modified.

The three full local reference windows were executed from the canonical
`pipeline-until-efit.json.gz` snapshots: 39915 (9 slices), 41524 (6 slices),
and 41672 (19 slices). NICE returned without a valid native equilibrium for
all 34 slices. The JSON summaries record every failed time and the exact input,
geometry, diagnostic, passive-current, source, build and solver provenance.
The trace figures show the EFIT reference and mark each failed NICE slice with
a red cross instead of silently dropping it.

The required 41672 at 331 ms focused run was also executed. This requested
time is not an EFIT output time in the local snapshot; 333 ms is the closest
reference slice. The full magnetic case exits zero but NICE rejects the plasma
on its magnetic-cost validity gate and writes no equilibrium tables. Reduced
families with only plasma current, flux loops, or B-pol terminate inside NICE
with status -11; diamagnetic flux is recorded as disabled because the pinned
standalone NICE objective has no diamagnetic constraint. Consequently a strict
LCFS/profile/scalar comparison cannot be made without misrepresenting invalid
solver output; this fact is recorded in `nice_41672_331ms_families.json`.

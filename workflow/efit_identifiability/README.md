# EFIT native constraint-identifiability study (#664)

This workflow uses EFIT's exported response matrix to determine which VEST
constraint families add independent information. It does not scan profile
forms and it does not alter routine EFIT defaults.

The five mandatory slices are:

| shot | time [ms] | role |
|---:|---:|---|
| 41672 | 331 | stable quasi-stationary negative control |
| 41672 | 342 | high native-condition stress case |
| 41672 | 347 | diamagnetic `beta_p`/`li` response case |
| 39915 | 319 | confirmation |
| 41524 | 332 | confirmation |

## Provenance gate

The clean control was built from EFIT commit `4d10ed5` in Release mode with
GNU Fortran 15.2 and NetCDF enabled. It passed 44/44 CTests. For the five
reference slices, its archived-#663 a-file content after the run-date line and
its m-files agree, and printed physics/outcomes are unchanged. The archived
a-files are not byte-identical because their first run-date line differs. The
separate same-date comparison between the clean control and the instrumented
build with diagnostics disabled is the one for which g-, a-, and m-files are
bit-for-bit identical.

Keep native instrumentation in a separate clean EFIT worktree. Every study
case is content-addressed by its executable, frozen k-file, table identity,
analysis settings, direction file, and restart parent. A completed result may
be resumed only when all hashes agree.

## Gated execution

Stage 1 runs exactly one frozen k-file per EFIT process and requires a complete
native main-solve and external-current sidecar. All five cases must reproduce
the native solve and pass the schema, curvature, family-share, condition, and
m-file diagnostic-chi-squared checks before Stage 2 is authorized. Before any
run, the workflow parses each k-file and proves the `(2,2)` zero-edge profile,
qualified seed, numerical controls, unit objective scales, active constraint
families, soft PF-relation rows, shot/time, and literal `TABLE_DIR`. A same-name
variant k-file is therefore rejected before it can inherit the frozen
scientific hash.

Stage 2 contains independent positive and negative directional chains and a
bidirectional diamagnetic path from `10^0` through `10^4` in 0.25-dex steps.
Every non-root case hashes and names its exact `esave.dat` parent; a failure is
never bridged. Transition intervals are bisected to at most 0.025 dex, while a
solver failure receives at most three refinements and two deterministic
repeats from the last valid restart.

VAFT also passes a native restart-control SHA-256 made from the original,
unstaged one-k-file bytes plus the direction-control bytes (or an explicit
null marker). It is computed before VAFT changes `ICINIT` or the `IOUT` restart
bit. EFIT may therefore restore saved stopping history only for unchanged
controls; a changed `FWTDLC` or direction still restores the physical state
but cannot reuse a stale objective history.

Generate a pending, content-addressed plan after collecting the five frozen
#663 k-files:

```bash
PYTHONPATH=$PWD python workflow/efit_identifiability/identifiability_study.py plan \
  --output /scratch/efit-identifiability \
  --kfile-root /scratch/efit-constraint-information \
  --executable /path/to/instrumented/efit \
  --scientific-sha256 SHA256_FROM_663 \
  --table-identity /path/to/table_identity.json \
  --build-provenance /path/to/build_provenance.json
```

The build record is required to contain `source_base_revision=4d10ed5`,
`source_base_is_ancestor=true`, the instrumented `source_revision` commit,
`source_dirty=false`, `build_type`, compiler identity, `cmake_options`,
`install_manifest_sha256`, nested control/instrumented CTest counts with no
additional failures, and the five-slice control comparison. The comparison
records printed-precision/outcome agreement with archived #663 and a separate
bit-identical g-/a-/m-file comparison between the same-date control and
diagnostics-disabled instrumented build. The executable identity and hash are
added by the workflow; table and k-file hashes are embedded separately in
every case. `table_identity.json` must include `sources_by_shot`, mapping each
of `41672`, `39915`, and `41524` to an object with the table `path`; this binds
each k-file's literal `TABLE_DIR` even though the primary and confirmation
shots use different archived table roots.

Run and analyze Stage 1 with the same arguments:

```bash
PYTHONPATH=$PWD python workflow/efit_identifiability/identifiability_study.py \
  execute-stage1 --output /scratch/efit-identifiability \
  --kfile-root /scratch/efit-constraint-information \
  --executable /path/to/instrumented/efit \
  --scientific-sha256 SHA256_FROM_663 \
  --table-identity /path/to/table_identity.json \
  --build-provenance /path/to/build_provenance.json
```

Then materialize and execute the native restart gate. A non-passing proof
returns status 3 and leaves all continuation commands blocked:

```bash
PYTHONPATH=$PWD python workflow/efit_identifiability/identifiability_study.py \
  plan-stage2 --output /scratch/efit-identifiability
PYTHONPATH=$PWD python workflow/efit_identifiability/identifiability_study.py \
  execute-restart-proof --output /scratch/efit-identifiability
```

Only after that proof passes, run the bidirectional diamagnetic path:

```bash
PYTHONPATH=$PWD python workflow/efit_identifiability/identifiability_study.py \
  execute-diamagnetic --output /scratch/efit-identifiability
```

The instrumented build also accepts native directional equalities. Run the
selected weak/borderline profile, PF, and diamagnetic-information modes with:

```bash
PYTHONPATH=$PWD python workflow/efit_identifiability/identifiability_study.py \
  execute-directional --output /scratch/efit-identifiability
```

Every mode first runs a mandatory warm-started `t=0` control. The native
constrained solve must preserve EFIT's baseline singular-value truncation while
adding exactly one target equality and leaving orthogonal parameters free. The
nonzero chain remains blocked unless the zero target reproduces the baseline
within ten times repeat noise.

Old #663 results normally have a different executable hash. Reproduce the
needed ablation and strength endpoints with the instrumented executable, then
finalize the evidence matrix:

```bash
PYTHONPATH=$PWD python workflow/efit_identifiability/identifiability_study.py \
  execute-663-reproduction --output /scratch/efit-identifiability
PYTHONPATH=$PWD python workflow/efit_identifiability/identifiability_study.py \
  finalize --output /scratch/efit-identifiability
```

`merge-663-evidence --input PATH` is available when an earlier result really
does match the executable, tables, scientific hash, and all five baseline
k-files. A mismatch is never silently mixed into the #664 result.

The workflow writes one atomic `case_manifest.json` per run, including failed
and refinement cases. It never bridges a failure. Branch jumps are localized
to at most 0.025 dex and receive a cold-start replica. JSON, Markdown, and PNG
outputs distinguish normalized diagnostic χ² from the scaled solver objective
and include both the main and external-current PF solves. The native family
diagnostic χ² is summed directly from exported statistical rows and compared
with `saisil`, `saimpi`, `chidflux`, and `chifcc` in the produced m-file. The
gate accepts either `5e-8` relative agreement or a `1e-12` absolute numerical
floor; Ip and soft PF relations are excluded from that comparison for the
explicit accounting reasons below.

Directional controls use `efit_direction_constraint_v1`: the file binds the
real baseline sidecar and parameter ordering by SHA-256 and carries `theta0`,
the fixed column scale, scaled-coordinate mode, independently checked physical
row, and target. EFIT applies that single row through its true `C,d` equality
block at every nonlinear iteration without promoting discarded modes into the
fit. The workflow never substitutes an external finite-difference direction or
an unrecorded cold start.

`chipasma` remains an accounting diagnostic because it includes fixed
`VCURRT`; it must never be reconstructed from or presented as the minimized Ip
response row.

Generated sidecars, restarts, JSON/Markdown reports, raw EFIT outputs, and PNG
figures remain in the selected output directory; they are not version-controlled.

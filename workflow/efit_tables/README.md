# EFIT Green tables: provenance, regeneration, and the 39915 A/B (issue #194)

EFIT reads its Green-function tables from `TABLE_DIR` and takes every
dimension from the `mhdin.dat` beside them, with no consistency check of its
own. The bundled `vaft/data/efit/` table has no recorded origin: which EFUND
build produced it, from which input, is not recoverable (git shows only a
2026-07 path move). Until this work, nothing in VAFT recorded which table an
equilibrium came from either. The #468 first pass found that under the local
EFIT build no slice of shot 39915 converges (`jflag = 0`) where the stored
pipeline a-file (`a039915.00319`, EFIT of 2023-03-28) has `jflag = 1`, and
the table was the one input that had never been controlled.

This directory holds the three tools that control it, and the record of
what they found for the legacy era shot 39915 belongs to.

| Tool | Does |
| --- | --- |
| `regenerate_legacy_table.py` | `build_static_ods(era)` → `efund_geometry_from_static` → `mhdin.dat` → `efund` → tables + `efund_table_manifest.json`, into a directory of your choosing. Never writes into `vaft/data/efit`. |
| `compare_tables.py` | Two table directories through one parser: namelist counts/grid/flags, per-element geometry, PF groups, then every record of `ep`, `ec`, `rfcoil`, `brzgfc`, `rv`, `re`, plus an inventory with hashes. |
| `ab_efit_table.py` | One constraints ODS, one `efit`, two tables. Refuses to report if the two k-file sets differ in anything but the `INPUT_DIR`/`TABLE_DIR` lines. |

```bash
export EFITHOME=~/git/efit/vaft-install        # one root for efit and efund (install/install_efit.sh)
PYTHONPATH=$PWD python workflow/efit_tables/regenerate_legacy_table.py --output /scratch/tables/legacy-39915
PYTHONPATH=$PWD python workflow/efit_tables/compare_tables.py vaft/data/efit /scratch/tables/legacy-39915 --markdown compare.md
ln -s /scratch/tables/legacy-39915 /scratch/t39915   # EFIT truncates TABLE_DIR at 100 characters
PYTHONPATH=$PWD python workflow/efit_tables/ab_efit_table.py --eddy-ods 39915_eddy.json --shot 39915 \
    --table-a vaft/data/efit/ --table-b /scratch/t39915/ --times 0.316 0.319 0.323 0.325 --output /scratch/ab
```

The eddy ODS is the shot's `pipeline-until-efit` product (the `equilibrium`
IDS is stripped by the tool). `reference/` keeps the manifest of the table
generated for this study and the full comparison record, with local paths
scrubbed.

## The controlled toolchain

Both binaries come from one configure of one revision, built by
`install/install_efit.sh` (see `install/README.md` for the EFIT licence
conditions; VAFT neither bundles nor fetches EFIT):

| | |
| --- | --- |
| source | gitlab `efit-ai/efit`, `4d10ed5` on `codex/macos-test-ordering`, **dirty** (10 modified EFIT sources; `sha256(git diff)` in the build manifest) |
| build | Release, GNU Fortran 15.2.0, NetCDF on, `-fconvert=big-endian` (the toolchain's own default: tables are big-endian) |
| `efit` | sha256 `b1d0bb21c949…` |
| `efund` | sha256 `2d9ff9d7ed5b…` — byte-identical to the July `build-mac` efund, so the July table experiments and this one used the same generator |
| `ctest` | 42/44 pass; `DIIID-esave` and its result check fail (`I/O past end of record` reading `esave.dat`, a restart feature VAFT does not use) |

Three facts about EFUND surfaced while making it run on VEST input, and are
pinned by tests:

- **`islpfc` belongs to `&in3`.** The bundled `mhdin.dat` also lists it under
  `&in5`; this EFUND rejects that group on the unknown name, returns before
  allocating its grid, and segfaults in the plasma-response stage. The bundled
  input therefore cannot have been consumed by this revision as it stands.
- **EFUND needs a large stack.** Its `nvsum×nvsum` work arrays are automatic;
  at 950 vessel segments the child needs the hard limit (just under 64 MB on
  macOS), which `EFUNDConfig(stack_size_kb="hard")` requests.
- **EFIT truncates `TABLE_DIR` at 100 characters** and then fails to open
  `lim.dat`. The A/B tool refuses a longer path; a symlink is the fix.
  `lim.dat` is an EFIT input read from `TABLE_DIR`, not an EFUND product, so a
  generated directory borrows the bundled one.

## Structural comparison: bundled table vs fresh table (129×129)

Generation took 2 minutes. `compare_tables.py` on `vaft/data/efit` (A) and
the fresh table (B):

**Inputs.** Vessel (950 segments), flux loops (11) and probes (64) are the
same geometry to 1e-16, in the same order (which is `em_coupling.passive_loops`
order). The probe angle is spelled `-270°` in A and `+90°` in B, the same
direction. `rsisvs` differs (A carries a resistivity, B the loop resistance);
EFUND does not use it. The F-coil description differs by design: A has one
rectangle per group, B the 302 canonical elements binned into the same 16
groups. Group centroids agree to ≤5 mm and turns to within one element
(PF1 segments 80/80/76/80 vs 79 each; total 632 both), PF5/6/9/10 exactly.

**Tables EFIT reads.**

| file / record | what it is | A vs B |
| --- | --- | --- |
| `ep` rsilpc, rmp2pc | plasma → loops, probes | identical to 5e-10 |
| `ec` gridpc, rgrid/zgrid | plasma → grid, the grid itself | identical to 2e-4 rel max, 6e-7 rms |
| `rv` rsilvs, rmp2vs, gridvs | vessel → loops, probes, grid | identical to 2e-11 |
| `ec` gridfc | F-coils → grid | PF1 groups differ up to 23 %, others ≤ 0.1 % |
| `rfcoil` gsilfc, gmp2fc | F-coils → loops, probes | PF1 groups differ up to 9 % (loops) and 84 % (the two or three inboard probes nearest PF1); others ≤ 1e-4 |

The plasma and vessel responses agree to round-off: the two EFUNDs computed
the same thing on the same grid, and the bundled table's vessel block is not
a suspect. All that differs is the PF1 near field, which is the single
rectangle vs filament discretization.

Records EFIT does not read (`brzgfc`, `rv` gfcvs/gvsvs, `mhdout`) differ
more, harmlessly: `gfcvs` is summed per element without the turn factor in
EFUND, so B/A equals the element count per group (20/12/24); `gvsvs`
off-diagonal entries of 1e-11 flip sign in round-off.

## The A/B on shot 39915

Constraints from the full 39915 eddy product, `fit = 0`, box average
±0.5 ms, slices 0.316/0.319/0.323/0.325 s, `EFITScientificConfig()`; the two
k-file sets differ in exactly the two directory lines (asserted). Same
`efit`, same run for both arms.

| slice | | jflag | lflag | chisq | iterations | GS error | R axis [m] | R LCFS max [m] | β_p | l_i | q95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.316 | A | 0 | 21 | 202.15 | 11 | 8.7e-3 | 0.4273 | 0.692 | 0.0159 | 0.646 | 8.65 |
| | B | 0 | 21 | 202.15 | 11 | 8.9e-3 | 0.4273 | 0.691 | 0.0157 | 0.646 | 8.74 |
| 0.319 | A | 0 | 21 | 74.43 | 11 | 8.5e-3 | 0.3734 | 0.591 | 0.0100 | 0.657 | 6.64 |
| | B | 0 | 21 | 74.43 | 11 | 8.5e-3 | 0.3734 | 0.589 | 0.0098 | 0.653 | 6.78 |
| | stored 2023 | **1** | 0 | 77.62 | – | 7.7e-4 | 0.4813 | 0.756 | 0.209 | 0.758 | 8.09 |
| 0.323 | A | 0 | – | – (no output: `findax` separatrix point off grid) | 11 | 9.3e-3 | – | – | – | – | – |
| | B | 0 | 21 | 10.56 | 11 | 8.4e-3 | 0.3105 | 0.477 | 0.0053 | 0.672 | 5.60 |
| 0.325 | A | 0 | 21 | 0.0559 | 12 | 7.1e-3 | 0.2746 | 0.410 | 0.0035 | 0.696 | 4.80 |
| | B | 0 | 21 | 0.0559 | 11 | 9.9e-3 | 0.2746 | 0.409 | 0.0034 | 0.687 | 5.00 |

Both arms terminate the same way: the fit exits on the `iconvr = 2`
chi-square criterion after 11 iterations with a GS error of ~8e-3 (the
k-file asks for `ERROR = 1e-5`), and the consistency check reports
"Failure #21, Bp not consistent". χ², plasma current and the axis position
are identical between arms to every printed digit; the boundary and the
shape scalars move by about 1 % (the PF1 near-field difference above). The
one qualitative difference is the 0.323 s slice, where arm A's boundary
finder puts the first separatrix point off the grid and writes nothing while
arm B writes a fit — a boundary-finder edge case, not a convergence change.

## Decision (plan step 5 / #194 step 12)

The decision rule was: if the fresh table restores `jflag = 1` at 0.319 s or
moves χ² by more than 10 % or the axis by more than 1 cm, the bundled table
is a confounder and the reference window must be rerun with it before any
further sweep. **None of that happened**: χ² +0.0 %, axis shift 0.00 cm,
`jflag` unchanged. The bundled table is not what separates the local runs
from the stored 2023 reconstruction. The negative result is recorded here
and on #468/#119, and the remaining suspects stay under #119/#171: the
stored fit reaches a GS error ten times smaller with a condition number
three orders of magnitude lower (6e6 vs 3.5e9), which points at the
constraint set and termination settings, not the Green functions.

What this PR therefore does **not** do, on purpose: it does not replace or
edit `vaft/data/efit`, does not add era-aware `table_dir` selection to the
pipeline, and does not package versioned tables. Those follow from a positive
result, and the result was negative. What it does leave behind is the
provenance chain: a table can now be generated from the canonical geometry
for a named era with a manifest, compared with any other, and every EFIT run
records the table it consumed (`efit_configuration.json` → `table`).

> **Superseded by the section below (#695).** The decision rule above is about
> χ², the axis and `jflag`, and every one of those readings still holds. The
> effect the table does have is the one dismissed here as "a boundary-finder
> edge case": arm A losing the separatrix off grid at 0.323 s. That was 1 of
> 4 slices in this sample and it is 15 of 82 across the reference set. The
> rule did not test for it.

## The table against the namelist, over the whole reference set (#695)

`table_geometry_ab.py` runs the 2×2 that the four-slice A/B above could not:
the packaged `.ddd` tables against regenerated ones, crossed with the
packaged `mhdin.dat` against a regenerated one, over all 82 plasma slices of
the three reference discharges. Everything else is held fixed, including one
acceptance envelope for all four cases.

| case | tables | `mhdin.dat` | equilibria | accepted | `bound` | `findax` |
| --- | --- | --- | --- | --- | --- | --- |
| PP | packaged | packaged | 31 | 18 | 41 | 17 |
| PF | packaged | regenerated | 31 | 18 | 41 | 17 |
| FP | regenerated | packaged | **46** | **30** | 41 | **2** |
| FF | regenerated | regenerated | 46 | 30 | 41 | 2 |

**The namelist is inert and the tables carry all of it.** PP and PF agree on
every count on every shot; so do FP and FF. That is what the file comparison
predicted: everything EFIT reads from `mhdin.dat` is identical between the
two — `turnfc`, which scales the coil currents it fits
(`data_input.F90:2575`), `rsi`, which normalises the flux loops (`:2601`),
and the probe positions — except that all 64 probe angles read −270° in one
file and +90° in the other, which is the same angle. What genuinely differs
is the F-coil description: 16 lumped conductors against 302 discrete
filaments, with `fcturn` following. Those are EFUND inputs and reach EFIT
only through the tables.

**What the table changes is where the boundary goes, not how well the fit
works.** `findax`, which rejects a separatrix point landing within two cells
of the grid edge, fails 17 times under the packaged table and twice under the
regenerated one. `bound` fails 41 times under both, so the 29-slice collapse
block is untouched — as it was by the seed (#588) and by the domain and grid
(#459). The packaged table accounts for the entire `findax` population that
#171 recorded and none of the collapse block.

**The extra yield is not evidence that the regenerated table is correct.** On
every slice both tables reconstruct, EFIT reports the same χ² to the digits
it prints, while the boundary moves: q95 by 2–4 %, the minor radius by under
1 %. The two tables agree about the measurements and disagree about the
plasma. Magnetics alone cannot separate them, which is what the kinetic arm
of the reference set exists for.

So the case for the regenerated table rests on provenance, not on yield: it
is generated from named, hashed era assets by a recorded EFUND build, and it
resolves each PF coil into 302 filaments rather than 16 boxes — a finer
representation of the same geometry. The packaged table has no recoverable
origin, and the `mhdin.dat` shipped beside it cannot be consumed by this
EFUND revision at all (`islpfc` sits in the wrong namelist group), so it is
not even the input that produced it.

**What this means for the studies already run.** The #171 baseline, the #588
seed study and the #459 domain/grid study all ran against the packaged table,
which is case PP. Their comparisons are internally consistent. Their absolute
yields are the packaged table's.

## The switch

`vaft/data/efit` now holds the regenerated table. The pipeline reads it
through `config.yaml`'s `table_dir: ${VAFT_DATA_DIR}/efit/`, so replacing the
files is the switch; nothing else had to change.

What moved, and what deliberately did not:

| file | | why |
| --- | --- | --- |
| `ec129129.ddd`, `ep129129.ddd`, `rv129129.ddd`, `rfcoil.ddd` | replaced | the tables EFIT consumes |
| `brzgfc.dat`, `mhdout.dat` | replaced | products of the same EFUND run; leaving them stale would make the manifest describe files that are not there |
| `efund_table_manifest.json` | **added** | the point of the exercise. Every EFIT run now records the table it consumed by identity instead of `provenance: unrecorded` |
| `mhdin.dat` | **kept** | see below |
| `lim.dat`, `dprobe.dat`, `rfcoil.txt`, the sample g/a-files | untouched | not EFUND products of this run |

`mhdin.dat` is kept as the legacy file on purpose, and it is the one
inconsistency in the shipped directory: it describes 16 lumped PF conductors
while the tables beside it were built from 302 filaments. Two reasons, and
the manifest states both rather than leaving them to be discovered.

- **EFIT does not care.** That is measured, not assumed: case FP above —
  regenerated tables with the legacy namelist — is identical to case FF on
  every count on every shot. EFIT takes the per-group coil response from the
  tables and only `nfsum` and `turnfc` from the namelist, and those agree.
- **It is an independent reference.** `test_efund_geometry.py` validates the
  canonical geometry against this file: centroids, turns, probe positions and
  angles. Overwriting it with a file generated from that same geometry would
  turn those checks into a comparison of the geometry with itself.

The acceptance envelope did not move. `&incheck` still carries its packaged
values; what those should be is #649's question, and changing them alongside
the table would confound the two.

**Verification of the switch**: the #171 baseline rerun against the switched
directory reproduces case FF exactly — `bound` 41, `findax` 2, 29 collapsed,
against the pre-switch 41/17/29.

**Reproducing the A/B after the switch.** `table_geometry_ab.py` defaulted its
"packaged" arm to `vaft/data/efit`, which is now the regenerated table; it
refuses to run when both arms are the same table and takes `--packaged` for a
copy recovered from git history. The pre-switch file hashes are recorded in
`test/data/efit_table_geometry_ab.json` under `packaged_table`.

## What actually differs between the two tables

`compare_tables.py vaft/data/efit <regenerated>`, reading every record of
every file. Relative differences, worst element:

| record | what it is | read by EFIT | max relative difference |
| --- | --- | --- | --- |
| `rfcoil/gmp2fc` | probe response to the PF coils | yes | **0.836** |
| `ec/gridfc` | grid flux from the PF coils | yes | **0.230** |
| `rfcoil/gsilfc` | flux-loop response to the PF coils | yes | **0.088** |
| `ec/gridpc`, `ep/rsilpc`, `ep/rmp2pc` | plasma-to-grid and plasma-to-diagnostic | yes | ≤ 1.6e-4 |
| `rv/gsilvs`, `rv/gmp2vs`, `rv/ggridvs` | vessel response | yes | ≤ 2e-11 |
| `ec/rgrid_zgrid` | the grid itself | yes | 2e-16 |
| `brzgfc/brgrfc`, `brzgfc/bzgrfc` | Br, Bz from the PF coils | **no** | ~2 |
| `rv/gfcvs`, `rv/gvsvs` | coil-to-vessel, vessel-to-vessel | **no** | 0.96, 1.9 |

**Everything EFIT reads and that matters is the PF-coil response, and within
that, the central solenoid.** Per F-coil group, `gmp2fc` differs by 35–84 %
for groups 1–8 and by 5e-5 for groups 9–16. Groups 1–8 are the solenoid stack
at R = 0.053 m; 9–16 are the outer shaping coils at R = 0.71 and 0.93 m.

The reason is the near field. The legacy file models each solenoid section as
a single conductor 0.30 m tall; the regenerated one resolves it into ~20
filaments over the same extent. VEST's inner magnetic probe array sits at
R = 0.089 m — 27 mm outside the solenoid's outer face — so what those probes
are told to expect from a solenoid current depends strongly on how the
solenoid is discretised:

| probes | count | median change in solenoid response | worst |
| --- | --- | --- | --- |
| inner array, R = 0.089 m | 27 | 4.5 % | **83.6 %** |
| outer, R > 0.1 m | 37 | 0.0 % | small |

That is the whole mechanism. The two tables tell EFIT nearly the same thing
about the plasma, the vessel and the outer coils, and materially different
things about what the innermost probes should read from the solenoid. It is
consistent with what the runs show: the fit to the measurements is unchanged
(χ² identical on shared slices) while the implied field — and so where the
separatrix lands — moves enough to take `findax` from 17 failures to 2.

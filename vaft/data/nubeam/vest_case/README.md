# VEST NUBEAM reference case

The inputs for one NUBEAM neutral-beam run on VEST. Inputs only: running them
regenerates the roughly 260 MB of Plasma State, marker and log output that
would otherwise have to be preserved for an analysis to be reproducible.

Load with `vaft.code.nubeam.packaged_vest_case()`, which pairs the directory
with its equilibrium; `notebooks/vest_nbi_analysis_with_nubeam.ipynb` runs it
into a temporary directory.

## Contents

| file | what it is |
| --- | --- |
| `inputf` | positional input to the Plasma State generator: time window, file names, run id |
| `profiles` | kinetic profiles and beam operating point — the modelling inputs |
| `nubeam_init.dat`, `nubeam_step.dat` | NUBEAM namelists: Monte Carlo particle counts, physics switches |
| `nubeam_init_files.dat`, `nubeam_step_files.dat` | `&NUBEAM_FILES` — which state file to read and which to write |
| `mdescr_VEST_190307.dat` | machine description, including the NBI geometry |
| `sconfig_VEST_190307.dat` | shot configuration |
| `g020000.015100` | the G-EQDSK this case was run against |

`inputf` line 2 still names `chease_g026537.031000`, the equilibrium of the
original author's run. Nothing reads that name: `prepare_nubeam_inputs`
rewrites the line to point at whichever equilibrium it stages. The line is
left as received rather than edited, so the case stays byte-identical to what
was validated.

## Provenance and caveats

Received as a working NUBEAM case for VEST and used unchanged to validate the
macOS build and the VAFT adapter (`external/nubeam/VALIDATION.md`). Two
server-specific launcher scripts that came with it, `run.sh` and
`run_slurm.sh`, are deliberately not included: they hard-code a `qsub` queue
that exists on one cluster, and `vaft.code.nubeam.runner` is what launches the
run here.

**The NBI geometry in `mdescr_VEST_190307.dat` is a model, not as-built.** It
is the source of the `nbi` block in `vaft/machine_mapping/vest.yaml`, which is
marked `mapping_status: partial` for exactly this reason; reconciliation
against the real hardware is issue #265.

**The profiles are a modelling input, not a measurement.** Beam energy and
launched power come from `profiles`, which is why they are mapped from a run
result rather than from the static machine description.

NUBEAM itself is not in this repository. NTCC requires each user to accept its
licence before downloading the source, so VAFT carries the build recipe
(`external/nubeam/`) and this adapter contract only. Point `$NUBEAMHOME` at
your installation.

# Gyrokinetic validation (Lane Y)

Linear CGYRO on Lane T's good states, compared with TGLF on the same local input, to
judge which TGLF saturation rule fits VEST (the TGLF SAT-rule sensitivity found SAT0–3
differing ~5x in Q/Q_GB at the median surface). Run record: issue "[Lane Y] CGYRO:
progress log" (#1484).

| script | does |
|---|---|
| `run_linear.py` | resolve the states (Lane T's `resolve_transport_state`), project each surface once, run linear CGYRO per (field model, ky) and TGLF linear (`USE_TRANSPORT_MODEL=F`) at the same ky |
| `build_linear.py` | `linear.csv`, `linear_summary.csv`, `schema.json` from a run tree; optional join with the SAT-rule `sensitivity.csv` (`transport_atlas/build_sensitivity.py`) |
| `convergence.py` | `convergence.csv` from a base run tree and resolution variants |

## Inputs and conventions

- State key: Lane T's provisional `(shot, time_efit_s, efit_lineage)`; surfaces r/a.
- Ti = Te (the `vest.yaml` policy), H+/C6+ at Z_eff 2, no rotation or ExB shear: exactly the TGLF atlas input.
- CGYRO's input is a renaming of TGLF's local input (`vaft.code.gacode.cgyro.inputs`),
  held against CGYRO's own `PROFILE_MODEL=2` projection (`test/test_cgyro_oracle.py`).
- Frequencies are written with the **ion diamagnetic direction negative** (TGLF's
  convention). CGYRO's native sign depends on IPCCW/BTCCW; the conversion uses the
  direction CGYRO itself prints in `out.cgyro.info`.
- `ky` is `k_y rho_s`, rates are `c_s/a` (GACODE normalisation, deuterium `c_s`).

## Running on tdst

This workflow imports Lane T's `vaft.process.transport_state` and
`workflow/transport_atlas/run_tglf.py`. Both are on `develop`; the production runs
were made from a detached worktree at `~/work/lane-y/src`, set up before they were
merged (Lane T's branch with the Lane Y files on top).

`~/work/lane-y/run.sh` puts that checkout first on `PYTHONPATH`, loads
`gcc/14.2 gnu/openmpi/5.0.10`, strips `SLURM_CPU_BIND*` and asserts `vaft.__file__`
before calling `run_linear.py` with the FileDB and the EFIT campaign slice labels. The conda env
`vaft` is non-editable and is not changed.

CGYRO was built in `~/git/gacode` (b493397, TDST_GNU) on a compute node with
`make -C cgyro`; `cgyro -rs reg01` passes.

Smoke first, on `lowpri-short`, then the grid. Run the driver *inside* one allocation so
the cluster sees one job, not hundreds:

```bash
sbatch -p lowpri-short -A tdst -n 32 --mem=32G -t 06:00:00 \
  --wrap "~/work/lane-y/run.sh --out ~/runs/gyrokinetic/linear --n-mpi 8 --workers 4"
```

`--backend slurm` (one `sbatch` per run, as Lane T does for TGLF) also works, with
`--mem-mb` set, but a 360-run grid would then flood the shared queue.

Every run is resumable: a `record.json` whose input hash matches and whose status is
`solved` or `decayed` (stable, amplitude underflow) is reused, so a timed-out allocation is continued by resubmitting.

## Products

`vestserver:~/runs/campaign/atlas/gyrokinetic_linear/`: `linear.csv`,
`linear_summary.csv`, `convergence.csv`, `schema.json`, and `runs/` with each native
CGYRO directory (`input.cgyro`, `input.cgyro.gen`, `out.cgyro.*`, `bin.cgyro.*`), its
`record.json`, `cgyro_outputs.json` and `gyrokinetics_local.json` (DD 3.41).

## Nonlinear runs

`run_nonlinear.py` runs one state and surface. Two things about it are not obvious:

- **The default field model is EM (`em-aperp`).** An electrostatic run with kinetic electrons carries a spurious high-frequency branch at low `k_y` (the ω_H mode, |ω| ~ 200 c_s/a). Finer θ resolution makes it worse, and finite β removes it. On 39915 r/a 0.7 it drove the whole ES run (run record #1484). An ES run with `n=1` at `k_y <= 0.2` is therefore refused unless you pass `--allow-es-low-ky`.
- **`--amp` sets CGYRO's `AMP`**, the seed amplitude of the `n>0` modes (CGYRO default 0.1). A 10x larger seed only saves about ln(10)/γ_max of linear growth.

Long runs are chained allocations with `--restart`. **CGYRO counts `MAX_TIME` from the restart**, not from t = 0. A continuation therefore passes the *extra* span: a run at t = 250 continued with `--max-time 500` stops at t = 750. With Slurm, chain the jobs with `sbatch -d afterany:<previous>`. The restart file survives a job that hits its wall-time limit.

`build_nonlinear.py` reports, for a window:
- the window means of Q_tot, Q_i and Q_e;
- a half-window drift test;
- a **batch-means standard error** (`--blocks`, default 4);
- the **zonal-fraction trace**;
- the TGLF SAT0-3 reference and the locality QA on the same window.

A bursty run, where turbulence and zonal flows trade energy, is not stationary on short windows. For such a run, quote the window mean ± its batch-means standard error, with blocks longer than the burst spacing. With `--blocks n` the standard error has only n - 1 degrees of freedom; the default 4 is a rough error bar, so quote more blocks when the run is long enough (39915 r/a 0.7: 9 blocks of 100 a/c_s). Do not quote the window standard deviation; it mostly measures the bursts.

## Reusing retained outputs (2-D fields)

Use `python -m workflow.gyrokinetic.build_result_reference --out /tmp/gk-reference`
for the offline compact real-native fixture, static fields, synchronized MP4/WebM,
window statistics and machine-readable mapping/retention audit. `--skip-video`
runs without PyAV. Full native fields remain outside IMAS.

`build_multimodel_reference` remaps the retained SAT0–3 state/native trees into one
ODS and optionally adds the #39915 nonlinear CGYRO source and linear scan. See
`docs/_guide/Gyrokinetic_results.md` for source layout, commands and qualifications.
`capture_reference` creates content-addressed contiguous native fixture slices.

Baseline: develop `9744dbd73eebc1ca3a03857f19e35360c4052488`.
Pinned native oracle: GACODE `b49339750a4aa4cf2b089fa9ff3afe098005f0f8`.
No solver is launched; no server source file is modified. Open: #1286 and #2019 (3-D
field rendering).

The same offline field/flux workflow can express time as SI elapsed time:

```bash
python -m workflow.gyrokinetic.build_result_reference --out /tmp/vaft-time-ms \
  --time-unit ms \
  --time-normalization test/data/gacode/cgyro_39915_time_normalization.json
```

The retained normalization is tied to the native input hash and plasma-state
identity. Native windows remain in `a/c_s`. The resulting millisecond coordinate
starts at the solver's origin; the input state's discharge time remains separate.

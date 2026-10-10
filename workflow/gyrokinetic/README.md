# Gyrokinetic validation (Lane Y, #1354)

Linear CGYRO on Lane T's good states, compared with TGLF on the same local input, to
judge which TGLF saturation rule fits VEST (#1482: SAT0–3 differ ~5x in Q/Q_GB at the
median surface). Progress log: issue "[Lane Y] CGYRO: progress log" (#1484).

| script | does |
|---|---|
| `run_linear.py` | resolve the states (Lane T's `resolve_transport_state`), project each surface once, run linear CGYRO per (field model, ky) and TGLF linear (`USE_TRANSPORT_MODEL=F`) at the same ky |
| `build_linear.py` | `linear.csv`, `linear_summary.csv`, `schema.json` from a run tree; optional #1482 `sensitivity.csv` join |
| `convergence.py` | `convergence.csv` from a base run tree and resolution variants |

## Inputs and conventions

- State key: Lane T's provisional `(shot, time_efit_s, efit_lineage)`; surfaces r/a.
- Ti = Te (#1414), H+/C6+ at Z_eff 2, no rotation or ExB shear: exactly the TGLF atlas input.
- CGYRO's input is a renaming of TGLF's local input (`vaft.code.gacode.cgyro.inputs`),
  held against CGYRO's own `PROFILE_MODEL=2` projection (`test/test_cgyro_oracle.py`).
- Frequencies are written with the **ion diamagnetic direction negative** (TGLF's
  convention). CGYRO's native sign depends on IPCCW/BTCCW; the conversion uses the
  direction CGYRO itself prints in `out.cgyro.info`.
- `ky` is `k_y rho_s`, rates are `c_s/a` (GACODE normalisation, deuterium `c_s`).

## Running on tdst

The Lane Y code is based on `develop`, but this workflow imports Lane T's
`vaft.process.transport_state` and `workflow/transport_atlas/run_tglf.py`, which are not
merged yet. The run checkout is therefore Lane T's branch with the Lane Y files on top:

```bash
git -C ~/git/vaft fetch origin claude/lane-t-tglf-sensitivity-1482
git -C ~/git/vaft worktree add --detach ~/work/lane-y/src FETCH_HEAD
# copy (or merge) the Lane Y branch's files over it
```

`~/work/lane-y/run.sh` puts that checkout first on `PYTHONPATH`, loads
`gcc/14.2 gnu/openmpi/5.0.10`, strips `SLURM_CPU_BIND*` and asserts `vaft.__file__`
before calling `run_linear.py` with the FileDB and the #1331 labels. The conda env
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

- **The default field model is EM (`em-aperp`).** An electrostatic run with kinetic electrons carries a spurious high-frequency branch at low `k_y` (the ω_H mode, |ω| ~ 200 c_s/a). Finer θ resolution makes it worse, and finite β removes it. On 39915 r/a 0.7 it drove the whole ES run (#1484). An ES run with `n=1` at `k_y <= 0.2` is therefore refused unless you pass `--allow-es-low-ky`.
- **`--amp` sets CGYRO's `AMP`**, the seed amplitude of the `n>0` modes (CGYRO default 0.1). A 10x larger seed only saves about ln(10)/γ_max of linear growth.

Long runs are chained allocations with `--restart`. **CGYRO counts `MAX_TIME` from the restart**, not from t = 0. A continuation therefore passes the *extra* span: a run at t = 250 continued with `--max-time 500` stops at t = 750. With Slurm, chain the jobs with `sbatch -d afterany:<previous>`. The restart file survives a job that hits its wall-time limit.

`build_nonlinear.py` reports, for a window:
- the window means of Q_tot, Q_i and Q_e;
- a half-window drift test;
- a **batch-means standard error** (`--blocks`, default 4);
- the **zonal-fraction trace**;
- the TGLF SAT0-3 reference and the locality QA on the same window.

A bursty run, where turbulence and zonal flows trade energy, is not stationary on short windows. For such a run, quote the window mean ± its batch-means standard error, with blocks longer than the burst spacing. With `--blocks n` the standard error has only n - 1 degrees of freedom; the default 4 is a rough error bar, so quote more blocks when the run is long enough (39915 r/a 0.7: 9 blocks of 100 a/c_s). Do not quote the window standard deviation; it mostly measures the bursts.

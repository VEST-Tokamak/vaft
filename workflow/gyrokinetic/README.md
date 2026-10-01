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
`solved` is reused, so a timed-out allocation is continued by resubmitting.

## Products

`vestserver:~/runs/campaign/atlas/gyrokinetic_linear/`: `linear.csv`,
`linear_summary.csv`, `convergence.csv`, `schema.json`, and `runs/` with each native
CGYRO directory (`input.cgyro`, `input.cgyro.gen`, `out.cgyro.*`, `bin.cgyro.*`), its
`record.json`, `cgyro_outputs.json` and `gyrokinetics_local.json` (DD 3.41).

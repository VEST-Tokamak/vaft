# Transport atlas (lane T, #1453)

This directory holds the drivers for the Tier A transport atlas. TGLF, NEO and classical model fluxes are keyed by (shot, time_efit_s, efit_lineage, r/a) and are computed on the #1331 slices labelled good or admissible only.

| script | issue | what it does |
|---|---|---|
| `run_tglf.py` | #1428 | Enumerates the states (good/admissible slices that carry a core_profiles slice within 0.5 ms), resolves each through `vaft.process.transport_state`, and runs native TGLF at r/a = 0.30, 0.40, …, 0.80 through the existing runner and execution backend. Projects the solved surfaces with `core_transport_from_tglf`. Writes `states.jsonl`, `run_manifest.json`, and one `state.json` plus the native directories per state. Resumable: a surface already solved under the same run identity is not run again. |

The shared resolver is `vaft.process.transport_state`:
- **State key** (provisional, until lane K's contract): `(shot, time_efit_s, efit_lineage)`, with `efit_lineage ∈ {magnetics-only, electron-kinetic}` and an `efit_label ∈ {good, admissible}` column. Unreconstructible slices are refused, not run.
- **Time pairing:** core_profiles is matched to the equilibrium slice by time, within an explicit tolerance, never by index.
- **Ion temperature:** measured, then a #1426 pressure-partition result when one is passed, then the `vest.yaml` ratio (#1414: Ti = Te ± 0.5, `assumed`). Otherwise the state is insufficient. Composition goes through `prepare_gacode_profile(impurity="C", z_eff=2)`.
- **Geometry:** EFIT products carry no `r_inboard`/`r_outboard`, so these are derived from the 2-D flux map. Some products also lack shape profiles (#1458); for those, elongation and triangularity come from closed contours inside the boundary outline. `provenance.shape.kind` says which applies.

Runs go on tdst (`lowpri-short`, GACODE `TDST_GNU`). The branch worktree sits on `PYTHONPATH` in front of the non-editable `vaft` env, and the wrapper asserts `vaft.__file__`. The inputs are a read-only subset of the campaign FileDB copied from vestserver. Results are copied back to `vestserver:~/runs/campaign/atlas/transport/`, never into the production FileDB. Pass `--mem-mb`: without `--mem`, tdst's default reserves the whole node for a one-core job.

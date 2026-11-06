# TGLF-NN training-domain snapshot (#2274, phase A)

These are the population tables behind section 11 of
`turbulent_transport_with_tglf.ipynb`. They were captured from
`user1@147.46.36.244:/home/user1/runs/campaign/atlas/transport/tglf_nn_domain_bounds`,
the output of `workflow/transport_atlas/audit_tglf_nn.py` run against the campaign
FileDB. The TurbulentTransport.jl 1.2.0 ONNX families it audited carry an `xbounds.txt`
exported by `workflow/tglf_nn/export_training_bounds.jl`. `snapshot_manifest.json`
records the producing revision, the run manifest (including every model file's
SHA-256), and the row count and SHA-256 of each copied table.

## Population

- **States:** the Tier A confinement table (`vaft/data/confinement/vest_tier_a_confinement.csv`, 133 magnetics-EFIT states).
- **Usable states:** only states with a Thomson profile within 0.5 ms of their EFIT slice can become a TGLF input. That leaves 64: 31 of the 59 `primary` states and all 16 `thomson_consistent` ones.
- **Surfaces:** six per state, r/a = 0.3, 0.4, ..., 0.8.
- **T_i scenarios** (`scenario`):
  - `ratio_1`: T_i = T_e, the pipeline's assumption.
  - `ratio_0.17`: the superseded pressure-matching ratio, which is what CES gives on 48224.
  - `pressure_partition_profile`: a BIC-chosen T_i/T_e(rho) fitted to the magnetics-EFIT pressure partition (`ti_inferred --ti-form profile`).

## Tables

### `summary.csv`

One row per subset × scenario × family:
- `in_domain_surfaces`: z ≤ 3 on every input.
- `contained_surfaces`: inside upstream's training bounds on every input.
- `surfaces_past_data_edge`: at least one input beyond an edge that lies inside upstream's ±3.3σ clip, that is, past where the training data ended.
- The binding inputs of each test.

### `coverage.csv`

One row per subset × scenario × family × input: the fraction of surfaces inside / below / above the bounds, and the min / median / max position on the range, where 0 = training min and 1 = training max. The `tier_a` subset is left out of this copy; `summary.csv` keeps it.

## Reading the bounds

Upstream's `xbounds` are the training data range clipped at `xm ± 3.3 xsigma`.
- An input outside a clipped edge restates the z-score.
- Only an edge inside the clip marks where the data ended.
- The notebook keeps the two apart.

## Regenerating

1. Export the bounds once:
   `julia --startup-file=no workflow/tglf_nn/export_training_bounds.jl <TurbulentTransport checkout> <copy>`.
2. Build the reference T_i products: run the `ti_inferred` stage with `--ti-form profile` on the composed core_profiles of each shot.
3. Run the audit:
   `python workflow/transport_atlas/audit_tglf_nn.py --filedb <FileDB> --profile-ti-dir <products> --model-dir <copy>/models --ratios 1.0 0.17 --out <dir>`.
4. Copy `summary.csv` here, and `coverage.csv` filtered to the `primary` and `thomson_consistent` subsets, then update the manifest.

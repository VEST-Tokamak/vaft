# Pipeline 2 — the Thomson lineage

Builds the products that depend on data uploaded *after* a shot was processed:
Thomson scattering, charge exchange (CES), the `core_profiles` mapped from them,
the assumed impurity composition, an inferred ion temperature, the Romero
resistive `Z_eff`, and the two kinetic EFIT lineages. A Snakemake stage-product
DAG, like pipeline 1; the stage-by-stage description is in
[`docs/_guide/Pipelines.md`](../../docs/_guide/Pipelines.md) (section *Pipeline 2*).
Operating it as a service is in [`DEPLOYMENT.md`](DEPLOYMENT.md).
(`README_external_diagnostics.md` is about the soft X-ray and camera ingest, which
this DAG does not build.)

```
thomson ─┐
         ├─> core_profiles ─> composition ─┬─> ti_inferred ─> resistive_zeff
ces ─────┘                                 ├─> electron_efit
                                           └─> kinetic_efit
```

Pipeline 1's magnetic EFIT and diagnostics products are consulted as parameters,
not waited for: a shot pipeline 1 never reconstructed records its stages as
`unavailable` instead of making Snakemake raise.

## Run it by hand

```bash
cd workflow/automatic_pipeline_2_corrective_data_update
snakemake --snakefile Snakefile --configfile config.yaml --cores 4 -n    # dry run
snakemake --snakefile Snakefile --configfile config.yaml --cores 4
```

`config.yaml` lists the shots (`shots: [48226]`), the upload directory
(`external.data_root`) and the settings of each stage. A run configuration may be
partial: it is merged over `config.yaml`, and `base_dir` and `shots` must be stated.

| Key | What it decides |
|---|---|
| `stages` | build and judge only these stages (a stage needs its upstream; see `vaft.database.corrective_lineage`) |
| `external.data_root`, `external.ces_options` | where the `.mat` uploads are, and the `CES_*` or `IDS_*` ion layout |
| `composition` | `ionization` (`transient` needs the plasma age), `normalization`, the ADF11 cache |
| `kinetic` | EFIT encoding and preset, and `dilution` (`species` or `none`) |
| `ti_inference` | the floor and window of `sigma(p_EFIT)` |
| `resistive_zeff` | Lane Z's estimator settings, all stated because it has no defaults |
| `plots.enabled` | the per-stage validation figures |
| `hsds.replicate` | one replication rule per stage |

## Reading a result

Every stage always writes a product and a manifest and exits 0. **Read the
manifest `status`**: `success`; `unavailable` or `no_output` (the normal answer for
most shots, never retried); `failed` (retried). Under `hsds.replicate` the
`core_profiles` IDS reaches `main` through four stages that replace one another in
a fixed order (`core_profiles`, `composition`, `ti_inferred`, `resistive_zeff`), so
`main` always holds the last one that succeeded.

## The worker

`vaft corrective-worker run --config corrective.yaml` watches the upload directory
and runs this pipeline for a shot when an upload for it appears or changes. Start
from [`corrective.example.yaml`](corrective.example.yaml).

## The legacy daemons

`legacy/` holds the pre-Snakemake polling scripts. They are not part of the
supported pipeline; see its README.

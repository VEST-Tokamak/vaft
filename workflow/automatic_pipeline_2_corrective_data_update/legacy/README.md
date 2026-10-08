# Legacy polling daemons (quarantined)

`update_thomson_scattering_and_core_profile.py`, `update_charge_exchange.py` and
`update_core_profile.py` are the pre-Snakemake Pipeline 2: long-running polling
loops that wrote the shot-first `public/` tree and the `processed_shots.h5`
registry directly.

They are **not** part of the supported pipeline and must not be run against
`main`. The supported replacement is the stage-product DAG in the parent
directory (`../Snakefile`): every stage writes a manifest and replicates through
`vaft.database.replication`.

The scripts are kept only because the registry readers
(`vaft.database.utils.read_legacy_processed_registry`) and a few regression tests
still describe their on-disk format. They import each other by bare module
name, so run them from this directory if you must.

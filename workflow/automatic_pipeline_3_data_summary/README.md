# Pipeline 3 summary materialization

Database-backed history scripts are thin wrappers around
`vaft.database.summary()` and `vaft.database.export_summary()`. Supported
presets are:

- `equilibrium_global` (also replaces the redundant EFIT g/m/a history scan)
- `core_profiles`
- `volume_averaged`
- `efit_magnetic_reliability`
- `efit_kinetic_reliability`
- `shot_overview`

Every wrapper accepts an optional inclusive `--shot-range START:END`. Omitting
the range discovers all numeric shots in the selected HSDS namespace. XLSX and
CSV files are materialized from the canonical DataFrame; workbooks are not a
database input.

CHEASE remains solver-file-backed because no distinct CHEASE IDS is available
in HSDS yet, but it uses the shared DataFrame exporter. DCON/RDCON stability
generation and joins are intentionally excluded from this migration.

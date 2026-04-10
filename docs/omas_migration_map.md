# OMAS Migration Map

## Goal

Restore the `vaft` package layout around the long-lived core boundaries:

- `vaft/database`
- `vaft/machine_mapping`
- `vaft/process`
- `vaft/formula`
- `vaft/omas`
- `vaft/plot`
- `vaft/code`
- `vaft/imas`

The donor implementation in `/Users/yun/git/vest_database/OMAS/` is still the
reference for behavior, but the implementation must land inside these existing
modules instead of new temporary package families.

## Target Package Layout

### `vaft/database`

- raw SQL access
- shot waveform lookup
- signal-name lookup
- OMAS JSON/NC/MAT convenience helpers

### `vaft/machine_mapping`

- canonical IDS builders and orchestration
- flat IDS-centered module names
- `utils.py` as the only shared machine-mapping helper layer

### `vaft/process`

- reusable signal processing
- reusable electromagnetic helper routines
- no direct SQL or file-format ownership

## Donor-to-VAFT Mapping

| Donor source | Restored target in `vaft` | Status |
| --- | --- | --- |
| `/Users/yun/git/vest_database/OMAS/data_loader/vest_sql.py` | `vaft/database/raw.py` | absorb |
| `/Users/yun/git/vest_database/OMAS/data_loader/vest_tools.py` | `vaft/process/signal_processing.py`, `vaft/process/electromagnetics.py`, `vaft/database/ods.py` | absorb |
| `/Users/yun/git/vest_database/OMAS/vest_ods_dynamic.py` | `vaft/machine_mapping/dataset_description.py` | absorb |
| `/Users/yun/git/vest_database/OMAS/ods_builders/diagnostics/vest_barometry.py` | `vaft/machine_mapping/barometry.py` | absorb |
| `/Users/yun/git/vest_database/OMAS/ods_builders/diagnostics/vest_tf.py` | `vaft/machine_mapping/tf.py` | absorb |
| `/Users/yun/git/vest_database/OMAS/ods_builders/diagnostics/vest_spectrometer_uv.py` | `vaft/machine_mapping/spectrometer_uv.py` | absorb |
| `/Users/yun/git/vest_database/OMAS/ods_builders/diagnostics/vest_thomson_scattering.py` | `vaft/machine_mapping/thomson_scattering.py` | absorb |
| `/Users/yun/git/vest_database/OMAS/ods_builders/diagnostics/vest_charge_exchange.py` | `vaft/machine_mapping/charge_exchange.py` | absorb |
| `/Users/yun/git/vest_database/OMAS/ods_builders/geometry/vest_pf_active.py` | `vaft/machine_mapping/pf_active.py` | absorb |
| `/Users/yun/git/vest_database/OMAS/physics_models/vest_magnetics.py` | `vaft/machine_mapping/magnetics.py` | absorb |
| `/Users/yun/git/vest_database/OMAS/physics_models/vest_plasma.py` | `vaft/machine_mapping/magnetics.py` private helpers | absorb |

## Design Rules

1. `machine_mapping` owns OMAS IDS construction and no other package may become a parallel builder surface.
2. `database` owns raw input access and file-format convenience.
3. `process` owns reusable computation kernels.
4. `machine_mapping.utils` stays thin:
   - dotted-path read/write
   - dict/ODS access helpers
   - static asset resolution
   - uncertainty helpers
   - common time-axis validation helpers
5. `plot`, `omas`, and `code` must consume canonical OMAS paths only.

## Migration Completion Criteria

1. No runtime imports remain from `vaft.builders`, `vaft.loaders`, `vaft.models`, or `vaft.compat`.
2. `vaft.__init__` exposes only the restored core package surfaces.
3. `vaft.machine_mapping` is the only public builder surface.
4. Contract, uncertainty, and import smoke tests pass without `omas` or `mysql` installed.

## Current First-Class Entry Points

- `from vaft.machine_mapping import vfit_dataset_description`
- `from vaft.machine_mapping import vfit_pf_active_for_shot`
- `from vaft.machine_mapping import vfit_tf_static`
- `from vaft.machine_mapping import vfit_magnetics_for_shot`
- `from vaft.machine_mapping import apply_default_constraint_uncertainties`

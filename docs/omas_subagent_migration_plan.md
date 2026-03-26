# OMAS Restoration Workstreams

## Objective

Execute the OMAS port as a restoration of the original `vaft` module
boundaries, not as a long-lived forked package tree.

## Recommended Workstreams

### 1. `schema-contract`

Ownership:

- OMAS contract table
- required path assertions
- synthetic and sample validation

Output:

- canonical IDS contract
- contract test expectations

### 2. `database-integration`

Ownership:

- SQL access consolidation into `vaft/database/raw.py`
- OMAS JSON/NC/MAT convenience in `vaft/database/ods.py`
- static lookup assets under `vaft/data`

Output:

- import-safe raw access
- stable file helper surface

### 3. `machine-mapping-restoration`

Ownership:

- absorb donor IDS builders into flat `vaft/machine_mapping/*`
- keep `filterscope` as a thin alias to `spectrometer_uv`
- keep shared helpers only in `vaft/machine_mapping/utils.py`

Output:

- canonical `vfit_*` entrypoints under `vaft.machine_mapping`

### 4. `process-kernel-restoration`

Ownership:

- reusable signal processing helpers
- reusable electromagnetic helpers
- no data-access coupling

Output:

- clean `process` kernel surface reused by `machine_mapping`

### 5. `consumer-verification`

Ownership:

- verify `vaft.plot`, `vaft.omas`, and `vaft.code`
- reject non-canonical path assumptions
- keep uncertainty handling centralized in `machine_mapping.utils`

Output:

- boundary tests
- cutover checklist

## Definition of Done

1. `vaft/builders`, `vaft/loaders`, `vaft/models`, and `vaft/compat` are removed.
2. `machine_mapping` no longer imports those removed packages.
3. contract, uncertainty, and boundary tests pass.
4. public imports are:
   - `import vaft`
   - `from vaft import machine_mapping`
   - `from vaft.machine_mapping import vfit_dataset_description, vfit_pf_active_for_shot, vfit_tf_static`

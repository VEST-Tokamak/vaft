# Kinetic profile chain: TS + IDS/CES → core_profiles (design)

Date: 2026-07-13
Branch: `feature/kinetic_profile_eq`
Status: approved by user (2026-07-13 conversation)

## Goal

Promote the ad-hoc script logic (TS + ion-Doppler kinetic profiles for VEST) into the
VAFT library, ending at **kinetic pressure inside `core_profiles`**. EFIT
pressure-constraint export and kinetic-vs-EFIT comparison are out of scope and their
scripts are deleted.

Pipeline per (shot, time):

```
magnetic-EFIT gfile ──to_omas──▶ ODS[equilibrium]
NeTe_<shot>.mat  ──machine_mapping.thomson_scattering──▶ ODS[thomson_scattering]   (pre-mapping)
IDS_<shot>.mat / CES_<shot>.mat ──machine_mapping.charge_exchange──▶ ODS[charge_exchange] (pre-mapping)
        │
        ├─ equilibrium_mapping_thomson_scattering / _charge_exchange  (positions → psi_N)
        ├─ profile_fitting_thomson_scattering  (ne, Te)
        ├─ profile_fitting_charge_exchange     (Ti, Vtor)
        ▼
ODS[core_profiles]  ← fitted ne/Te/Ti/Vtor + pressure_thermal        (post-mapping)
        ▼
save: OMAS JSON  +  IMAS IDS export (local, via vaft.imas)
```

Both the **pre-mapping** raw diagnostic IDSs (`thomson_scattering`, `charge_exchange`)
and the **post-mapping** fitted `core_profiles` live in the same ODS, and the whole ODS
is saved to OMAS JSON and exported to IMAS IDS form.

## Physics assumptions (user-confirmed)

- Ohmic plasma: all electrons thermal (`density_thermal = density`).
- Quasi-neutrality with no impurity dilution: **n_i ≈ n_e** (confirmed).
- Main ion H+ temperature equals measured impurity (C3+) temperature: Ti(H+) = Ti(C3+).
- Kinetic pressure: `pressure_thermal = e · n_e · (T_e + T_i)` [Pa], with T in eV,
  e = 1.602176634e-19.

## Coordinate convention

- `equilibrium_mapping_*` produces **psi_N** (normalized poloidal flux,
  `fluxSurfaces['levels']`), so all fit functions are functions of psi_N.
- `core_profiles.profiles_1d.grid` is taken from the equilibrium in the ODS:
  `rho_tor_norm` and `psi` from `equilibrium.time_slice.<i>.profiles_1d`. Fits are
  evaluated at `psi_N(grid)` computed from the equilibrium `psi` array
  (`(psi - psi[0]) / (psi[-1] - psi[0])`), so profile values are correct on the
  `rho_tor_norm` grid. This replaces the previous mislabeling (uniform psi_N grid
  stored as `rho_tor_norm`).

## Components

### 1. `vaft/process/profile.py` — extend `core_profiles()`

New optional arguments (backward compatible):

- `T_i_function=None` — callable Ti(psi_N); when given, `ion.0.temperature` uses real
  Ti instead of the Ti=Te fallback.
- `V_tor_function=None` — callable Vtor(psi_N); when given, written to
  `ion.0.velocity.toroidal`.
- Equilibrium-grid handling: when the ODS carries `equilibrium`, use its
  `rho_tor_norm`/`psi` grid as described above; otherwise fall back to the current
  uniform grid (documented as psi_N-based).
- Always write H+ ion metadata (`label`, `z_ion`, `element.0.{a,z_n,atoms_n}`) and
  `pressure_thermal` when Ti is available.
- Ti fit metadata: `ion.0.temperature_fit.{rho_tor_norm,measured,reconstructed}` in the
  same pattern as the existing electron `*_fit` blocks (measured points from
  `charge_exchange.channel[:].ion[i].t_i.data` at the matched time).
- `core_profiles.ids_properties.homogeneous_time = 1` and `core_profiles.time`
  maintained consistently with appended profiles_1d entries.

### 2. `notebooks/build_kinetic_profiles.py` — new CLI script

`build_omas_for_kinetic_efit.py` minus all EFIT-bundle packaging:

- Args: `--shots` (default 48224 48226 48233), `--times` (default 299 300 301 ms;
  298 excluded as artifact), `--source` (`auto`|`ids`|`ces`, auto = pick by which MAT
  file exists), `--outdir` (default `notebooks/kinetic_profiles/`), fit-mode flags
  (`--te-mode polynomial`, `--ne-mode free_exponential`, `--ti-mode polynomial`,
  `--vtor-mode polynomial`).
- Per (shot, time): load magnetic-EFIT gfile (`vaft/data/efit_scaled_f/<shot>/`),
  build ODS as in the pipeline diagram, call the extended `core_profiles()`, then:
  - `save_omas_json(ods, outdir/ods_<shot>_<time>ms.json)`
  - IMAS IDS export via `vaft.imas` — default netCDF (`ods_<shot>_<time>ms.nc`); AL5
    data-entry export only if netCDF is unavailable in the runtime.
    **No writes to the remote HSDS/MySQL database.**
- `summary.csv` with ne0/Te0/Ti0/Vtor0/p_th0 per slice.
- TS MAT-key adaptation (`Te_<shot>` → `Te` etc.) kept as a helper.

### 3. Deletions

- `notebooks/ts_ids_efit_pressure.py`
- `notebooks/export_kinetic_pressure_for_efit.py`
- `notebooks/generate_scaled_efit_ids_plots.py`
- `notebooks/build_omas_for_kinetic_efit.py` (superseded by
  `build_kinetic_profiles.py`)

Kept: bundle artifacts (`efit_*_bundle*`, comparison md) and
`vaft/data/efit_scaled_f/` gfiles (equilibrium input required for mapping); all are
untracked and will not be committed.

### 4. TS review integration

A parallel review of the TS fitting chain is running. Confirmed findings that touch
`fit_profile` / `profile_fitting_*` / the TS builder are fixed **before** the new chain
is built on top of them.

## Error handling

- Missing MAT file or gfile for a (shot, time): skip with a clear message; continue
  the sweep (matches current script behavior).
- `curve_fit` non-convergence: surface the failure (do not silently return garbage);
  exact policy follows the TS review findings.
- CES data with a single time point: time matching falls back to nearest (existing
  `profile_fitting_charge_exchange` behavior).

## Testing / verification

- Unit-level: synthetic-ODS test for the extended `core_profiles()` (Ti/Vtor written,
  pressure_thermal computed, grid taken from equilibrium, duplicate-time replacement
  still works).
- End-to-end: run `build_kinetic_profiles.py` for 48224/48226/48233 × 299/300/301 with
  the conda `vaft` interpreter (`/opt/anaconda3/envs/vaft/bin/python`) and check the
  produced JSON/IDS files and summary numbers are physical.

## Out of scope

- EFIT pressure-constraint files (k-file/p-file) and kinetic-vs-EFIT comparisons.
- Remote database (HSDS/MySQL) writes.
- Impurity ion species as separate `core_profiles.ion` entries (single H+ with ne≈ni).

# 001 — Canonical subject/view taxonomy and naming migration (issue #251)

Status: **accepted** (decisions ratified 2026-09-01)

## Context

Canonical plot identity moves from `<domain>_<view>_<quantity>` (IDS-centered)
to `<subject>_<view>[_<quantity>]` (physics-centered). `domain` stays on
`PlotSpec` as IDS-ownership metadata; `subject` becomes the public identity.
See `000-architecture.md` for layering.

## Decisions

### B1 — Subject vocabulary lives in `vaft/plot/taxonomy.py`

Frozen dataclasses, mirroring the registry style:

- `Subject(name, aliases, description)` — canonical subject + strict aliases.
- `QuantityFamily(name, members, aliases)` — named group of distinct quantities
  (not synonyms; never stored in the alias map).
- Module-level registries `SUBJECTS`, `FAMILIES` and resolvers
  `resolve_subject(term)` / `resolve_family(term)`; alias collisions raise at
  import time; unknown terms raise `KeyError` with suggestions.

`PlotSpec` gains a `subject: str` field validated against `SUBJECTS`.
Registration with an unknown subject fails loudly.

### B2 — Aliases resolve only in discovery/search

Strict aliases (`ip` → `plasma_current`) are consumed by `available_plots()`
queries (#262) and documentation. They are **not** resolvable at
`render()`/adapter call sites; rename compatibility is a separate mechanism
(B4). Alias registration rule: only when both terms confidently denote the same
concept (`Rogowski coil` ≠ `plasma_current`).

### B3 — Big-bang rename, one PR

All 34 stem renames (table below) land in one mechanical PR together with the
migration rows, so the grammar test enforces exactly one naming scheme at every
commit. The other 69 stems already fit the subject grammar and do not move.

### B4 — Compatibility via migration-table wrappers

Old names become DEPRECATED rows in `vaft/plot/_migration.py` with thin
wrappers that warn and delegate; window: two minor releases (A4). Registry
stays at 103 canonical specs.

### B5 — `evolution` view

`VIEWS` gains `evolution` = `quantity(t, x)` (time × spatial coordinate, e.g.
`n_e(t, rho)`). Distinct from `time` (`quantity(t)`), `profile`
(`quantity(x)` at one time), and `spectrogram` (`spectral_quantity(t, f)`).
No existing spec re-views; first users are future
`electron_density_evolution` / `electron_temperature_evolution` /
`q_evolution` plots. `interactive`, `3d`, `comparison`, `validation` are
explicitly **not** views (they are capabilities, #261).

### B6 — Tutorials and notebooks update in the rename PR

`tutorial/` and `notebooks/` switch to canonical names in the same PR; no
window where documentation teaches deprecated API.

### Naming judgment calls (maintainer-decided)

- **Ip / diamagnetic flux**: the diagnostic plots take the physical subject
  (`magnetics_time_ip` → `plasma_current_time`,
  `magnetics_time_diamagnetic_flux` → `diamagnetic_flux_time`); the
  `equilibrium_time_plasma_current` / `equilibrium_time_diamagnetic_flux`
  reconstruction views keep the `equilibrium` subject. Phase G overlays the
  reconstruction as synthetic points on the diagnostic waveform (#261 §9).
- **Kinetic quantities**: `core_profiles` electron/ion plots take physical
  subjects (`electron_density_*`, `electron_temperature_*`,
  `ion_temperature_profile`, `thermal_pressure_profile`); Thomson /
  charge-exchange plots stay diagnostic-named. Phase G makes the
  physical-subject plots source-aware. `core_profiles` remains a registered
  subject only for the `core_profiles_time_volume_averaged` composite
  (revisited in phase G).
- **`soft_x_rays` stays plural** (matches the IDS and current stems);
  `soft_x_ray` and `sxr` are aliases.
- **`summary`**: only `summary_time_beta` moves — it reads the equilibrium IDS
  and is exactly the beta family plot → `equilibrium_time_beta`. The
  genuinely cross-IDS composites (`summary_time_energy`,
  `summary_time_power_balance`, `summary_time_voltage_consumption`) keep the
  `summary` subject and are revisited as overview recipes in phase G.
- **`limiter_current_time`** (subject `limiter_current`, alias
  `limiter_shunt`), mirroring `plasma_current_time`.
- **IMPA** is a subject (`impa`, alias `hall_probe_array`):
  `impa_time_field`, `impa_time_voltage`, `impa_overview`,
  `impa_profile_field`.
- **`electromagnetics_time_current` → `current_overview`** (subject
  `current`): a purpose-driven currents summary (plasma + PF coil; vessel eddy
  may join in phase G). The `electromagnetics` domain label leaves the public
  name.
- **`coil_3d`** (aliases `coils_non_axisymmetric`, `3d_coil`) for the
  non-axisymmetric coil views. RMP/error-field are uses, not synonyms — not
  aliased.
- **`w_mag` keeps its name**: the recipe reads
  `global_quantities.energy_mag` (generic magnetic stored energy); `w_mag_p`
  is only introduced if/when a specifically poloidal energy is stored.

## Subject vocabulary (initial)

| Subject | Aliases | Kind |
|---|---|---|
| plasma_current | ip, I_p | physical quantity |
| diamagnetic_flux | | physical quantity |
| electron_density | ne, n_e | physical quantity |
| electron_temperature | te, T_e | physical quantity |
| ion_temperature | ti, T_i | physical quantity |
| thermal_pressure | kinetic_pressure | physical quantity |
| limiter_current | limiter_shunt | physical quantity |
| current | | quantity group (overview only) |
| flux_loop | | diagnostic |
| b_field_probe | b_pol_probe, bpol_probe | diagnostic |
| mirnov | mirnov_coil | diagnostic |
| impa | hall_probe_array | diagnostic |
| soft_x_rays | soft_x_ray, sxr | diagnostic |
| interferometer | | diagnostic |
| thomson_scattering | thomson | diagnostic |
| charge_exchange | | diagnostic |
| spectrometer_uv | | diagnostic |
| barometry | | diagnostic |
| camera_visible | | diagnostic |
| magnetics | | diagnostic system |
| wall | | machine |
| pf_coil | pf_active | machine |
| tf_coil | tf | machine |
| passive_structure | pf_passive | machine |
| coil_3d | coils_non_axisymmetric, 3d_coil | machine |
| machine | | machine |
| equilibrium | | reconstruction |
| core_profiles | | reconstruction |
| mhd_linear | | model |
| chease | | code |
| summary | | composite |

Equilibrium quantity aliases (#251 §11): `q`←safety_factor, `q0`←q_axis,
`q95`, `beta_n`←beta_normal/beta_norm, `beta_t`←beta_tor/beta_toroidal,
`beta_p`←beta_pol/beta_poloidal, `li`←internal_inductance,
`w_mhd`←mhd_energy, `w_mag`←magnetic_energy, `w_tot`←total_energy.

Quantity families (not aliases):

```
beta   = { beta_n, beta_p, beta_t }     family plot: equilibrium_time_beta
energy = { w_mhd, w_mag, w_tot }        aliases: w   (family plot: phase G)
```

## Rename table (34 stems)

| Old canonical stem | New canonical stem |
|---|---|
| magnetics_time_ip | plasma_current_time |
| magnetics_time_diamagnetic_flux | diamagnetic_flux_time |
| magnetics_time_flux_loop_flux | flux_loop_time_flux |
| magnetics_time_flux_loop_voltage | flux_loop_time_voltage |
| magnetics_time_b_field_pol_probe_field | b_field_probe_time_field |
| magnetics_time_mirnov_voltage | mirnov_time_voltage |
| magnetics_spectrum_mirnov | mirnov_spectrum |
| magnetics_spectrogram_mirnov | mirnov_spectrogram |
| magnetics_time_limiter_current | limiter_current_time |
| magnetics_time_impa_field | impa_time_field |
| magnetics_time_impa_voltage | impa_time_voltage |
| magnetics_overview_impa | impa_overview |
| magnetics_profile_impa_tf | impa_profile_field |
| pf_active_time_current | pf_coil_time_current |
| pf_active_time_current_turns | pf_coil_time_current_turns |
| pf_active_geometry_poloidal | pf_coil_geometry_poloidal |
| pf_passive_geometry_poloidal | passive_structure_geometry_poloidal |
| tf_time_coil_current | tf_coil_time_current |
| tf_time_b_field_tor | tf_coil_time_b_t |
| tf_time_b_field_tor_vacuum_r | tf_coil_time_b_t_vacuum_r |
| electromagnetics_time_current | current_overview |
| equilibrium_time_beta_pol | equilibrium_time_beta_p |
| equilibrium_time_beta_tor | equilibrium_time_beta_t |
| summary_time_beta | equilibrium_time_beta |
| core_profiles_time_electron_density | electron_density_time |
| core_profiles_time_electron_temperature | electron_temperature_time |
| core_profiles_profile_electron_density | electron_density_profile |
| core_profiles_profile_electron_temperature | electron_temperature_profile |
| core_profiles_profile_ion_temperature | ion_temperature_profile |
| core_profiles_profile_pressure | thermal_pressure_profile |
| core_profiles_field_electron_density | electron_density_field |
| core_profiles_field_electron_temperature | electron_temperature_field |
| coils_non_axisymmetric_geometry3d | coil_3d_geometry3d |
| coils_non_axisymmetric_geometry_topview | coil_3d_geometry_topview |

The remaining 69 stems keep their names; their leading token becomes their
registered subject (`equilibrium_*`, `thomson_scattering_*`, `soft_x_rays_*`,
`interferometer_*`, `charge_exchange_*`, `camera_visible_*`, `machine_*`,
`wall_*`, `barometry_*`, `spectrometer_uv_*`, `mhd_linear_*`, `chease_*`,
`magnetics_geometry_poloidal` + `magnetics_overview*`, `summary_time_energy` /
`_power_balance` / `_voltage_consumption`,
`core_profiles_time_volume_averaged`).

Phase-G forward notes: `camera_visible_image_efit_overlay` /
`_image_field_line` fold into `plot_camera_visible_image(overlay=…)`;
`equilibrium_overview` is redefined as the representative-slice summary (#261).
Neither changes names in phase B.

## Implementation plan

- **PR-B1**: `vaft/plot/taxonomy.py` (subjects, aliases, families, resolvers)
  + `PlotSpec.subject` (validated; every registration site declares it) +
  `VIEWS += ("evolution",)` + taxonomy tests. Names unchanged; grammar test
  still domain-based.
- **PR-B2**: the rename table above applied to registry names, recipe keys,
  `vaft.omas` adapters, `vaft/validation.py` references; DEPRECATED wrappers +
  migration rows for the 34 old names; grammar test switches to
  `name == f"{subject}_{view}"` or `f"{subject}_{view}_{quantity}"`;
  tutorials/notebooks updated.

## Test plan

- Taxonomy: alias uniqueness across subjects, families disjoint from the alias
  map, resolver determinism, unknown-term errors.
- Registry: every spec's `subject` registered; grammar
  `subject_view[_quantity]` (after B2); `evolution` accepted; `interactive`
  etc. rejected as views.
- Migration: 34 new DEPRECATED rows; wrappers warn and delegate
  (`test_plot_migration.py` machinery).

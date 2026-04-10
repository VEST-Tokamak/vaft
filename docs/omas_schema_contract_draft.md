# OMAS Schema Contract Draft

## Purpose

This document freezes the OMAS contract for the VEST port.
The source of truth is the official OMAS schema, and the canonical
implementation surface inside this repository is `vaft.machine_mapping`.

References:

- OMAS schema index: <https://gafusion.github.io/omas/schema.html>
- PF active schema: <https://gafusion.github.io/omas/schema/schema_pf%20active.html>
- Whole schema: <https://gafusion.github.io/omas/schema/whole_schema.html>

## Core Decisions

1. The official OMAS schema is the contract.
2. `/Users/yun/git/vest_database/OMAS/` is the donor implementation.
3. `vaft.machine_mapping` is the only canonical OMAS builder/orchestration layer.
4. New IDS builders must emit canonical OMAS paths directly.
5. Thin aliases may remain only when they preserve historical flat module names, such as `filterscope`.

## Global Rules

1. Use IDS root paths defined by OMAS.
2. When the IDS uses homogeneous time, write the OMAS value required by that IDS.
3. For homogeneous-time IDS, prefer IDS-level shared time arrays such as `pf_active.time`, `magnetics.time`, and `tf.time` as the governing time axis for time series.
4. Per-signal `*.time` leaves are optional unless the donor builder explicitly emits them or the IDS really needs signal-local time.
5. Do not invent VAFT-only namespaces when an OMAS IDS already exists.
6. Flat `machine_mapping` entrypoints may keep historical names, but implementation must stay inside the existing core package structure.

## Canonical IDS Targets

### `dataset_description`

- Required roots:
  - `dataset_description.data_entry.machine`
  - `dataset_description.data_entry.pulse`
  - `dataset_description.data_entry.run`
  - `dataset_description.ids_properties.*`
- Donor source:
  - `/Users/yun/git/vest_database/OMAS/vest_ods_dynamic.py`

### `pf_active`

- Required roots:
  - `pf_active.ids_properties.homogeneous_time`
  - `pf_active.time`
  - `pf_active.coil.{i}.name`
  - `pf_active.coil.{i}.identifier`
  - `pf_active.coil.{i}.current.data`
  - `pf_active.coil.{i}.element.{j}.turns_with_sign`
  - `pf_active.coil.{i}.element.{j}.geometry.rectangle.*`
- Canonical donor:
  - `/Users/yun/git/vest_database/OMAS/ods_builders/geometry/vest_pf_active.py`

### `magnetics`

- Required roots:
  - `magnetics.ids_properties.homogeneous_time`
  - `magnetics.time`
  - `magnetics.ip.0.data`
  - `magnetics.diamagnetic_flux.0.data`
  - `magnetics.flux_loop.{i}.flux.data`
  - `magnetics.flux_loop.{i}.position.{k}.r`
  - `magnetics.flux_loop.{i}.position.{k}.z`
  - `magnetics.b_field_pol_probe.{i}.field.data`
  - `magnetics.b_field_pol_probe.{i}.position.r`
  - `magnetics.b_field_pol_probe.{i}.position.z`
- Time semantics:
  - `magnetics.time` is the canonical governing time axis.
  - `magnetics.ip.0.time` and `magnetics.diamagnetic_flux.0.time` may be present for donor parity.
- Canonical donor:
  - `/Users/yun/git/vest_database/OMAS/physics_models/vest_magnetics.py`

### `tf`

- Required roots:
  - `tf.ids_properties.homogeneous_time`
  - `tf.time`
  - `tf.coil.0.current.data`
  - `tf.b_field_tor_vacuum_r.data`
  - `tf.r0`
- Time semantics:
  - `tf.time` is the canonical governing time axis.
  - `tf.coil.0.current.time` and `tf.b_field_tor_vacuum_r.time` are optional donor-parity outputs.
- Canonical donor:
  - `/Users/yun/git/vest_database/OMAS/ods_builders/diagnostics/vest_tf.py`

### `barometry`

- Required roots:
  - `barometry.ids_properties.homogeneous_time`
  - `barometry.gauge.0.name`
  - `barometry.gauge.0.pressure.time`
  - `barometry.gauge.0.pressure.data`
- Time semantics:
  - First-port donor parity keeps `barometry.gauge.0.pressure.time` as the governing time axis.
- Canonical donor:
  - `/Users/yun/git/vest_database/OMAS/ods_builders/diagnostics/vest_barometry.py`

### `spectrometer_uv`

- Required roots:
  - `spectrometer_uv.ids_properties.homogeneous_time`
  - `spectrometer_uv.time`
  - `spectrometer_uv.channel.{i}.name`
  - `spectrometer_uv.channel.{i}.processed_line.{j}.label`
  - `spectrometer_uv.channel.{i}.processed_line.{j}.wavelength_central`
  - `spectrometer_uv.channel.{i}.processed_line.{j}.intensity.data`
- Time semantics:
  - Canonical donor behavior uses `spectrometer_uv.time` as the governing time axis for all processed lines.
- Canonical donor:
  - `/Users/yun/git/vest_database/OMAS/ods_builders/diagnostics/vest_spectrometer_uv.py`

### `thomson_scattering`

- Required roots:
  - `thomson_scattering.ids_properties.homogeneous_time`
  - `thomson_scattering.time`
  - `thomson_scattering.channel.{i}.position.r`
  - `thomson_scattering.channel.{i}.position.z`
  - `thomson_scattering.channel.{i}.t_e.data`
  - `thomson_scattering.channel.{i}.n_e.data`
- Canonical donor:
  - `/Users/yun/git/vest_database/OMAS/ods_builders/diagnostics/vest_thomson_scattering.py`

### `charge_exchange`

- Required roots:
  - `charge_exchange.ids_properties.homogeneous_time`
  - `charge_exchange.time`
  - `charge_exchange.channel.{i}.position.r.data`
  - `charge_exchange.channel.{i}.ion.{j}.label`
  - `charge_exchange.channel.{i}.ion.{j}.intensity.data`
  - `charge_exchange.channel.{i}.ion.{j}.velocity_tor.data`
  - `charge_exchange.channel.{i}.ion.{j}.t_i.data`
- Canonical donor:
  - `/Users/yun/git/vest_database/OMAS/ods_builders/diagnostics/vest_charge_exchange.py`

## Non-goals for the First Port

- `mhd_linear` workflow parity
- stability runner parity
- script-by-script compatibility with every legacy entrypoint

## Contract Validation Strategy

1. Build shot fixtures for a small number of representative shots.
2. Validate required paths exist.
3. Validate each time series has an effective governing time axis, preferring IDS-level `*.time` for homogeneous-time IDS.
4. Validate required paths have the expected rank and approximate shapes.
5. Validate downstream readers in `vaft.plot`, `vaft.omas`, and `vaft.code` run against the canonical ODS without alias hacks.

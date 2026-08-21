# VAFT API layer boundaries

Issue [#103](https://github.com/VEST-Tokamak/vaft/issues/103) defines the
target data flow:

```text
raw source -> VEST machine mapping -> generic processing -> OMAS/IMAS mapping
```

New APIs must use the layer that owns their responsibility:

- `vaft.machine_mapping` owns VEST hardware, field codes, acquisition eras,
  calibration policy, geometry, and legacy source formats.
- `vaft.process` owns machine-independent numerical and signal operations.
- `vaft.omas` and `vaft.imas` own data-dictionary paths and representation
  conversion.
- `vfit_*` identifies legacy compatibility only; it is not a naming convention
  for new VAFT APIs.

## Initial compatibility inventory

The first migration phase removes legacy names from
`vaft.machine_mapping.__all__`. Direct package imports continue to work with a
`DeprecationWarning`, while direct diagnostic-module imports remain available
for code that cannot migrate yet.

| Legacy surface | Classification | Canonical direction |
| --- | --- | --- |
| `vfit_signal_start_end`, `vfit_signal_startend` | Generic processing | `vaft.process.detect_active_window` |
| `signal_onoffset` | Generic processing alias | `vaft.process.signal_on_offset` |
| `VEST_CoilCurrentNoiseReduction` | Generic processing alias | `vaft.process.vest_coil_current_noise_reduction` |
| `vfit_barometry_*` | Mixed source/schema mapper | `barometry` now; split source policy from schema mapping later |
| `vfit_charge_exchange`, misspelled ion-Doppler alias | Mixed legacy-file/schema mapper | `charge_exchange` |
| `vfit_dataset_description` | Schema mapper | `dataset_description` |
| `vfit_filterscope` | Mixed source/schema mapper | `spectrometer_uv` |
| `vfit_magnetics_*`, `vfit_mirnov_raw_dynamic` | Mixed source/process/schema mapper | `magnetics` and diagnostic-specific raw mappers |
| `vfit_pf_active_*` | Mixed geometry/source/schema mapper | `pf_active` |
| `vfit_soft_x_rays_*` | Schema mapper | `soft_x_rays` |
| `vfit_tf_dynamic`, `vfit_tf_static` | Mixed source/process/schema mapper | `tf` |
| `vfit_thomson_scattering_*` | Mixed legacy-file/schema mapper | `thomson_scattering` |
| `vfit_md`, `vfit_plasma_current`, `vfit_pf`, `vfit_tf_current`, `vfit_tf_bt_r` | VEST source policy | Keep in diagnostic modules until canonical physical-data APIs exist |
| `vfit_plasma_mgods_startend` | ODS-aware processing | Split ODS extraction from generic window detection |
| `VEST_DiamagneticFlux` and mixed-case aliases | Compatibility aliases | Use their lowercase snake-case functions |

The unimplemented `machine_mapping.pf_plasma` placeholder is not a supported
API and has been removed. Further phases should migrate one diagnostic at a
time, preserving numerical behavior with focused tests before moving its
schema writer into `vaft.omas` or `vaft.imas`.

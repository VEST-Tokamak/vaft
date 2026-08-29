# vaft/data data catalog

`vaft/data` contains runtime resources and repository-only samples. Data assets
are grouped one level deep only; Python package files stay at this directory
root. The GitHub repository contains every file listed below; the PyPI
distribution includes only the runtime geometry resources, GPEC templates,
`legacy/sql_table.txt`, `legacy/langmuir_probe_positions.csv`, and
`omas/39915.json`. Clone the repository to access the archived EFIT, kinetic-EFIT, IMAS,
legacy diagnostic, digitizer, and additional OMAS samples.

## Layout

| Directory | Files | Purpose |
| --- | --- | --- |
| `geometry/` | `Coil_info.mat`, `MD.yaml`, `VEST_DiscretizedCoilGeometry_Full_ver_1906.mat`, `VEST_DiscretizedCoilGeometry_Full_ver_2507.mat`, `VEST_em_coupling_pf_versions.npz`, `VEST_static_geometry.json.gz`, `VEST_MagneticsGeometry_Full_ver_2302.yaml`, `line_of_sight_endpoints.csv`, `table.yaml` | VEST magnetic, PF, electromagnetic-coupling, wall/passive, and soft X-ray geometry metadata |
| `efit/` | `g039020.031180`, `g039915.00317`, `g039915.00319`, `g040330.00320`, `g040330.00321`, `g040330.00323`, `a039915.00319`, EFIT table files | GEQDSK/AEQDSK samples and EFIT reference tables |
| `omas/` | `39915.json`, `41524.json`, `41672.json`, `thomson_scattering.json` | OMAS/ODS sample and contract-test payloads |
| `kineticEfit/` | `g048224.00300`, `g048224.00300.kinetic_efit`, `g048224.00300.chease`, `NeTe_48224.mat`, `IDS_48224.mat`, `ods_48224_300ms.json` | Paired kinetic-EFIT sample for shot 48224 @ 300 ms (equilibrium + Thomson + ion Doppler) and the stored kinetic-profile ODS |
| `imas/` | `vest_imas_3.40.1.nc` | IMAS-format sample container |
| `legacy/` | `41514.h5`, `46051_NeTe.mat`, `CES_47514.mat`, `IDS_47518.mat`, `NeTe_Shot39915_v9_rev.mat`, `digitizer_17592_45531.csv`, `digitizer_22577_45531.csv`, `47230_056789_LID_1_100.mat`, `47230_ALL_LID_1_100.mat`, `shot_44740.json.gz`, `langmuir_probe_positions.csv`, `langmuir_probes_42699.json.gz`, `sql_table.txt` | Legacy diagnostic samples, raw SQL dump, and DB lookup table |
| `gpec/` | `*.in`, `vest_*.dat` | VEST GPEC-suite namelist templates and coil data |

## Access

Use `vaft.data.resources.data_path()` with explicit category paths:

```python
from vaft.data.resources import data_path

ods_path = data_path("omas/39915.json")
```

Repository-only samples such as `efit/g039915.00319` and
`legacy/46051_NeTe.mat` are available after cloning the repository, not after
`pip install vaft`.

Flat calls such as `data_path("39915.json")` are intentionally unsupported.
Deleted duplicate assets from older checkouts are not recreated here.

`geometry/VEST_em_coupling_pf_versions.npz` is a compact extraction of the
active-active, passive-active, and passive-passive matrices from the legacy
VEST coupling assets. `VEST_static_geometry.json.gz` contains only wall and
passive-loop geometry; it deliberately contains no shot waveform. The source
SHA-256 checksums for the geometry-dependent coupling assets are
`0f0d34ea98a14c32791db7bf5804bce537782993ab1cd7a9ca809b62d925eddf`
(1909) and
`71c10a410b4bb180d5366f1bb7191a1a14e9277142af2cd20246af181f5b6830`
(2507). The 1909 matrix is the coupling counterpart of VAFT's 1906 PF
geometry; the differing suffixes are retained from the source asset names.
`VEST_MagneticsGeometry_Full_ver_2302.yaml` retains its historical filename
for API compatibility, while its source metadata, channel order, and
calibration values reflect the production 2409 magnetic geometry.

`kineticEfit/ods_48224_300ms.json` is the canonical kinetic-profile ODS sample: the
result of running the kinetic chain once on shot 48224 at 300 ms with polynomial
`T_e`/`n_e`/`T_i`/`V_tor` fits. It carries the g-file equilibrium, the mapped
`thomson_scattering` and `charge_exchange` channels, and the generated
`core_profiles.profiles_1d.0`, so notebooks, examples, and tests can use
representative profiles offline without `omfit_classes` or the `.mat` inputs
(`notebooks/kinetic_efit_end_to_end.ipynb` loads it and only rebuilds when it is
absent). The recipe below is deterministic -- rerunning it reproduces the
committed file byte for byte, provided `user` stays pinned (`dataset_description`
otherwise stamps `$USER`, which would turn a regeneration into a spurious 2 MB
diff). Regenerate it with:

```python
import vaft
from vaft.data.resources import data_path

vaft.apply_omfit_compat_patches()
from omas import save_omas_json
from omfit_classes.omfit_eqdsk import OMFITgeqdsk

from vaft.code.efit import build_kinetic_core_profiles
from vaft.machine_mapping.charge_exchange import charge_exchange
from vaft.machine_mapping.dataset_description import dataset_description
from vaft.machine_mapping.thomson_scattering import thomson_scattering

root = data_path("kineticEfit")
geq = OMFITgeqdsk(str(root / "g048224.00300"))
geq["fluxSurfaces"].load()
ods = geq.to_omas()
ods["equilibrium.ids_properties.homogeneous_time"] = 1
dataset_description(
    ods, source=48224,
    options={"source_type": "shot",
             "user": "vaft",  # fixed, so regeneration does not stamp $USER
             "description": "VAFT canonical kinetic-profile sample (shot 48224 @ 300 ms)"},
)
thomson_scattering(ods, 48224, str(root / "NeTe_48224.mat"))
charge_exchange(ods, shotnumber=48224, options="ids", mat_file=str(root / "IDS_48224.mat"))
ods = build_kinetic_core_profiles(
    ods, geq, 300.0,
    te_mode="polynomial", ne_mode="polynomial",
    ti_mode="polynomial", vtor_mode="polynomial",
)
save_omas_json(ods, str(root / "ods_48224_300ms.json"))
```

`legacy/47230_056789_LID_1_100.mat` and `legacy/47230_ALL_LID_1_100.mat` are
downsampled (1/100) postprocessed line-integrated-density samples for shot
47230, used by `vaft.machine_mapping.interferometer` (94 GHz horizontal and
282 GHz vertical systems respectively).

`legacy/langmuir_probe_positions.csv` is the VEST shot-log table of measured
mid/upper triple-Langmuir-probe radial positions (`shot`, `mid TP
position[m]`, `upper TP position[m]`); `vaft.machine_mapping.langmuir_probes`
loads it by default to populate `langmuir_probes.embedded.{0,1}.position.r`
for issue #152. `legacy/langmuir_probes_42699.json.gz` is a repository-only
sample `langmuir_probes` IDS built from that pipeline against shot 42699's
real SQL-backed raw signals (both mid and upper assemblies present, plasma
pulse near t=0.35-0.46 s).

`legacy/sxr_te_ratio_be_al.csv` is the VEST soft X-ray two-filter
electron-temperature calibration table (`te` [eV], `ratio` = Be/Al filtered
signal ratio), used by `vaft.process.soft_x_rays.load_te_ratio_calibration`.
It originates from the validated VEST SXR Viewer analysis tool (`ratio.csv`,
2024-11-22) documented in the 2026 VEST SXR thesis presentation; the ratio ->
Te inversion assumes the Al channel gain correction and validity threshold
applied by `sxr_electron_temperature`.

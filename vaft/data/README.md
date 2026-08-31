# vaft/data data catalog

`vaft/data` contains runtime resources, a compact paired reference sample, and
repository-only full-shot examples and regeneration inputs. The PyPI
distribution includes runtime resources, all sample manifests, and both
compact representations of shot 39915. Clone the repository to access the
full native IMAS examples for shots 41524 and 41672, archived EFIT and
kinetic-EFIT data, and legacy diagnostic and digitizer samples.

## Layout

| Directory | Files | Purpose |
| --- | --- | --- |
| `geometry/` | `Coil_info.mat`, `MD.yaml`, `VEST_DiscretizedCoilGeometry_Full_ver_1906.mat`, `VEST_DiscretizedCoilGeometry_Full_ver_2507.mat`, `VEST_em_coupling_pf_versions.npz`, `VEST_static_geometry.json.gz`, `VEST_MagneticsGeometry_Full_ver_2302.yaml`, `line_of_sight_endpoints.csv`, `table.yaml` | VEST magnetic, PF, electromagnetic-coupling, wall/passive, and soft X-ray geometry metadata |
| `efit/` | `g039020.031180`, `g039915.00317`, `g039915.00319`, `g040330.00320`, `g040330.00321`, `g040330.00323`, `a039915.00319`, EFIT table files | GEQDSK/AEQDSK samples and EFIT reference tables |
| `samples/39915/` | `manifest.yaml`, `omas.json.gz`, `imas.nc` | One compact logical reference dataset in paired OMAS and native IMAS representations |
| `samples/39915/source/` | frozen raw input, configuration, stage manifests, canonical ODS | Repository-only regeneration inputs through the EFIT stage |
| `samples/41524/` | `manifest.yaml`, `imas.nc` | Complete repository-only native IMAS example composed from the current pipeline through EFIT |
| `samples/41524/source/` | frozen SQL raw input, configuration, stage manifests, canonical ODS | Repository-only regeneration inputs for the 41524 pipeline run through EFIT |
| `samples/41672/` | `manifest.yaml`, `imas.nc` | Complete repository-only native IMAS example composed from the current pipeline through EFIT |
| `samples/41672/source/` | frozen SQL raw input, configuration, stage manifests, canonical ODS | Repository-only regeneration inputs for the 41672 pipeline run through EFIT |
| `kineticEfit/` | `g048224.00300`, `g048224.00300.kinetic_efit`, `g048224.00300.chease`, `NeTe_48224.mat`, `IDS_48224.mat`, `ods_48224_300ms.json` | Paired kinetic-EFIT sample for shot 48224 @ 300 ms (equilibrium + Thomson + ion Doppler) and the stored kinetic-profile ODS |
| `legacy/` | `41514.h5`, `46051_NeTe.mat`, `CES_47514.mat`, `IDS_47518.mat`, `NeTe_Shot39915_v9_rev.mat`, `digitizer_17592_45531.csv`, `digitizer_22577_45531.csv`, `47230_056789_LID_1_100.mat`, `47230_ALL_LID_1_100.mat`, `shot_44740.json.gz`, `shot_45531.json.gz`, `langmuir_probe_positions.csv`, `langmuir_probes_42699.json.gz`, `sql_table.txt` | Legacy diagnostic samples, raw SQL dump, and DB lookup table |
| `gpec/` | `*.in`, `vest_*.dat` | VEST GPEC-suite namelist templates and canonical 3D coil geometry |

## Access

Discover samples through the representation-neutral registry and pass the
returned path to the adapter being tested:

```python
import vaft

omas_artifact = vaft.data.sample(39915, representation="omas")
imas_artifact = vaft.data.sample(39915, representation="imas")
ods = vaft.omas.load(omas_artifact)
with vaft.imas.load(imas_artifact) as handle:
    equilibrium = handle.get("equilibrium")

# Storage is independent from the consuming adapter. Both calls resolve the
# same repository-only IMAS artifact and either loader can consume it.
legacy_as_omas = vaft.omas.load(vaft.data.sample(41524, "omas"))
with vaft.imas.load(vaft.data.sample(41524, "imas")) as handle:
    legacy_equilibrium = handle.get("equilibrium")
```

Repository-only samples such as `efit/g039915.00319` and
`legacy/46051_NeTe.mat` are available after cloning the repository, not after
`pip install vaft`.

The repository copy of the 39915 pair retains every successful EFIT time
slice. Its wheel build replaces those checkout artifacts with a separately
manifested three-slice variant generated from the same canonical ODS, keeping
the installed package small without making a repository checkout less useful.
`workflow/reference_validation/generate_paired_sample.py` generates both
forms; neither representation is maintained independently. The 39915 sample
carries `pf_passive` geometry only -- all 950 loop outlines, resistances and
resistivities, but no loop currents -- and omits `em_coupling` entirely. Both
are derivable: the coupling matrices come from
`geometry/VEST_em_coupling_pf_versions.npz` through
`vaft.machine_mapping.em_coupling`, keyed by the shot's PF geometry version.
Materializing them into the sample would store a reconstruction rather than a
measurement, so an example that needs them should call that API instead. Shots 41524 and 41672 are
regenerated from frozen current-pipeline products by
`generate_pipeline_imas_sample.py`; each retains its successful EFIT slices
and records unavailable optional channels and unsuccessful EFIT times in its
manifest. Both native samples normalize DD-only metadata and use +Bz probe
direction metadata. The old format-grouped JSON files and unrelated IMAS
NetCDF sample are intentionally not recreated.

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

`gpec/vest_UP.dat`, `gpec/vest_MID.dat`, and `gpec/vest_LOW.dat` are the
canonical VEST non-axisymmetric 3D coil geometries in GPEC coil format. Each
file's header is four fields `ncoil nsec npts nw` (coils per set, sections,
points per coil, winding-turn multiplier), followed by `ncoil * npts` rows of
Cartesian `x y z` coordinates in metres. Each coil is a *single* closed
geometric filament; the physical winding count is carried by `nw = 20`, so
the magnetic effect is 20 x the per-turn current. All three sets have 6
sectors at 60-degree toroidal spacing: `UP` and `LOW` are the upper
(z = +0.62 to +1.12 m) and lower (z = -1.12 to -0.62 m) saddle arrays
(420 points per filament), and `MID` is the mid-plane 12-inch circular coil
array (loop radius = 0.15 m at machine R = 0.80 m, 100 points per filament),
taken from the shot-48226 @ 300 ms ideal-GPEC reference run
(`VEST_3Dcoil_12inch20turn_200A_48226_300ms` bundle, where it was named
`vest_12inch_20turn.dat`). The `UP`/`LOW` headers previously carried an
erroneous `nw = 100.00` (their bodies were already identical to the corrected
files); the geometry and the 20-turn interpretation were reviewed with 3D
coil developer Gwang-geun Seo. `vaft.machine_mapping.coil_geometry_3d` is the
canonical loader; the metadata (identifiers, sector angles, provenance) lives
in its `VEST_3D_COIL_SETS` constant.

`kineticEfit/ods_48224_300ms.json` is the canonical kinetic-profile ODS sample: the
result of running the kinetic chain once on shot 48224 at 300 ms with polynomial
`T_e`/`n_e`/`T_i`/`V_tor` fits. It carries the g-file equilibrium, the mapped
`thomson_scattering` and `charge_exchange` channels, and the generated
`core_profiles.profiles_1d.0`, so notebooks, examples, and tests can use
representative profiles offline without the `.mat` inputs
(`notebooks/kinetic_efit_end_to_end.ipynb` loads it and only rebuilds when it is
absent).

The committed file was produced by an earlier version of the recipe that read
the g-file through `omfit_classes`, which VAFT no longer depends on (issue
#192). The native recipe below is deterministic and produces the same
`core_profiles` content, but **not** the same file: OMFIT supplied its
flux-surface solve's derived equilibrium quantities
(`profiles_1d.gm1`..`gm9`, `dvolume_dpsi`, `b_field_average`, `area`,
`elongation`, `global_quantities.beta_pol`/`li_3`, the X-point locations, and
~550 other leaves). Since issue #236 the native `to_omas` also writes `psi`
in Wb as the IMAS DD requires (the earlier Wb/rad discrepancy this paragraph
used to document is fixed); `from_omas` tells legacy Wb/rad artifacts apart
from DD-conformant ones via the `dphi/dpsi` vs `q` slope, so both this OMFIT
sample and older native ODS files read back correctly. The committed sample is
kept as a frozen artifact rather than regenerated -- do not overwrite it
casually. Keep `user`
pinned if you do regenerate (`dataset_description` otherwise stamps `$USER`,
turning a regeneration into a spurious 2 MB diff). The recipe:

```python
from omas import save_omas_json

from vaft.code.efit import build_kinetic_core_profiles
from vaft.data import read_geqdsk
from vaft.data.resources import data_path
from vaft.machine_mapping.charge_exchange import charge_exchange
from vaft.machine_mapping.dataset_description import dataset_description
from vaft.machine_mapping.thomson_scattering import thomson_scattering

root = data_path("kineticEfit")
geq = read_geqdsk(root / "g048224.00300")
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

`legacy/shot_45531.json.gz` is a trimmed raw-signal archive of VEST shot 45531
(the discharge whose soft X-ray digitizer records are packaged alongside it),
holding the 117 field codes used by `notebooks/fluctuation_diagnostics_analysis.ipynb`:
the equilibrium magnetics channels, the 30 outboard fluctuation Mirnov coils
(2 MHz / 500 kHz), the plasma-current Rogowski and reference flux loop, the
diamagnetic-flux and TF signals, and the filterscope set.

Both raw archives (`shot_44740.json.gz`, `shot_45531.json.gz`) use the
self-describing timebase schema: every field records its corrected start time
`t0` and measured cadence `dt`, so loading reproduces the live-DB time axis at
any sampling rate. (The older two-rate schema labeled fields only "fast"/"slow"
and reconstructed every fast channel at 250 kHz, which stretched the 2 MHz
outboard-Mirnov timebase eightfold; `shot_44740.json.gz` was re-dumped from the
VEST SQL database to repair its 52 affected channels.) Data are verified
bit-identical to the DB waveforms at dump time; reconstructed times agree with
the DB's stored time strings to within the DB's own ~0.5 microsecond string
quantization.

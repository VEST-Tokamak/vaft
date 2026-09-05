# VAFT - Versatile Analysis Framework for Tokamak

<!-- README.md -->
English | [한국어](README.ko.md)

[PyPI](https://pypi.org/project/vaft/)
[Python](https://pypi.org/project/vaft/)
[License](LICENSE)

> **Integrate fusion science knowledge so it can be discovered, verified, compared, and studied.**

**VAFT is a standardized, verifiable, and interoperable scientific infrastructure
for machine-agnostic tokamak research.** It integrates experimental data,
reconstructed and simulated plasma states, and analysis workflows so that fusion
science knowledge can be discovered, verified, compared, and studied. Its full
end-to-end implementation on the [VEST tokamak](https://eng.snu.ac.kr/) at Seoul
National University supports routine experimental data processing, validation,
modeling, physics analysis, and shared scientific use across collaborating
researchers and institutions, while serving as the reference implementation for
modern, reproducible, and data-driven fusion research.

> Hong-Sik Yun, Sunjae Lee *et al* 2025 *Plasma Phys. Control. Fusion* **67** 115021
> ([doi:10.1088/1361-6587/ae1b6a](https://doi.org/10.1088/1361-6587/ae1b6a))

## What VAFT is

Four things, which together are what "infrastructure" means here.

### Integrated Standardized Interface

Integrate standardized data representations, scientific data processing,
validation, visualization, and physics codes into one consistent scientific
workflow.
Machine-specific VEST signals, [IMAS](https://imas.iter.org/)/[OMAS](https://gafusion.github.io/omas/)
representations, VAFT processing and plotting, verification and validation, and
community physics codes — EFIT, CHEASE, GPEC, TokaMaker, VFIT — interoperate
rather than being reimplemented here.

### Version-Controlled Data Pipeline

Produce traceable and reproducible workflows and data products across the whole
workflow, from machine design and data acquisition to reconstructed and
simulated physics states, so scientific results can be verified against their
provenance and assumptions. Versioning covers more than source code: machine
descriptions and geometry, diagnostic mappings, calibration, conventions,
processing logic, validation criteria, model configuration, and schema versions.

### IMAS-FAIR Database

Preserve, discover, access, and share validated data through both native and
standardized representations, following the FAIR principles — Findability,
Accessibility, Interoperability, Reusability. IMAS/OMAS, FileDB and native
artifacts, [HSDS](https://github.com/HDFGroup/hsds)-backed storage, lazy and
partial access, and programmatic APIs provide a foundation for discovering
available experimental and modeling information. Standardized access
**complements** native scientific artifacts rather than replacing them.

### Machine & Research Archive

A living archive of the VEST tokamak and its research ecosystem since operation
began in 2012 — machine history, technical documentation, experimental practices,
tutorials, example notebooks, and reproducible research knowledge, kept usable
for long-term verification, comparison, and scientific study across generations
of researchers and collaborating institutions.

## What can I do with VAFT?

| I want to... | Start here |
| --- | --- |
| See what VEST data looks like, with no setup | [Tutorial 01](tutorial/01_getting_started_with_vaft.ipynb) — runs offline from packaged data |
| Load a real shot and plot it | [Quick Start](#quick-start), then [`notebooks/README.md`](notebooks/README.md) |
| Reconstruct or refine an equilibrium | [`notebooks/`](notebooks/README.md) — EFIT, CHEASE, TokaMaker, TES workflows |
| Interpret a diagnostic | [`notebooks/`](notebooks/README.md) — magnetics, soft X-ray, fluctuation, fast-camera |
| Analyse many shots at once | [`notebooks/`](notebooks/README.md) — database-scale and statistical workflows |
| Understand the data model | [Fusion data structure and IMAS concepts](https://vest-tokamak.github.io/vaft/reference/imas-concepts/) |

## Research on VEST

### Research historically performed on VEST

Operated at Seoul National University since 2012, VEST has served as an in-house
experimental tokamak for machine operation, diagnostic development, discharge
optimization, and plasma-physics research, accumulating more than a decade of
machine-specific experimental knowledge and analysis practice.

- **Compact and spherical tokamak operation** — low-aspect-ratio plasmas with
  `R ≈ 0.4` m, `I_p < 300` kA, pulse durations up to about 40 ms
- **Diagnostic development and experimental analysis** — magnetic, kinetic,
  imaging and spectroscopic diagnostics, from calibration to interpretation
- **Discharge formation, heating, and current drive** — start-up; optimization of
  coil operation, fuelling, wall conditions and magnetic topology; NBI, EC,
  helicity-injection, trapped-particle and merging-configuration scenarios
- **Disruptions and transient MHD phenomena** — vertical displacement events,
  tearing modes, internal reconnection events
- **Equilibrium reconstruction and interpretation** — the lab-developed VFIT
  framework, including element-fitting and Grad–Shafranov flux reconstruction
- **Plasma confinement and performance** — high-`I_p`, long-pulse operation,
  operational limits, and comparison with spherical-tokamak scaling

### What VAFT enables next

VAFT extends that ecosystem into shareable, interoperable infrastructure.

1. **Collaborative and open research** — shared access to validated data and
   reproducible workflows across institutions
2. **Database-scale physics studies** — statistical analysis of large discharge
   populations rather than selected shots
3. **Integrated data analysis and modeling** — multiple diagnostics, reconstructed
   states, stability and plasma-response models in one workflow
4. **Multi-machine studies** — standardized representations extending beyond VEST
5. **Data-driven and AI-enabled research** — validated, traceable datasets for
   event detection, surrogate modeling and machine learning
6. **Scientific knowledge management** — preservation of data products, procedures,
   provenance and practice so results can be reproduced and transferred

> **Maturity.** Items 1–3 are implemented and in routine use. Item 4 is in active
> development. Items 5–6 are partly implemented: validated datasets and provenance
> exist today, while semantic knowledge graphs, machine-actionable provenance,
> digital-twin integration and autonomous research agents are **long-term
> direction, not current functionality.**

## Key Features


| Capability                  | Description                                                                                                                                                             |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Remote Database Access**  | Load per-shot OMAS ODS data from the VEST HSDS server with a single function call                                                                                       |
| **Machine Mapping**         | Convert native VEST diagnostic signals into standardized IMAS IDS (magnetics, Thomson scattering, barometry, PF active, TF, spectrometer UV, charge exchange, etc.)     |
| **Equilibrium & Stability** | Interfaces for EFIT, CHEASE, GPEC(DCON/RDCON) — read/write code I/O in IDS format                                                                                       |
| **Physics Formulas**        | Equilibrium quantities (poloidal/toroidal flux, safety factor), stability metrics (beta limits, ballooning), confinement scaling laws (ITER89P, H98y2), Green's functions |
| **Signal Processing**       | Smoothing, baseline subtraction, noise reduction, electromagnetic field calculations, eddy current modeling                                                             |
| **Profile Fitting**         | Map kinetic diagnostics (Thomson scattering, CES) onto equilibrium flux surfaces; fit with GP, polynomial, or exponential models                                        |
| **Visualization**           | Time traces, 1D/2D profiles, flux surface contours, top-view, and operational-space maps                                                                                |
| **IMAS Interoperability**   | Convert between OMAS ODS and IMAS-Python (AL5) data structures; export to NetCDF                                                                                        |



## Architecture

```
VEST Data Analysis Platform
├── Automated Pipeline (Snakemake)     ── experiment → postprocessing → simulation
├── Database (IMAS-HSDS)                ── per-shot HDF5 storage via REST API
└── Interface (VAFT)                    ── data access, mapping, processing, visualization
```


### Available IMAS IDSs in the VEST Database

**Experimental:**
`dataset_description` · `magnetics` · `tf` · `pf_active` · `barometry` · `spectrometer_uv` · `thomson_scattering` · `charge_exchange`

**Modelling:**
`wall` · `em_coupling` · `pf_passive` · `equilibrium` (EFIT/CHEASE) · `core_profiles` · `mhd_linear` (DCON/RDCON)


## Quick Start

### Installation

New to VAFT, or setting up a teaching/course machine? Follow
[`install/README.md`](install/README.md): it has a one-command bootstrap for
Linux, macOS, native Windows, and WSL2, an environment checker, and the
procedure for updating an existing checkout.

```bash
git clone https://github.com/VEST-Tokamak/vaft.git
cd vaft
bash install/linux.sh          # or macos.sh / windows_wsl.sh / windows_native.ps1
conda run -n vaft python install/check_vaft_environment.py
```

Install from source manually:

```bash
git clone https://github.com/VEST-Tokamak/vaft.git
cd vaft
python -m pip install -e .
```

```bash
# Development tooling
python -m pip install -e ".[dev]"
```

#### Legacy NumPy 1 installation

Use this only for an external package that still requires NumPy 1. Because
`h5pyd==0.20.0` declares a NumPy 2 requirement, install it with `--no-deps`
after replacing NumPy:

```bash
python -m pip install -e .
python -m pip install --force-reinstall --no-deps "numpy>=1.26.4,<2"
python -m pip install --force-reinstall --no-deps h5pyd==0.20.0
```

This is a legacy compatibility option; `pip check` may report the intentionally
bypassed NumPy requirement.

#### Install the released package from PyPI

```bash
pip install vaft
```

This installs the latest published release. Install from source instead when
you need unreleased changes from `develop`.


**Supported Python**: 3.10 -- 3.13
**Numerical stack default**: NumPy 2.x (`numpy>=2.0.0,<3`)


### Initialize external fusion codes

Set the installation roots for the codes you use before starting VAFT:

```bash
export GPECHOME=/path/to/gpec
export CHEASEHOME=/path/to/chease
export EFITHOME=/path/to/efit
export TESHOME=/path/to/tes
```

Each executable belongs under its root's `bin/` directory. See
[Initialize external fusion codes](notebooks/initialize_external_fusion_codes.ipynb)
for layouts, compatibility variables, FileDB configuration, and validation.


### Connect to the VEST Database

If you will use the remote VEST HSDS database, configure your HSDS credentials:

```bash
hsconfigure
```

Enter the following when prompted:


| Field           | Value                                                             |
| --------------- | ----------------------------------------------------------------- |
| Server endpoint | `http://147.46.36.244:5101`                                       |
| Username        | contact [peppertonic18@snu.ac.kr](mailto:peppertonic18@snu.ac.kr) |
| Password        | contact [peppertonic18@snu.ac.kr](mailto:peppertonic18@snu.ac.kr) |


A `connection ok` message confirms you are connected. See the [detailed guide](https://vest-tokamak.github.io/vaft/guide/Quick_start_guide/) for more information.


### Basic Usage

```python
import vaft

# Load a shot from the remote database
ods = vaft.database.load(39915)

# Access IMAS-structured data directly
time = ods['magnetics.time']
ip = ods['magnetics.ip.0.data']
```


## Library Modules

```
vaft/
├── cli/               # Command-line workflow dispatch
├── database/          # HSDS/SQL access and canonical FileDB layout
├── machine_mapping/   # Native-to-IDS diagnostic conversion (70+ functions)
├── formula/           # Physics formulas (equilibrium, stability, Green's functions)
├── process/           # Signal processing, EM modeling, profile fitting
├── plot/              # Visualization (time, 1D, 2D, top-view, analysis)
├── omas/              # ODS utilities (shot metadata, sample data)
├── imas/              # IMAS-Python (AL5) interoperability
├── code/              # Code interfaces (EFIT, CHEASE, GPEC, TES, TokaMaker, Snakemake)
└── data/              # Sample data, geometry assets, calibration tables
```


## Example Notebooks


| Notebook                                                                                                                               | Description                                 |
| -------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| [initialize_external_fusion_codes](notebooks/initialize_external_fusion_codes.ipynb)                                                   | Configure and verify external code roots    |
| [database_initialization_and_load](notebooks/database_initialization_and_load.ipynb)                                                   | Core data loading and framework basics      |
| [plotting_sample_using_vaft_plot_module](notebooks/plotting_sample_using_vaft_plot_module.ipynb)                                       | Visualization examples with the plot module |
| [profile_fitting_using_equilibrium_and_kinetic_diagnostics](notebooks/profile_fitting_using_equilibrium_and_kinetic_diagnostics.ipynb) | Thomson/CES mapping and profile fitting     |
| [read_and_convert_data_structure](notebooks/read_and_convert_data_structure.ipynb)                                                     | ODS/IMAS data structure conversion          |
| [imas_omas_data_conversion](notebooks/imas_omas_data_conversion.ipynb)                                                                 | IMAS ↔ OMAS interoperability                |
| [vest_experimental_data_list](notebooks/vest_experimental_data_list.ipynb)                                                             | Browse the VEST shot database               |
| [confinement_time_scaling](notebooks/confinement_time_scaling.ipynb)                                                                   | Energy confinement time scaling analysis    |
| [vest_daily_monitoring](notebooks/vest_daily_monitoring.ipynb)                                                                         | Daily experiment monitoring dashboard       |
| [publication_figures](notebooks/publication_figures.ipynb)                                                                             | Reproduce figures from publications         |
| [verify_exist_shot_and_load](notebooks/verify_exist_shot_and_load.ipynb)                                                               | Verify shot availability and load TS/CX data |
| [tokamak_power_balance](notebooks/tokamak_power_balance.ipynb)                                                                         | Tokamak power balance and radiation decomposition |
| [verification_and_validation](notebooks/verification_and_validation.ipynb)                                                             | Verification and validation examples        |
| [soft_x_ray_signal_analysis](notebooks/soft_x_ray_signal_analysis.ipynb)                                                               | Soft X-ray signal analysis                  |
| [equilibrium_refinement_using_chease](notebooks/equilibrium_refinement_using_chease.ipynb)                                             | Equilibrium refinement with CHEASE          |
| [forward_equilibrium_using_TES](notebooks/forward_equilibrium_using_TES.ipynb)                                                         | Forward equilibrium reconstruction with TES |
| [forward_equilibrium_using_TokaMaker](notebooks/forward_equilibrium_using_TokaMaker.ipynb)                                             | Forward free-boundary equilibrium with TokaMaker (Open FUSION Toolkit) |
| [time_dependent_equilibrium_using_TokaMaker](notebooks/time_dependent_equilibrium_using_TokaMaker.ipynb)                             | Vessel eddy currents, wall modes, and quasi-static evolution with TokaMaker |
| [free_boundary_pf_coil_scan](notebooks/free_boundary_pf_coil_scan.ipynb)                                                               | Free-boundary PF-coil scans and topology transitions with TokaMaker |
| [kinetic_efit_end_to_end](notebooks/kinetic_efit_end_to_end.ipynb)                                                                     | End-to-end kinetic-EFIT workflow            |


## Related Resources

**In this repository**

- **Tutorial course**: [`tutorial/`](tutorial/README.md) — six guided sessions, starting offline
- **Example notebooks**: [`notebooks/`](notebooks/README.md) — research workflows by topic
- **Installation and environment**: [`install/`](install/README.md) — per-platform bootstrap and checks
- **Contributing**: [`CONTRIBUTING.md`](CONTRIBUTING.md) · **Third-party notices**: [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)

**External**

- **Documentation site**: [vest-tokamak.github.io/vaft](https://vest-tokamak.github.io/vaft/) — workflows, API and reference
- **Paper**: H.-S. Yun, S. Lee *et al*, "Developing an IMAS-compatible platform for the university-scale tokamak VEST and its application to operating characteristics analysis", *Plasma Phys. Control. Fusion* **67** 115021 (2025). [doi:10.1088/1361-6587/ae1b6a](https://doi.org/10.1088/1361-6587/ae1b6a)
- **OMAS**: [gafusion.github.io/omas](https://gafusion.github.io/omas/) — Python API for IMAS data structures
- **OMFIT**: [omfit.io](https://omfit.io/) — Integrated modeling and experimental data analysis framework for tokamak research
- **HSDS**: [github.com/HDFGroup/hsds](https://github.com/HDFGroup/hsds) — HDF5 REST-based data service
- **IMAS**: [github.com/iterorganization/IMAS-Data-Dictionary](https://github.com/iterorganization/IMAS-Data-Dictionary) — ITER Integrated Modelling & Analysis Suite


## Contributing

Contributions are welcome. Please open an [issue](https://github.com/VEST-Tokamak/vaft/issues) or submit a pull request.

Notebook hygiene, documentation conventions and the branch policy are in
[`CONTRIBUTING.md`](CONTRIBUTING.md).

For database write access, contact [peppertonic18@snu.ac.kr, satelite2517@snu.ac.kr](mailto:peppertonic18@snu.ac.kr).


## Acknowledgements

The authors would like to thank O Meneghini and J McClenaghan at General Atomics for their technical advice. Some parts of the data processing were performed using the code API in the OMFIT integrated modeling framework [1]. This research was supported by the National Research Foundation of Korea (NRF) grant funded by the Korean Government (MSIT) (RS-2021-NR057187, RS-2023-00281276, RS-2024-00409564, and RS-2025-02304810).


## Reference

Deeper technical material. These sections are migrating to the
[documentation site](https://vest-tokamak.github.io/vaft/); they remain here until
they have a home there.

### EFIT slice status

`vaft.code.run_efit()` preserves its backward-compatible process-level
`result.ok` property. Use `result.usable` and `result.slice_statuses` when the
scientific usability of the generated equilibria matters:

```python
for status in result.slice_statuses:
    print(status.time, status.overall_status, status.failure_codes)
```

Each slice reports runtime, output, numerical, and physical status separately.
The stable failure taxonomy is available as `vaft.code.EFIT_FAILURE_CODES`, and
each status round-trips through JSON with `to_dict()` and `from_dict()`.


### EFIT scientific configuration

Routine k-file settings are available as typed, validated objects instead of
generator literals. Defaults preserve the existing VEST routine semantics:

```python
from vaft.code import (
    EFITConfig,
    EFITNumericsConfig,
    EFITProfileConfig,
    prepare_efit_inputs,
)

config = EFITConfig(
    shot=39915,
    workdir="efit/39915/work",
    profile=EFITProfileConfig(kppcur=3, kffcur=2),
    numerics=EFITNumericsConfig(relaxation=0.8, max_iterations=200),
    provenance={"geometry_version": "vest-2025-07", "source": "main"},
)
inputs = prepare_efit_inputs(ods, config)
```

Preparation writes `efit_configuration.json` with the resolved configuration,
its stable hash, VAFT version, provenance, and k-file checksums. Use
`vaft.code.efit_parameter_grid()` with dotted paths such as
`profile.kppcur` or `constraints.group_weights.bpol_probe` for deterministic
convergence scans that do not require the EFIT binary.

Every remote call names an HSDS *source* — one namespace per analysis lineage,
so an EFIT baseline and its CHEASE refinement of the same shot never overwrite
each other. `source` defaults to `main`, the VAFT-native pipeline's namespace.
`public` is the pre-VAFT pipeline's output: still readable, never written.

| Source | Purpose |
| --- | --- |
| `main` | Default. VAFT EFIT baseline. |
| `chease-mhd-stability` | CHEASE-refined equilibrium plus DCON/RDCON/GPEC linear-MHD stability. |
| `vfit-element` | VFIT element-fitting equilibrium. |
| `vfit-gse` | VFIT Grad-Shafranov-equilibrium fitting result. |
| `electron-efit` | Kinetic EFIT from Thomson scattering with an assumed Ti/Te ratio. |
| `kinetic-efit` | Kinetic EFIT for shots with Thomson scattering and CES/ion-Doppler spectroscopy. |
| `public` | **Read-only** legacy source from the previous pipeline. |

`python -m vaft.cli summary sources` prints the same list. The historical
`directory=`/`target=` keywords still work and warn. To use a namespace outside
the catalog, list it in `VAFT_HSDS_EXTRA_SOURCES`.

```python
ods = vaft.database.load(39915)                       # reads main
legacy = vaft.database.load(39915, source="public")   # legacy reference
vaft.database.save(refined, 39915, source="chease-mhd-stability")
```

`load` is the eager path for complete ODS exports and workflows that need a
local IMAS staging set. Without `paths` it stages the complete shot; with
`paths=["equilibrium"]` it stages only that IDS plus `dataset_description` and
uses a validated local domain cache by default. For exploratory access to
selected leaves, use the direct lazy path, which opens only the requested IDS
domain and transfers only the dataset selections that are read:

When byte-exact per-IDS images are available, eager loads use them by default
to avoid the many requests made by `hsget`. Use `transport="canonical"` to
bypass derived images or `transport="h5image"` to require them. Direct lazy
`open()` always keeps canonical selection-based access.

```python
with vaft.database.open(39915, paths="equilibrium") as ods:
    psi = ods["equilibrium.time_slice.0.profiles_2d.0.psi"]
```

The lazy API supports occurrence 0 in this first version. Native IDS use the
explicit remote representation:

```python
equilibrium = vaft.database.load(
    39915, representation="imas", paths="equilibrium"
)
```

Remote saves keep canonical IMAS images authoritative and can publish derived
caches alongside them. `derived_cache="auto"` creates per-IDS images; the
historical full-ODS cache remains readable but is only created explicitly. The choices are
`"none"`, `"imas-images"`, `"omas"`, and `"both"`.

For experimental native lazy access without a local staging directory, open an
IMAS handle. It returns a read-only, lazy `IDSToplevel`; each requested leaf is
read directly from the corresponding HSDS IDS domain. This first version
supports occurrence 0 and an exact stored IMAS DD version.

```python
with vaft.database.open(
    39915, representation="imas", paths="equilibrium"
) as handle:
    psi = handle.get().time_slice[0].profiles_2d[0].psi
```

Local artifacts are deliberately separate from the HSDS API. They are
content-detected rather than selected by a format flag:

```python
ods = vaft.omas.load("./shot/master.h5")
with vaft.imas.load("./equilibrium.nc") as entry:
    equilibrium = entry.get("equilibrium")
```


### Profile Fitting

```python
# Map Thomson scattering data onto equilibrium flux coordinates, then fit profiles
mapped_rho = vaft.process.equilibrium_mapping_thomson_scattering(ods, geq)
vaft.process.profile_fitting_thomson_scattering(
    ods, time_ms, mapped_rho, fitting_function_te='gp', fitting_function_ne='gp'
)
```


### IMAS Conversion

```python
# Write an OMAS ODS as an IMAS HDF5 image set or a native IMAS NetCDF file
vaft.imas.save(ods, "./shot")
vaft.imas.save(ods, "./shot.nc")
```


### Parametric Equilibrium Analysis

`EquilibriumData` is VAFT's lightweight, single-slice, axisymmetric working
model for numerical algorithms. It is not a persistence schema: GEQDSK, ODS,
and native IDS remain the authoritative storage and interchange formats.

```python
from vaft.data.resources import sample_geqdsk
from vaft.process.equilibrium import as_equilibrium, derive_global_descriptors

# An EFIT g-file stores psi in weber/radian, so it is a COCOS 1-8 index.
equilibrium = as_equilibrium(sample_geqdsk(), convention=1)
descriptors = derive_global_descriptors(equilibrium)
print(descriptors["beta_t"].value, descriptors["beta_t"].provenance)
```

Every `DerivedValue` records its SI unit, implemented definition, source
fields, convention, method, tolerances, and quality information. Missing or
ambiguous inputs produce an unavailable result with a reason. In particular,
VAFT does not infer one COCOS index when the observable signs admit several;
an explicit convention is required before conversion.

Shape descriptors follow the conventional definitions, so `major_radius` is
`(R_out+R_in)/2` and triangularity is measured from it, matching IMAS
`boundary.geometric_axis` and `boundary.triangularity`. The LCFS area centroid
is reported separately as `area_centroid_r`/`area_centroid_z` because that, not
the geometric centre, is the radius Pappus's theorem needs for `volume`. The
descriptors also cover the boundary-length-averaged poloidal field and the Lao
virial internal inductance. Poloidal fields honour the COCOS `e_Bp` factor, so
dimensionless quantities such as `beta_p` and `li` agree whether an equilibrium
is expressed in weber or weber-per-radian. Normalized coordinates are
`psi_n=(psi-psi_axis)/(psi_boundary-psi_axis)`,
`rho_pol_n=sqrt(psi_n)`, and
`rho_tor_n=sqrt(integral(q dpsi)/integral_boundary(q dpsi))`. A non-monotonic
toroidal-flux mapping is reported rather than repaired with absolute values.

Local Miller fits use bounded symmetric contour least squares and report RMS,
maximum, and Hausdorff errors. Fits at `psi_n >= 0.995` or within `0.05a` of an
X-point are flagged because the local form is not meaningful there. The
analytic Solov'ev model is restricted to axisymmetric constant-`p'` and
constant-`FF'` solutions; it is a regression/example model, not a general
experimental equilibrium solver. Edge `dRsep` is always the outboard-midplane
quantity `R_out(psi_X,upper)-R_out(psi_X,lower)`, never an absolute X-point
coordinate, and it is reported only for a diverted configuration.

Boundary topology is decided from the flux map, with no machine-specific
geometry. Stationary points of `psi` are located and split into O-points and
saddles by the sign of the Hessian determinant. A saddle is promoted to a
physical X-point only when it is relevant to the boundary: its flux must match
the boundary flux within a window derived from its own curvature and the grid
spacing, and the confined region's level set just inside the boundary must
reach it on the scale that curvature implies. At least one such X-point gives
`UPPER_SINGLE_NULL`, `LOWER_SINGLE_NULL`, or `DOUBLE_NULL` (all
`Topology.is_diverted`); none, with an LCFS in contact with the wall, gives
`LIMITED`. A grid-clipped confined region, a missing wall, or an LCFS bounded
by neither gives `AMBIGUOUS` with a reason rather than a guess. Real
reconstructions routinely contain numerical saddles far from the plasma; those
are returned in `x_points` with `active=False` instead of being filtered by
hard-coded geometry.



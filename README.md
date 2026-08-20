# VAFT - Versatile Analysis Framework for Tokamak

<!-- README.md -->
English | [한국어](README.ko.md)

[PyPI](https://pypi.org/project/vaft/)
[Python](https://pypi.org/project/vaft/)
[License](LICENSE)

**VAFT** is an open-source Python library that functions both as a dedicated data platform for the [VEST (Versatile Experiment Spherical Torus)](https://eng.snu.ac.kr/) tokamak at Seoul National University and as a machine- and code-generic data analysis framework built upon the IMAS data model, providing an [IMAS](https://imas.iter.org/)-compliant data interface built on the [OMAS](https://gafusion.github.io/omas/) interface library and an [HSDS](https://github.com/HDFGroup/hsds) remote HDF5 database.

> Hong-Sik Yun, Sunjae Lee *et al* 2025 *Plasma Phys. Control. Fusion* **67** 115021
> ([doi:10.1088/1361-6587/ae1b6a](https://doi.org/10.1088/1361-6587/ae1b6a))

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

Install from source (recommended):

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

Install from PyPI (obsolete):

```bash
pip install vaft
```


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
with vaft.database.open(39915, source="public", paths="equilibrium") as ods:
    psi = ods["equilibrium.time_slice.0.profiles_2d.0.psi"]
```

The lazy API supports occurrence 0 in this first version. Native IDS use the
explicit remote representation:

```python
equilibrium = vaft.database.load(
    39915, source="public", representation="imas", paths="equilibrium"
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
    39915, source="public", representation="imas", paths="equilibrium"
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

See the [HSDS lazy and per-IDS h5image report](docs/hsds_lazy_h5image_report.md)
for the architecture, cache policy, and shot 39915 benchmark results.

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
├── code/              # Code interfaces (EFIT, CHEASE, GPEC, Snakemake)
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
| [kinetic_efit_end_to_end](notebooks/kinetic_efit_end_to_end.ipynb)                                                                     | End-to-end kinetic-EFIT workflow            |


## Related Resources

- **Documentation**: [vest-tokamak.github.io/vaft](https://vest-tokamak.github.io/vaft/)
- **Paper**: H.-S. Yun, S. Lee *et al*, "Developing an IMAS-compatible platform for the university-scale tokamak VEST and its application to operating characteristics analysis", *Plasma Phys. Control. Fusion* **67** 115021 (2025). [doi:10.1088/1361-6587/ae1b6a](https://doi.org/10.1088/1361-6587/ae1b6a)
- **OMAS**: [gafusion.github.io/omas](https://gafusion.github.io/omas/) — Python API for IMAS data structures
- **OMFIT**: [omfit.io](https://omfit.io/) — Integrated modeling and experimental data analysis framework for tokamak research
- **HSDS**: [github.com/HDFGroup/hsds](https://github.com/HDFGroup/hsds) — HDF5 REST-based data service
- **IMAS**: [github.com/iterorganization/IMAS-Data-Dictionary](https://github.com/iterorganization/IMAS-Data-Dictionary) — ITER Integrated Modelling & Analysis Suite

## Contributing

Contributions are welcome. Please open an [issue](https://github.com/VEST-Tokamak/vaft/issues) or submit a pull request.

For database write access, contact [peppertonic18@snu.ac.kr, satelite2517@snu.ac.kr](mailto:peppertonic18@snu.ac.kr).

## Acknowledgements

The authors would like to thank O Meneghini and J McClenaghan at General Atomics for their technical advice. Some parts of the data processing were performed using the code API in the OMFIT integrated modeling framework [1]. This research was supported by the National Research Foundation of Korea (NRF) grant funded by the Korean Government (MSIT) (RS-2021-NR057187, RS-2023-00281276, RS-2024-00409564, and RS-2025-02304810).

## Third-party Notices

### OPEN-ADAS atomic routines

Parts of VAFT's OPEN-ADAS ADF11 parsing, interpolation, default-file selection,
and ionization-equilibrium logic are adapted from software distributed under
the following license.

> MIT License
>
> Copyright (c) 2021 Francesco Sciortino
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

### OMFIT classes compatibility port

VAFT's native EQDSK compatibility and interoperability paths include behavior
ported or adapted from `omfit_classes`. VAFT also provides compatibility shims
for the corresponding legacy NumPy, SciPy, and xarray interfaces. The original
OMFIT classes software is distributed under the following license.

> Copyright 2013-2021 the OMFIT contributors
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

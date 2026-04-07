# VAFT - Versatile Analytical Framework for Tokamak

[PyPI](https://pypi.org/project/vaft/)
[Python](https://pypi.org/project/vaft/)
[License](LICENSE)

**VAFT** is an open-source Python library that functions both as a dedicated data platform for the [VEST (Versatile Experiment Spherical Torus)](https://eng.snu.ac.kr/) tokamak at Seoul National University and as a machine- and code-generic data analysis framework built upon the IMAS data model, providing an [IMAS](https://imas.iter.org/)-compliant data interface built on the [OMAS](https://gafusion.github.io/omas/) interface library and an [HSDS](https://github.com/HDFGroup/hsds) remote HDF5 database.

> Hong-Sik Yun, Sunjae Lee *et al* 2025 *Plasma Phys. Control. Fusion* **67** 115021
> ([doi:10.1088/1361-6587/ad9ba7](https://doi.org/10.1088/1361-6587/ad9ba7))

## Key Features


| Capability                  | Description                                                                                                                                                             |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Remote Database Access**  | Load per-shot OMAS ODS data from the VEST HSDS server with a single function call                                                                                       |
| **Machine Mapping**         | Convert native VEST diagnostic signals into standardized IMAS IDS (magnetics, Thomson scattering, barometry, PF active, TF, spectrometer UV, charge exchange, etc.)     |
| **Equilibrium & Stability** | Interfaces for EFIT, CHEASE, GPEC(DCON/RDCON) — read/write code I/O in IDS format                                                                                       |
| **Physics Formulas**        | Equilibrium quantities (poloidal/toroidal flux, safety factor), stability metrics (beta limits, ballooning), confinement scaling laws (IPB89, H98y2), Green's functions |
| **Signal Processing**       | Smoothing, baseline subtraction, noise reduction, electromagnetic field calculations, eddy current modeling                                                             |
| **Profile Fitting**         | Map kinetic diagnostics (Thomson scattering, CES) onto equilibrium flux surfaces; fit with GP, polynomial, or exponential models                                        |
| **Visualization**           | Time traces, 1D/2D profiles, flux surface contours, top-view, and operational-space maps                                                                                |
| **IMAS Interoperability**   | Convert between OMAS ODS and IMAS-Python (AL5) data structures; export to NetCDF                                                                                        |


## Architecture

```
VEST Data Analysis Platform
├── Automated Pipeline (Snakemake)     ── experiment → postprocessing → simulation
├── IMAS Database (OMAS-HSDS)          ── per-shot HDF5 storage via REST API
└── VAFT Library (this repo)           ── data access, mapping, processing, visualization
```

### Available IMAS IDSs in the VEST Database

**Experimental:**
`dataset_description` · `magnetics` · `tf` · `pf_active` · `barometry` · `spectrometer_uv` · `thomson_scattering` · `charge_exchange`

**Modelling:**
`wall` · `em_coupling` · `pf_passive` · `equilibrium` (EFIT/CHEASE) · `core_profiles` · `mhd_linear` (DCON/RDCON)

## Quick Start

### Installation

Install from PyPI:

```bash
pip install vaft
```

Install from source (recommended for development):

```bash
git clone https://github.com/VEST-Tokamak/vaft.git
cd vaft
python -m pip install -e .
```

Install optional extras:

```bash
# Development tooling
python -m pip install -e ".[dev]"

# HSDS database client (source install)
python -m pip install -e ".[hsds]"

# HSDS database client (PyPI install)
python -m pip install "vaft[hsds]"
```

**Supported Python**: 3.10 -- 3.13
**Numerical stack default**: NumPy 2.x (`numpy>=2,<3`)

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

### Profile Fitting

```python
# Map Thomson scattering data onto equilibrium flux coordinates and fit profiles
vaft.process.equilibrium_mapping_thomson_scattering(ods)
vaft.process.profile_fitting_thomson_scattering(ods, method='gp')
```

### IMAS Conversion

```python
# Convert OMAS ODS ↔ IMAS-Python data entry
from vaft.imas import omas_imas
omas_imas.save_omas_to_imas(ods, pulse=39915, run=0)
```

## Library Modules

```
vaft/
├── database/          # Remote database access (HSDS, raw SQL)
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
| [database_initialization_and_load](notebooks/database_initialization_and_load.ipynb)                                                   | Core data loading and framework basics      |
| [plotting_sample_using_vaft_plot_module](notebooks/plotting_sample_using_vaft_plot_module.ipynb)                                       | Visualization examples with the plot module |
| [profile_fitting_using_equilibrium_and_kinetic_diagnostics](notebooks/profile_fitting_using_equilibrium_and_kinetic_diagnostics.ipynb) | Thomson/CES mapping and profile fitting     |
| [read_and_convert_data_structure](notebooks/read_and_convert_data_structure.ipynb)                                                     | ODS/IMAS data structure conversion          |
| [imas_omas_data_conversion](notebooks/imas_omas_data_conversion.ipynb)                                                                 | IMAS ↔ OMAS interoperability                |
| [vest_experimental_data_list](notebooks/vest_experimental_data_list.ipynb)                                                             | Browse the VEST shot database               |
| [confinement_time_scaling](notebooks/confinement_time_scaling.ipynb)                                                                   | Energy confinement time scaling analysis    |
| [vest_daily_monitoring](notebooks/vest_daily_monitoring.ipynb)                                                                         | Daily experiment monitoring dashboard       |
| [publication_figures](notebooks/publication_figures.ipynb)                                                                             | Reproduce figures from publications         |


## Related Resources

- **Documentation**: [vest-tokamak.github.io/vaft](https://vest-tokamak.github.io/vaft/)
- **Paper**: H.-S. Yun *et al*, "Development of an IMAS-compliant integrated data analysis platform for the VEST tokamak", *Plasma Phys. Control. Fusion* **67** 115021 (2025). [doi:10.1088/1361-6587/ad9ba7](https://doi.org/10.1088/1361-6587/ad9ba7)
- **OMAS**: [gafusion.github.io/omas](https://gafusion.github.io/omas/) — Python API for IMAS data structures
- **OMFIT**: [omfit.io](https://omfit.io/) — Integrated modeling and experimental data analysis framework for tokamak research
- **HSDS**: [github.com/HDFGroup/hsds](https://github.com/HDFGroup/hsds) — HDF5 REST-based data service
- **IMAS**: [github.com/iterorganization/IMAS-Data-Dictionary](https://github.com/iterorganization/IMAS-Data-Dictionary) — ITER Integrated Modelling & Analysis Suite

## Contributing

Contributions are welcome. Please open an [issue](https://github.com/VEST-Tokamak/vaft/issues) or submit a pull request.

For database write access, contact [peppertonic18@snu.ac.kr, satelite2517@snu.ac.kr](mailto:peppertonic18@snu.ac.kr).

## Acknowledgements

The authors would like to thank O Meneghini at Proxima Fusion and J McClenaghan at General Atomics, O Hoenen at ITER Organization for their technical advice. Some parts of the data processing were performed using the code API in the OMFIT integrated modeling framework. This research was supported by the National Research Foundation of Korea (NRF) grant funded by the Korean Government (MSIT) (RS-2023-00281276, RS-202400409564, and RS-2025-02304810).
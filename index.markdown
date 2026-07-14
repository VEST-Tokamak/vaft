---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults
layout: home
title: VAFT — Versatile Analytical Framework for Tokamak
---

![VAFT]({{ site.baseurl }}/assets/images/IMG_3873.jpg)

**VAFT** is an open-source Python library that functions both as a dedicated data platform for the
[VEST (Versatile Experiment Spherical Torus)](https://eng.snu.ac.kr/) tokamak at Seoul National
University and as a machine- and code-generic data analysis framework built upon the IMAS data model.
It provides an [IMAS](https://imas.iter.org/)-compliant data interface built on the
[OMAS](https://gafusion.github.io/omas/) interface library and an
[HSDS](https://github.com/HDFGroup/hsds) remote HDF5 database. Read access to the VEST database is
open to all users; writing to the database is restricted to authorized accounts.

> Hong-Sik Yun, Sunjae Lee *et al* 2025 *Plasma Phys. Control. Fusion* **67** 115021 —
> "Developing an IMAS-compatible platform for the university-scale tokamak VEST and its application
> to operating characteristics analysis"
> ([doi:10.1088/1361-6587/ae1b6a](https://doi.org/10.1088/1361-6587/ae1b6a))

## Key features

| Capability | Description |
| --- | --- |
| **Remote database access** | Load per-shot OMAS ODS data from the VEST HSDS server with a single function call |
| **Machine mapping** | Convert native VEST diagnostic signals into standardized IMAS IDSs (magnetics, Thomson scattering, barometry, PF active, TF, spectrometer UV, charge exchange, etc.) |
| **Equilibrium & stability** | Interfaces for EFIT, CHEASE, GPEC (DCON/RDCON) — read/write code I/O in IDS format |
| **Physics formulas** | Equilibrium quantities (poloidal/toroidal flux, safety factor), stability metrics (beta limits, ballooning), confinement scaling laws (ITER89P, H98y2), Green's functions |
| **Signal processing** | Smoothing, baseline subtraction, noise reduction, electromagnetic field calculations, eddy current modeling |
| **Profile fitting** | Map kinetic diagnostics (Thomson scattering, CES) onto equilibrium flux surfaces; fit with GP, polynomial, or exponential models |
| **Visualization** | Time traces, 1D/2D profiles, flux surface contours, top-view, and operational-space maps |
| **IMAS interoperability** | Convert between OMAS ODS and IMAS-Python (AL5) data structures; export to NetCDF |

## Architecture

```text
VEST Data Analysis Platform
├── Automated Pipeline (Snakemake)     ── experiment → postprocessing → simulation
├── IMAS Database (OMAS-HSDS)          ── per-shot HDF5 storage via REST API
└── VAFT Library                       ── data access, mapping, processing, visualization
```

### Available IMAS IDSs in the VEST database

**Experimental:**
`dataset_description` · `magnetics` · `tf` · `pf_active` · `barometry` · `spectrometer_uv` ·
`thomson_scattering` · `charge_exchange`

**Modelling:**
`wall` · `em_coupling` · `pf_passive` · `equilibrium` (EFIT/CHEASE) · `core_profiles` ·
`mhd_linear` (DCON/RDCON)

## Quick start

```python
import vaft

ods = vaft.database.load(39915)     # OMAS ODS for one shot, from the public HSDS directory
ip = ods['magnetics.ip.0.data']     # IMAS-structured access
```

Install from source, add the HSDS client, then configure your connection:

```bash
git clone https://github.com/VEST-Tokamak/vaft.git
cd vaft
python -m pip install -e .
python -m pip install --no-deps h5pyd==0.20.0
hsconfigure
```

See [Installation]({{ site.baseurl }}/guide/Installation/) for the full dependency matrix and
[Quick start guide]({{ site.baseurl }}/guide/Quick_start_guide/) for the HSDS credentials and a first
end-to-end session.

## Documentation

| Guide | Contents |
| --- | --- |
| [Installation]({{ site.baseurl }}/guide/Installation/) | Supported Python versions, dependencies, editable install, HSDS client |
| [Quick start guide]({{ site.baseurl }}/guide/Quick_start_guide/) | Connect to the database and load your first shot |
| [Database and data access]({{ site.baseurl }}/guide/Database/) | HSDS remote store, ODS/IDS loading and saving, raw DAQ signals |
| [Data structures (ODS, IDS, IMAS)]({{ site.baseurl }}/guide/Data_structures/) | How VEST data is organized under the IMAS data model |
| [Machine mapping]({{ site.baseurl }}/guide/Machine_mapping/) | Native VEST diagnostics → standardized IMAS IDSs |
| [Magnetics]({{ site.baseurl }}/guide/Magnetics/) | Magnetic probes, flux loops, plasma current, diamagnetic flux |
| [Equilibrium]({{ site.baseurl }}/guide/Equilibrium/) | EFIT and CHEASE reconstruction and refinement |
| [MHD stability]({{ site.baseurl }}/guide/Stability/) | Linear stability with DCON, RDCON, STRIDE and GPEC |
| [Kinetic profiles and fitting]({{ site.baseurl }}/guide/Profiles/) | Thomson scattering and CES mapping onto flux coordinates |
| [Signal processing and EM modeling]({{ site.baseurl }}/guide/Processing/) | Smoothing, baselines, electromagnetic fields, eddy currents |
| [Physics formulas]({{ site.baseurl }}/guide/Formula/) | Equilibrium, stability, confinement scaling, Green's functions |
| [Plotting]({{ site.baseurl }}/guide/Plotting/) | Time traces, 1D/2D profiles, flux surfaces, top view |
| [Automated pipelines]({{ site.baseurl }}/guide/Pipelines/) | Snakemake workflows from raw DAQ to stability analysis |
| [Examples]({{ site.baseurl }}/guide/examples/) | Runnable examples based on the repository notebooks |
| [API reference]({{ site.baseurl }}/guide/API_reference/) | Module-by-module reference for the `vaft` package |

## Resources

- **Source**: [github.com/VEST-Tokamak/vaft](https://github.com/VEST-Tokamak/vaft)
- **OMAS**: [gafusion.github.io/omas](https://gafusion.github.io/omas/) — Python API for IMAS data structures
- **OMFIT**: [omfit.io](https://omfit.io/) — integrated modeling and experimental data analysis framework
- **HSDS**: [github.com/HDFGroup/hsds](https://github.com/HDFGroup/hsds) — HDF5 REST-based data service
- **h5pyd**: [github.com/HDFGroup/h5pyd](https://github.com/HDFGroup/h5pyd) — Python client for HSDS
- **IMAS**: [IMAS-Data-Dictionary](https://github.com/iterorganization/IMAS-Data-Dictionary) — ITER Integrated Modelling & Analysis Suite

## Contact and support

All users may read the VEST datasets. Saving and storing data in the VEST database is restricted to
authorized users — request write access at
[peppertonic18@snu.ac.kr](mailto:peppertonic18@snu.ac.kr).

Questions, bug reports and feature requests are welcome on the
[issue tracker](https://github.com/VEST-Tokamak/vaft/issues), or by email at
[satelite2517@snu.ac.kr](mailto:satelite2517@snu.ac.kr). For more about the laboratory and the VEST
device, visit [Nuplex](http://nuplex.snu.ac.kr).

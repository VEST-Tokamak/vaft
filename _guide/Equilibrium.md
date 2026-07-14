---
title: Equilibrium
author: VEST team
date: 2026-07-01 10:00
category: guide
layout: post
---

__This section describes how to load, analyze, and plot equilibrium data, and how to drive the external equilibrium codes shipped with VAFT.__

Equilibrium support in VAFT is split across four layers:

| Layer | Module | Purpose |
| --- | --- | --- |
| File I/O | `vaft.data.eqdsk` | Read/write EFIT GEQDSK (g-file) files |
| Physics kernels | `vaft.formula.equilibrium` | Closed-form scalar/profile relations |
| Derived quantities | `vaft.process.equilibrium` | Coordinate mapping, volume integrals, Shafranov integrals |
| External codes | `vaft.code.efit`, `vaft.code.chease`, `vaft.code.gpec` | Reconstruction, refinement, 3D response |

Plotting helpers live in `vaft.plot`.

Get G-file
=====
`vaft.data` exposes a standalone GEQDSK parser. `read_geqdsk(path)` returns a
`GEQDSK` object whose `mapping` holds the standard EFIT quantities.

```python
from vaft.data import read_geqdsk
from vaft.data.resources import data_path

geqdsk = read_geqdsk(data_path('efit/g039915.00319'))

print(geqdsk['NW'], geqdsk['NH'])      # grid size
print(geqdsk['RMAXIS'], geqdsk['ZMAXIS'])  # magnetic axis
print(geqdsk['SIMAG'], geqdsk['SIBRY'])    # psi on axis / boundary
print(geqdsk['CURRENT'])                   # plasma current [A]
```

Available keys include `RDIM`, `ZDIM`, `RCENTR`, `RLEFT`, `ZMID`, `BCENTR`,
the 1D profiles `FPOL`, `PRES`, `FFPRIM`, `PPRIME`, `QPSI`, the 2D flux map
`PSIRZ`, the boundary `RBBBS`/`ZBBBS`, and the limiter `RLIM`/`ZLIM`.

A packaged sample is available without touching the database:

```python
from vaft.data import sample_geqdsk

geqdsk = sample_geqdsk()                       # defaults to 'efit/g039915.00319'
```

Convert to an ODS, or write the g-file back out:

```python
from vaft.data import write_geqdsk

ods = geqdsk.to_omas(ods=None, time_index=0)   # equilibrium subtree
write_geqdsk(geqdsk, '/tmp/g039915.00319')
```

Loading equilibrium from the database
=====
Equilibrium data stored in the VEST database is loaded like any other IDS:

```python
import vaft

ods = vaft.database.load_ods(39915, directory="public")
eq = ods['equilibrium.time_slice.0']

psi = eq['profiles_2d.0.psi']              # psi(R, Z)
q = eq['profiles_1d.q']                    # safety factor
```

Coordinate mapping
=====
`vaft.process.equilibrium` converts between the radial coordinate $R$, the
poloidal flux $\psi$, and the normalized radius $\rho_N$.

```python
from vaft.process.equilibrium import (
    radial_to_psi, psi_to_rho, rho_to_psi, psi_to_rz, psi_to_radial,
)

# R -> psi, interpolated along Z = 0
psi_val = radial_to_psi(r, psi_R, psi_Z, psi)

# psi <-> rho_N, via q-profile integration
rho = psi_to_rho(psi_val, q_profile, psi_axis, psi_boundary)
psi_back = rho_to_psi(rho, q_profile, psi_axis, psi_boundary, tol=1e-6)

# Map a 1D profile f(psi_N) onto the 2D (R, Z) grid.
# Values outside the LCFS (psi_N < 0 or > 1) are set to 0.
f_RZ, psiN_RZ = psi_to_rz(psiN_1d, f_1d, psi_RZ, psi_axis, psi_lcfs)

# 1D psi profile -> inboard/outboard radii
r_in, r_out = psi_to_radial(psi_1d, psi_2d_slice, grid_r, boundary_r, r_axis)
```

Volume averages use $dV = 2\pi R\,dR\,dZ$ and integrate only the plasma cells
($0 \le \psi_N \le 1$):

```python
from vaft.process.equilibrium import volume_average

f_avg = volume_average(f_RZ, psiN_RZ, R, Z)
```

Formulas
=====
`vaft.formula.equilibrium` holds the closed-form relations. They take plain
arrays and floats, so they are usable outside an ODS.

```python
from vaft.formula.equilibrium import (
    psi_normalised, q_from_phi, q_from_rhoN,
    volume_from_RZ_boundary, elongation_from_RZ_boundary,
    triangularity_from_RZ_boundary, bootstrap_current_fraction,
)

psiN = psi_normalised(psi, psi_axis, psi_boundary)  # (psi - psi_a)/(psi_b - psi_a)
q = q_from_phi(psi, phi)                            # q = dPhi/dpsi
q = q_from_rhoN(psiN, rhoN, C=1.0)                  # q = C * rho_N * drho_N/dpsi_N

V = volume_from_RZ_boundary(R_bdry, Z_bdry)         # 2*pi * A_poly * R_bar
kappa = elongation_from_RZ_boundary(R_bdry, Z_bdry)
delta = triangularity_from_RZ_boundary(R_bdry, Z_bdry)

f_bs = bootstrap_current_fraction(n_e, T_e_keV, R0, a, q_95)
```

See [Physics formulas]({{ site.baseurl }}/guide/Formula/) for the full catalogue.

Diamagnetism and Shafranov integrals
=====
The diamagnetism $\mu_i$ follows the volume-integral definition
$\mu_i = \frac{1}{B_{pa}^2 \Omega} \int_\Omega (B_{tv}^2 - B_t^2)\, dV$,
with $B_t = F(\psi)/R$ and $B_{tv} = F_{vac}/R$:

```python
from vaft.process.equilibrium import calculate_diamagnetism

mu_i = calculate_diamagnetism(
    R_grid, Z_grid, psi_RZ, psi_axis, psi_lcfs,
    psiN_1d, f_1d, f_vac_val, B_pa,
    V_p=None,          # plasma volume; recomputed from the grid when None
)
```

The poloidal field on the LCFS, and the Shafranov integrals $S_1, S_2, S_3$
used for $\beta_p$ and $l_i$, are computed from the flux grid:

```python
from vaft.process.equilibrium import (
    poloidal_field_at_boundary, shafranov_integrals, efit_virial_volume_integrals,
)

B_p_bdry, B_R_bdry, B_Z_bdry = poloidal_field_at_boundary(
    R_grid_1d, Z_grid_1d, psi_grid, R_bdry, Z_bdry,
)

S1, S2, S3, alpha = shafranov_integrals(
    R_bdry, Z_bdry, B_p_bdry,
    R_grid, Z_grid, B_R_grid, B_Z_grid,
    p_boundary=0.0,
)

integrals = efit_virial_volume_integrals(
    R_grid, Z_grid, R_bdry, Z_bdry, B_R_grid, B_Z_grid,
    p_tot_grid=p_tot_grid, B_phi_grid=B_phi_grid,
)
```

Plotting
=====
```python
import vaft

ods = vaft.database.load_ods(39915, directory="public")

vaft.plot.equilibrium_1d_radial(ods, time_slices=None)   # psi, J_tor, p, q vs R
vaft.plot.equilibrium_2d_profiles(ods, time_slice=None)  # psi, p, j / B_r, B_z, B_phi
vaft.plot.time_equilibrium_analysis(ods, xunit='s', xlim='plasma')
```

Single global quantities have their own time traces, e.g.

```python
vaft.plot.time_equilibrium_plasma_current(ods, yunit='MA')
vaft.plot.time_equilibrium_q95(ods)
vaft.plot.time_equilibrium_li(ods)
vaft.plot.time_equilibrium_beta_pol(ods)
vaft.plot.time_equilibrium_w_mhd(ods)
```

Equilibrium codes
=====

### EFIT — magnetic reconstruction
`vaft.code.efit` prepares k-files from an ODS, runs EFIT, and collects the
resulting g/a/m files back into an `EFITResult`.

```python
from vaft.code import EFITConfig, prepare_efit_inputs, run_efit, collect_efit_outputs

config = EFITConfig(
    executable='efit',
    workdir='/tmp/efit-run',
    shot=39915,
    times=[0.315, 0.320, 0.325],
    npprime=2,
    nffprime=2,
)

inputs = prepare_efit_inputs(ods, config)
result = run_efit(inputs, config)

if result.ok:                      # returncode == 0
    print(result.gfiles, result.afiles)

# Or re-collect an existing working directory
result = collect_efit_outputs('/tmp/efit-run', config)
```

### CHEASE — equilibrium refinement
`vaft.code.chease` takes a GEQDSK (path, `GEQDSK`, or mapping) and produces a
refined, high-resolution equilibrium.

```python
from pathlib import Path
from vaft.data.resources import data_path
from vaft.code.chease import (
    CHEASEConfig, find_chease_executable, prepare_chease_inputs,
    run_chease, collect_chease_outputs, refine_equilibrium,
)

config = CHEASEConfig(
    workdir=Path('/tmp/chease-run'),
    target_psin=0.993,
    nw=513,
    create_plot=True,
    cleanup=False,
    timeout=90,
)

print(find_chease_executable(config))   # None if CHEASE is not installed

# One-shot: prepare + run + collect
result = refine_equilibrium(data_path('efit/g039915.00319'), config)

# Or drive the steps explicitly
inputs = prepare_chease_inputs(data_path('efit/g039915.00319'), config)
result = run_chease(inputs, config)
result = collect_chease_outputs('/tmp/chease-run', config)
```

### GPEC — perturbed equilibrium and 3D response
`vaft.code.gpec` drives the DCON / RDCON / STRIDE / GPEC suite for one
shot/time GEQDSK. The installation root is read from `$GPECHOME` unless
`gpec_home` is set.

```python
from pathlib import Path
from vaft.code import (
    GPECSuiteConfig, GPECCaseInputs,
    prepare_gpec_suite_case, run_gpec_suite_case, collect_gpec_suite_outputs,
)

config = GPECSuiteConfig(
    modules=('dcon', 'gpec'),
    modes=(1, 2),
    run_mode='run_if_available',
    psihigh=0.994,
)

inputs = GPECCaseInputs(
    shot=39915,
    time_ms=319,
    geqdsk=Path('/tmp/g039915.00319'),
    workdir=Path('/tmp/gpec-run'),
)

prepare_gpec_suite_case(inputs, config)     # write input decks only
result = run_gpec_suite_case(inputs, config)

outputs = collect_gpec_suite_outputs('/tmp/gpec-run')
```

Notebooks
=====
Worked, runnable examples:

- [`forward_equilibrium_using_TES.ipynb`](https://github.com/VEST-Tokamak/vaft/blob/main/notebooks/forward_equilibrium_using_TES.ipynb) — forward Grad-Shafranov solve through the `vaft.code.tes` adapter
- [`equilibrium_refinement_using_chease.ipynb`](https://github.com/VEST-Tokamak/vaft/blob/main/notebooks/equilibrium_refinement_using_chease.ipynb) — CHEASE refinement of a packaged sample g-file

Outline notebooks, currently drafted as documentation shells:

- [`mhd_equilibrium_analysis.ipynb`](https://github.com/VEST-Tokamak/vaft/blob/main/notebooks/mhd_equilibrium_analysis.ipynb)
- [`magnetic_equilibrium_reconstruction_with_efit.ipynb`](https://github.com/VEST-Tokamak/vaft/blob/main/notebooks/magnetic_equilibrium_reconstruction_with_efit.ipynb)
- [`perturbed_equilibrium_and_3d_response_with_gpec.ipynb`](https://github.com/VEST-Tokamak/vaft/blob/main/notebooks/perturbed_equilibrium_and_3d_response_with_gpec.ipynb)

Source
=====
- [`vaft/process/equilibrium.py`](https://github.com/VEST-Tokamak/vaft/blob/main/vaft/process/equilibrium.py)
- [`vaft/formula/equilibrium.py`](https://github.com/VEST-Tokamak/vaft/blob/main/vaft/formula/equilibrium.py)
- [`vaft/data/eqdsk.py`](https://github.com/VEST-Tokamak/vaft/blob/main/vaft/data/eqdsk.py)
- [`vaft/code/efit.py`](https://github.com/VEST-Tokamak/vaft/blob/main/vaft/code/efit.py)
- [`vaft/code/chease.py`](https://github.com/VEST-Tokamak/vaft/blob/main/vaft/code/chease.py)
- [`vaft/code/gpec.py`](https://github.com/VEST-Tokamak/vaft/blob/main/vaft/code/gpec.py)

See also [Examples]({{ site.baseurl }}/guide/examples/) and
[Data structures]({{ site.baseurl }}/guide/Data_structures/).

Credit : Hongsik-yun (peppertonic18@snu.ac.kr)

---
title: Plasma models, orderings, and scales
author: VEST team
date: 2026-10-06 09:00
category: guide
layout: post
permalink: /reference/plasma-models/
guide:
  architecture: What physical model each VAFT backend mode represents, from an audit of upstream sources and primary literature (issue #1725, under #1723).
  prerequisites: The Computational layers page, for where a computation lives in VAFT; this page says what physics it represents.
  expected: Which equations a mode solves, which ordering it assumes, what "kinetic" means in each VAFT name, and which classifications are still open.
---

Two pages answer different questions. [Computational layers]({{ '/reference/computational-layers/' | relative_url }}) says
*where* a computation belongs in VAFT (Formula, Process, Code). This page says *what physical model* the computation represents.

**What this page is.** It is the Phase A audit of #1723, under issue #1725: a reviewed scientific classification.

**What it is not.** It is not an API: no enum and no model class is introduced here. Phase B (#1727) will encode only the
distinctions that survive this audit.

**How confidence is marked.** Every classification carries a confidence:

| confidence | meaning |
| --- | --- |
| **verified** | backed by a primary theory publication, upstream solver documentation, or upstream source |
| **provisional** | backed by literature, but not checked against source, or a citation detail is unconfirmed |
| **unresolved** | the evidence needed has not been found |

A VAFT name is never evidence for the physics.

## 1. Why one "fidelity" axis does not work

The models VAFT drives differ along several independent axes:

- the governing equation for the bulk plasma;
- which populations, if any, are treated kinetically;
- how a kinetic response is coupled back to a fluid model;
- the distribution representation;
- the spatial domain;
- the regime;
- the field model.

**Resistive MHD and gyrokinetics are not two rungs of one ladder.** Their orderings differ:
- resistive MHD keeps a resistive layer and averages over all particle phase space;
- gyrokinetics keeps $k_\perp\rho\sim1$ and phase-space structure, but removes the fast gyro-motion.

Neither contains the other.

The axes the audit found to be physically meaningful are:

| axis | question it answers | examples of values |
| --- | --- | --- |
| scientific operation | what is being computed | stability, perturbed equilibrium, neoclassical transport, turbulence, orbit/loss, source deposition |
| bulk description | what represents the bulk plasma | fluid, hybrid (fluid + kinetic response), kinetic, particle (test particles), reduced |
| fluid model | which fluid closure, when there is one | ideal MHD, resistive MHD, extended MHD |
| kinetic equation | which kinetic equation is solved, when one is | none, drift kinetic, gyrokinetic, Fokker–Planck (test species) |
| kinetic population | which species obey it | all species, thermal ions (+ electrons), fast ions, energetic particles |
| kinetic coupling | how a kinetic result re-enters a fluid model | pressure/energy (δW_k), current, sources only, none |
| distribution formulation | $f$ split | δf, full-f, N/A |
| orbit representation | particle motion | full orbit, guiding centre, bounce/transit averaged, N/A |
| spatial domain | radial extent | local (flux tube or single surface), radially global, whole volume, orbit |
| regime | dynamics | linear, nonlinear, quasilinear, steady state, static (marginal) |
| field model | perturbed fields | MHD displacement, electrostatic, electromagnetic ($A_\parallel$, $\delta B_\parallel$) |

**Two points this audit settled:**
- Orbit representation is a separate axis from the kinetic equation. ASCOT5 and NUBEAM solve a Fokker–Planck equation
  *by* Monte Carlo orbit following, while SIMPLE follows orbits and evolves no distribution.
- "Kinetic-profile input" is not an axis at all. It is a property of the inputs (§6).

## 2. Theory foundations

### 2.1 Fluid hierarchy

**Ideal MHD.**
- Single fluid, ideal Ohm's law $E + v\times B = 0$, scalar pressure with an adiabatic or incompressible closure.
- Valid at $\rho/L\ll1$ and $\omega\ll\Omega_i$, on scales above $d_i$.
- References: Bernstein, Frieman, Kruskal & Kulsrud, Proc. R. Soc. A **244**, 17 (1958); Freidberg, *Ideal MHD* (2014).

**Resistive MHD.**
- Adds $\eta J$ to Ohm's law. In a tokamak this matters only in thin layers around rational surfaces.
- Glasser, Greene & Johnson, Phys. Fluids **18**, 875 (1975).

**Hall, two-fluid and extended MHD.**
- These add, separately, the Hall term $J\times B/ne$, the electron pressure gradient, electron inertia, gyroviscosity
  (FLR) and anisotropic pressure. They come from Braginskii's two-fluid equations, Rev. Plasma Phys. **1**, 205 (1965).
- "Extended MHD" is a family of fluid closures.
- It is **not** a synonym for kinetic MHD (§2.5).

**Reduced MHD.**
- An expansion in $\epsilon=a/R$, $\beta$ and $k_\parallel/k_\perp$; Strauss, Phys. Fluids **19**, 134 (1976).
- It is an ordering applied to a fluid model, not a different fluid model. Low aspect ratio weakens its premise (#1627).

### 2.2 Drift kinetics

**Ordering.**
- $\omega/\Omega_c\ll1$ and $\rho/L\ll1$.
- The gyro-phase is averaged out.
- Particles move as guiding centres: parallel streaming, $E\times B$ drift, $\nabla B$ and curvature drifts.
- Collisions are retained as needed.

The equation is derived systematically by Hazeltine, Plasma Phys. **15**, 77 (1973). The neoclassical application is
reviewed by Hinton & Hazeltine, Rev. Mod. Phys. **48**, 239 (1976).

**Three related things that are not equivalent:**
- **Guiding-centre dynamics:** single-particle equations of motion. Littlejohn, J. Plasma Phys. **29**, 111 (1983);
  Cary & Brizard, Rev. Mod. Phys. **81**, 693 (2009).
- **The drift-kinetic equation:** evolves the guiding-centre *distribution*.
- **Neoclassical transport:** one application of the drift-kinetic equation, as a steady first-order solution driven by
  equilibrium gradients.

### 2.3 Gyrokinetics

**Ordering.**
- Low frequency, $\omega/\Omega_i\sim\rho_*$.
- Strong anisotropy, $k_\parallel/k_\perp\sim\rho_*$.
- Small fluctuations, $\delta f/F\sim\rho_*$.
- But $k_\perp\rho\sim1$ is allowed: the gyro-phase is removed by gyro-averaging, not by assuming $k_\perp\rho\ll1$.

References: Frieman & Chen, Phys. Fluids **25**, 502 (1982); Brizard & Hahm, Rev. Mod. Phys. **79**, 421 (2007). The
multiscale treatment with rotation is Abel, Plunk, Wang, Barnes, Cowley, Dorland & Schekochihin, Rep. Prog. Phys.
**76**, 116201 (2013).

**Relation to drift kinetics.** In the limit $k_\perp\rho\to0$, the gyrokinetic equation reduces to a drift-kinetic
form. Gyrokinetics is therefore not "higher-fidelity drift kinetics":
- it keeps finite-$k_\perp\rho$ physics that drift kinetics orders out;
- it usually works in δf with a fluctuation ordering;
- neoclassical drift-kinetic solvers do not use such an ordering.

### 2.4 Particle-orbit descriptions

There are three descriptions:
- **full orbit:** the Lorentz force;
- **guiding-centre orbit:** first-order guiding-centre equations;
- **distribution evolution:** a kinetic equation for $f$.

**Monte Carlo codes connect the first two to the third.** A test-particle Fokker–Planck equation can be solved by
following markers with stochastic collision kicks (ASCOT5, NUBEAM). This is still a statement about a test population
in a fixed background, not about the bulk plasma.

**An orbit integrator alone is not a drift-kinetic plasma solver.** SIMPLE is one example: it evolves no distribution.

### 2.5 Kinetic-MHD and hybrid models

**Shared structure.** The bulk plasma is fluid (MHD). A kinetic equation for selected populations supplies part of
the force through a pressure tensor, a current, or an energy term. The family:
- Kruskal & Oberman, Phys. Fluids **1**, 275 (1958): the collisionless kinetic energy principle;
- Antonsen & Lane, Phys. Fluids **23**, 1205 (1980);
- Cheng, Phys. Rep. **211**, 1 (1992): the kinetic-MHD review;
- Park, Belova, Fu, Tang, Strauss & Sugiyama, Phys. Fluids B **4**, 2033 (1992): hybrid gyrokinetic-MHD.

**Distinctions to keep:**
- MHD plus a drift-kinetic response (DCON kinetic, MARS-K);
- MHD plus kinetic energetic particles (M3D-K, MEGA, NIMROD kinetic);
- a fully kinetic or gyrokinetic plasma (CGYRO, GENE, GTC, ORB5).

**Kinetic-MHD is a family, not one model.** What varies inside it is the coupling (pressure or current), the
population, and the orbit treatment (§8).

## 3. The GPEC suite, mode by mode

**Sources.**
- Upstream: `~/git/GPEC` at e68d7ac2 (2026-08-04).
- VAFT adapter: `vaft/code/gpec/`. Packaged namelists: `vaft/data/gpec/*.in`.

**Main finding.** DCON's kinetic mode is not a separate kinetic solver:
- with `kin_flag=t`, DCON calls PENTRC's bounce-averaged drift-kinetic operator at every surface and bounce harmonic;
- it adds the complex result into the ideal Euler–Lagrange coefficient matrices (`dcon/fourfit.F:1062-1085,
  1153-1158`).

| mode | operation | bulk | fluid model | kinetic equation | kinetic population | coupling | FLR | collisions | regime | confidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DCON ideal (`kin_flag=f`) | $n\neq0$ stability: Newcomb criterion on the Euler–Lagrange ODE, δW eigenvalues | fluid | linear ideal MHD energy principle | none | — | — | N/A | none | linear, static | verified |
| DCON kinetic (`kin_flag=t`) | same ODE with a complex, non-self-adjoint δW + δW_k; Im δW gives the torque | hybrid | ideal MHD | bounce/transit-averaged linear drift kinetics (PENTRC operator) | one thermal main ion; electrons if `electron_flag=t`; Maxwellian only, no energetic particles | energy (δW_k matrices added to the EL coefficients), self-consistent | none (zero Larmor radius) | `pentrc.in` `nutype`: zero / small / Krook / harmonic | linear, zero lab-frame frequency | verified (equation class); provisional (the name "drift-kinetic MHD") |
| GPEC ideal | perturbed equilibrium under an external 3-D field | fluid | ideal MHD | none | — | — | N/A | none | linear, static | verified |
| GPEC kinetic | self-consistent kinetic perturbed equilibrium; `dw_flag` energy and torque profiles | hybrid | ideal MHD | inherited from DCON's matrices, nothing separate | inherited | inherited | none | inherited | linear, static | verified |
| GPEC thresholds (`singthresh_*`) | critical resonant field / island width per surface | reduced layer models | SLAYER: "linear drift MHD" slab; Callen: analytic cubic | none (kinetic *profiles* only) | — | — | unresolved | unresolved | linear layer | provisional |
| PENTRC on a GPEC ξ | NTV torque, non-ambipolar flux for a given displacement | kinetic closure on a fluid displacement, not self-consistent | (ideal or kinetic GPEC ξ is input) | bounce-averaged linear drift kinetics; reduced variants: large aspect ratio, CGL fluid limit | one species per run: main ion or electrons | torque from the anti-Hermitian part | none | zero / small / Krook / harmonic | linear, static | verified |
| RDCON + RMATCH | Δ′ matrix and matched resistive growth rates | fluid | outer: ideal MHD; inner: linear resistive MHD (GGJ) | **none** | — | — | N/A | scalar η per surface | linear eigenvalue | verified |
| STRIDE | ideal stability and free-boundary Δ′ | fluid | ideal MHD (outer region only; no η) | none | — | — | N/A | none | linear | verified |
| MATCH (`ideal_flag=t`, as shipped) | reconstruct DCON's ideal eigenfunction | fluid | ideal MHD | none | — | — | N/A | none | linear | provisional |

### 3.1 DCON ideal

**The governing model.**
- Linear ideal MHD energy principle, the BFKK δW.
- Integrated as Newcomb's Euler–Lagrange ODE in a Fourier poloidal basis.
- The ideal jump condition at rational surfaces.

References: Newcomb, Ann. Phys. **10**, 232 (1960); Glasser, Phys. Plasmas **23**, 072505 (2016).

**Numerical and boundary-condition choices, not physics:**
- `vac_flag`: free or fixed boundary;
- `psiedge`, `sas_flag`, `qhigh`: edge truncation;
- `delta_m*`;
- `con_flag`: integrate through singular layers instead of applying the ideal jump. It changes eigenvalues and turns
  off GPEC `singfld`.

**Local diagnostics alongside:**
- the Mercier criterion;
- high-$n$ ballooning: Connor, Hastie & Taylor, Proc. R. Soc. A **365**, 1 (1979);
- the GGJ $D_I$, $D_R$, $H$ from `resist_eval`, as diagnostics only.

### 3.2 DCON kinetic

The flags, from source:

| flag | effect | source |
| --- | --- | --- |
| `kin_flag` | builds PENTRC kinetic matrices and adds them to the EL coefficients; singular surfaces then move off the rationals and are found as zeros of det F (`ksing_find`) | `dcon_mod.f:108`; `dcon.F:214-245`; `sing.f:1486` |
| `passing_flag`, `trapped_flag` | choose the pitch-angle domain, i.e. the PENTRC method prefix f/t/p. Upstream defaults F/T, so a default kinetic run is **trapped-only**. VAFT's template sets both to t | `fourfit.F:959-968` |
| `ion_flag`, `electron_flag` | add main-ion and/or electron δW_k; the contributions are summed | `fourfit.F:1066-1085` |
| `dcon_kin_threads` | OpenMP thread count; numerical only | `dcon.F:89-95` |

**Kinetic physics (all from PENTRC).**
- Linear δf about a Maxwellian, bounce/transit averaged.
- The drive term contains $\omega_E+\omega_{*n}+\omega_{*T}(x-3/2)$.
- The denominator is $i(\ell\omega_b\sqrt{x}+n(\omega_E+\omega_D x))-\nu$ (`pentrc/energy.f90:374`). The mode
  frequency does not appear: the response is at zero lab-frame frequency.
- Resonances retained: bounce, transit, precession and $E\times B$.
- No gyro-average appears anywhere in `pentrc/`, so FLR is zero. Zero orbit width is inferred from the
  flux-surface bounce averaging, not stated in the source.

**The species model.** One thermal ion, $Z$ and $m$ from `pentrc.in`. The collision frequencies and $\ln\Lambda$ are
computed internally from $n$ and $T$, and $Z_{\rm eff}$ from $n_i/n_e$ with one impurity.

**Configuration split.** DCON-kinetic physics is partly configured outside `dcon.in`: `nl`, `nutype`, `f0type`, `zi`,
`mi` and the kinetic file come from `pentrc.in`. The kinetic-file rotation column is $\omega_E$, not $\omega_\phi$
(`pentrc/inputs.f90:162-167`).

References (upstream-cited): Park, Phys. Plasmas **18**, 110702 (2011), the kinetic energy principle; Logan, Park,
Kim, Wang & Berkery, Phys. Plasmas **20**, 122507 (2013); Park & Logan, Phys. Plasmas **24**, 032505 (2017).

### 3.3 GPEC kinetic

**Not a `gpec.in` switch.** GPEC reads `kin_flag` and `con_flag` back from DCON's `euler.bin` (`gpec/idcon.f:62`), so
its kinetic content is entirely inherited from DCON.

**Thresholds (`singthresh_*`).** They use kinetic **profiles** through `initialize_pentrc(op_kin=T)`, but no kinetic
equation.

### 3.4 RDCON, STRIDE and RMATCH

**No kinetic equation in either.** Neither `rdcon/` nor `rmatch/` reads a kinetic file.

**What VAFT's inputs do.** VAFT supplies $T_e$, $n_e$, $Z_{\rm eff}$ and $\ln\Lambda$ (`RDCONOptions`). These fill
only RMATCH's per-surface `eta` (NRL Spitzer) and `massden`, through `vaft/code/gpec/_solvers.py:385-457`.

**Classification.** This is *kinetic-profile-informed resistive parameters*. It is not
`kinetic_equation = drift_kinetic`.

References: Glasser, Wang & Park, Phys. Plasmas **23**, 112506 (2016); Glasser & Kolemen, Phys. Plasmas **25**,
082502 (2018).

### 3.5 PENTRC

**Classification:**
- bounce-averaged linear drift kinetics;
- the general-aspect-ratio methods are `fgar`, `tgar` and `pgar` (full = trapped + passing);
- reduced variants: `rlar` and `clar` (large aspect ratio), and `fcgl`, a CGL fluid limit;
- one species per run;
- torque from the anti-Hermitian part.

**Use with GPEC.** It is a kinetic *closure* evaluated on a fluid displacement, not a self-consistent model. Inside
DCON-kinetic, the same operator *is* self-consistent.

**Factorization.** PENTRC is best described by attributes: drift kinetic, bounce averaged, linear δf, closure on a
given ξ. A single "neoclassical" label does not cover it.

References: Park, Boozer & Menard, Phys. Rev. Lett. **102**, 065002 (2009); Logan et al. (2013).

## 4. GACODE

**Sources.** Upstream `~/git/gacode`; VAFT `vaft/code/gacode/`.

| mode | operation | bulk | kinetic equation | population | distribution | FLR | collisions | domain | regime | confidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NEO as VAFT runs it | neoclassical fluxes, bootstrap $j_\parallel$, poloidal flow | kinetic (all species) | drift kinetic, first order: steady $f_1$ driven by $\nabla F_M$ and $E_r$ | all species (≤6); electrons kinetic | δf | none | full linearized Fokker–Planck (COLLISION_MODEL=4) | local; several radii are independent local solves | steady state | verified |
| NEO analytic companions (HS, NCLASS, Sauter) | closed-form neoclassical fluxes | reduced | none | — | N/A | N/A | model-specific | local | steady | verified (existence) |
| CGYRO linear ES/EM (VAFT default ES) | microstability: γ, ω, eigenfunction | kinetic | gyrokinetic | all species, gyrokinetic electrons | δf | full gyro-average | Sugama (4) by default | local flux tube | linear initial value | verified |
| CGYRO nonlinear | turbulent fluxes | kinetic | gyrokinetic, nonlinear | as above | δf | full | as above | local | nonlinear | verified |
| TGLF (single $k_y$ / QL with SAT 0–3) | linear eigenmodes; quasilinear fluxes | reduced | **none solved**: a gyro-Landau-fluid moment system derived from gyrokinetics | trapped and passing species | N/A | approximate (gyro-averaged moments) | reduced model | local | linear / quasilinear | verified (reduction) |
| TGLF-NN | quasilinear fluxes | reduced (learned surrogate) | none; inherits its parent TGLF variant | as parent | N/A | inherited | inherited | local | quasilinear | provisional |

**NEO.**
- Neoclassical *drift kinetics* here means solving for the steady first-order distribution of all species.
- That is a different operation from DCON-kinetic's orbit-averaged response inside δW, though both are drift
  kinetic.
- References: Belli & Candy, Plasma Phys. Control. Fusion **50**, 095010 (2008); **54**, 015015 (2012).

**CGYRO.**
- References: Candy, Belli & Bravenec, J. Comput. Phys. **324**, 73 (2016). Collisions: Sugama, Watanabe & Nunami,
  Phys. Plasmas **16**, 112503 (2009).
- Its existing record, `vaft.code.gacode.cgyro.formalism()`, maps onto the axes in §1 as below.

| CGYRO field | values today | broad axis (#1723) | gyrokinetic-specific (#1353) | implementation detail |
| --- | --- | --- | --- | --- |
| `distribution_formulation` | `delta_f` | yes | — | — |
| `spatial_domain` | `local` | yes | the global-spectral variant | — |
| `numerical_representation` | `continuum` | no | continuum vs PIC matters for GK comparisons | yes |
| `field_model` | `es`, `em-aperp`, `em-aperp-bpar` | coarse: ES vs EM | the $A_\parallel$ / $\delta B_\parallel$ split | the integer `N_FIELD` |
| `regime` | `linear`, `nonlinear` | yes | — | — |
| `topology_domain` | `closed_flux_surface` | yes | — | — |
| `geometry_model` | `miller` | no | Miller / MXH / s-α | partly |
| `species_model` | `kinetic_electrons` | as kinetic population | adiabatic vs gyrokinetic electrons | — |

Not in the record, though they are physics: `kinetic_equation = gyrokinetic` (implicit), the collision model, and the
fact that VAFT pins $\gamma_E=\gamma_p=M=0$ (no rotation or shear).

**TGLF.**
- References: Staebler, Kinsey & Waltz, Phys. Plasmas **14**, 055909 (2007), with Hammett & Perkins, Phys. Rev. Lett.
  **64**, 3019 (1990) for the Landau-fluid closure.
- SAT1: Staebler, Candy, Howard & Holland, Phys. Plasmas **23**, 062518 (2016). SAT2: Staebler et al., Nucl. Fusion
  **61**, 116007 (2021). SAT3: Dudding et al., Nucl. Fusion **62**, 096005 (2022).

**TGLF-NN.**
- Its physical meaning is that of the TGLF variant it was trained on.
- Its computational realization is a learned surrogate, which is a separate fact.
- In VAFT the variant is known only from the model name, and a SAT mismatch is reported but not refused
  (`tglf/surrogate/inference.py:149-178`).

## 5. Orbit and fast-ion codes

| mode | operation | bulk | equation | population | orbit | collisions | coupling | confidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ASCOT5 `SIM_MODE=1` | fast-ion orbits, distribution, losses | particle (test) | Fokker–Planck for the test species, by Monte Carlo | minority / test species | full orbit | test-particle Coulomb on a Maxwellian background | none (fixed fields and background) | verified (upstream docs) |
| ASCOT5 `SIM_MODE=2` (default) | same | particle (test) | same | same | guiding centre (first-order transform) | guiding-centre operator | none | verified |
| ASCOT5 `SIM_MODE=3` | same, wall loads | particle (test) | same | same | guiding centre, switching to full orbit near the wall | as above | none | verified |
| NUBEAM (standard FLR) | beam deposition, slowing down, heating, current drive, torque | particle (test) inside a transport code | Fokker–Planck for fast ions, by Monte Carlo | beam ions, fusion products | guiding centre + FLR position sampling | drag, energy diffusion, pitch scattering; charge exchange | sources and fast-ion pressure returned; no field feedback | verified (code) |
| NUBEAM VEST case (`nlbgflr=1`) | as above | as above | as above | as above | **enhanced FLR** (gyro-orbit deformed by $\nabla B$), meant for low-field STs | as above | as above | verified |
| SIMPLE | collisionless guiding-centre confinement / loss fraction | particle (test) | **none**: orbit integration | alphas / test particles | guiding centre, symplectic | none | none | provisional; no VAFT adapter |

**ASCOT5.** Option names come from `a5py/ascot5io/options.py`. References: Hirvijoki et al., Comput. Phys. Commun.
**185**, 1310 (2014) for ASCOT4; Varje et al., arXiv:1908.02482 (2019) for ASCOT5.

**NUBEAM.** References: Goldston et al., J. Comput. Phys. **43**, 61 (1981); Pankin et al., Comput. Phys. Commun.
**159**, 157 (2004).

**SIMPLE.** Albert, Kasilov & Kernbichler, J. Comput. Phys. **403**, 109065 (2020).

**VAFT coverage.** ASCOT5 appears in VAFT only as a diagram catalogue entry. SIMPLE does not appear at all.

## 6. What "kinetic" means in VAFT names

The word carries at least six meanings across VAFT's public surface. Each name below is classified so that
documentation can say which one it means. **No name is changed by this audit.**

| category | meaning | representative names | say instead |
| --- | --- | --- | --- |
| A. measurement/data | Thomson, CES/ion Doppler data, or slices that have them | "kinetic diagnostics" (`Profiles.md`), `kinetic_overview_profiles`, `n_kinetic` / `supports_kinetic_efit` (`vest_upstream.py`), `kineticEfit/` sample directory | profile diagnostics; slices with measured $T_i$ |
| B. profile container | $n$, $T$, ω on a radial grid | `KineticProfiles`, `read_kin`/`write_kin`, `SyntheticKineticProfiles`, `build_kinetic_core_profiles`, GPEC/PENTRC `kinetic_file` | kinetic-profile set |
| C. kinetically constrained reconstruction | EFIT with a pressure constraint from profiles | `vaft.code.efit.kinetic`, `run_kinetic_efit`, stages `kinetic_efit` / `electron_efit`, lineage `electron_kinetic`, `KINETIC_FAMILIES`, `equilibrium_family: kinetic` | kinetic-pressure-constrained EFIT (measured $T_i$, or assumed $T_i/T_e$) |
| D. profile-informed closure or coefficient | profiles set a closure or coefficient of a fluid calculation | `chease_kinetic_iteration`, `closure="kinetic_pressure"`, RDCON's $T_e$/$n_e$ → η | kinetic-profile-pressure closure; profile-informed resistivity |
| E. hybrid fluid-kinetic model | a kinetic response inside a fluid model | `kin_flag` / "kinetic DCON", NTV "kinetic response" | kinetic-MHD (δW_k) DCON; drift-kinetic response |
| F. kinetic governing equation | the code solves a kinetic equation | NEO ("drift-kinetic solver"), `gyrokinetics_local_from_cgyro`, "kinetic electrons" (CGYRO) | drift-kinetic neoclassical; gyrokinetic; gyrokinetic (non-adiabatic) electrons |
| G. derived from kinetic theory | a reduced or analytic model fitted to kinetic results | Sauter/Redl fits, TGLF, NTV formulas, Porcelli "kinetic effects" | drift-kinetic-derived fit; quasilinear gyro-Landau-fluid |

The names most likely to mislead:

1. `kinetic_efit` versus `run_kinetic_efit`.
   - The function runs both the `electron_efit` stage (Thomson + assumed $T_i/T_e$) and the `kinetic_efit` stage
     (measured $T_i$).
   - The stage name means "measured $T_i$" only.
2. The lineage `electron_kinetic` (`vaft/validation/kinetic_state.py`) is the `electron_efit` stage under a second
   name.
3. "Kinetic state" names three unrelated things:
   - the lane-K Thomson-vs-reconstruction slice table;
   - a synthetic self-consistent equilibrium/profile state (`Profiles.md`);
   - the NEO-ready local plasma state.
4. `kinetic_pressure` means five things:
   - measured thermal pressure in validation;
   - EFIT constraint points;
   - a CHEASE closure mode;
   - a synthetic-profile pressure constraint;
   - a plot alias.
5. `kin_flag` versus the `.kin` file.
   - `kin_flag=t` switches DCON to kinetic-MHD.
   - A `.kin` file with `kin_flag=f` only feeds profiles to PENTRC or the threshold models.
6. `EFIT_KINETIC_FAMILIES` (`_summary.py`) includes MSE and $j_\phi$, so there "kinetic" means "non-magnetic".
7. `kinetic_energy*`:
   - `kinetic_energy_from_beta_p_B_pa_V_p` is the thermal stored energy;
   - `virial_kinetic_energy` is bulk-flow energy.

**Kinetic-profile input never implies a kinetic governing equation.** The RDCON row of §3, the CHEASE iteration and
the EFIT names all show the same pattern: a fluid model whose inputs or closures come from measured or assumed
profiles.

## 7. Orderings and scales

**Purpose of this section.** It is a compact map from scales to the classifications above. The full reference for
characteristic scales and orderings is #1724. The ordering parameters themselves are #1627.

**Frequencies,** typically ordered as

$$\Omega_{ci} \gg \omega_{t},\ \omega_{b} \gg \omega_{D},\ \omega_E \quad;\quad
\tau_A^{-1} \gtrsim \omega_{t,i} \gg \tau_E^{-1} \gg \tau_R^{-1}.$$

The second chain is the conventional tokamak hierarchy; VEST's short pulse can compress it.

**Lengths:**
- $\rho_s$ and $\rho_i$ against gradient lengths $L_n$, $L_T$ and the minor radius $a$;
- $k_\perp\rho$;
- the connection length $qR$;
- the banana orbit width $\sim q\rho/\sqrt\epsilon$;
- the resistive layer width;
- the radial simulation domain.

| model family | what it removes | ordering that justifies it |
| --- | --- | --- |
| ideal MHD | particle phase space; resistivity | $\rho/L\ll1$, $\omega\ll\Omega_i$, $S\gg1$, scales $\gg d_i$ |
| resistive MHD (layer) | phase space; keeps η only in the layer | as ideal MHD outside the layer |
| drift kinetics | the gyro-phase | $\omega/\Omega\ll1$, $\rho/L\ll1$, $k_\perp\rho\ll1$ |
| gyrokinetics | the gyro-phase, keeping $k_\perp\rho\sim1$ | $\omega/\Omega\sim\rho_*$, $k_\parallel/k_\perp\sim\rho_*$, $\delta f/F\sim\rho_*$ |
| bounce-averaged drift kinetics (PENTRC, DCON-kinetic) | gyro-phase and bounce phase | $\omega\ll\omega_b,\omega_t$ for the averaged population |
| guiding-centre orbit following | the gyro-phase for one particle | $\rho/L_B\ll1$ (fails for fast ions at low field, hence NUBEAM's FLR options) |

**Why the hierarchy is not a ladder.**
- Ideal MHD averages over phase space completely; drift kinetics keeps phase space but drops the gyro-phase.
- Gyrokinetics keeps phase space at finite $k_\perp\rho$, but orders $\omega/\Omega$ and the fluctuation amplitude.
- Resistive MHD keeps a thin resistive layer that none of the kinetic models above contains.
- A guiding-centre orbit integrator follows a particle and evolves no distribution.

## 8. Unresolved

Each item says what evidence is missing.

**DCON and GPEC:**
- Whether DCON-kinetic should be called "drift-kinetic MHD". The operator is drift kinetic and bounce averaged at zero
  lab-frame frequency. The exact ordering statement of Park (2011) and Park & Logan (2017) has not been checked against
  it.
- The Kruskal–Oberman correspondence. PENTRC implements a CGL limit (`f0type="cgl"`); no KO-labelled limit exists in
  the source.
- How the fluid compressional term and δW_k are partitioned (double counting). No adiabatic-γ term was found in DCON's
  matrix construction.
- Finite orbit width in PENTRC: inferred absent, not stated.
- The "harmonic" collision model: an energy- and ℓ-dependent Krook model; its derivation reference was not found.
- SLAYER's physics: only "linear drift MHD" in its README. Callen's threshold rests on an internal report (UW-CPTC
  16-4, 2016).

**GACODE:**
- The CGYRO collision-operator class, and whether NEO's and CGYRO's operators belong to one generic category. NEO's
  "4" and CGYRO's "4" are different operators.
- The FLR content of TGLF's gyro-Landau-fluid moments.
- The training provenance of the TGLF-NN models VAFT resolves.

**Taxonomy design:**
- NUBEAM's guiding centre + FLR sampling (and `nlbgflr`) does not fit a guiding-centre / full-orbit binary.
- Whether full orbit belongs under the kinetic equation or the orbit representation. This audit puts it under orbit
  representation (§1); Phase B must confirm.
- Pressure versus current coupling. MEGA's coupling scheme, and MARS-K's FLR and collision content, are unconfirmed
  from source.
- How kinetic reduced MHD (KRMHD) terminology should be represented.

**External codes:** GTC-X has no identified primary reference.

## 9. Stress test against external codes

These rows check that the axes are not overfitted to VAFT's backends. All are provisional: from literature, not
source.

| code | bulk | fluid model | kinetic equation | population | coupling | domain | reference |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MARS-F | fluid | linear resistive MHD + flow, wall | none | — | — | whole volume + vacuum | Liu et al., Phys. Plasmas **7**, 3681 (2000) |
| MARS-K | hybrid | linear MHD | bounce/precession-resonant drift kinetics, perturbative or self-consistent | thermal + energetic | pressure ($p_\parallel$, $p_\perp$) | whole volume | Liu, Chu, Gimblett & Hastie, Phys. Plasmas **15**, 112503 (2008) |
| GENE (local / global) | kinetic | — | gyrokinetic | all | — | flux tube / radially global | Jenko et al., Phys. Plasmas **7**, 1904 (2000); Görler et al., J. Comput. Phys. **230**, 7053 (2011) |
| GENE-X | kinetic | — | gyrokinetic, full-f, EM, across the separatrix | all | — | whole volume incl. SOL | Michels et al., Comput. Phys. Commun. **264**, 107986 (2021) |
| GTC | kinetic (PIC) | fluid-kinetic hybrid electrons in EM | gyrokinetic ions/EP, drift-kinetic electrons | all | — | global | Lin et al., Science **281**, 1835 (1998) |
| ORB5 | kinetic (PIC) | — | gyrokinetic, δf with control variates | all | — | global | Jolliet et al., Comput. Phys. Commun. **177**, 409 (2007); Lanti et al., Comput. Phys. Commun. **251**, 107072 (2020) |
| M3D-K | hybrid | nonlinear resistive MHD | drift-/gyro-kinetic energetic particles (PIC) | energetic particles | pressure | global | Fu et al., Phys. Plasmas **13**, 052517 (2006) |
| MEGA | hybrid | nonlinear MHD | guiding-centre energetic particles | energetic particles | current (unresolved) | global | Todo & Sato, Phys. Plasmas **5**, 1321 (1998) |
| NIMROD kinetic | hybrid | extended (two-fluid) MHD | drift-kinetic δf PIC | energetic particles, kinetic ions | pressure | global | Kim, Sovinec & Parker, Comput. Phys. Commun. **164**, 448 (2004) |

**Fit.** Every row is expressible with the §1 axes, without a fidelity score and without inventing values: an absent
axis is "N/A".

**Strain.** The axes are strained in only two places, both already listed in §8: the coupling type of hybrid codes,
and the orbit-representation question.

## 10. Recommended minimal vocabulary for Phase B (#1727)

Proposed, not implemented.

**Required fields:**
- `scientific_operation`;
- `bulk_description`, one of `fluid` / `hybrid` / `kinetic` / `particle` / `reduced`.

**Conditional fields,** each `None` when not applicable:
- `fluid_model`: `ideal_mhd` / `resistive_mhd` / `extended_mhd`;
- `kinetic_equation`: `drift_kinetic` / `gyrokinetic` / `fokker_planck_test_particle`;
- `kinetic_population`;
- `kinetic_coupling`: `energy` / `pressure` / `current` / `sources` / `closure_on_given_displacement`.

**Independent axes,** carried over from CGYRO's record where they exist:
- `distribution_formulation`;
- `orbit_representation`: `full_orbit` / `guiding_centre` / `bounce_averaged`;
- `spatial_domain`;
- `topology_domain`;
- `regime`;
- `field_model`, coarse: ES/EM.

**Not in the common record:**
- input provenance: kinetic profiles measured, fitted or assumed;
- collision operators, saturation rules, geometry models, the integer encodings;
- confidence.

Input provenance stays with the inputs (§6). The rest stays solver-specific. Confidence describes our knowledge, not
the model, so it stays separate.

**Distinguish unknown from not applicable.** `None` means not applicable; an explicit `"unknown"` marks a legacy mode
whose physics is not documented.

**Backend gaps this audit found, for Phase C:**
- **GPEC (#1734).** VAFT exposes no DCON kinetic options. Its `mhd_linear` provenance does not record `kin_flag`,
  `con_flag` or the `pentrc.in` settings, so a kinetic run would be indistinguishable from an ideal one.
- **GACODE (#1735):**
  - `cgyro.formalism()` hard-codes values that `extra_parameters` can contradict (`AE_FLAG`, `EQUILIBRIUM_MODEL`,
    `GLOBAL_FLAG`), and it omits the collision model and the zeroed rotation;
  - NEO's `equilibrium_model=0` silently means s-α if `profile_model=1` is ever used.

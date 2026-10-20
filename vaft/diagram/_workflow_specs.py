"""The state-to-simulation spine (#1585): physics workflows from measured state to standardized IMAS results.

Each builder draws one :class:`~vaft.diagram._workflow.WorkflowSpec` from
:data:`WORKFLOWS`. Every node says how its quantity is obtained in VAFT
today -- read from the implementation, not idealised -- and where VAFT has no
IMAS mapping or does not yet implement the physically complete step, the spec
says so (``mapping_todo`` on the node, ``todos`` on the workflow) instead of
drawing an unsupported capability. ``api`` paths name the implementing
function and ``equation`` paths the ``vaft.formula`` definition;
``test/test_diagram_workflows.py`` resolves both.
"""

from __future__ import annotations

from typing import Dict

from ._render import Diagram
from ._workflow import Node as N, WorkflowSpec, _render_workflow

_STATE = "vaft.process.transport_state.resolve_transport_state"
_CP = "core_profiles.profiles_1d[:]"
_EQ = "equilibrium.time_slice[:]"
#: works cited by the inference figures (each already referenced in the vaft docstrings it names)
_LAO = "L. L. Lao et al., Nucl. Fusion 25 (1985) 1611"
_WESSON = "J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011), Sec. 4.25"
_ROMERO = "J. A. Romero and JET-EFDA contributors, Nucl. Fusion 50 (2010) 115002"
_SPITZER = "L. Spitzer and R. Harm, Phys. Rev. 89 (1953) 977"
_SAUTER = "O. Sauter, C. Angioni and Y. R. Lin-Liu, Phys. Plasmas 6 (1999) 2834; 9 (2002) 5140 (erratum)"
_REDL = "A. Redl et al., Phys. Plasmas 28 (2021) 022502"
_BRAGINSKII = "S. I. Braginskii, Rev. Plasma Phys. 1 (1965) 205"
_BELLI_2008 = "E. A. Belli and J. Candy, Plasma Phys. Control. Fusion 50 (2008) 095010"
_BELLI_2012 = "E. A. Belli and J. Candy, Plasma Phys. Control. Fusion 54 (2012) 015015"
_STAEBLER = "G. M. Staebler, J. E. Kinsey and R. E. Waltz, Phys. Plasmas 14 (2007) 055909"
_CANDY_2016 = "J. Candy, E. A. Belli and R. V. Bravenec, J. Comput. Phys. 324 (2016) 73"

_SPECS = (
    # ------------------------------------------------------------------ inference
    WorkflowSpec(
        key="plasma_parameter_inference", title="Plasma parameter inference: ion temperature and species",
        family="inference", status="implemented",
        summary="Measured electron profiles and the equilibrium resolved into an ion temperature, a species mix "
                "and the local quantities transport codes read. Each ion-temperature branch is kept apart, and "
                "the resolved state is held in memory (ResolvedTransportState), not written back to IMAS.",
        nodes=(
            N("te_ne", "Electron profiles", "measured", ids=f"{_CP}.electrons.{{temperature, density_thermal}}",
              symbols=r"T_e(\rho),\ n_e(\rho)"),
            N("eq", "Equilibrium", "reconstructed", ids=f"{_EQ}.profiles_1d.{{pressure, r_inboard, r_outboard}}",
              symbols=r"p_{\mathrm{eq}}(\psi),\ r(\psi)"),
            N("ti_cx", "Measured ion temperature (CX)", "measured", ids=f"{_CP}.ion[:].temperature",
              symbols=r"T_i^{\mathrm{CX}}(\rho)"),
            N("ti_pressure", "Pressure-partition ion temperature", "inferred",
              api="vaft.process.kinetic_closure.infer_ti_pressure_partition",
              relation=r"T_i = \frac{p_{\mathrm{eq}} - e\,n_e T_e}{e\,f_i\,n_e}"),
            N("ti_ratio", "Temperature ratio", "prior", api="vaft.machine_mapping.core_profiles.vest_core_profiles_policy",
              relation=r"T_i = r\,T_e,\ \ r\ \mathrm{from\ record\ or\ machine\ policy}"),
            N("select", "Ion-temperature branch selection", "derived", api=_STATE,
              symbols=r"\mathrm{measured} \succ \mathrm{pressure\ partition} \succ \mathrm{ratio}"),
            N("zeff", "Zeff and one impurity species", "prior", ids=f"{_CP}.zeff",
              symbols=r"Z_{\mathrm{eff}},\ Z_I\ (\mathrm{default\ C})"),
            N("species", "Species resolution", "derived", api="vaft.code.gacode.inputs.impurity_fractions",
              equation="vaft.formula.atomic.impurity_fraction_from_effective_charge",
              relation=r"\frac{n_H}{n_e} = \frac{Z_I - Z_{\mathrm{eff}}}{Z_I - 1}\qquad "
                       r"\mathrm{hydrogenic\ main\ ion\ +\ one\ impurity,\ quasineutral}"),
            N("gradients", "Normalized gradients", "derived", api="vaft.code.gacode.tglf.prepare_tglf_input",
              relation=r"\frac{a}{L_X} = -a\,\frac{d\ln X}{dr}\qquad r = \tfrac12(R_{\mathrm{out}} - R_{\mathrm{in}}),"
                       r"\ a = r_{\mathrm{edge}}"),
            N("collisionality", "Electron collision rate", "derived", api="vaft.code.gacode.tglf.prepare_tglf_input",
              relation=r"\hat\nu_{ee} = \nu_{ee}\,a/c_s\qquad n\,[10^{19}\mathrm{m^{-3}}],\ T\,[\mathrm{keV}],"
                       r"\ \ln\Lambda = 24 - \ln(\sqrt{10^{13}n}/10^{3}T)"),
            N("state", "Resolved local state", "derived", api=_STATE,
              symbols=r"T_e,\ T_i,\ n_e,\ n_H,\ n_I,\ a/L_{T,n},\ \hat\nu_{ee}",
              mapping_todo="in memory only; not written to core_profiles"),
        ),
        rows=(("te_ne", "eq"), ("ti_cx", "ti_pressure", "ti_ratio"), ("select",), ("species",),
              ("gradients", "collisionality"), ("state",)),
        edges=(("te_ne", "ti_pressure", ""), ("eq", "ti_pressure", ""), ("te_ne", "ti_ratio", ""),
               ("ti_cx", "select", ""), ("ti_pressure", "select", ""), ("ti_ratio", "select", ""),
               ("select", "species", ""), ("species", "gradients", ""), ("species", "collisionality", ""),
               ("gradients", "state", ""), ("collisionality", "state", "")),
        side=(("zeff", "species"),),
        todos=("The resolved state (T_i choice, n_H, n_I) is not written back to core_profiles.",
               "Normalized gradients and the GACODE collision rate are computed inside the TGLF input projection, "
               "not as vaft.formula definitions."),
    ),
    WorkflowSpec(
        key="romero_transformer_balance", title="Plasma resistance from Romero's transformer balance",
        family="inference", status="implemented",
        summary="The reconstructed boundary flux, plasma current and internal inductance close Romero's voltage "
                "balance; what the inductive part does not explain is the resistive voltage, from which the "
                "plasma resistance follows.",
        nodes=(
            N("eq", "Reconstructed equilibrium", "reconstructed",
              ids=f"{_EQ}.global_quantities.{{ip, psi_boundary, psi_axis, li_3}}",
              symbols=r"I_p(t),\ \psi_B(t),\ l_{i,3}(t)"),
            N("sign", "Romero convention: full flux", "convention",
              symbols=r"V = -\dot\psi,\ \psi\ \mathrm{in\ Wb}\qquad \mathrm{sign\ from}\ (\psi_a - \psi_B)I_p"),
            N("li", "Internal inductance and equilibrium flux", "derived",
              api="vaft.omas.process_wrapper.compute_romero_flux_balance_ods",
              relation=r"L_i = \tfrac12\mu_0 R_0\,l_{i,3}\qquad \psi_C = \psi_B + L_i I_p"),
            N("voltages", "Loop voltages", "derived", api="vaft.process.equilibrium.romero_flux_balance",
              relation=r"V_B = -\dot\psi_B,\quad V_C = -\dot\psi_C\qquad V_I = L_i\dot I_p + \tfrac12 I_p\dot L_i"),
            N("ini", "Non-inductive current", "prior", symbols=r"I_{\mathrm{ni}}\ (0\ \mathrm{stated\ for\ Ohmic})"),
            N("vr", "Resistive voltage", "derived",
              equation="vaft.formula.transformer.resistive_voltage_from_R_p_I_p_I_ni", symbols=r"V_R = V_B - V_I"),
            N("rp", "Plasma resistance", "inferred", api="vaft.omas.process_wrapper.compute_romero_flux_balance_ods",
              symbols=r"R_p = V_R/(I_p - I_{\mathrm{ni}})", mapping_todo="returned dict only; no IDS path"),
        ),
        rows=(("eq",), ("li",), ("voltages",), ("vr",), ("rp",)),
        edges=(("eq", "li", ""), ("li", "voltages", ""), ("voltages", "vr", ""), ("vr", "rp", "")),
        side=(("sign", "voltages"), ("ini", "rp")),
        todos=("psi_C is formed as psi_B + L_i I_p from li_3; the current-weighted integral "
               "(transformer.current_weighted_flux_from_psi_j_dS) exists but is not used here.",
               "No IMAS storage for V_B, V_I, V_R or R_p (e.g. summary.global_quantities.v_loop)."),
    ),
    WorkflowSpec(
        key="resistive_zeff_inference", title="Resistively equivalent effective charge",
        family="inference", status="implemented",
        summary="One scalar Zeff over a time window: the effective charge a chosen parallel-conductivity model "
                "needs to reproduce the plasma resistance observed through Romero's balance. It is a model-"
                "inferred, resistively equivalent value, not a measured or radially resolved Zeff(rho).",
        nodes=(
            N("cp", "Electron profiles", "measured", ids=f"{_CP}.electrons.{{temperature, density}}",
              symbols=r"T_e(\rho),\ n_e(\rho)"),
            N("eq", "Flux-surface geometry", "reconstructed",
              ids=f"{_EQ}.profiles_1d.{{gm5, volume, f, trapped_fraction}}",
              symbols=r"\langle J\!\cdot\!B\rangle,\ \langle B^2\rangle,\ V(\psi),\ f_t"),
            N("observed", "Observed resistive voltage", "inferred",
              api="vaft.process.resistive_zeff.observed_resistance",
              symbols=r"V_R^{\mathrm{obs}}(t),\ R_p^{\mathrm{obs}}(t)\ \ (\mathrm{Romero\ balance})"),
            N("model", "Conductivity model", "convention",
              symbols=r"\mathrm{spitzer\_nrl \mid sauter\_spitzer \mid sauter \mid redl}"),
            N("lnlambda", "Coulomb logarithm prescription", "convention", symbols=r"\ln\Lambda\ \mathrm{fixed\ or\ Sauter}"),
            N("sigma", "Parallel conductivity", "derived", api="vaft.process.resistive_zeff.parallel_conductivity",
              equation="vaft.formula.neoclassical.sauter_spitzer_conductivity", symbols=r"\sigma_\parallel(\rho; Z)"),
            N("rp_model", "Model resistance", "derived", api="vaft.process.resistive_zeff.model_resistance",
              relation=r"R_p^{\mathrm{model}}(Z) = \frac{1}{I_p^2}\int \frac{\langle E\cdot B\rangle"
                       r"\langle J\cdot B\rangle}{\langle B^2\rangle}\,dV"),
            N("bounds", "Fit bounds and weights", "prior", symbols=r"Z_{\min} \le Z \le Z_{\max}\ (\mathrm{caller\ set}),\ \ w_t\ \mathrm{uniform}"),
            N("fit", "Bounded scalar fit", "derived", api="vaft.process.resistive_zeff.infer_resistive_zeff",
              relation=r"\min_Z \sum_t w_t\,[V_R^{\mathrm{obs}} - R_p^{\mathrm{model}}(Z)(I_p - I_{\mathrm{ni}})]^2"),
            N("zeff", "Resistive Zeff (scalar)", "inferred",
              symbols=r"Z_{\mathrm{eff}}^{\mathrm{res}} \pm \sigma_Z\qquad \mathrm{residual\ rms},\ \mathrm{bound\ hit}",
              mapping_todo="CSV/JSON product; core_profiles.zeff untouched"),
        ),
        rows=(("cp", "eq"), ("sigma",), ("rp_model", "observed"), ("fit",), ("zeff",)),
        edges=(("cp", "sigma", ""), ("eq", "sigma", ""), ("sigma", "rp_model", ""), ("rp_model", "fit", ""),
               ("observed", "fit", ""), ("fit", "zeff", "")),
        side=(("model", "sigma"), ("lnlambda", "sigma"), ("bounds", "fit")),
        todos=("The observed resistance carries no propagated uncertainty (smoothing 'none' by default); the Zeff "
               "uncertainty is the fit's sqrt(J/(n-1)/sum w (dV/dZ)^2).",
               "No IMAS mapping for the scalar resistive Zeff."),
    ),
    # ------------------------------------------------------------------ reconstruction
    WorkflowSpec(
        key="magnetic_efit", title="Magnetic equilibrium reconstruction (EFIT)", family="reconstruction",
        status="implemented",
        summary="A free-boundary Grad-Shafranov inverse problem: magnetic measurements with their uncertainties, "
                "modelled vessel currents and the machine's Green-function tables constrain a low-order p' and "
                "FF' basis. EFIT fits; it does not forward-solve.",
        nodes=(
            N("pf", "PF coil currents", "measured", ids="pf_active.coil[:].current.data", symbols=r"I_{\mathrm{PF},j}(t)"),
            N("magnetics", "Magnetic diagnostics", "measured",
              ids="magnetics.{ip, b_field_pol_probe, flux_loop, diamagnetic_flux}",
              symbols=r"I_p,\ B_{p,k},\ \psi_{\mathrm{FL},k},\ \Phi_{\mathrm{dia}}"),
            N("circuit", "Vessel circuit model", "machine", symbols=r"R,\ L,\ M\ \mathrm{of\ the\ passive\ loops}"),
            N("eddy", "Vessel eddy currents (modelled)", "derived",
              api="vaft.process.electromagnetics.solve_eddy_currents", ids="pf_passive.loop[:].current",
              relation=r"L\,\dot{\mathbf I}_{\mathrm{pass}} + R\,\mathbf I_{\mathrm{pass}} = "
                       r"-M_{\mathrm{PF}}\dot{\mathbf I}_{\mathrm{PF}} - M_p\dot I_p"),
            N("uncertainty", "Weights, uncertainty floor, exclusions", "prior",
              symbols=r"\sigma_k = \max(\sigma_k^{\mathrm{meas}}/s_g,\ 0.02\,\mathrm{median}|y|)\qquad \mathrm{diagonal\ weights}"),
            N("constraints", "Measurement constraints", "code_input", api="vaft.code.efit.generate_constraints_ods",
              symbols=r"y_k \pm \sigma_k,\ \ w_k \in \{0, 1\}"),
            N("basis", "Profile basis", "convention",
              symbols=r"p'(\psi): 2,\ FF'(\psi): 1\ \mathrm{polynomial\ terms}\qquad \mathrm{zero\ at\ the\ edge}"),
            N("geometry", "Green tables, limiter", "machine", symbols=r"G(R,Z;R',Z'),\ \mathrm{limiter\ (free\ boundary)}"),
            N("kfile", "k-file", "code_input", api="vaft.code.efit.prepare_efit_inputs"),
            N("efit", "EFIT inverse solve", "solver", api="vaft.code.efit.run_efit",
              equation="vaft.formula.equilibrium.toroidal_current_density_from_p_prime_ff_prime"),
            N("gfile", "g-, a-, m-files", "native_result", api="vaft.code.efit.collect_efit_outputs",
              symbols=r"\psi(R,Z),\ p'(\psi),\ FF'(\psi),\ \chi^2"),
            N("equilibrium", "Equilibrium", "standardized", api="vaft.code.efit.gfile_to_omas",
              ids=f"{_EQ}.{{profiles_1d, profiles_2d, boundary, global_quantities}}",
              symbols=r"\psi,\ q,\ p,\ \beta_p,\ l_i"),
        ),
        rows=(("pf",), ("eddy", "magnetics"), ("constraints",), ("kfile",), ("efit",), ("gfile",), ("equilibrium",)),
        edges=(("pf", "eddy", ""), ("magnetics", "eddy", ""), ("eddy", "constraints", ""),
               ("magnetics", "constraints", ""), ("constraints", "kfile", ""), ("kfile", "efit", ""),
               ("efit", "gfile", ""), ("gfile", "equilibrium", "")),
        side=(("circuit", "eddy"), ("uncertainty", "constraints"), ("basis", "kfile"), ("geometry", "efit")),
        todos=("Vessel currents are modelled from a circuit driven by measured PF currents and I_p filaments, not "
               "fitted (IFITVS=0 by default); no flux loop constrains them.",
               "Measurement weighting is diagonal; no covariance enters EFIT.",
               "Native k/g/a/m files are recorded only by path and hash under equilibrium.code.parameters."),
    ),
    WorkflowSpec(
        key="kinetic_efit", title="Kinetically constrained equilibrium reconstruction", family="reconstruction",
        status="implemented",
        summary="The magnetic constraints plus a kinetic pressure profile: Thomson T_e, n_e and a measured or "
                "ratio-assumed T_i give pressure points with propagated uncertainty, placed in real space so "
                "EFIT maps them to psi on its own solution.",
        nodes=(
            N("ts", "Thomson scattering", "measured", ids="thomson_scattering.channel[:].{t_e, n_e, position.r}",
              symbols=r"T_e \pm \sigma_{T_e},\ n_e \pm \sigma_{n_e}\ \mathrm{at}\ R_k"),
            N("ti", "Ion temperature", "measured", ids="charge_exchange.channel[:].ion[0].t_i",
              symbols=r"T_i \pm \sigma_{T_i}\qquad \mathrm{or}\ T_i = rT_e,\ r \pm \sigma_r\ \mathrm{from\ machine\ policy}"),
            N("mapping", "Major radius to normalized flux", "derived", ids=f"{_EQ}.profiles_2d[0].psi",
              symbols=r"\psi_N(R, Z{=}0)\ \mathrm{for\ the}\ T_i\ \mathrm{fit}"),
            N("pressure", "Kinetic pressure points", "derived", api="vaft.code.efit.kinetic_pressure_points",
              relation=r"p_k = e\,n_e(T_e + T_i)\qquad "
                       r"\sigma_p = e\sqrt{((T_e{+}T_i)\sigma_{n_e})^2 + (n_e\sigma_{T_e})^2 + (n_e\sigma_{T_i})^2}"),
            N("floor", "Minimum pressure uncertainty", "prior", symbols=r"\sigma_p \ge 0.05\,p"),
            N("magnetic", "Magnetic constraints", "code_input", api="vaft.code.efit.generate_constraints_ods",
              symbols=r"y_k \pm \sigma_k"),
            N("kfile", "k-file with pressure block", "code_input", api="vaft.code.efit.inject_pressure_constraint",
              symbols=r"\mathrm{KPRFIT}=1:\ (R_k, 0, p_k, \sigma_{p,k})\qquad \mathrm{separatrix}\ p = 0 \pm 0.05\,p_{\max}"),
            N("efit", "EFIT inverse solve", "solver", api="vaft.code.efit.run_kinetic_efit"),
            N("equilibrium", "Kinetic equilibrium", "standardized", api="vaft.code.efit.run_kinetic_chain",
              ids=f"{_EQ}.{{profiles_1d.pressure, profiles_2d}}", symbols=r"\psi,\ p(\psi),\ q"),
        ),
        rows=(("ts", "ti"), (None, "mapping"), ("pressure",), ("magnetic", "kfile"), ("efit",), ("equilibrium",)),
        edges=(("ti", "mapping", ""), ("ts", "pressure", ""), ("mapping", "pressure", ""), ("pressure", "kfile", ""),
               ("magnetic", "kfile", ""), ("kfile", "efit", ""), ("efit", "equilibrium", "")),
        side=(("floor", "pressure"),),
        todos=("Pressure assumes one ion species with n_i = n_e: no impurity dilution or Zeff enters p_kin.",
               "The pressure points and the Ip scale run_kinetic_efit settles on are not stored in IMAS "
               "(manifest only); the work directory with the k-file is deleted.",
               "Single pass against the magnetic equilibrium: no kinetic <-> equilibrium iteration."),
    ),
    # ------------------------------------------------------------------ equilibrium
    WorkflowSpec(
        key="analytic_mhd_equilibrium", title="Analytic MHD equilibrium models", family="equilibrium",
        status="implemented",
        summary="Two uses of closed-form Grad-Shafranov solutions. Forward generation builds an equilibrium from "
                "shape parameters and a profile class; fitting projects a reconstructed equilibrium onto a "
                "Solov'ev basis and reports how well it is represented.",
        nodes=(
            N("shape", "Shape and topology", "prior",
              symbols=r"R_0,\ a,\ \kappa,\ \delta,\ \mathrm{limited\ /\ single\ /\ double\ null}"),
            N("eq_in", "Reconstructed equilibrium", "reconstructed", ids=f"{_EQ}.profiles_2d[0].psi",
              symbols=r"\psi(R,Z)\ \mathrm{inside\ the\ LCFS}"),
            N("solovev", "Solov'ev / Cerfon-Freidberg", "solver", api="vaft.process.solve_solovev_constraints",
              relation=r"p' = \mathrm{const},\ FF' = \mathrm{const}\qquad \psi = \psi_p + \sum_k c_k\psi_k"),
            N("gf", "Guazzotto-Freidberg", "solver", api="vaft.process.solve_guazzotto_freidberg",
              relation=r"p,\ F^2\ \mathrm{quadratic\ in}\ \psi\qquad p',\ FF'\ \mathrm{linear,\ eigenvalue}\ \alpha"),
            N("basis", "Solov'ev basis", "convention",
              symbols=r"\mathrm{classic}\ 5 \mid \mathrm{CF\ even}\ 7\qquad \mathrm{CF}\ 12\ \mathrm{terms}"),
            N("fit", "Least-squares projection", "derived", api="vaft.process.fit_solovev",
              relation=r"\min_{c_k,\,p',\,FF'} \|\psi - \psi_{\mathrm{Sol}}\|_2\ \ (\psi_N \le \psi_{N,\max})"),
            N("psi", "Analytic flux", "native_result", symbols=r"\psi(R,Z),\ c_k"),
            N("residual", "Fit fidelity", "native_result",
              symbols=r"\psi_{\mathrm{rms}},\ \mathrm{boundary\ rms},\ \mathrm{topology\ match}"),
            N("equilibrium", "Equilibrium", "standardized", api="vaft.data.eqdsk.to_omas",
              ids=f"{_EQ}.{{profiles_1d, profiles_2d}}", symbols=r"\psi,\ q,\ p"),
        ),
        rows=(("shape", "eq_in"), ("solovev", "gf", "fit"), ("psi", "residual"), ("equilibrium",)),
        edges=(("shape", "solovev", ""), ("shape", "gf", ""), ("eq_in", "fit", ""), ("solovev", "psi", ""),
               ("gf", "psi", ""), ("fit", "residual", ""), ("psi", "equilibrium", "via GEQDSK")),
        side=(("basis", "fit"),),
        todos=("No direct ODS writer: forward results reach IMAS through EquilibriumData -> GEQDSK -> to_omas.",
               "Fit results (SolovevFit) are not written to IMAS."),
    ),
    WorkflowSpec(
        key="chease_coupling", title="Fixed-boundary equilibrium refinement (CHEASE)", family="equilibrium",
        status="implemented",
        summary="A reconstructed boundary and profiles re-solved at fixed boundary; the COCOS transform into and "
                "out of CHEASE is explicit.",
        nodes=(
            N("eq_in", "Reconstructed equilibrium", "reconstructed", ids=f"{_EQ}.{{boundary.outline, profiles_1d}}",
              symbols=r"R_b,\ Z_b,\ p'(\psi),\ FF'(\psi)"),
            N("cocos_in", "COCOS transform", "convention", symbols=r"\mathrm{COCOS}\ 11 \to 2 \to 11"),
            N("inputs", "EXPEQ, namelist", "code_input", api="vaft.code.chease.prepare_chease_inputs"),
            N("chease", "CHEASE fixed-boundary solve", "solver", api="vaft.code.chease.run_chease",
              equation="vaft.formula.equilibrium.toroidal_current_density_from_p_prime_ff_prime"),
            N("native", "EQDSK, output files", "native_result", api="vaft.code.chease.collect_chease_outputs",
              symbols=r"\psi(R,Z),\ q(\psi)"),
            N("equilibrium", "Refined equilibrium", "standardized", api="vaft.code.chease.refine_equilibrium",
              ids=f"{_EQ}.{{profiles_1d, profiles_2d}}", symbols=r"\psi,\ q,\ \langle\cdot\rangle_\psi"),
        ),
        rows=(("eq_in",), ("inputs",), ("chease",), ("native",), ("equilibrium",)),
        edges=(("eq_in", "inputs", ""), ("inputs", "chease", ""), ("chease", "native", ""),
               ("native", "equilibrium", "")),
        side=(("cocos_in", "inputs"),),
    ),
    WorkflowSpec(
        key="tokamaker_coupling", title="Free-boundary equilibrium (TokaMaker)", family="equilibrium",
        status="implemented",
        summary="Machine geometry, measured coil currents and power-law profile shapes solved at free boundary "
                "on a finite-element mesh; the plasma boundary is part of the solution.",
        nodes=(
            N("coils", "Coil currents", "measured", ids="pf_active.coil[:].current.data", symbols=r"I_{\mathrm{PF},j}"),
            N("targets", "Plasma current and vacuum field", "measured", ids="{equilibrium | magnetics}.ip; tf",
              symbols=r"I_p,\ F_0 = R_0 B_0"),
            N("geometry", "Wall, coils, vessel", "machine", ids="{wall, pf_active.coil, pf_passive.loop}"),
            N("inputs", "TokaMaker inputs", "code_input", api="vaft.code.tokamaker.prepare_tokamaker_inputs"),
            N("mesh", "Finite-element mesh", "code_input", api="vaft.code.tokamaker.build_tokamaker_mesh"),
            N("profiles", "Profile shape", "prior",
              symbols=r"p',\ FF' \propto (1-\hat\psi^{\alpha_a})^{\alpha_b},\ \ p_{\mathrm{ax}},\ I_{FF'}/I_{p'}"),
            N("solve", "TokaMaker free-boundary solve", "solver", api="vaft.code.tokamaker.run_tokamaker",
              equation="vaft.formula.equilibrium.toroidal_current_density_from_p_prime_ff_prime"),
            N("native", "Flux, boundary, statistics", "native_result",
              api="vaft.code.tokamaker.collect_tokamaker_outputs", symbols=r"\psi(R,Z),\ R_b,\ Z_b"),
            N("equilibrium", "Equilibrium", "standardized", ids=f"{_EQ}.{{profiles_1d, profiles_2d, boundary}}",
              symbols=r"\psi,\ q,\ \beta_p"),
        ),
        rows=(("coils", "targets"), ("inputs",), ("mesh",), ("solve",), ("native",), ("equilibrium",)),
        edges=(("coils", "inputs", ""), ("targets", "inputs", ""), ("inputs", "mesh", ""), ("mesh", "solve", ""),
               ("solve", "native", ""), ("native", "equilibrium", "")),
        side=(("geometry", "inputs"), ("profiles", "solve")),
    ),
    # ------------------------------------------------------------------ stability and response
    WorkflowSpec(
        key="dcon_rdcon_stability", title="Ideal and resistive MHD stability (DCON / RDCON)", family="stability",
        status="implemented",
        summary="An equilibrium tested for ideal stability (DCON energy principle) and for tearing (RDCON "
                "matching at the rational surfaces); the solver-native energies and Delta-prime come before any "
                "verdict.",
        nodes=(
            N("equilibrium", "Equilibrium", "reconstructed", ids=_EQ, symbols=r"\psi,\ q(\psi),\ p(\psi)"),
            N("modes", "Toroidal modes and flux range", "convention",
              symbols=r"n = 1, 2,\ \ \psi_N \in [0.01, 0.994]\qquad \Delta m = 8\ (\mathrm{RDCON}\ 16)"),
            N("case", "DCON / RDCON inputs", "code_input", api="vaft.code.gpec.prepare_gpec_suite_case"),
            N("wall", "Vacuum boundary", "convention", symbols=r"\mathrm{free\ boundary,\ no\ wall\ (far\ wall)}"),
            N("layer", "Resistive inner layers", "derived", api="vaft.code.gpec._solvers.write_rmatch_resistive_layers",
              symbols=r"\eta(T_e, Z_{\mathrm{eff}}, \ln\Lambda),\ \rho_m\ \mathrm{at}\ q = m/n"),
            N("dcon", "DCON ideal energy principle", "solver", api="vaft.code.gpec.run_gpec_suite_case"),
            N("rdcon", "RDCON resistive matching", "solver", api="vaft.code.gpec.run_gpec_suite_case"),
            N("dw", "Energy and eigenfunctions", "native_result", api="vaft.code.gpec.read_dcon_output",
              symbols=r"\delta W_n = \delta W_p + \delta W_v,\ \ \xi_{m,n}(\psi)"),
            N("dprime", "Delta-prime at rational surfaces", "native_result",
              api="vaft.code.gpec.read_pest3_matching_output", symbols=r"\Delta'_{m/n}\ \mathrm{at}\ q = m/n"),
            N("mhd_linear", "Ideal mode", "standardized",
              ids="mhd_linear.time_slice[:].toroidal_mode[:].{energy_perturbed, plasma}",
              symbols=r"n,\ \delta W_n,\ \xi_\perp"),
            N("ntms", "Tearing stability index", "standardized", ids="ntms.time_slice[:].mode[:].deltaw[0]",
              symbols=r"\Delta'_{m/n}"),
        ),
        rows=(("equilibrium",), ("case", "layer"), ("dcon", "rdcon"), ("dw", "dprime"), ("mhd_linear", "ntms")),
        edges=(("equilibrium", "case", ""), ("case", "dcon", ""), ("case", "rdcon", ""), ("layer", "rdcon", ""),
               ("dcon", "dw", ""),
               ("rdcon", "dprime", ""), ("dw", "mhd_linear", ""), ("dprime", "ntms", "")),
        side=(("modes", "case"), ("wall", "dcon")),
        todos=("energy_perturbed carries the DCON-normalised total delta W, not joules; the plasma/vacuum split "
               "and the full Delta-prime matrices stay native.",
               "Wall position, qlow and delta_mlow/high come from the packaged templates, not from VAFT options."),
    ),
    WorkflowSpec(
        key="gpec_plasma_response", title="Ideal plasma response to 3-D fields (GPEC)", family="response",
        status="implemented",
        summary="An applied non-axisymmetric coil field and the plasma's ideal response to it: coil geometry is "
                "machine data, the excitation is prescribed, and the total, plasma and resonant fields are kept "
                "apart.",
        nodes=(
            N("equilibrium", "Equilibrium", "reconstructed", ids=_EQ, symbols=r"\psi,\ q,\ p,\ F"),
            N("coil_geometry", "3-D coil geometry", "machine", symbols=r"\mathrm{VEST\ UP/MID/LOW\ coils,\ coil.in}"),
            N("excitation", "Coil current and phasing", "prior",
              api="vaft.machine_mapping.coils_non_axisymmetric_geometry.CoilExcitation.from_mode",
              relation=r"I_k = A\cos(n\phi_k + \phi_0)"),
            N("case", "GPEC inputs", "code_input", api="vaft.code.gpec.prepare_gpec_suite_case"),
            N("response_model", "Response model", "convention",
              symbols=r"\mathrm{ideal,\ static}\ (\omega = 0)\qquad \mathrm{no\ rotation,\ no\ kinetic\ terms}"),
            N("gpec", "GPEC ideal response", "solver", api="vaft.code.gpec.run_gpec_suite_case"),
            N("fields", "Perturbed fields", "native_result", api="vaft.code.gpec.read_gpec_netcdf",
              symbols=r"\delta\mathbf B_{\mathrm{total}} = \delta\mathbf B_{\mathrm{vac}} + \delta\mathbf B_{\mathrm{plasma}}"),
            N("resonant", "Resonant response", "native_result", api="vaft.code.gpec.read_gpec_netcdf",
              symbols=r"\Phi_{\mathrm{res}},\ w_{\mathrm{isl}}\qquad K_{\mathrm{Chirikov}},\ \delta W"),
            N("mhd_linear", "Perturbed normal field", "standardized",
              ids="mhd_linear.time_slice[:].toroidal_mode[:].plasma.b_field_perturbed",
              symbols=r"\delta\mathbf B\cdot\nabla\psi",
              mapping_todo="dB(R,Z,phi), its vacuum part and resonant quantities"),
        ),
        rows=(("equilibrium",), ("case",), ("gpec",), ("fields", "resonant"), ("mhd_linear",)),
        edges=(("equilibrium", "case", ""), ("case", "gpec", ""), ("gpec", "fields", ""), ("gpec", "resonant", ""),
               ("fields", "mhd_linear", "")),
        side=(("coil_geometry", "case"), ("excitation", "case"), ("response_model", "gpec")),
        todos=("GPEC returns total and plasma fields; the vacuum part (total minus plasma) is not computed by any "
               "mapping.",
               "No frequency, rotation or kinetic response inputs: the response is ideal and static.",
               "Energies go to code.parameters; resonant quantities and cylindrical fields are not mapped."),
    ),
    WorkflowSpec(
        key="flare_field_line_topology", title="Magnetic field-line topology (FLARE)", family="response",
        status="implemented",
        summary="Field lines traced through the axisymmetric background plus the 3-D perturbation: Poincare maps, "
                "connection lengths and strike-point footprints are the native products; a heat load needs an "
                "explicit model, here a relative proxy.",
        nodes=(
            N("background", "Axisymmetric background field", "reconstructed", ids=_EQ, symbols=r"\mathbf B_0(R,Z)"),
            N("perturbation", "3-D perturbation from GPEC", "native_result",
              api="vaft.code.flare.write_helicity_flipped_field", symbols=r"\delta\mathbf B(R,Z,\phi)"),
            N("scales", "Field-direction convention", "convention", api="vaft.code.flare.flare_equilibrium_scales",
              symbols=r"\mathrm{scale}_{I_p},\ \mathrm{scale}_{B_t}\ \mathrm{from\ COCOS}"),
            N("targets", "Wall and target geometry", "machine", symbols=r"\mathrm{in\ the\ FLARE\ control\ file}"),
            N("tracing", "Tracing configuration", "convention", symbols=r"\mathrm{in\ the\ FLARE\ control\ file}"),
            N("flare", "FLARE field-line tracing", "solver", api="vaft.code.flare.run_flare"),
            N("poincare", "Poincare map", "native_result", api="vaft.data.flare_products.read_flare_product",
              symbols=r"(R, Z)\ \mathrm{crossings\ at\ fixed}\ \phi", mapping_todo="no IMAS path"),
            N("lc", "Connection length", "native_result", ids="plasma_initiation.b_field_lines", symbols=r"L_c(R,Z)"),
            N("footprint", "Strike-point footprint", "native_result",
              symbols=r"\psi_{\min},\ \alpha\ \mathrm{at\ the\ target}"),
            N("proxy", "Relative heat-load proxy", "derived",
              api="vaft.process.field_line_topology.footprint_heat_load_proxy",
              relation=r"q_{\mathrm{rel}} = w_\psi\,|\sin\alpha|\,w_L\ \ (\mathrm{unitless})"),
            N("divertors", "Incident power fractions", "standardized",
              ids="divertors[:].target[:].power_incident_fraction", symbols=r"f_{\mathrm{inc}}"),
        ),
        rows=(("background", "perturbation"), ("flare",), ("poincare", "lc", "footprint"), ("proxy",), ("divertors",)),
        edges=(("background", "flare", ""), ("perturbation", "flare", ""), ("flare", "poincare", ""),
               ("flare", "lc", ""), ("flare", "footprint", ""), ("lc", "proxy", ""), ("footprint", "proxy", ""),
               ("proxy", "divertors", "")),
        side=(("scales", "flare"), ("targets", "flare"), ("tracing", "flare")),
        todos=("Wall/target geometry and tracing settings live in the user's FLARE control file, not in VAFT.",
               "No island width or stochasticity metric is computed from FLARE output (GPEC's Chirikov K is the "
               "only one in VAFT).",
               "The heat load is a relative proxy; no parallel-transport or q_perp model is implemented."),
    ),
    # ------------------------------------------------------------------ transport
    WorkflowSpec(
        key="neo_neoclassical", title="Neoclassical transport: closed-form fits and drift-kinetic solution",
        family="transport", status="implemented",
        summary="The same resolved local state through the Sauter and Redl fits and the drift-kinetic solver NEO, "
                "so the two can be compared at one state.",
        nodes=(
            N("state", "Resolved local state", "derived", api=_STATE, symbols=r"T_s,\ n_s,\ q,\ \epsilon,\ f_t"),
            N("fits", "Sauter / Redl fits", "derived", api="vaft.formula.neoclassical.redl_bootstrap_current",
              equation="vaft.formula.neoclassical.sauter_bootstrap_current"),
            N("case", "NEO input", "code_input", api="vaft.code.gacode.neo.prepare_neo_case"),
            N("neo", "NEO drift-kinetic solve", "solver", api="vaft.code.gacode.neo.run_neo"),
            N("native", "Fluxes, bootstrap current", "native_result", api="vaft.code.gacode.neo.collect_neo_outputs",
              symbols=r"\Gamma_s,\ Q_s,\ \langle j_{\mathrm{bs}}B\rangle"),
            N("transport", "Neoclassical transport", "standardized", ids="core_transport.model[:].profiles_1d[:]",
              symbols=r"\Gamma_s,\ Q_s,\ j_{\mathrm{bs}}"),
        ),
        rows=(("state",), ("fits", "case"), (None, "neo"), (None, "native"), ("transport",)),
        edges=(("state", "fits", ""), ("state", "case", ""), ("case", "neo", ""), ("neo", "native", ""),
               ("native", "transport", ""), ("fits", "transport", "compared at the same state")),
    ),
    WorkflowSpec(
        key="tglf_cgyro_local_transport", title="Local turbulent transport: quasilinear and gyrokinetic",
        family="transport", status="implemented",
        summary="One local gyrokinetic state projected into TGLF (quasilinear) and CGYRO (local delta-f, linear or "
                "nonlinear); their outputs are different physical objects and are kept apart.",
        nodes=(
            N("eq", "Equilibrium", "reconstructed", ids=f"{_EQ}.profiles_1d", symbols=r"q,\ r,\ R,\ \kappa,\ \delta"),
            N("cp", "Kinetic profiles", "measured", ids=_CP, symbols=r"T_s,\ n_s,\ Z_{\mathrm{eff}}"),
            N("state", "Resolved local gyrokinetic state", "derived", api=_STATE,
              equation="vaft.formula.equilibrium.shear_from_r_q",
              symbols=r"n_s, T_s, Z_s, m_s,\ a/L_{n_s}, a/L_{T_s},\ T_i/T_e\qquad "
                      r"Z_{\mathrm{eff}},\ \hat\nu_{ee},\ \beta_e,\ q,\ \hat s,\ \kappa, s_\kappa,\ \delta, s_\delta"),
            N("rotation", "Rotation and ExB shear", "prior", symbols=r"\gamma_E = 0,\ M = 0\ (\mathrm{not\ derived})"),
            N("tglf_in", "input.tglf", "code_input", api="vaft.code.gacode.tglf.prepare_tglf_input"),
            N("cgyro_in", "input.cgyro", "code_input", api="vaft.code.gacode.cgyro.prepare_cgyro_input"),
            N("tglf_cfg", "TGLF model choices", "convention",
              symbols=r"\mathrm{SAT\_RULE},\ \delta B_\perp, \delta B_\parallel\qquad \mathrm{XNU\_MODEL},\ k_y\ \mathrm{grid}"),
            N("tglf", "TGLF quasilinear model", "solver", api="vaft.code.gacode.tglf.run_tglf"),
            N("cgyro", "CGYRO local delta-f", "solver", api="vaft.code.gacode.cgyro.run_cgyro"),
            N("cgyro_cfg", "CGYRO model choices", "convention",
              symbols=r"\mathrm{linear \mid nonlinear},\ N_{\mathrm{field}}\qquad \mathrm{Sugama\ collisions},\ \mathrm{Miller\ geometry}"),
            N("tglf_out", "Quasilinear fluxes and spectrum", "native_result",
              api="vaft.code.gacode.tglf.collect_tglf_outputs", symbols=r"Q_s,\ \Gamma_s;\ \gamma(k_y),\ \omega(k_y)"),
            N("cgyro_lin", "Linear eigenmodes", "native_result", api="vaft.code.gacode.cgyro.collect_cgyro_outputs",
              symbols=r"\gamma,\ \omega,\ \phi(\theta)"),
            N("cgyro_nl", "Saturated fluxes (nonlinear)", "native_result",
              api="vaft.code.gacode.cgyro.collect_cgyro_outputs",
              symbols=r"\langle Q_s\rangle_t,\ \langle\Gamma_s\rangle_t"),
            N("ct_tglf", "Turbulent transport (TGLF)", "standardized",
              api="vaft.machine_mapping.turbulence.core_transport_from_tglf", ids="core_transport.model[:]",
              symbols=r"Q_s,\ \Gamma_s\ \mathrm{(quasilinear)}",
              mapping_todo="no TGLF to gyrokinetics_local writer"),
            N("gk_local", "Gyrokinetic run (CGYRO)", "standardized",
              api="vaft.machine_mapping.gyrokinetics.gyrokinetics_local_from_cgyro",
              ids="gyrokinetics_local.{linear.wavevector[:].eigenmode[:], non_linear.fluxes_1d}",
              symbols=r"\gamma,\ \omega;\ \langle Q_s\rangle_t,\ \langle\Gamma_s\rangle_t"),
        ),
        rows=(("eq", "cp"), ("state",), ("tglf_in", "cgyro_in"), ("tglf", "cgyro"),
              ("tglf_out", "cgyro_lin", "cgyro_nl"), ("ct_tglf", "gk_local")),
        edges=(("eq", "state", ""), ("cp", "state", ""), ("state", "tglf_in", ""), ("state", "cgyro_in", ""),
               ("tglf_in", "tglf", ""), ("cgyro_in", "cgyro", ""), ("tglf", "tglf_out", ""), ("cgyro", "cgyro_lin", ""),
               ("cgyro", "cgyro_nl", ""), ("tglf_out", "ct_tglf", ""), ("cgyro_lin", "gk_local", ""),
               ("cgyro_nl", "gk_local", "")),
        side=(("rotation", "state"), ("tglf_cfg", "tglf"), ("cgyro_cfg", "cgyro")),
        todos=("Rotation and ExB shear are zero in both projections (TGLF VEXB_SHEAR, CGYRO GAMMA_E, MACH): their "
               "derivation from data is not implemented (#553).",
               "Squareness zeta enters as 0 (VEST equilibria carry no squareness); Z_EFF is not written to "
               "input.cgyro by design: CGYRO recomputes it from the species list (Z_EFF_METHOD=2).",
               "No implemented local-gyrokinetic validity criterion (rho*): compare_with_oracle checks only the "
               "input translation against CGYRO's own projection.",
               "No TGLF to gyrokinetics_local mapping."),
    ),
    # ------------------------------------------------------------------ plasma parameter inference (#1601)
    WorkflowSpec(
        key="parameter_inference_overview", title="Plasma parameter inference: completing the kinetic profiles",
        family="parameter_inference", status="implemented",
        summary="Where inference sits: diagnostics and the equilibrium give reconstructed but incomplete "
                "kinetic profiles; explicit assumptions and the closures VAFT implements complete them, and every "
                "completed quantity keeps its origin on the way to simulation.",
        nodes=(
            N("diagnostics", "Thomson scattering", "measured", ids="thomson_scattering.channel[:]",
              symbols=r"T_e,\ n_e\ \mathrm{at}\ (R, Z)"),
            N("cx", "Charge exchange (optional)", "measured", ids="charge_exchange.channel[:].ion[0].t_i",
              symbols=r"T_i^{\mathrm{CX}}\ \mathrm{at}\ (R, Z)"),
            N("eq", "Magnetic equilibrium", "reconstructed", ids=f"{_EQ}.profiles_1d", references=(_LAO,),
              symbols=r"p_{\mathrm{eq}}(\psi),\ q,\ \langle j\cdot B\rangle,\ \mathrm{geometry}"),
            N("reconstructed", "Reconstructed kinetic profiles", "reconstructed", ids=f"{_CP}.electrons",
              symbols=r"T_e(\rho),\ n_e(\rho)\ (\mathrm{fitted}),\ [T_i(\rho)]\qquad p_{\mathrm{eq}}(\rho),\ q(\rho)"),
            N("fitting", "Fitting assumptions", "convention",
              api="vaft.process.profile.profile_fitting_thomson_scattering",
              symbols=r"\mathrm{polynomial\ |\ core\text{-}poly/edge\text{-}exp\ |\ GP}\qquad "
                      r"\mathrm{order}\ N,\ x = \rho_{\mathrm{tor},N}\ \mathrm{or}\ \psi_N\qquad "
                      r"T, n > 0,\ \sigma\text{-}\mathrm{weighted}"),
            N("assumptions", "Closure assumptions", "prior",
              symbols=r"\mathrm{common}\ T_i,\ \mathrm{impurity\ species},\ Z_{\mathrm{eff}}\qquad "
                      r"\sigma_\parallel:\ \mathrm{Spitzer\ or\ neoclassical},\ \ln\Lambda"),
            N("thermo", "Thermodynamic closure", "inferred",
              api="vaft.process.kinetic_closure.infer_ti_pressure_partition",
              relation=r"p_{\mathrm{eq}} = e\,n_e T_e + e\sum_s n_s T_i\qquad T_i = \frac{p_{\mathrm{eq}} - e n_e T_e}"
                       r"{e f n_e}"),
            N("composition", "Impurity closure", "inferred",
              api="vaft.process.impurity.resolve_impurity_composition", references=(_WESSON,),
              equation="vaft.formula.atomic.impurity_fraction_from_effective_charge",
              relation=r"n_e = \sum_s Z_s n_s\qquad Z_{\mathrm{eff}} = \frac{\sum_s Z_s^2 n_s}{n_e}"),
            N("resistive", "Resistive closure", "inferred", api="vaft.process.resistive_zeff.infer_resistive_zeff",
              references=(_ROMERO, _SPITZER, _SAUTER, _REDL),
              symbols=r"V_R^{\mathrm{obs}} \to Z_{\mathrm{eff}}^{\mathrm{res}}\ (\mathrm{scalar})\qquad "
                      r"\sigma_\parallel:\ \mathrm{Spitzer\ |\ neoclassical\ (Sauter,\ Redl)}"),
            N("state", "Completed kinetic profiles", "derived", ids=f"{_CP}.{{electrons, ion[:], zeff}}",
              symbols=r"n_e, T_e, T_i, n_s, Z_{\mathrm{eff}}\qquad \mathrm{each\ with\ origin=\ldots;\ method=\ldots}"),
            N("simulation", "Simulation-ready", "code_input",
              symbols=r"\mathrm{every\ input\ quantity\ with\ its\ origin}"),
        ),
        rows=(("diagnostics", "cx", "eq"), ("reconstructed",), ("thermo", "composition", "resistive"), ("state",),
              ("simulation",)),
        edges=(("diagnostics", "reconstructed", ""), ("cx", "reconstructed", "optional"),
               ("eq", "reconstructed", "mapping"),
               ("reconstructed", "thermo", ""), ("reconstructed", "composition", ""),
               ("reconstructed", "resistive", ""), ("thermo", "state", ""), ("composition", "state", ""),
               ("resistive", "state", ""), ("state", "simulation", "")),
        side=(("assumptions", "reconstructed"), ("fitting", "reconstructed")),
        todos=("Atomic-model-constrained Z_eff(rho) (transient charge states projected through the resistive "
               "closure) is on develop as resolve_radial_composition(normalization='resistive_closure') "
               "(#1565, #1566, PR #1659); this diagram still draws the scalar resistive closure only.",
               "Rotation, E_r and the ExB shear are not inferred: downstream codes receive gamma_E = 0 as an "
               "explicit assumption (#553).",
               "Provenance is recorded per quantity where it exists (core_profiles origin=...; method=... records "
               "for composition, the T_i result's origin/method fields); there is no single runtime provenance "
               "object, by design (#1601 is documentation-first)."),
    ),
    WorkflowSpec(
        key="parameter_inference_dependency_graph", title="Inferred quantities depend on inferred quantities",
        family="parameter_inference", status="implemented",
        summary="Provenance propagates: a composition assumption fixes the ion densities, which set the "
                "pressure-partition T_i, whose gradients reach the gyrokinetic input beside an assumed zero ExB "
                "shear. Every link below the measurements carries the assumptions above it.",
        nodes=(
            N("te_ne", "Electron profiles", "measured", ids=f"{_CP}.electrons.{{temperature, density_thermal}}",
              symbols=r"T_e(\rho),\ n_e(\rho)"),
            N("eq", "Equilibrium pressure", "reconstructed", ids=f"{_EQ}.profiles_1d.pressure", references=(_LAO,),
              symbols=r"p_{\mathrm{eq}}(\psi)\ \ (\mathrm{independent\ lineage})"),
            N("composition", "Impurity source", "prior",
              symbols=r"\mathrm{measured} \succ \mathrm{explicit} \succ \mathrm{derived} \succ \mathrm{assumed}"),
            N("ions", "Ion densities", "derived", api="vaft.process.impurity.resolve_impurity_composition",
              references=(_WESSON,),
              equation="vaft.formula.impurity.main_ion_density_from_species"),
            N("ti", "Pressure-partition ion temperature", "inferred",
              api="vaft.process.kinetic_closure.infer_ti_pressure_partition",
              relation=r"T_i = \frac{p_{\mathrm{eq}}}{e f n_e} - \frac{T_e}{f}\qquad \sigma_{T_i}\ \mathrm{propagated};"
                       r"\ p_i \le \sigma_{p_i}\ \mathrm{flagged,\ never\ filled}"),
            N("pi", "Ion pressure", "derived", equation="vaft.formula.equilibrium.ion_pressure",
              symbols=r"p_i(\rho),\ dp_i/dr"),
            N("gradients", "Normalized gradients", "derived", api="vaft.code.gacode.tglf.prepare_tglf_input",
              symbols=r"a/L_{T_i},\ a/L_{n_i},\ T_i/T_e"),
            N("gamma", "ExB shear", "prior", symbols=r"\gamma_E = 0\ \ (\mathrm{assumed})"),
            N("gk", "Gyrokinetic input", "code_input", api="vaft.code.gacode.cgyro.prepare_cgyro_input",
              symbols=r"\mathrm{local\ state\ with\ origins}"),
        ),
        rows=(("te_ne", "eq"), ("ions",), ("ti",), ("pi",), ("gradients",), ("gk",)),
        edges=(("te_ne", "ions", ""), ("eq", "ti", ""), ("ions", "ti", "with electron profiles"),
               ("ti", "pi", ""), ("pi", "gradients", ""), ("gradients", "gk", "")),
        side=(("composition", "ions"), ("gamma", "gk")),
        todos=("E_r, rotation and gamma_E are not inferred from data (#553): gamma_E = 0 enters the gyrokinetic "
               "input as an assumption, and force balance is future scope.",),
    ),
    # ------------------------------------------------------------------ reconstruction inputs (#1901)
    WorkflowSpec(
        key="eddy_current_model", title="Vessel eddy currents: a circuit driven by the coils and the plasma",
        family="reconstruction", status="implemented",
        summary="The passive structure is a set of resistive loops driven by the changing PF and plasma currents. "
                "VAFT solves that circuit forward from rest -- it does not fit the vessel currents to the "
                "magnetics -- and checks the result against the pre-plasma flux loops and probes.",
        nodes=(
            N("pf", "PF coil currents", "measured", ids="pf_active.coil[:].current.data",
              api="vaft.machine_mapping.pf_active.vfit_pf_active_dynamic", symbols=r"I_{\mathrm{PF},j}(t)"),
            N("ip", "Plasma current", "measured", ids="magnetics.ip[0].data", symbols=r"I_p(t)"),
            N("filaments", "Plasma filaments", "prior",
              symbols=r"(R_k, Z_k),\ f_k\ \mathrm{set\ by\ the\ caller}\qquad I_k = f_k I_p"),
            N("machine", "Loops and mutual inductances", "machine",
              api="vaft.omas.process_wrapper.ensure_em_coupling",
              ids="{pf_passive.loop[:].resistance, em_coupling.mutual_passive_passive}",
              symbols=r"R = \mathrm{diag}(R_k),\ M_{\mathrm{pp}},\ M_{\mathrm{pa}}"),
            N("matrices", "Circuit matrices", "derived",
              api="vaft.process.electromagnetics.compute_impedance_matrices",
              symbols=r"M,\ R,\ L = [M_{\mathrm{pa}},\ M_{\mathrm{plasma}}]\qquad M_{\mathrm{plasma}}\ "
                      r"\mathrm{from\ Green's\ functions}"),
            N("solve", "Forward circuit solve", "solver", api="vaft.process.electromagnetics.solve_eddy_currents",
              relation=r"M\,\dot{\mathbf I} + R\,\mathbf I = -L\,\frac{d}{dt}[\mathbf I_{\mathrm{PF}},\ I_p]"
                       r"\qquad e^{-M^{-1}R\,\Delta t},\ \Delta t = 50\ \mu\mathrm{s},\ \mathbf I(0) = 0"),
            N("currents", "Vessel loop currents", "standardized", api="vaft.omas.vest_upstream.build_eddy_ods",
              ids="pf_passive.loop[:].current", symbols=r"I_{\mathrm{loop},k}(t)"),
            N("vacuum", "Pre-plasma magnetics", "measured", ids="magnetics.{flux_loop, b_field_pol_probe}",
              symbols=r"\psi_{\mathrm{FL}},\ B_p\ \mathrm{before\ breakdown}"),
            N("check", "Eddy improvement", "derived", api="vaft.omas.vacuum_magnetics.eddy_improvement",
              relation=r"1 - \frac{\mathrm{RMS}(m - m_{\mathrm{coil}} - m_{\mathrm{eddy}})}"
                       r"{\mathrm{RMS}(m - m_{\mathrm{coil}})}"),
        ),
        rows=(("matrices",), ("pf", "solve", "ip"), ("currents", "vacuum"), ("check",)),
        edges=(("pf", "solve", ""), ("ip", "solve", ""), ("matrices", "solve", ""), ("solve", "currents", ""),
               ("currents", "check", ""), ("vacuum", "check", "")),
        side=(("machine", "matrices"), ("filaments", "matrices")),
        todos=("Vessel currents are a forward model, not fitted to the magnetics; EFIT then treats them as known "
               "(IFITVS=0).",
               "The plasma filaments are placed by the caller, not taken from the equilibrium.",
               "The circuit is constant in time and starts from zero current; the 50 us substep is coarser than "
               "the 40 us diagnostics grid (#425).",
               "A missing PF channel (PF5 48644-48760, PF6 48534-48537) leaves NaN currents and the stage refuses "
               "to run."),
    ),
    WorkflowSpec(
        key="kinetic_profile_fitting", title="Kinetic profile fitting: point measurements to profiles",
        family="reconstruction", status="implemented",
        summary="Thomson and charge-exchange points are placed on flux surfaces by the equilibrium and fitted to "
                "smooth profiles with declared fitting choices; each fitted profile keeps its origin record.",
        nodes=(
            N("ts", "Thomson scattering", "measured", ids="thomson_scattering.channel[:].{t_e, n_e, position}",
              symbols=r"T_e,\ n_e \pm \sigma\ \mathrm{at}\ (R, Z)"),
            N("eq", "Magnetic equilibrium", "reconstructed", ids="equilibrium.time_slice[:].{profiles_2d, profiles_1d.q}",
              symbols=r"\psi(R, Z),\ q(\psi)"),
            N("cx", "Charge exchange (optional)", "measured",
              ids="charge_exchange.channel[:].ion[:].{t_i, velocity_tor}",
              symbols=r"T_i,\ v_\phi\ \mathrm{at}\ (R, Z)"),
            N("mapping", "Flux-surface mapping", "derived",
              api="vaft.process.profile.equilibrium_mapping_thomson_scattering",
              symbols=r"(R,Z) \to \psi_N \to \rho_{\mathrm{pol}} = \sqrt{\psi_N}\qquad \rho_{\mathrm{tor},N}\ "
                      r"\mathrm{from}\ \Phi = \int q\,d\psi;\ \mathrm{outside\ the\ LCFS:\ dropped}"),
            N("coordinate", "Fitting coordinate", "convention",
              symbols=r"x = \rho_{\mathrm{tor},N}\ (\mathrm{default})\ |\ \psi_N\ |\ \rho_{\mathrm{pol},N}"),
            N("fit_e", "Electron profile fit", "derived",
              api="vaft.process.profile.profile_fitting_thomson_scattering",
              symbols=r"T_e(x),\ n_e(x)"),
            N("fit_i", "Ion profile fit", "derived", api="vaft.process.profile.profile_fitting_charge_exchange",
              symbols=r"T_i(x),\ v_\phi(x)"),
            N("model", "Fit model and order", "convention", api="vaft.formula.utils.fit_profile",
              symbols=r"\mathrm{polynomial\ |\ core\text{-}poly/edge\text{-}exp\ |\ tanh\ |\ GP},\ N = 3"
                      r"\qquad \sigma\text{-}\mathrm{weighted};\ T > 0:\ \mathrm{order\ stepped\ down}"),
            N("ratio", "Ion temperature without CX", "prior", api="vaft.process.profile.fit_ti_te_ratio",
              symbols=r"T_i = r\,T_e,\ r\ \mathrm{fitted\ or\ assumed}"),
            N("profiles", "Kinetic profiles", "standardized", api="vaft.process.profile.core_profiles",
              ids="core_profiles.profiles_1d[:].{electrons, ion[0], *_fit}",
              symbols=r"T_e, n_e, T_i, v_\phi\qquad \mathrm{each\ with\ origin=\ldots;\ method=\ldots}"),
        ),
        rows=(("ts", "eq", "cx"), ("mapping",), ("fit_e", "fit_i"), ("profiles",)),
        edges=(("ts", "mapping", ""), ("eq", "mapping", ""), ("cx", "mapping", ""), ("mapping", "fit_e", ""),
               ("mapping", "fit_i", ""), ("fit_e", "profiles", ""), ("fit_i", "profiles", "")),
        side=(("coordinate", "mapping"), ("model", "fit_e"), ("ratio", "fit_i")),
        todos=("Thomson and charge exchange are fitted separately and combined only in core_profiles; there is "
               "no joint multi-diagnostic fit.",
               "The kinetic-profile evidence framework (#1204) is not yet referenced by the fit records."),
    ),
    # ------------------------------------------------------------------ transport hierarchy (#1901)
    WorkflowSpec(
        key="transport_hierarchy", title="Transport hierarchy: classical, neoclassical, turbulent",
        family="transport", status="implemented",
        summary="One resolved local state and five independently computed routes; none feeds another. "
                "'Always available' means the route needs no external code, not that its physics applies "
                "everywhere. Each route with a writer lands in its own core_transport.model entry, found by its "
                "identifier.index and code name -- the array position k is assigned at write time.",
        nodes=(
            N("state", "Resolved local state", "derived", api="vaft.process.transport_state.resolve_transport_state",
              symbols=r"T_s,\ n_s,\ a/L_{T,n},\ q,\ \hat s,\ \hat\nu"),
            N("classical", "Classical: Braginskii", "derived", api="vaft.process.transport_state.classical_heat_fluxes",
              references=(_BRAGINSKII,),
              symbols=r"\mathrm{perpendicular\ conductive}\ q_{\perp,e},\ q_{\perp,i}\qquad "
                      r"\mathrm{strongly\ magnetized}:\ \kappa_\perp \propto nT/(m\Omega^2\tau)\qquad "
                      r"\mathrm{software:\ always\ available}"),
            N("neo_fit", "Neoclassical closures: Sauter / Redl", "derived",
              api="vaft.omas.neoclassical.compute_bootstrap_current", references=(_SAUTER, _REDL),
              symbols=r"j_{\mathrm{bs}},\ \sigma_\parallel\ \mathrm{only:\ no}\ \Gamma_s,\ Q_s\qquad "
                      r"\mathrm{software:\ always\ available}"),
            N("neo_code", "Neoclassical: NEO", "solver", api="vaft.code.gacode.neo.runner.run_neo_pair",
              references=(_BELLI_2008, _BELLI_2012),
              symbols=r"\Gamma_s,\ Q_s,\ j_{\mathrm{bs}}\ (\mathrm{transport\ run})\qquad "
                      r"\sigma_\parallel\ (\mathrm{separate}\ E_\parallel\ \mathrm{run})\qquad "
                      r"\mathbf{optional}\ \mathrm{external\ code}"),
            N("turb_ql", "Turbulent: TGLF", "solver", api="vaft.code.gacode.tglf.run_tglf", references=(_STAEBLER,),
              symbols=r"Q_s,\ \Gamma_s\ (\mathrm{quasilinear})\qquad \mathbf{optional}\ \mathrm{external\ code}"),
            N("turb_gk", "Turbulent: CGYRO", "solver", api="vaft.code.gacode.cgyro.run_cgyro", references=(_CANDY_2016,),
              symbols=r"\mathrm{nonlinear}\ Q_s(t),\ \Gamma_s(t)\qquad \mathbf{optional}\ \mathrm{external\ code}"),
            N("window", "Saturation window", "prior", symbols=r"(t_0, t_1)\ [a/c_s],\ \mathrm{chosen\ by\ the\ caller}"),
            N("ct_classical", "Classical transport", "standardized",
              api="vaft.machine_mapping.classical.core_transport_from_classical",
              ids="core_transport.model[k] (identifier.index -1)", symbols=r"\chi_e,\ \chi_i\ (\mathrm{private\ index})"),
            N("ct_neo", "Neoclassical transport", "standardized",
              api="vaft.machine_mapping.neoclassical.core_transport_from_neo",
              ids="core_transport.model[k] (identifier.index 5)", symbols=r"\Gamma_s,\ Q_s\ (\mathrm{NEO})"),
            N("ct_tglf", "Anomalous transport (TGLF)", "standardized",
              api="vaft.machine_mapping.turbulence.core_transport_from_tglf",
              ids="core_transport.model[k] (identifier.index 6, code TGLF)", symbols=r"\Gamma_s,\ Q_s,\ \Pi_\phi"),
            N("ct_cgyro", "Anomalous transport (CGYRO)", "standardized",
              api="vaft.machine_mapping.gyrokinetics.core_transport_from_cgyro",
              ids="core_transport.model[k] (identifier.index 6, code CGYRO)",
              symbols=r"\langle\Gamma_s\rangle_t,\ \langle Q_s\rangle_t\ \mathrm{over}\ (t_0, t_1)"),
        ),
        rows=(("state",), ("classical", "neo_fit", "neo_code", "turb_ql", "turb_gk"),
              ("ct_classical", None, "ct_neo", "ct_tglf", "ct_cgyro")),
        edges=(("state", "classical", ""), ("state", "neo_fit", ""), ("state", "neo_code", ""),
               ("state", "turb_ql", ""), ("state", "turb_gk", ""), ("classical", "ct_classical", ""),
               ("neo_code", "ct_neo", ""), ("turb_ql", "ct_tglf", ""), ("turb_gk", "ct_cgyro", "")),
        side=(("window", "ct_cgyro"),),
        todos=("The closed-form Sauter/Redl results have no core_transport writer (and carry no particle or heat "
               "flux to write); they are returned, not stored.",
               "Classical transport uses a private identifier.index (-1): the data dictionary has no classical "
               "entry. The Braginskii forms assume a strongly magnetized plasma, and ln Lambda is flagged invalid "
               "below 10 eV.",
               "core_transport.model[k]: k is the array position the writer finds or appends (matching "
               "identifier.index and code.name); it is not the identifier.index.",
               "A nonlinear CGYRO run writes no fluxes without a saturation window: choosing it is the caller's "
               "judgement.",
               "The TGLF-NN surrogate (vaft.code.gacode.tglf.surrogate) refuses VEST inputs as out of domain and "
               "has no IMAS writer."),
    ),
    WorkflowSpec(
        key="neoclassical_model_comparison", title="Sauter / Redl against NEO at one state",
        family="transport", status="implemented",
        summary="One shared source state, two independently prepared input paths, and an end-to-end comparison "
                "against NEO as the numerical reference -- not as ground truth. The bootstrap current is compared "
                "in one definition, <J.B>/B0 with the vacuum B0 and its sign, on one grid; a plasma mismatch is "
                "refused, while differences in grid construction, trapped fraction and collisionality between the "
                "two input paths remain part of the gap.",
        nodes=(
            N("state", "Shared source state", "reconstructed",
              ids="{core_profiles.profiles_1d[:], equilibrium.time_slice[:]}",
              symbols=r"T_s,\ n_s,\ Z_{\mathrm{eff}},\ \mathrm{impurity},\ \psi,\ q\ (\mathrm{one\ time\ slice})"),
            N("analytic_in", "Analytic-model inputs (VAFT)", "derived",
              api="vaft.omas.neoclassical.compute_bootstrap_current",
              symbols=r"\rho_{\mathrm{tor},N}\ \mathrm{from\ the\ equilibrium};\ f_t\ \mathrm{from\ the\ equilibrium}"
                      r"\qquad \nu^*\ (\mathrm{Sauter\ definition});\ Z_{\mathrm{eff}},\ n_i/n_e"),
            N("neo_in", "NEO inputs (GACODE conversion)", "code_input", api="vaft.code.gacode.neo.inputs.prepare_neo_case",
              symbols=r"\mathrm{input.gacode\ grid\ and\ geometry};\ f_t,\ \nu\ \mathrm{as\ NEO\ forms\ them}"
                      r"\qquad \mathrm{species\ list}\ (Z_{\mathrm{eff}}\ \mathrm{from\ it})"),
            N("sauter", "Sauter closure", "derived", api="vaft.validation.neoclassical.bootstrap_models",
              equation="vaft.formula.neoclassical.sauter_bootstrap_current", references=(_SAUTER,)),
            N("redl", "Redl closure", "derived", api="vaft.validation.neoclassical.bootstrap_models",
              references=(_REDL,),
              symbols=r"j_{\mathrm{bs}}\ \mathrm{with\ Redl's\ refitted}\ L_{31}, L_{32}, \alpha"),
            N("neo_tr", "NEO transport solve", "solver", api="vaft.code.gacode.neo.runner.run_neo_case",
              references=(_BELLI_2008, _BELLI_2012),
              symbols=r"j_{\mathrm{bs}},\ \Gamma_s,\ Q_s\qquad \mathbf{optional}\ \mathrm{external\ code}"),
            N("neo_cond", "NEO conductivity solve", "solver", api="vaft.code.gacode.neo.runner.run_neo_conductivity_case",
              symbols=r"\sigma_\parallel = j_\parallel / E_\parallel\qquad E_\parallel = 1,\ \mathrm{all\ gradients\ off}"),
            N("mapping", "NEO into the IMAS definition", "standardized",
              api="vaft.machine_mapping.neoclassical.core_profiles_from_neo",
              ids="core_profiles.profiles_1d[:].{j_bootstrap, conductivity_parallel}",
              symbols=r"\langle J\cdot B\rangle/B_{\mathrm{unit}} \to \langle J\cdot B\rangle/B_0\qquad "
                      r"\mathrm{signed}\ B_{\mathrm{unit}}\ (\mathrm{GACODE\ COCOS\ 2})"),
            N("basis", "Comparison basis", "convention",
              symbols=r"j_{\mathrm{bs}} = \langle J\cdot B\rangle/B_0,\ \mathrm{vacuum}\ B_0,\ \mathrm{signed}"
                      r"\qquad \mathrm{same}\ \rho_{\mathrm{tor},N}\ \mathrm{grid};\ Z_{\mathrm{eff}},\ n_i/n_e\ "
                      r"\mathrm{must\ match}"),
            N("compare", "Model agreement", "derived", api="vaft.validation.neoclassical.model_comparison",
              symbols=r"\mathrm{reference = NEO\ (numerical,\ not\ ground\ truth)}\qquad "
                      r"\mathrm{integrated,\ RMS/peak,\ peak\ differences}"),
        ),
        rows=(("state",), ("analytic_in", None, "neo_in", None), ("sauter", "redl", "neo_tr", "neo_cond"),
              ("compare", None, "mapping", None)),
        edges=(("state", "analytic_in", "VAFT path"), ("state", "neo_in", "GACODE path"), ("analytic_in", "sauter", ""),
               ("analytic_in", "redl", ""), ("neo_in", "neo_tr", ""), ("neo_in", "neo_cond", ""),
               ("neo_tr", "mapping", ""), ("neo_cond", "mapping", ""), ("sauter", "compare", ""),
               ("redl", "compare", ""), ("mapping", "compare", "reference")),
        side=(("basis", "compare"),),
        todos=("End-to-end, not like-for-like: the two input paths differ in grid construction, trapped fraction "
               "and collisionality definition, so part of any gap is the input path. The like-for-like check "
               "against NEO's own Sauter/Redl columns (agreement to 1e-7) is a test, not a product.",
               "A stored NEO profile on a different rho_tor_norm grid is refused, not interpolated; a run whose "
               "Z_eff or n_i/n_e differs is refused when the solver's values are passed.",
               "Without NEO there is no reference and no comparison; the closed-form results are not written to "
               "IMAS."),
    ),
    # ------------------------------------------------------------------ experimental processing (#1901 part B)
    WorkflowSpec(
        key="machine_model_history", title="Machine model and shot history: one shot number selects the machine",
        family="experimental_processing", status="implemented",
        summary="The shot number is the only key. It selects the machine era (PF and wall geometry), each "
                "diagnostic's revision (wiring, gains, baseline windows) and the shot-log record; the static "
                "machine description is built once per era and the processing era is reported per diagnostic.",
        nodes=(
            N("shot", "Shot number", "measured", ids="dataset_description.data_entry.pulse", symbols=r"s"),
            N("era", "Machine era", "machine", api="vaft.omas.vest_upstream.machine_era_for_shot",
              symbols=r"\mathrm{PF\ 1906\ /\ 2507},\ \mathrm{wall\ 1512\ /\ 2409}"),
            N("revisions", "Diagnostic revisions", "machine", api="vaft.machine_mapping.utils.resolve_vest_diagnostic",
              symbols=r"\mathrm{from\_shot} \le s \le \mathrm{to\_shot}\qquad \mathrm{wiring,\ gain,\ baseline}"),
            N("schedule", "Shot log", "measured", api="vaft.machine_mapping.pulse_schedule.mapping.pulse_schedule",
              ids="pulse_schedule", symbols=r"\mathrm{set\ points,\ gas,\ timing}"),
            N("conventions", "IMAS sign and angle conventions", "convention",
              api="vaft.machine_mapping.registry.port_phi",
              symbols=r"\phi = (-30^\circ\,h) \bmod 360^\circ\qquad \mathrm{discharge\ sign\ contract}"),
            N("static", "Static machine description", "standardized", api="vaft.omas.vest_upstream.build_static_ods",
              ids="{wall | pf_active | pf_passive | em_coupling | magnetics | tf}",
              symbols=r"\mathrm{geometry},\ M_{ij},\ \mathrm{sensor\ positions}"),
            N("provenance", "Processing era per diagnostic", "derived",
              api="vaft.machine_mapping.provenance.vest_processing_provenance",
              symbols=r"\mathrm{calibration\ lineage}",
              mapping_todo="returned as a dict; not written to an IDS"),
        ),
        rows=(("shot",), ("era", "revisions", "schedule"), ("static", "provenance")),
        edges=(("shot", "era", ""), ("shot", "revisions", ""), ("shot", "schedule", ""), ("era", "static", ""),
               ("revisions", "static", "wiring"), ("revisions", "provenance", "")),
        side=(("conventions", "static"),),
        todos=("The VEST COCOS index is not resolved yet (vaft.machine_mapping.conventions."
               "equilibrium_orientation_is_resolved); the discharge sign contract stands in for it.",
               "The shot-log mapper writes pulse_schedule when called (vaft shotlog CLI) but "
               "build_diagnostics_ods does not call it.",
               "Wall and limiter history is the era table only; there is no per-component calibration lineage.",
               "There is no single shot-to-machine object: era, revisions and provenance are separate lookups."),
    ),
    WorkflowSpec(
        key="signal_processing", title="Signal processing: raw digitizer records to a diagnostic IDS",
        family="experimental_processing", status="implemented",
        summary="A representative chain, in the order the magnetic probes use it: anti-alias low-pass on the "
                "native grid, calibration, time integration of the inductive voltage, a linear baseline over the "
                "era's leading samples, then interpolation onto the analysis grid. Other diagnostics reorder or "
                "skip steps.",
        nodes=(
            N("raw", "Raw digitizer record", "measured", api="vaft.database.raw.load",
              symbols=r"V(t)\ \mathrm{per\ DAQ\ field}"),
            N("fields", "DAQ field codes", "machine", api="vaft.machine_mapping.raw_daq_sources.raw_sources",
              symbols=r"(\mathrm{diagnostic},\ s) \rightarrow \mathrm{field\ codes}"),
            N("lowpass", "Anti-alias low-pass", "derived", api="vaft.process.signal_processing.fir_filter",
              symbols=r"\mathrm{FIR\ on\ the\ native\ fast\ grid}"),
            N("calibrate", "Calibration", "derived", api="vaft.machine_mapping.utils.calibrate_vest_signal",
              relation=r"x = c\,V\ \mathrm{(linear)}\qquad \mathrm{or\ an\ exponential\ gauge\ law}"),
            N("revision", "Calibration revision", "machine", api="vaft.machine_mapping.utils.resolve_vest_diagnostic",
              symbols=r"c,\ \mathrm{sign},\ \mathrm{baseline\ window}"),
            N("integrate", "Time integration", "derived",
              api="vaft.process.magnetics.vest_b_field_pol_probe_legacy",
              relation=r"B(t) = -\int_0^t \dot B\,dt'\qquad \dot B\ \mathrm{the\ calibrated\ record}"),
            N("baseline", "Baseline subtraction", "derived", api="vaft.process.signal_processing.subtract_baseline",
              symbols=r"\mathrm{linear\ fit\ over\ the\ era's\ leading\ samples}"),
            N("resample", "Resampling", "derived", api="vaft.process.signal_processing.resample_to_time",
              symbols=r"t_k = t_{\mathrm{start}} + k\,\Delta t"),
            N("grid", "Analysis time grid", "convention",
              api="vaft.machine_mapping.utils.resolve_diagnostics_time_policies",
              symbols=r"t_{\mathrm{start}},\ t_{\mathrm{end}},\ \Delta t"),
            N("ids", "Diagnostic IDS", "standardized", api="vaft.omas.vest_upstream.build_diagnostics_ods",
              ids="{magnetics | pf_active | tf | spectrometer_uv | barometry}",
              symbols=r"\psi_{\mathrm{FL}},\ B_p,\ I_p,\ I_{\mathrm{PF}}"),
        ),
        rows=(("raw",), ("lowpass",), ("calibrate",), ("integrate",), ("baseline",), ("resample",), ("ids",)),
        edges=(("raw", "lowpass", ""), ("lowpass", "calibrate", ""), ("calibrate", "integrate", ""),
               ("integrate", "baseline", ""), ("baseline", "resample", ""), ("resample", "ids", "")),
        side=(("fields", "raw"), ("revision", "calibrate"), ("grid", "resample")),
        todos=("The order is per diagnostic: PF coils subtract a baseline mean, filter, scale and align; H-alpha "
               "shifts in time and resamples with an anti-aliased resampler; flux loops integrate per radian.",
               "There is no drift-correction stage separate from the baseline fit, and no cross-DAQ "
               "synchronization module: each mapper applies its own time shift and interpolation.",
               "Integration is inline in each mapper; vaft.process has no shared integration helper.",
               "Thomson scattering and the interferometer read post-processed files, so this chain does not apply "
               "to them."),
    ),
    WorkflowSpec(
        key="quality_validation", title="Quality and validation: measurements against references to evidence",
        family="experimental_processing", status="implemented",
        summary="Three separate kinds of evidence: per-sample source validity (written to the IDS validity "
                "nodes), agreement with the vacuum model before the plasma (a report), and equilibrium fit "
                "quality turned into a slice class. Model disagreement never sets a channel's validity.",
        nodes=(
            N("meas", "Processed magnetics", "measured", ids="magnetics.{flux_loop, b_field_pol_probe, ip}",
              symbols=r"m_i(t)"),
            N("model", "Synthetic vacuum magnetics", "derived",
              api="vaft.omas.vacuum_magnetics.synthetic_vacuum_magnetics",
              symbols=r"\hat m_i = m_{\mathrm{coil}} + m_{\mathrm{eddy}}"),
            N("eq", "Magnetic equilibrium", "reconstructed", ids="equilibrium.time_slice[:]",
              symbols=r"\psi(R, Z),\ \chi^2"),
            N("source", "Source validity checks", "derived", api="vaft.validation.magnetics.validate_magnetics_signals",
              symbols=r"\mathrm{per\ sample,\ per\ channel}"),
            N("residual", "Residual metrics", "derived", api="vaft.omas.vacuum_magnetics.vacuum_magnetics_metrics",
              symbols=r"m_i - \hat m_i:\ \mathrm{RMS},\ \mathrm{correlation}"),
            N("efitq", "Fit and convergence quality", "derived", api="vaft.omas.efit_quality.efit_quality_metrics",
              symbols=r"\chi^2_\nu,\ \mathrm{residual\ bias},\ \mathrm{convergence}"),
            N("validity", "Channel validity", "standardized", api="vaft.validation.imas.write_validity",
              ids="magnetics.{flux_loop[:].flux | b_field_pol_probe[:].field | ip[:]}.validity_timed",
              symbols=r"0\ \mathrm{valid},\ 1\ \mathrm{flagged},\ -2\ \mathrm{rejected}"),
            N("gate", "Vacuum-model quality gate", "derived", api="vaft.omas.vacuum_magnetics.quality_gate",
              symbols=r"\mathrm{pass\ /\ fail\ per\ sensor\ group}", mapping_todo="a report; not written to an IDS"),
            N("classes", "Slice quality class", "derived",
              api="vaft.validation.equilibrium_quality.equilibrium_quality_table",
              symbols=r"\mathrm{good\ /\ admissible\ /\ unreconstructible}",
              mapping_todo="a table; not written to an IDS"),
            N("criteria", "Acceptance criteria", "convention", symbols=r"\mathrm{versioned\ thresholds}"),
        ),
        rows=(("meas", "model", "eq"), ("source", "residual", "efitq"), ("validity", "gate", "classes")),
        edges=(("meas", "source", ""), ("meas", "residual", ""), ("model", "residual", "reference"),
               ("eq", "efitq", ""), ("source", "validity", ""), ("residual", "gate", ""),
               ("efitq", "classes", "")),
        side=(("criteria", "classes"),),
        todos=("Only source validity reaches IMAS; model-agreement and fit-quality metrics are returned as "
               "dicts and tables in the stage manifests.",
               "Flux-loop agreement with the vacuum model is report-only by default; only source validity can "
               "reject a loop.",
               "The vacuum check covers the pre-plasma window only; there is no plasma-phase synthetic "
               "magnetics check outside the EFIT residuals."),
    ),
    WorkflowSpec(
        key="fault_anomaly_detection", title="Fault and anomaly evidence: diagnostic-specific checks",
        family="experimental_processing", status="implemented",
        summary="There is no single anomaly model: each diagnostic has its own checks. Magnetics separate hard "
                "faults from soft flags and honour the recorded fault list; the Rogowski coil has a square-wave "
                "test; clipped PF, diamagnetic and TF records are repaired over the clipped interval; H-alpha "
                "channels are screened before they time the plasma.",
        nodes=(
            N("signals", "Processed channels", "measured", ids="{magnetics | pf_active | tf | spectrometer_uv}",
              symbols=r"x_i(t)"),
            N("known", "Recorded faults", "machine", api="vaft.machine_mapping.magnetics.known_magnetics_faults",
              symbols=r"(\mathrm{kind},\ i) \rightarrow \mathrm{reason}"),
            N("mag", "Magnetics channel checks", "derived", api="vaft.validation.magnetics.validate_magnetics_signals",
              symbols=r"\mathrm{hard:\ non\text{-}finite,\ flatline,\ magnitude,\ outlier}\qquad"
                      r"\mathrm{soft:\ spike,\ offset\ jump,\ drift,\ noise}"),
            N("rog", "Rogowski square-wave test", "derived", api="vaft.validation.plasma_current.assess_plasma_rogowski",
              symbols=r"I_p(t)\ \mathrm{shape}"),
            N("clip", "Clipping detection", "derived", api="vaft.process.signal_processing.detect_clipped_samples",
              symbols=r"\mathrm{PF,\ diamagnetic,\ TF}"),
            N("halpha", "H-alpha usability", "derived", api="vaft.omas.plasma_timing.halpha_usability",
              symbols=r"\mathrm{finite,\ not\ railed,\ live\ baseline}"),
            N("validity", "Channel validity", "standardized", api="vaft.validation.magnetics.project_validity",
              ids="magnetics.{flux_loop[:].flux | b_field_pol_probe[:].field | ip[:]}.validity_timed",
              symbols=r"0,\ 1,\ -2"),
            N("repair", "Repaired interval", "derived", api="vaft.process.signal_processing.repair_clipped_interval",
              symbols=r"x_i(t)\ \mathrm{across\ the\ clipped\ samples}"),
            N("usable", "Light sources for timing", "derived", api="vaft.omas.plasma_timing.plasma_timing",
              symbols=r"\mathrm{usable\ H_\alpha\ channels}"),
        ),
        rows=(("signals",), ("mag", "rog", "clip", "halpha"), ("validity", None, "repair", "usable")),
        edges=(("signals", "mag", ""), ("signals", "rog", ""), ("signals", "clip", ""), ("signals", "halpha", ""),
               ("mag", "validity", ""), ("rog", "validity", ""), ("clip", "repair", ""), ("halpha", "usable", "")),
        side=(("known", "mag"),),
        todos=("No general anomaly model runs in the pipeline: vaft.process.ml provides the framework "
               "(PCA autoencoder, k-NN, one-class SVM) but no trained model ships with the package.",
               "The spectrometer and barometry fault registry (known_diagnostic_faults) has no pipeline consumer.",
               "Missing raw data marks the component unavailable or partial; it is not a per-channel fault flag."),
    ),
    WorkflowSpec(
        key="shot_classification", title="Shot classification: rules over derived features",
        family="experimental_processing", status="implemented",
        summary="A shot class is a rule applied to derived features -- the plasma window, the gauge-pressure "
                "response, the peak current -- under stated thresholds, and it records which rule decided. It is "
                "not a measurement. A separate pre-run check classifies DAQ completeness, and equilibrium slices "
                "get a phase label from their own rule.",
        nodes=(
            N("fields", "Raw field inventory", "measured", symbols=r"\mathrm{field\ codes\ in\ the\ DAQ\ database}"),
            N("pressure", "Gauge pressure", "measured", ids="barometry.gauge[0].pressure.data", symbols=r"p(t)"),
            N("signals", "Plasma current and H-alpha", "measured", ids="magnetics.ip[0].data; spectrometer_uv",
              symbols=r"I_p(t),\ H_\alpha(t)"),
            N("eq", "Equilibrium slices", "reconstructed", ids="equilibrium.time_slice[:].global_quantities.ip",
              symbols=r"I_p(t_k)"),
            N("daq", "DAQ completeness", "derived", api="vaft.database.worker.classify.RawFieldClassifier",
              symbols=r"\mathrm{daq\_missing\ /\ raw\_incomplete}"),
            N("response", "Pressure response", "derived", api="vaft.omas.shot_class.pressure_response",
              symbols=r"\Delta p\ \mathrm{during\ the\ shot}"),
            N("timing", "Plasma window", "derived", api="vaft.omas.plasma_timing.plasma_timing",
              symbols=r"t_{\mathrm{start}},\ t_{\mathrm{end}}"),
            N("regime", "Slice phase", "derived", api="vaft.validation.equilibrium_regime.classify_equilibrium_regimes",
              symbols=r"\mathrm{ramp\text{-}up\ /\ flat\ /\ ramp\text{-}down}:\ \mathrm{level}\ 0.9\,|I_p|_{\max}"),
            N("rule", "Shot class", "derived", api="vaft.omas.shot_class.shot_class",
              symbols=r"\mathrm{Plasma\ /\ BD\ failure\ /\ Vacuum},\ \mathrm{decided\ by}",
              mapping_todo="kept in the database worker record, not in an IDS"),
            N("thresholds", "Default thresholds", "convention",
              symbols=r"p_{\mathrm{th}} = 10^{-2},\ I_{p,\min} = 2\ \mathrm{kA}"),
        ),
        rows=(("fields", "pressure", "signals", "eq"), ("daq", "response", "timing", "regime"),
              (None, "rule", None, None)),
        edges=(("fields", "daq", ""), ("pressure", "response", ""), ("signals", "timing", ""), ("eq", "regime", ""),
               ("response", "rule", ""), ("timing", "rule", "")),
        side=(("thresholds", "rule"),),
        todos=("The four-way physics class of #57 is not implemented; the class is Plasma, BD failure or Vacuum.",
               "There are no disruption labels and no label-provenance store; vaft.process.ml datasets take "
               "targets from the caller.",
               "Events passed to the slice classifier (VDE, IRE, disruption) are caller annotations, not "
               "detections."),
    ),
    WorkflowSpec(
        key="event_detection", title="Event detection: onsets and windows from a robust baseline",
        family="experimental_processing", status="implemented",
        summary="Each waveform gets a robust baseline and spread from its quiet reference, a threshold that is "
                "the larger of a fraction of the peak and a number of robust sigmas, and an active window that "
                "must stay above it. The plasma window is then confirmed across signals: H-alpha light against "
                "the plasma current. Coil onsets anchor the loop-voltage time base.",
        nodes=(
            N("light", "H-alpha and plasma current", "measured", ids="spectrometer_uv; magnetics.ip[0].data",
              symbols=r"H_\alpha(t),\ I_p(t)"),
            N("coils", "Coil currents", "measured", ids="pf_active.coil[:].current.data",
              symbols=r"I_{\mathrm{PF},j}(t)"),
            N("baseline", "Robust baseline", "derived", api="vaft.process.onset.robust_baseline",
              symbols=r"b,\ \sigma\ \mathrm{from\ the\ median\ and\ MAD}"),
            N("rules", "Detection rules", "convention", symbols=r"f,\ k,\ t_{\mathrm{hold}}\ \mathrm{per\ signal}"),
            N("threshold", "Activity threshold", "derived", api="vaft.process.onset.excess_threshold",
              relation=r"x_{\mathrm{th}} = b + \max(f\,x_{\mathrm{peak}},\ k\,\sigma)"),
            N("window", "Active window", "derived", api="vaft.process.onset.active_window",
              symbols=r"x > x_{\mathrm{th}}\ \mathrm{for\ at\ least}\ t_{\mathrm{hold}}"),
            N("confirm", "Light against current", "derived", api="vaft.omas.plasma_timing.plasma_timing",
              symbols=r"\mathrm{consistent,\ I_p\ before\ H_\alpha,\ offset\ disagreement}"),
            N("plasma", "Plasma window", "derived", symbols=r"t_{\mathrm{start}},\ t_{\mathrm{end}}",
              mapping_todo="a dataclass; not written to an IDS"),
            N("onsets", "Coil onsets", "derived", api="vaft.omas.discharge_timing.coil_onsets",
              symbols=r"t_{\mathrm{on},j},\ t_{\mathrm{OH}}"),
            N("vloop", "Loop-voltage event", "derived", api="vaft.omas.discharge_timing.loop_voltage_event",
              symbols=r"\mathrm{zero\ crossing\ after\ the\ OH\ onset}",
              mapping_todo="a dataclass; not written to an IDS"),
        ),
        rows=(("light", "coils"), ("baseline", "onsets"), ("threshold", "vloop"), ("window", None),
              ("confirm", None), ("plasma", None)),
        edges=(("light", "baseline", ""), ("baseline", "threshold", ""), ("threshold", "window", ""),
               ("window", "confirm", ""), ("confirm", "plasma", ""), ("coils", "onsets", ""),
               ("onsets", "vloop", "")),
        side=(("rules", "threshold"),),
        todos=("Breakdown time is the plasma-window start; there is no separate breakdown detector.",
               "There is no flat-top detector (only the post-hoc slice phase) and no IRE, sawtooth or disruption "
               "detector.",
               "current_quench and current_spike (vaft.process.transients) measure a quench or spike in a given "
               "window; nothing in the pipeline calls them, and they do not label it an IRE or a disruption."),
    ),
)

#: every spine workflow, by key, in reading order
WORKFLOWS: Dict[str, WorkflowSpec] = {spec.key: spec for spec in _SPECS}


def plasma_parameter_inference(*, labels: bool = True) -> Diagram:
    """Electron profiles and the equilibrium resolved into T_i, the species mix and the local state."""
    return _render_workflow(WORKFLOWS["plasma_parameter_inference"], labels=labels)


def romero_transformer_balance(*, labels: bool = True) -> Diagram:
    """Plasma resistance from Romero's exact transformer balance."""
    return _render_workflow(WORKFLOWS["romero_transformer_balance"], labels=labels)


def resistive_zeff_inference(*, labels: bool = True) -> Diagram:
    """The resistively equivalent scalar Zeff a conductivity model needs to match the observed resistance."""
    return _render_workflow(WORKFLOWS["resistive_zeff_inference"], labels=labels)


def magnetic_efit(*, labels: bool = True) -> Diagram:
    """Magnetic equilibrium reconstruction: a free-boundary inverse problem."""
    return _render_workflow(WORKFLOWS["magnetic_efit"], labels=labels)


def kinetic_efit(*, labels: bool = True) -> Diagram:
    """Kinetically constrained reconstruction: magnetic constraints plus kinetic pressure points."""
    return _render_workflow(WORKFLOWS["kinetic_efit"], labels=labels)


def analytic_mhd_equilibrium(*, labels: bool = True) -> Diagram:
    """Analytic MHD equilibria: forward generation and projection onto a Solov'ev basis."""
    return _render_workflow(WORKFLOWS["analytic_mhd_equilibrium"], labels=labels)


def chease_coupling(*, labels: bool = True) -> Diagram:
    """Fixed-boundary refinement with CHEASE, COCOS transform explicit."""
    return _render_workflow(WORKFLOWS["chease_coupling"], labels=labels)


def tokamaker_coupling(*, labels: bool = True) -> Diagram:
    """Free-boundary equilibrium with TokaMaker."""
    return _render_workflow(WORKFLOWS["tokamaker_coupling"], labels=labels)


def dcon_rdcon_stability(*, labels: bool = True) -> Diagram:
    """Ideal and resistive MHD stability with DCON and RDCON."""
    return _render_workflow(WORKFLOWS["dcon_rdcon_stability"], labels=labels)


def gpec_plasma_response(*, labels: bool = True) -> Diagram:
    """The ideal plasma response to applied 3-D fields with GPEC."""
    return _render_workflow(WORKFLOWS["gpec_plasma_response"], labels=labels)


def flare_field_line_topology(*, labels: bool = True) -> Diagram:
    """Magnetic field-line topology with FLARE."""
    return _render_workflow(WORKFLOWS["flare_field_line_topology"], labels=labels)


def neo_neoclassical(*, labels: bool = True) -> Diagram:
    """Neoclassical transport: Sauter / Redl fits and NEO at one state."""
    return _render_workflow(WORKFLOWS["neo_neoclassical"], labels=labels)


def tglf_cgyro_local_transport(*, labels: bool = True) -> Diagram:
    """Local turbulent transport: TGLF and CGYRO from one local state."""
    return _render_workflow(WORKFLOWS["tglf_cgyro_local_transport"], labels=labels)


def parameter_inference_overview(*, labels: bool = True, references: bool = False) -> Diagram:
    """Where plasma parameter inference sits between reconstruction and simulation (#1601).

    ``references=True`` numbers each closure's representative papers and lists them under the legend.
    """
    return _render_workflow(WORKFLOWS["parameter_inference_overview"], labels=labels, references=references)


def parameter_inference_dependency_graph(*, labels: bool = True, references: bool = False) -> Diagram:
    """How provenance propagates through inferred plasma parameters (#1601).

    ``references=True`` numbers the representative papers and lists them under the legend.
    """
    return _render_workflow(WORKFLOWS["parameter_inference_dependency_graph"], labels=labels, references=references)


def eddy_current_model(*, labels: bool = True, references: bool = False) -> Diagram:
    """Vessel eddy currents from the PF and plasma current circuit (#1901)."""
    return _render_workflow(WORKFLOWS["eddy_current_model"], labels=labels, references=references)


def kinetic_profile_fitting(*, labels: bool = True, references: bool = False) -> Diagram:
    """Kinetic profile fitting from Thomson and charge exchange (#1901)."""
    return _render_workflow(WORKFLOWS["kinetic_profile_fitting"], labels=labels, references=references)


def transport_hierarchy(*, labels: bool = True, references: bool = False) -> Diagram:
    """Classical, neoclassical and turbulent transport from one state (#1901)."""
    return _render_workflow(WORKFLOWS["transport_hierarchy"], labels=labels, references=references)


def neoclassical_model_comparison(*, labels: bool = True, references: bool = False) -> Diagram:
    """Sauter / Redl against NEO at one kinetic state (#1901)."""
    return _render_workflow(WORKFLOWS["neoclassical_model_comparison"], labels=labels, references=references)


def machine_model_history(*, labels: bool = True, references: bool = False) -> Diagram:
    """Shot number to machine era, diagnostic revisions and the static machine description (#1901)."""
    return _render_workflow(WORKFLOWS["machine_model_history"], labels=labels, references=references)


def signal_processing(*, labels: bool = True, references: bool = False) -> Diagram:
    """Raw digitizer records to a diagnostic IDS, in the magnetic-probe order (#1901)."""
    return _render_workflow(WORKFLOWS["signal_processing"], labels=labels, references=references)


def quality_validation(*, labels: bool = True, references: bool = False) -> Diagram:
    """Source validity, vacuum-model agreement and fit quality as separate evidence (#1901)."""
    return _render_workflow(WORKFLOWS["quality_validation"], labels=labels, references=references)


def fault_anomaly_detection(*, labels: bool = True, references: bool = False) -> Diagram:
    """Diagnostic-specific fault checks; no universal anomaly model (#1901)."""
    return _render_workflow(WORKFLOWS["fault_anomaly_detection"], labels=labels, references=references)


def shot_classification(*, labels: bool = True, references: bool = False) -> Diagram:
    """Shot and slice classes as rules over derived features (#1901)."""
    return _render_workflow(WORKFLOWS["shot_classification"], labels=labels, references=references)


def event_detection(*, labels: bool = True, references: bool = False) -> Diagram:
    """Plasma window and coil onsets from robust-baseline thresholds (#1901)."""
    return _render_workflow(WORKFLOWS["event_detection"], labels=labels, references=references)

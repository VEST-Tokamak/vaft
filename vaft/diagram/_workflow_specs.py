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
              api="vaft.validation.kinetic_state.infer_ti_pressure_partition",
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

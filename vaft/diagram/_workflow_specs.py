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
    # ------------------------------------------------------------------ derived physics (#1901)
    WorkflowSpec(
        key="vacuum_field_proxies", title="Vacuum fields and startup proxies: coils and vessel, no plasma",
        family="derived", status="implemented",
        summary="The field of the PF coils and the vessel currents alone, evaluated from geometric Green's "
                "functions on a grid or at a point, and the breakdown proxies read from it. Everything is "
                "returned to the caller; nothing is written to an IDS except the vessel currents the evaluators "
                "solve when the ODS has none.",
        nodes=(
            N("pf", "PF coil currents", "measured", ids="pf_active.coil[:].current.data", symbols=r"I_{\mathrm{PF},j}(t)"),
            N("tf", "Toroidal field", "measured", ids="tf.b_field_tor_vacuum_r.data", symbols=r"R_0 B_0\ [\mathrm{T\,m}]"),
            N("gauge", "Prefill pressure", "measured", api="vaft.omas.process_wrapper.compute_prefill_pressure_ods",
              ids="barometry.gauge[:].pressure.data", symbols=r"p\ [\mathrm{Pa}],\ \mathrm{median\ before\ the\ onset}"),
            N("vessel", "Vessel loop currents", "standardized", api="vaft.omas.process_wrapper.compute_eddy_currents",
              ids="pf_passive.{time, loop[:].current}",
              symbols=r"I_{\mathrm{loop},k}(t)\qquad \mathrm{solved\ and\ written\ when}\ \mathtt{pf\_passive}\ "
                      r"\mathrm{has\ no\ time}"),
            N("geometry", "Conductor and limiter geometry", "machine",
              ids="{pf_active.coil[:].element, pf_passive.loop[:].element, wall.description_2d[0].limiter}",
              symbols=r"(R, Z)\ \mathrm{of\ every\ filament},\ \mathrm{limiter\ unit\ 0}"),
            N("field", "Vacuum field", "derived", api="vaft.omas.process_wrapper.compute_vacuum_field_map",
              equation="vaft.formula.green.green_psi_exact",
              symbols=r"\psi\ [\mathrm{Wb,\ not\ Wb/rad}],\ B_R,\ B_Z,\ \partial\psi/\partial t\ \mathrm{on}\ (R, Z)"
                      r"\qquad \mathrm{time\ snapped\ to\ a\ PF\ sample}"),
            N("proxies", "Point proxies", "derived", api="vaft.omas.process_wrapper.compute_startup_proxies_ods",
              relation=r"V_{\mathrm{loop}} = -\frac{d\psi}{dt},\quad n = -\frac{R}{B_Z}\frac{dB_Z}{dR}"
                       r"\qquad \mathrm{at}\ (0.4, 0)\ \mathrm{m,\ every\ PF\ sample}"),
            N("length", "Connection length", "derived",
              api="vaft.omas.process_wrapper.compute_connection_length_map_ods",
              symbols=r"L\ \mathrm{to\ the\ limiter,\ both\ directions},\ \le 150\ \mathrm{m\ each}"),
            N("merit", "Lloyd figure of merit", "derived",
              api="vaft.omas.process_wrapper.compute_vacuum_midplane_profiles_ods",
              equation="vaft.formula.startup.lloyd_figure_of_merit",
              symbols=r"|E_\phi| = |\partial\psi/\partial t|/(2\pi R),\ B_\phi = R_0 B_0/R"),
            N("margin", "Lloyd breakdown margin", "derived",
              api="vaft.omas.process_wrapper.compute_vacuum_midplane_profiles_ods",
              equation="vaft.formula.startup.breakdown_margin",
              symbols=r"E_{BD}(p, L)\ \mathrm{(Lloyd)};\ \mathrm{NaN\ where}\ A p L \le 1"),
            N("summary", "Startup summary", "derived", api="vaft.omas.startup_summary.startup_summary",
              symbols=r"\mathrm{one\ flat\ dict;\ evaluated\ on\ a\ private\ copy}",
              mapping_todo="maps, profiles and proxies are returned, never written to an IDS"),
        ),
        rows=(("pf", "vessel"), ("field", "tf"), ("proxies", "length", "merit"), ("margin", "gauge"), ("summary",)),
        edges=(("pf", "vessel", "if absent"), ("pf", "field", ""), ("vessel", "field", ""),
               ("field", "proxies", ""), ("field", "length", ""), ("tf", "length", ""), ("field", "merit", ""),
               ("tf", "merit", ""), ("length", "margin", ""), ("gauge", "margin", ""), ("merit", "summary", ""),
               ("margin", "summary", ""), ("proxies", "summary", "")),
        side=(("geometry", "field"),),
        todos=("No vacuum field, map, proxy or breakdown margin is written to an IDS: every evaluator returns a "
               "dict, and the startup summary is a dict too.",
               "compute_vacuum_field_map and compute_startup_proxies_ods solve the vessel currents and store them "
               "in the caller's pf_passive when it has no time base; startup_summary avoids that by evaluating "
               "on a private copy.",
               "Only vacuum sources contribute (coils and vessel, no plasma); above roughly 10 kA of plasma "
               "current the map no longer describes the machine.",
               "The loop voltage is a model quantity from the coil and vessel flux; it is to be compared with the "
               "measured flux-loop voltage, not substituted for it.",
               "No empirical threshold is applied to the Lloyd figure of merit."),
    ),
    WorkflowSpec(
        key="derived_mhd_parameters", title="Derived MHD parameters: global quantities of an equilibrium",
        family="derived", status="implemented",
        summary="Global quantities computed from a reconstructed equilibrium. The DD-defined betas, l_i3 and "
                "W_MHD are written to equilibrium.global_quantities; the descriptor set and the edge-q proxies "
                "are returned with their derivation records. Two poloidal betas exist on purpose: the DD leaf "
                "and the circumference form EFIT reports.",
        nodes=(
            N("eq", "Magnetic equilibrium", "reconstructed",
              ids="equilibrium.time_slice[:].{profiles_1d.{pressure, volume}, profiles_2d, boundary}",
              symbols=r"p(\psi),\ V(\psi),\ \psi(R, Z),\ I_p"),
            N("ip", "Plasma current", "measured", ids="magnetics.ip[0].data", symbols=r"I_p(t)"),
            N("reference", "Reference radius and field", "convention",
              api="vaft.omas.update.resolve_reference_major_radius",
              ids="{equilibrium.vacuum_toroidal_field.r0, equilibrium.vacuum_toroidal_field.b0, tf.r0}",
              symbols=r"R_0:\ \mathtt{tf.r0}\ \mathrm{when\ the\ stored}\ r_0\ \mathrm{is\ corrupt},\ "
                      r"B_0 = \mathtt{b0}"),
            N("surfaces", "Flux-surface integrals", "derived", api="vaft.process.equilibrium.flux_surface_quantities",
              symbols=r"\int p\,dV,\ \int B_p^2\,dV,\ L_{\mathrm{pol}},\ a"),
            N("descriptors", "Global descriptors", "derived", api="vaft.process.equilibrium.derive_global_descriptors",
              relation=r"\beta_p = 2\mu_0\langle p\rangle_V / \langle B_p\rangle_{\mathrm{LCFS}}^2",
              symbols=r"\kappa,\ \delta,\ V,\ \beta_t,\ \beta_N,\ q_{95},\ \mathrm{virial}\ l_i"
                      r"\qquad \mathrm{each\ a\ DerivedValue\ with\ its\ record}",
              mapping_todo="returned, not written"),
            N("edge_q", "Edge safety-factor proxies", "derived", api="vaft.omas.edge_q.edge_q_estimate",
              equation="vaft.formula.equilibrium.estimated_q95",
              symbols=r"q_{95}^{\mathrm{est}},\ q^*_{\mathrm{Menard}},\ q^*_{\mathrm{Freidberg}},\ I_N"
                      r"\qquad \mathrm{source:\ equilibrium\ |\ magnetics}",
              mapping_todo="returned (EdgeQEstimate), not written"),
            N("shape", "Machine default shape", "prior", api="vaft.machine_mapping.edge_q_estimate.vest_edge_q_estimate_policy",
              symbols=r"a,\ R_0,\ \kappa,\ \delta\ \mathrm{from\ vest.yaml\ without\ an\ equilibrium}"),
            N("beta_pol", "Poloidal beta (DD)", "standardized",
              api="vaft.omas.update.update_equilibrium_global_quantities_beta_li",
              equation="vaft.formula.equilibrium.beta_poloidal_from_pressure_integral",
              ids="equilibrium.time_slice[:].global_quantities.beta_pol"),
            N("beta_tor", "Toroidal beta", "standardized",
              api="vaft.omas.update.update_equilibrium_global_quantities_beta_li",
              equation="vaft.formula.equilibrium.beta_toroidal_from_p_B0",
              ids="equilibrium.time_slice[:].global_quantities.beta_tor"),
            N("li", "Internal inductance", "standardized",
              api="vaft.omas.update.update_equilibrium_global_quantities_beta_li",
              equation="vaft.formula.equilibrium.li_3_from_Bp2_volume_integral",
              ids="equilibrium.time_slice[:].global_quantities.li_3"),
            N("beta_n", "Normalized beta", "standardized",
              api="vaft.omas.update.update_equilibrium_global_quantities_beta_li",
              equation="vaft.formula.equilibrium.beta_normal_from_beta_tor",
              ids="equilibrium.time_slice[:].global_quantities.beta_normal", symbols=r"\beta_N\ [-]"),
            N("energy", "Stored energy", "standardized", api="vaft.omas.update.update_equilibrium_stored_energy",
              equation="vaft.formula.equilibrium.thermal_energy_from_p_V",
              ids="equilibrium.time_slice[:].global_quantities.energy_mhd",
              symbols=r"W_{\mathrm{MHD}}\ [\mathrm{J}]\ \mathrm{from\ the\ reconstructed}\ p"),
        ),
        rows=(("eq", "ip"), ("surfaces", "energy", "descriptors", "edge_q"), ("beta_pol", "beta_tor", "li", None),
              (None, "beta_n", None, None)),
        edges=(("eq", "surfaces", ""), ("eq", "descriptors", ""), ("eq", "edge_q", ""),
               ("ip", "edge_q", ""), ("surfaces", "beta_pol", ""), ("surfaces", "beta_tor", ""),
               ("surfaces", "li", ""), ("beta_tor", "beta_n", ""), ("eq", "energy", "")),
        side=(("reference", "surfaces"), ("shape", "edge_q")),
        todos=("beta_pol holds the DD definition; EFIT normalizes by the LCFS-circumference field instead and "
               "differs by 26% on the packaged reference (#318). derive_global_descriptors reports the "
               "boundary-average form as beta_p_boundary_average; the DD leaf is not matched to OMFIT.",
               "equilibrium.vacuum_toroidal_field.r0 is corrupt on the VEST database and b0 drifts within a shot "
               "(#325); R_0 is replaced by tf.r0, loudly, and b0 is used as stored.",
               "beta_tor with the stored b0 does not reproduce the OMFIT reference (ratio 1.041); the reference "
               "field is unidentified (#238).",
               "li_3 is the ITER/Jackson third definition: the DD documents no formula for it.",
               "energy_mhd is (3/2) int p dV of the reconstructed pressure: a reconstruction-derived energy, not a "
               "kinetic measurement.",
               "The global descriptors and the edge-q proxies are returned, not written to equilibrium or summary."),
    ),
    WorkflowSpec(
        key="synthetic_diagnostics", title="Synthetic diagnostics: predicted signals from a modelled state",
        family="derived", status="implemented",
        summary="A synthetic diagnostic predicts what an instrument would record from a modelled or "
                "reconstructed state. VAFT has three: vacuum magnetics from the coil and vessel currents, the "
                "equilibrium's diamagnetic flux, and soft-X-ray chords through a prescribed island. "
                "Assumption-completed kinetic profiles (generate_synthetic_kinetic_profiles) are a synthetic "
                "state, not a diagnostic, and are not drawn here.",
        nodes=(
            N("pf", "PF coil currents", "measured", ids="pf_active.coil[:].current.data", symbols=r"I_{\mathrm{PF},j}(t)"),
            N("vessel", "Vessel loop currents", "standardized", ids="pf_passive.loop[:].current",
              symbols=r"I_{\mathrm{loop},k}(t)\ \mathrm{(forward\ circuit\ model)}"),
            N("eq", "Magnetic equilibrium", "reconstructed",
              ids="equilibrium.time_slice[:].{profiles_2d[0].{psi, b_field_tor}, profiles_1d.f}",
              symbols=r"\psi(R, Z),\ F(\psi),\ B_\phi(R, Z)"),
            N("dia_meas", "Diamagnetic flux", "measured", ids="magnetics.diamagnetic_flux[0].data",
              symbols=r"\Phi_{\mathrm{dia}}^{\mathrm{meas}}(t)"),
            N("sensors", "Magnetic sensor positions", "machine",
              ids="magnetics.{flux_loop[:].position, b_field_pol_probe[:].position}",
              symbols=r"(R, Z)\ \mathrm{of\ each\ selected\ channel}"),
            N("mag_meas", "Measured magnetics", "measured", ids="magnetics.{flux_loop, b_field_pol_probe}",
              symbols=r"\psi_{\mathrm{FL}},\ B_p\ \mathrm{with\ validity}"),
            N("vacuum", "Vacuum magnetics forward model", "derived",
              api="vaft.omas.vacuum_magnetics.synthetic_vacuum_magnetics",
              symbols=r"m_{\mathrm{coil}},\ m_{\mathrm{coil+eddy}}\ \mathrm{on\ the\ PF\ time\ grid}"
                      r"\qquad \mathrm{measured\ interpolated\ onto\ it,\ no\ extrapolation}"),
            N("diamagnetic", "Diamagnetic flux of the equilibrium", "derived",
              api="vaft.omas.process_wrapper.compute_reconstructed_diamagnetic_flux",
              relation=r"\Phi_{\mathrm{dia}} = \int_S (B_\phi - B_{\phi,\mathrm{vac}})\,dA"
                       r"\qquad \mathrm{negative\ for\ a\ diamagnetic\ plasma}"),
            N("island", "Prescribed island", "prior",
              symbols=r"(m, n),\ W\ \mathrm{at\ the\ outboard\ midplane},\ \mathrm{phase},\ \omega"
                      r"\qquad \epsilon_0(\psi_N)\ \mathrm{background}"),
            N("sxr", "Island soft-X-ray forward model", "derived",
              api="vaft.process.soft_x_rays.synthetic_island_soft_x_rays",
              symbols=r"\epsilon\ \mathrm{flattened\ in\ the\ island},\ \mathrm{pencil\ chords\ clipped\ to\ the\ limiter}"),
            N("vac_out", "Synthetic vacuum magnetics", "derived",
              symbols=r"\mathrm{measured},\ m_{\mathrm{coil}},\ m_{\mathrm{coil+eddy}}\ \mathrm{per\ channel}",
              mapping_todo="returned as VacuumChannel records, not written"),
            N("dia_out", "Diamagnetic constraint", "standardized",
              api="vaft.omas.update.update_equilibrium_constraints_diamagnetic_flux",
              ids="equilibrium.time_slice[:].constraints.diamagnetic_flux.{measured, reconstructed}",
              symbols=r"\Phi_{\mathrm{dia}}^{\mathrm{meas}}\ \mathrm{vs}\ \Phi_{\mathrm{dia}}^{\mathrm{rec}}"),
            N("sxr_out", "Synthetic SXR brightness", "synthetic",
              symbols=r"b(t, \mathrm{chord})\ [\epsilon\cdot\mathrm{m}],\ \mathrm{before\ any\ detector\ response}",
              mapping_todo="returned (SyntheticSXRResult), not written to soft_x_rays"),
        ),
        rows=(("pf", "vessel", "eq"), ("mag_meas", "vacuum", "diamagnetic", "sxr"),
              ("vac_out", "dia_meas", "dia_out", "sxr_out")),
        edges=(("pf", "vacuum", ""), ("vessel", "vacuum", ""), ("mag_meas", "vacuum", ""), ("eq", "diamagnetic", ""),
               ("eq", "sxr", ""), ("vacuum", "vac_out", ""), ("diamagnetic", "dia_out", ""),
               ("dia_meas", "dia_out", ""), ("sxr", "sxr_out", "")),
        side=(("sensors", "mag_meas"), ("island", "sxr")),
        todos=("There is no synthetic Thomson, interferometer, visible-camera or bolometer model: those "
               "measurements cannot be compared with a prediction of the reconstructed state.",
               "The vacuum magnetics are returned with their measured counterparts and written nowhere; EFIT's "
               "own fitted magnetics (equilibrium constraints.*.reconstructed) are the solver's native output, "
               "not a VAFT forward model.",
               "The vacuum forward model has no plasma current: it predicts the magnetics only before breakdown "
               "or in plasma-free windows.",
               "The SXR model is forward only: a rigidly rotating island of fixed width, pencil-beam chords, no "
               "filter, etendue or noise; it is comparable with a measurement only up to a per-channel "
               "calibration, and is not written to soft_x_rays."),
    ),
    WorkflowSpec(
        key="coordinate_cocos_conversions", title="Coordinates and COCOS: one transform per layer",
        family="derived", status="implemented",
        summary="Two COCOS transforms, at two layers: the ODS is normalized in place to COCOS 11 (weber), and an "
                "in-memory equilibrium record is converted to any target COCOS. The ODS coordinate writers "
                "resolve the flux storage of each slice themselves (as_equilibrium), so they also run on an "
                "unnormalized ODS; the record-level radii and angle are COCOS-independent and returned.",
        nodes=(
            N("src", "Equilibrium as stored", "reconstructed",
              ids="equilibrium.time_slice[:].{profiles_1d.psi, profiles_2d[0].psi, global_quantities}",
              symbols=r"\psi\ [\mathrm{Wb\ or\ Wb/rad}],\ I_p,\ B_0,\ q,\ F"),
            N("declared", "Declared COCOS", "convention", ids="equilibrium.code.parameters.cocos",
              symbols=r"\mathrm{trusted\ as\ provenance;\ absent\ before}\ \#236"),
            N("normalize", "ODS normalization to COCOS 11", "derived",
              api="vaft.omas.general.equilibrium_psi_to_weber",
              symbols=r"\mathrm{declared\ 1\text{-}8}:\ \mathrm{OMAS\ cocos\_transform\ to\ 11}\qquad "
                      r"\mathrm{declared\ 11\text{-}18:\ unchanged}\qquad "
                      r"\mathrm{undeclared:\ storage\ probed,\ Wb/rad\ taken\ as\ COCOS\ 1}"),
            N("ods11", "Equilibrium in COCOS 11", "standardized", api="vaft.omas.general.set_ods_cocos",
              ids="{equilibrium.time_slice[:], equilibrium.code.parameters.cocos}",
              symbols=r"\psi\ [\mathrm{Wb}],\ \mathrm{declared}\ 11;\ \mathrm{constraints\ untouched}"),
            N("psi_norm", "Normalized poloidal flux", "standardized",
              api="vaft.omas.update.update_equilibrium_profiles_1d_normalized_psi",
              equation="vaft.formula.equilibrium.psi_normalised",
              ids="equilibrium.time_slice[:].profiles_1d.psi_norm", symbols=r"\psi_N\ [-]"),
            N("rho_tor", "Toroidal flux coordinate", "standardized",
              api="vaft.omas.update.update_equilibrium_profiles_1d_toroidal_flux",
              equation="vaft.formula.equilibrium.toroidal_flux_from_q_psi",
              ids="equilibrium.time_slice[:].profiles_1d.{phi, rho_tor, rho_tor_norm}",
              symbols=r"\rho_{\mathrm{tor}} = \sqrt{|\Phi|/(\pi |B_0|)}"),
            N("grid", "Flux-surface grid", "standardized",
              api="vaft.omas.update.update_equilibrium_profiles_2d_sfl_coordinates",
              relation=r"R = R_{\mathrm{ax}} + \sqrt{\psi_N}\,r_{\mathrm{LCFS}}(\theta)\cos\theta,\ "
                       r"\theta\ \mathrm{geometric}",
              ids="equilibrium.time_slice[:].profiles_2d[1].{grid.dim1, grid.dim2, r, z, theta}",
              symbols=r"(\psi_N, \theta)\ \mathrm{grid},\ \mathrm{surfaces\ =\ scaled\ LCFS}",
              mapping_todo="stored as grid_type 11 (straight field line), though theta is geometric"),
            N("record", "Equilibrium record", "derived", api="vaft.process.equilibrium.as_equilibrium",
              symbols=r"\mathrm{Wb\ vs\ Wb/rad\ from\ Amp\grave{e}re's\ law\ round\ the\ LCFS}\qquad "
                      r"\mathrm{COCOS\ declared,\ or\ candidates\ from\ the\ signs\ of}\ B_0, I_p, q"),
            N("target", "Target COCOS", "convention", symbols=r"1\text{-}18\ (\mathrm{no}\ 9, 10),\ \mathrm{chosen\ by\ the\ caller}"),
            N("convert", "Record conversion", "derived", api="vaft.process.equilibrium.convert_cocos",
              symbols=r"\psi,\ F,\ q,\ p',\ FF',\ I_p,\ B_0\ \times\ \mathrm{their\ cocos\_transform\ factors}"
                      r"\qquad p\ \mathrm{and\ the\ axis\ untouched;\ ambiguous\ source\ refused}"
                      r"\qquad \mathrm{returns\ a\ copy\ in\ the\ target\ COCOS,\ in\ memory}"),
            N("radial", "Normalized radii", "derived", api="vaft.process.equilibrium.derive_radial_coordinates",
              symbols=r"\psi_N,\ \rho_{\mathrm{pol},N} = \sqrt{\psi_N},\ \rho_{\mathrm{tor},N}\qquad "
                      r"\mathrm{COCOS\text{-}independent;\ unavailable\ with\ a\ reason}"),
            N("sfl", "Straight-field-line angle", "derived", api="vaft.process.equilibrium.straight_field_line_map",
              equation="vaft.formula.equilibrium.straight_field_line_angle",
              symbols=r"\theta^*\ (\mathrm{PEST})\ \mathrm{at\ any}\ (R, Z)"),
        ),
        rows=(("src",), ("normalize", "record"), ("ods11", "convert"),
              ("psi_norm", "grid", "rho_tor", "radial", "sfl")),
        edges=(("src", "normalize", ""), ("src", "record", ""), ("normalize", "ods11", ""),
               ("ods11", "psi_norm", ""), ("ods11", "rho_tor", ""), ("ods11", "grid", ""),
               ("record", "radial", ""), ("record", "sfl", ""), ("record", "convert", "")),
        side=(("declared", "normalize"), ("target", "convert")),
        todos=("The ODS writer of the 2-D flux-surface grid uses a geometric approximation (the boundary scaled "
               "by sqrt(psi_N) about the axis, the geometric angle) for every requested convention, yet labels "
               "it grid_type 11; the true PEST angle exists only in memory "
               "(vaft.process.equilibrium.straight_field_line_map).",
               "equilibrium_psi_to_weber leaves equilibrium.time_slice[:].constraints untouched: VAFT copies "
               "them from the DD-conformant magnetics and pf_active IDSs.",
               "rho_tor integrates q from the axis, so EFIT's often outlying q[0] moves rho_tor_norm by up to "
               "about 0.04 on the innermost surfaces; packaged samples older than #192 still hold the "
               "sqrt(psi_N) proxy.",
               "convert_cocos shifts the identified-candidate list on the assumption that every candidate "
               "shares the source's storage family, which can produce an out-of-range index (#601)."),
    ),
    WorkflowSpec(
        key="power_balance_confinement", title="Power balance and confinement: experimental and modelled",
        family="derived", status="implemented",
        summary="Two power balances answer different questions. The experimental Ohmic chain splits the measured "
                "loop voltage with l_i3 and takes the stored energy of the equilibrium; the modelled budget uses "
                "Spitzer heating and modelled radiation. Both are returned; neither writes to summary.",
        nodes=(
            N("ip", "Plasma current", "measured", ids="magnetics.ip[0].{data, time}", symbols=r"I_p(t),\ dI_p/dt"),
            N("vloop", "Loop voltage", "measured", api="vaft.omas.discharge_timing.inboard_midplane_loop",
              ids="magnetics.flux_loop[:].{voltage, flux}",
              symbols=r"V_{\mathrm{loop}}\ \mathrm{(inboard\ midplane\ loop,\ off\ the\ boundary)}"),
            N("eq", "Magnetic equilibrium", "reconstructed",
              ids="equilibrium.time_slice[:].{global_quantities.li_3, profiles_1d.{pressure, volume}}",
              symbols=r"l_{i,3},\ W_{\mathrm{MHD}} = \tfrac32\int p_{\mathrm{EFIT}}\,dV"),
            N("cp", "Kinetic profiles", "standardized", ids="core_profiles.profiles_1d[:].electrons",
              symbols=r"n_e(\rho),\ T_e(\rho)"),
            N("w_source", "Stored-energy source", "convention",
              symbols=r"\mathrm{efit\_pressure\ (default)\ |\ core\_profiles\_thermal}"),
            N("split", "Resistive loop voltage", "derived", api="vaft.process.confinement.resistive_loop_voltage",
              equation="vaft.formula.equilibrium.inductive_voltage_from_dW_magdt_I_p",
              symbols=r"V_{\mathrm{res}} = V_{\mathrm{loop}} - V_{\mathrm{ind}},\ L_i = \mu_0 R_0 l_{i,3}/2"),
            N("balance", "Ohmic power balance", "derived", api="vaft.process.confinement.confinement_power_balance",
              equation="vaft.formula.equilibrium.ohmic_heating_power_from_I_p_V_res",
              symbols=r"P_{\mathrm{net}} = P_{\mathrm{OH}} - dW/dt\ (\mathrm{DB5\ PLTH})\qquad "
                      r"P_{\mathrm{transport}}\ \mathrm{needs\ a\ measured}\ P_{\mathrm{rad}}:\ \mathrm{NaN}"),
            N("tau", "Confinement time", "derived", api="vaft.omas.confinement.ohmic_confinement_ods",
              equation="vaft.formula.equilibrium.confinement_time_from_P_loss_W_th",
              symbols=r"\tau_E\ \mathrm{per\ validated\ slice,\ with\ its\ assumptions}",
              mapping_todo="returned (OhmicConfinementODS), not written to summary"),
            N("scaling", "Slice qualification and scaling", "inferred",
              api="vaft.process.confinement.fit_confinement_scaling",
              equation="vaft.formula.equilibrium.confinement_time_from_engineering_parameters",
              symbols=r"\mathrm{evidence\ per\ slice;\ thresholds\ are\ arguments}"),
            N("impurity", "Impurity policy", "prior", api="vaft.machine_mapping.core_profiles.vest_core_profiles_policy",
              symbols=r"\mathrm{C,\ O\ at}\ n_I/n_e = 1/86,\ Z_{\mathrm{eff}} = 2\ (\mathrm{assumed})"),
            N("spitzer", "Spitzer Ohmic dissipation", "derived",
              api="vaft.omas.process_wrapper.compute_ohmic_heating_power_from_core_profiles",
              relation=r"P_{\mathrm{ohm,diss}} = \int_V \eta\,J_\phi^2\,dV,\ T_i = T_e"),
            N("radiation", "Modelled radiation", "inferred",
              api="vaft.process.atomic.compute_line_radiation_power_series",
              equation="vaft.formula.atomic.line_cooling_coefficient",
              symbols=r"P_{\mathrm{line}} = \int n_e n_I L_Z\,dV\ \mathrm{(ADF11)}"),
            N("budget", "Modelled power budget", "derived", api="vaft.omas.formula_wrapper.compute_power_balance",
              equation="vaft.formula.equilibrium.loss_power_from_p_heat_dWdt_p_rad",
              symbols=r"P_{\mathrm{rad}} = P_{\mathrm{line}} + P_{\mathrm{sync}} + P_{\mathrm{Br}},\ P_{\mathrm{heat}} = P_{\mathrm{ohm,diss}}\qquad dW/dt = 0\ \mathrm{for\ one\ slice}",
              mapping_todo="returned as a dict, not written to summary"),
        ),
        rows=(("ip", "vloop", "eq", "cp"), ("split", None, "spitzer", "radiation"), ("balance", None, "budget", None),
              ("tau",), ("scaling",)),
        edges=(("ip", "split", ""), ("vloop", "split", ""), ("eq", "split", ""), ("split", "balance", ""),
               ("eq", "balance", ""), ("cp", "spitzer", ""), ("cp", "radiation", ""), ("spitzer", "budget", ""),
               ("radiation", "budget", ""), ("balance", "tau", ""), ("tau", "scaling", "")),
        side=(("w_source", "balance"), ("impurity", "radiation")),
        todos=("Neither power balance nor the confinement time is written to summary: both are returned with "
               "their assumptions.",
               "VEST has no bolometer, so no measured P_rad enters: the experimental chain's transport loss power "
               "is NaN and its default is p_net (DB5 PLTH); the modelled budget always subtracts a modelled "
               "P_rad, which is a sensitivity study, not a measurement.",
               "The experimental chain pairs the measured current with EFIT's l_i3 (a hybrid), and its flux loop "
               "is not on the plasma boundary; both are recorded assumptions.",
               "The stored energy is reconstruction-derived (efit_pressure) or assumes T_i = T_e "
               "(core_profiles_thermal); neither is a global energy (#1713)."),
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


def vacuum_field_proxies(*, labels: bool = True, references: bool = False) -> Diagram:
    """Vacuum fields and the startup proxies read from them (#1901)."""
    return _render_workflow(WORKFLOWS["vacuum_field_proxies"], labels=labels, references=references)


def derived_mhd_parameters(*, labels: bool = True, references: bool = False) -> Diagram:
    """Global MHD quantities derived from a reconstructed equilibrium (#1901)."""
    return _render_workflow(WORKFLOWS["derived_mhd_parameters"], labels=labels, references=references)


def synthetic_diagnostics(*, labels: bool = True, references: bool = False) -> Diagram:
    """Predicted diagnostic signals from a modelled or reconstructed state (#1901)."""
    return _render_workflow(WORKFLOWS["synthetic_diagnostics"], labels=labels, references=references)


def coordinate_cocos_conversions(*, labels: bool = True, references: bool = False) -> Diagram:
    """COCOS transforms at the ODS and record layers, and the coordinates (#1901)."""
    return _render_workflow(WORKFLOWS["coordinate_cocos_conversions"], labels=labels, references=references)


def power_balance_confinement(*, labels: bool = True, references: bool = False) -> Diagram:
    """The experimental Ohmic chain and the modelled power budget (#1901)."""
    return _render_workflow(WORKFLOWS["power_balance_confinement"], labels=labels, references=references)

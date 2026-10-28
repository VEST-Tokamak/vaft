"""Equilibrium, MHD and transport joined through shared representations of a plasma state.

One figure across the physics-workflow spine (#1585): where the per-solver
figures (``magnetic_efit``, ``dcon_rdcon_stability``, ``transport_hierarchy``,
...) draw the steps inside one coupling, this one draws the relations between
the domains -- which state a calculation reads, which methods are alternatives
for a related quantity and which are complementary, what reaches an IMAS IDS
and what stays in the solver's own form. It is the technical companion of
:func:`~vaft.diagram.cross_domain_scientific_architecture`, which draws the
same architecture at the level of a manuscript figure.

Every :class:`Element` is drawn from nodes of the existing
:data:`~vaft.diagram._workflow_specs.WORKFLOWS` (``sources``); its API and IDS
are read from them, never restated, and ``dd_path`` names one Data Dictionary
leaf the tests check against the installed DD. Relations come in four kinds
(:data:`CONNECTIONS`), each drawn its own way so they stay apart in grayscale:
a data flow is an arrow, alternative methods share a dash-dot enclosure
(:data:`ALTERNATIVES`), and a result sits directly under the operation that
produced it, in the IMAS band when it is mapped to an IDS and in the native
band when the solver keeps it.

What the figure claims, and what it leaves out on purpose:

- One ``equilibrium`` IDS holds equilibria of different origin. Shared
  representation is not shared state: only the reconstructed entry carries
  the experimental state the transport resolver keys on
  (``TransportStateKey`` = shot, EFIT time, EFIT lineage).
  The resolver takes either EFIT lineage (``magnetics`` or
  ``electron_kinetic``), so the EFIT box reads the kinetic profiles too.
  CHEASE re-solves that reconstructed entry at fixed boundary; it is a
  refinement, not an alternative.
- MHD and transport are complementary consumers of an equilibrium, not
  consecutive stages; no edge joins them. Inside MHD, GPEC reads DCON's
  ideal eigenfunctions (``euler.bin``, ``vaft.code.gpec`` validates it as a
  prerequisite), so that one edge is drawn.
- ``state_identity`` keys runs, not IDS entries: NEO, classical, TGLF and
  CGYRO read :func:`~vaft.process.transport_state.resolve_transport_state`,
  ``run_identity`` folds the identity into each run's hash, and the
  transport atlas joins its NEO, TGLF and classical rows on it
  (``workflow/transport_atlas/build_atlas.py``; CGYRO in
  ``workflow/gyrokinetic``). It is drawn on the resolved state, not as a
  join of the IDS results: ``core_transport_from_neo`` records no identity.
  The NEO product the pipeline stores (``vaft.omas.vest_upstream``) is
  gated by input readiness rather than the resolver; it is the same writer.
- The Sauter / Redl closures read the ODS directly
  (``vaft.omas.neoclassical.compute_bootstrap_current``) and have no
  ``core_transport`` writer, so they are retained, not mapped; the kinetic
  profiles they share with the resolver are drawn once.
- CGYRO is drawn as linear eigenmodes only. A nonlinear run can be written to
  ``core_transport`` (``core_transport_from_cgyro``) and to
  ``gyrokinetics_local.non_linear`` (``gyrokinetics_local_from_cgyro``), each
  given a saturation window, but no nonlinear result is a demonstrated
  benchmark, so neither route is drawn. TGLF's ``gyrokinetics_local`` writer
  is likewise omitted: no stage stores that IDS for TGLF (#2020).
- DCON's ``energy_perturbed`` is a normalized eigenvalue in a field the DD
  documents in joules; the mismatch is recorded in ``code.parameters`` by
  the mapper and stated on the node. RDCON writes only the classical
  (diagonal) Delta-prime to ``ntms``; the matrix stays native.
- Nothing here says a route ran on a given machine or how often: an edge is
  an implemented interface. Execution evidence lives in the handoff and the
  per-solver documentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ._concept import Box, box, connector, escape_latex
from ._render import Diagram
from ._scene import Arrow, Label, Polyline, Scene
from ._workflow import KINDS, kind_legend, kind_style
from ._workflow_specs import WORKFLOWS

#: the four relations the figure states, in legend order
CONNECTIONS: Tuple[str, ...] = ("flow", "alternative", "mapping", "native")
CONNECTION_LABEL: Dict[str, str] = {
    "flow": "consumed by (data flow)",
    "alternative": "alternative methods, related quantity",
    "mapping": "mapped to an IMAS IDS",
    "native": "kept in native form",
}
#: how each is drawn: an arrow, a dash-dot enclosure, and the two result bands -- distinct without colour
CONNECTION_STYLE: Dict[str, str] = {
    "flow": "connector strong",
    "alternative": "draw=driftred!85!black,dash dot,line width=1.2pt,rounded corners=6pt",
    "mapping": "draw=absequil!90!black,line width=1.2pt,rounded corners=6pt,fill=absequil!7",
    "native": "draw=black!45,dashed,line width=1.0pt,rounded corners=6pt,fill=black!4",
}
DOMAINS: Tuple[str, ...] = ("equilibrium", "mhd", "transport")
#: the drawn regions: (key, title, subtitle); ``hub``, ``imas`` and ``native`` are result bands
REGIONS: Tuple[Tuple[str, str, str], ...] = (
    ("equilibrium", "Equilibrium", "three alternative routes and a refinement"),
    ("hub", "Equilibrium representation", "one IDS type; each entry keeps its own plasma state"),
    ("mhd", "MHD stability and plasma response", "complementary analyses"),
    ("transport", "Transport", "independent models"),
    ("imas", "IMAS representation", "each IDS result sits under the operation that wrote it"),
    ("native", "Native results", "retained in the solver's own form, not mapped"),
)


@dataclass(frozen=True)
class Element:
    """One drawn quantity, method or result, and the spine nodes it is drawn from.

    ``label`` is the physical role, ``method`` the solver or model name drawn
    under it and ``basis`` the approximation it rests on; ``detail`` is inline
    LaTeX math for the representative quantities.
    ``api`` is set only where no source node names one (a mapper the per-solver
    figure leaves implicit); ``dd_path`` is one IMAS leaf in the
    ``omas_info_node`` spelling (``:`` for every array index).
    """

    key: str
    label: str
    kind: str
    domain: str
    sources: Tuple[Tuple[str, str], ...]
    method: str = ""
    basis: str = ""
    detail: str = ""
    api: Optional[str] = None
    dd_path: Optional[str] = None
    note: str = ""

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"element {self.key!r}: kind must be one of {KINDS}, not {self.kind!r}")
        if self.domain not in DOMAINS:
            raise ValueError(f"element {self.key!r}: domain must be one of {DOMAINS}, not {self.domain!r}")
        if not self.sources:
            raise ValueError(f"element {self.key!r}: name the spine nodes it is drawn from")
        if (self.kind == "standardized") != (self.dd_path is not None):
            raise ValueError(f"element {self.key!r}: a standardized element, and only one, names its dd_path")

    @property
    def source_nodes(self):
        return tuple(WORKFLOWS[spec].node(node) for spec, node in self.sources)

    @property
    def apis(self) -> Tuple[str, ...]:
        names = [n.api for n in self.source_nodes if n.api] + ([self.api] if self.api else [])
        return tuple(dict.fromkeys(names))

    @property
    def ids(self) -> Tuple[str, ...]:
        return tuple(dict.fromkeys(n.ids for n in self.source_nodes if n.ids))

    @property
    def ids_name(self) -> str:
        return self.dd_path.split(".")[0] if self.dd_path else ""


E = Element
_EFIT, _CHEASE, _TOKAMAKER, _ANALYTIC = "magnetic_efit", "chease_coupling", "tokamaker_coupling", "analytic_mhd_equilibrium"
_STAB, _GPEC = "dcon_rdcon_stability", "gpec_plasma_response"
_HIER, _GK, _NEOCMP, _PPI = "transport_hierarchy", "tglf_cgyro_local_transport", "neoclassical_model_comparison", \
    "plasma_parameter_inference"
_EQ_LEAF = "equilibrium.time_slice.:.profiles_2d.:.psi"

ELEMENTS: Tuple[Element, ...] = (
    # ---------------------------------------------------------------- equilibrium: inputs, beside their methods
    E("an_shape", "Shape, profile class", "prior", "equilibrium", ((_ANALYTIC, "shape"),),
      detail=r"R_0,\ a,\ \kappa,\ \delta"),
    E("tm_targets", "Targets", "measured", "equilibrium", ((_TOKAMAKER, "targets"),), detail=r"I_p,\ F_0"),
    E("tm_profiles", "Profile shape", "prior", "equilibrium", ((_TOKAMAKER, "profiles"),), detail=r"p',\ FF'"),
    # read by both the free-boundary model and the reconstruction: drawn once, between them
    E("eq_coils", "Coil currents", "measured", "equilibrium", ((_TOKAMAKER, "coils"), (_EFIT, "pf")),
      detail=r"I_{\mathrm{PF}}"),
    E("eq_machine", "Machine", "machine", "equilibrium", ((_TOKAMAKER, "geometry"), (_EFIT, "geometry")),
      detail=r"\mathrm{wall,\ coils}"),
    E("ef_magnetics", "Magnetics", "measured", "equilibrium", ((_EFIT, "magnetics"),),
      detail=r"I_p,\ B_p,\ \psi_{\mathrm{FL}}"),
    E("ef_kinetic", "Kinetic profiles", "measured", "equilibrium", (("kinetic_efit", "ts"),), detail=r"T_e,\ n_e",
      note="kinetic EFIT only: Thomson pressure points constrain the reconstruction"),
    E("ef_eddy", "Vessel currents", "derived", "equilibrium", ((_EFIT, "eddy"),), detail=r"I_{\mathrm{vessel}}",
      note="modelled from the coil currents and magnetics, then fitted alongside them"),
    E("ef_basis", "Profile basis", "convention", "equilibrium", ((_EFIT, "basis"),), detail=r"p',\ FF'\ \mathrm{basis}",
      note="the polynomial terms the reconstruction is allowed"),
    E("ch_cocos", "COCOS", "convention", "equilibrium", ((_CHEASE, "cocos_in"),), detail=r"\mathrm{COCOS}\ 11 \to 2"),
    # ---------------------------------------------------------------- equilibrium: methods
    E("analytic", "Analytic Grad-Shafranov", "solver", "equilibrium", ((_ANALYTIC, "solovev"), (_ANALYTIC, "gf")),
      method="Solov'ev, Cerfon-Freidberg, Guazzotto-Freidberg"),
    E("tokamaker", "Free-boundary model", "solver", "equilibrium", ((_TOKAMAKER, "solve"),), method="TokaMaker",
      basis="measured coils, assumed profiles"),
    E("efit", "Reconstruction", "solver", "equilibrium", ((_EFIT, "efit"), ("kinetic_efit", "efit")), method="EFIT",
      basis="magnetic or kinetic"),
    E("chease", "Fixed-boundary equilibrium", "solver", "equilibrium", ((_CHEASE, "chease"),), method="CHEASE",
      basis="re-solves a reconstruction",
      note="accepts any GEQDSK or ODS equilibrium; the coupling applies it to a reconstruction (by convention, "
           "not enforced, unlike the transport resolver)"),
    # ---------------------------------------------------------------- equilibrium: one IDS, four kinds of state
    E("eq_analytic", "analytic", "standardized", "equilibrium", ((_ANALYTIC, "equilibrium"),), dd_path=_EQ_LEAF,
      note="reaches IMAS through EquilibriumData -> GEQDSK -> to_omas; the Solov'ev fit is not written"),
    E("eq_model", "free-boundary model", "standardized", "equilibrium", ((_TOKAMAKER, "equilibrium"),),
      dd_path=_EQ_LEAF, note="a modelled state: measured coil currents with assumed profile shapes"),
    E("eq_reconstructed", "reconstructed", "standardized", "equilibrium", ((_EFIT, "equilibrium"),),
      dd_path=_EQ_LEAF, note="the experimental state; the transport resolver keys on its time and lineage"),
    E("eq_refined", "refined", "standardized", "equilibrium", ((_CHEASE, "equilibrium"),), dd_path=_EQ_LEAF,
      note="the reconstructed boundary and profiles re-solved at fixed boundary"),
    # ---------------------------------------------------------------- MHD
    E("dcon_in", "Mode numbers, wall", "convention", "mhd", ((_STAB, "modes"), (_STAB, "wall")),
      detail=r"n,\ \mathrm{far\ wall}"),
    E("rdcon_in", "Resistive layers", "derived", "mhd", ((_STAB, "layer"),), detail=r"\eta(T_e, Z_{\mathrm{eff}})"),
    E("gpec_coils", "3-D coils", "machine", "mhd", ((_GPEC, "coil_geometry"),),
      api="vaft.machine_mapping.coils_non_axisymmetric.coils_non_axisymmetric", detail=r"\mathrm{3\text{-}D\ coils}"),
    E("gpec_drive", "Excitation", "prior", "mhd", ((_GPEC, "excitation"),), detail=r"I_k,\ n,\ \phi_0"),
    E("dcon", "Ideal stability", "solver", "mhd", ((_STAB, "dcon"),), method="DCON", basis="energy principle"),
    E("rdcon", "Resistive stability", "solver", "mhd", ((_STAB, "rdcon"),), method="RDCON", basis="inner-layer matching"),
    E("gpec", "Plasma response", "solver", "mhd", ((_GPEC, "gpec"),), method="GPEC", basis="ideal response"),
    E("ml_dcon", "Ideal mode", "standardized", "mhd", ((_STAB, "mhd_linear"),),
      api="vaft.machine_mapping.mhd_linear.mhd_linear", dd_path="mhd_linear.time_slice.:.toroidal_mode.:.energy_perturbed",
      detail=r"\delta W_n\ (\mathrm{normalized})",
      note="a dimensionless eigenvalue in a field documented in J; the mismatch is stated in code.parameters"),
    E("ntms_rdcon", "Tearing index", "standardized", "mhd", ((_STAB, "ntms"),),
      api="vaft.machine_mapping.mhd_linear.mhd_linear", dd_path="ntms.time_slice.:.mode.:.deltaw.:.value",
      detail=r"\Delta'_{m/n}\ (\mathrm{diagonal})", note="the classical single-surface value only"),
    E("ml_gpec", "Perturbed field", "standardized", "mhd", ((_GPEC, "mhd_linear"),),
      api="vaft.machine_mapping.gpec_ideal.gpec_ideal",
      dd_path="mhd_linear.time_slice.:.toroidal_mode.:.plasma.b_field_perturbed.coordinate1.real",
      detail=r"\delta\mathbf B\cdot\nabla\psi", note="the total and plasma fields; the vacuum part (total minus plasma) is not written"),
    E("dcon_native", "Energy split", "native_result", "mhd", ((_STAB, "dw"),),
      detail=r"\delta W_p,\ \delta W_v,\ \xi_{m,n}"),
    E("rdcon_native", "Matching data", "native_result", "mhd", ((_STAB, "dprime"),),
      detail=r"\Delta'\ \mathrm{matrix}"),
    E("gpec_native", "Resonant response", "native_result", "mhd", ((_GPEC, "resonant"),),
      detail=r"\Phi_{\mathrm{res}},\ w_{\mathrm{isl}}"),
    # ---------------------------------------------------------------- transport
    E("sr_geometry", "Equilibrium", "reconstructed", "transport", ((_NEOCMP, "state"),), detail=r"q,\ f_t,\ B_0"),
    E("sr_closures", "Closures", "prior", "transport", ((_PPI, "zeff"), (_PPI, "ti_ratio")),
      detail=r"Z_{\mathrm{eff}},\,T_i",
      note="a T_i stored in the ODS (measured or a recorded ratio) and Z_eff from the ODS or the caller"),
    # read by the closed-form fits and by the resolver: drawn once, between them
    E("tr_profiles", "Kinetic profiles", "measured", "transport", ((_GK, "cp"),), detail=r"n_e,\ T_e"),
    E("tr_closures", "Closures", "prior", "transport", ((_PPI, "ti_ratio"), (_PPI, "zeff"), (_GK, "rotation")),
      detail=r"T_i\ \mathrm{rule},\ Z_{\mathrm{eff}},\ \gamma_E = 0"),
    E("state", "Resolved local state", "derived", "transport", ((_HIER, "state"),),
      api="vaft.process.transport_state.ResolvedTransportState.identity",
      detail=r"T_s,\ n_s,\ a/L_{T,n},\ q,\ \hat s,\ \hat\nu"),
    E("sauter_redl", "Neoclassical", "derived", "transport", ((_HIER, "neo_fit"),), method="Sauter / Redl", basis="closed-form fits"),
    E("neo", "Neoclassical", "solver", "transport", ((_HIER, "neo_code"),), method="NEO", basis="drift-kinetic"),
    E("classical", "Classical", "derived", "transport", ((_HIER, "classical"),), method="Braginskii",
      basis="magnetized"),
    E("tglf", "Turbulent", "solver", "transport", ((_HIER, "turb_ql"),), method="TGLF", basis="QL gyrofluid"),
    # gyrokinetic, linear or nonlinear; only the linear eigenmodes are drawn mapped (see the module docstring)
    E("cgyro", "Gyrokinetic", "solver", "transport", ((_GK, "cgyro"),), method="CGYRO", basis="gyrokinetic"),
    E("sr_native", "Closed-form result", "native_result", "transport", ((_HIER, "neo_fit"),),
      detail=r"j_{\mathrm{bs}},\ \sigma_\parallel", note="returned to the caller; no core_transport writer"),
    E("ct_neo", "Neoclassical fluxes", "standardized", "transport", ((_HIER, "ct_neo"), (_NEOCMP, "mapping")),
      dd_path="core_transport.model.:.profiles_1d.:.electrons.energy.flux", detail=r"\Gamma_s,\ Q_s",
      note="fluxes in core_transport (identifier.index 5); j_bs and sigma_par in core_profiles"),
    E("ct_classical", "Classical fluxes", "standardized", "transport", ((_HIER, "ct_classical"),),
      dd_path="core_transport.model.:.profiles_1d.:.electrons.energy.flux", detail=r"q_{\perp,e},\ q_{\perp,i}",
      note="a private identifier.index (-1): the DD has no classical entry"),
    E("ct_tglf", "Turbulent fluxes", "standardized", "transport", ((_HIER, "ct_tglf"),),
      dd_path="core_transport.model.:.profiles_1d.:.electrons.energy.flux", detail=r"Q_s,\ \Gamma_s"),
    E("gk_cgyro", "Linear eigenmodes", "standardized", "transport", ((_GK, "gk_local"),),
      dd_path="gyrokinetics_local.linear.wavevector.:.eigenmode.:.growth_rate_norm", detail=r"\gamma,\ \omega",
      note="the writer exists; no pipeline stage stores gyrokinetics_local yet (#2020)"),
)
ELEMENT: Dict[str, Element] = {e.key: e for e in ELEMENTS}

#: (source, target, connection, label); a source or target is an element or a region. Flows are arrows;
#: a mapping or native edge is drawn by placing the result under its source, in the band of that kind
EDGES: Tuple[Tuple[str, str, str, str], ...] = (
    # equilibrium: what each method reads
    ("an_shape", "analytic", "flow", ""),
    ("tm_targets", "tokamaker", "flow", ""),
    ("tm_profiles", "tokamaker", "flow", ""),
    ("eq_coils", "tokamaker", "flow", ""),
    ("eq_coils", "efit", "flow", ""),
    ("eq_machine", "tokamaker", "flow", ""),
    ("eq_machine", "efit", "flow", ""),
    ("ef_magnetics", "efit", "flow", ""),
    ("ef_kinetic", "efit", "flow", ""),
    ("ef_eddy", "efit", "flow", ""),
    ("ef_basis", "efit", "flow", ""),
    ("ch_cocos", "chease", "flow", ""),
    ("eq_reconstructed", "chease", "flow", ""),
    ("analytic", "eq_analytic", "mapping", ""),
    ("tokamaker", "eq_model", "mapping", ""),
    ("efit", "eq_reconstructed", "mapping", ""),
    ("chease", "eq_refined", "mapping", ""),
    # the representation, consumed downstream: each model still checks its own inputs
    ("hub", "mhd", "flow", "an eligible equilibrium, per-model inputs"),
    ("eq_reconstructed", "state", "flow", "reconstructed only (either lineage)"),
    # MHD: three complementary analyses, no alternative among them; GPEC needs DCON's eigenfunctions
    ("dcon_in", "dcon", "flow", ""),
    ("rdcon_in", "rdcon", "flow", ""),
    ("gpec_coils", "gpec", "flow", ""),
    ("gpec_drive", "gpec", "flow", ""),
    ("dcon", "gpec", "flow", ""),
    ("dcon", "ml_dcon", "mapping", ""),
    ("rdcon", "ntms_rdcon", "mapping", ""),
    ("gpec", "ml_gpec", "mapping", ""),
    ("dcon", "dcon_native", "native", ""),
    ("rdcon", "rdcon_native", "native", ""),
    ("gpec", "gpec_native", "native", ""),
    # transport: closed-form neoclassical fits read the IMAS state; the rest read the resolved state
    ("sr_geometry", "sauter_redl", "flow", ""),
    ("sr_closures", "sauter_redl", "flow", ""),
    ("tr_profiles", "sauter_redl", "flow", ""),
    ("tr_profiles", "state", "flow", ""),
    ("tr_closures", "state", "flow", ""),
    ("state", "neo", "flow", ""),
    ("state", "classical", "flow", ""),
    ("state", "tglf", "flow", ""),
    ("state", "cgyro", "flow", ""),
    ("sauter_redl", "sr_native", "native", ""),
    ("neo", "ct_neo", "mapping", ""),
    ("classical", "ct_classical", "mapping", ""),
    ("tglf", "ct_tglf", "mapping", ""),
    ("cgyro", "gk_cgyro", "mapping", ""),
)
#: methods that are alternatives for a related quantity: one dash-dot enclosure per set, within one domain.
#: CHEASE refines a reconstruction and DCON / GPEC / RDCON answer different questions, so neither is here
ALTERNATIVES: Tuple[Tuple[str, ...], ...] = (
    ("analytic", "tokamaker", "efit"),
    ("sauter_redl", "neo"),
    ("tglf", "cgyro"),
)
#: the models whose runs one ``state_identity`` keys (``run_identity``; the atlas joins NEO, TGLF and classical
#: rows on it, ``build_atlas.py``: ``join_key_across_models``); it keys runs, not the IDS entries they write
STATE_IDENTITY_JOIN: Tuple[str, ...] = ("neo", "classical", "tglf", "cgyro")

# ---------------------------------------------------------------------------
# layout [cm]: x from 0 to _WIDTH, y downward from 0
# ---------------------------------------------------------------------------

_WIDTH = 28.0
#: (x0, x1, y_top, y_bottom) of each region
_REGION_BOUNDS: Dict[str, Tuple[float, float, float, float]] = {
    "equilibrium": (0.0, _WIDTH, 0.0, -3.55),
    "hub": (0.0, _WIDTH, -3.95, -5.45),
    "mhd": (0.0, 11.6, -6.25, -10.9),
    "transport": (12.1, _WIDTH, -6.25, -10.9),
    "imas": (0.0, _WIDTH, -11.3, -13.25),
    "native": (0.0, _WIDTH, -13.55, -15.15),
}
_EQ_X = (3.0, 9.9, 18.0, 24.7)  # analytic, TokaMaker, EFIT, CHEASE: EFIT clear of the transport title below
_MHD_X = (1.95, 5.95, 9.75)  # DCON, GPEC (reads DCON's eigenfunctions, so beside it), RDCON
_TR_X = (13.85, 16.8, 20.0, 23.2, 26.15)  # Sauter / Redl, NEO, classical, TGLF, CGYRO
_EQ_W, _MHD_W, _TR_W = 5.2, 3.45, 2.6
_ROW_EQ_CHIPS, _ROW_EQ, _ROW_HUB = -0.95, -2.45, -4.4
_ROW_CHIPS, _ROW_STATE, _ROW_METHOD, _ROW_IDS, _ROW_NATIVE = -7.35, -8.45, -9.95, -12.45, -14.5
#: key -> (x, y, width, height)
_PLACE: Dict[str, Tuple[float, float, float, float]] = {
    "an_shape": (_EQ_X[0], _ROW_EQ_CHIPS, 3.0, 0.55),
    "tm_targets": (_EQ_X[1] - 1.75, _ROW_EQ_CHIPS, 1.5, 0.55),
    "tm_profiles": (_EQ_X[1] + 0.05, _ROW_EQ_CHIPS, 1.6, 0.55),
    # stacked between the two methods that read them, so their four arrows fan out without crossing
    "eq_machine": (13.95, -0.55, 1.9, 0.45),
    "eq_coils": (13.95, -1.15, 1.2, 0.45),
    "ef_magnetics": (16.6, _ROW_EQ_CHIPS, 2.1, 0.55),
    "ef_kinetic": (18.35, _ROW_EQ_CHIPS, 1.2, 0.55),
    "ef_eddy": (19.75, _ROW_EQ_CHIPS, 1.2, 0.55),
    "ef_basis": (21.75, _ROW_EQ_CHIPS, 2.4, 0.55),
    "ch_cocos": (_EQ_X[3], _ROW_EQ_CHIPS, 2.6, 0.55),
    "analytic": (_EQ_X[0], _ROW_EQ, _EQ_W, 1.45),
    "tokamaker": (_EQ_X[1], _ROW_EQ, _EQ_W, 1.45),
    "efit": (_EQ_X[2], _ROW_EQ, _EQ_W, 1.45),
    "chease": (_EQ_X[3], _ROW_EQ, _EQ_W, 1.45),
    "eq_analytic": (_EQ_X[0], _ROW_HUB, 4.6, 0.6),
    "eq_model": (_EQ_X[1], _ROW_HUB, 4.6, 0.6),
    "eq_reconstructed": (_EQ_X[2], _ROW_HUB, 4.6, 0.6),
    "eq_refined": (_EQ_X[3], _ROW_HUB, 4.6, 0.6),
    "dcon_in": (_MHD_X[0], _ROW_STATE, 2.6, 0.55),
    "gpec_coils": (_MHD_X[1] - 0.85, _ROW_STATE, 1.5, 0.55),
    "gpec_drive": (_MHD_X[1] + 0.85, _ROW_STATE, 1.6, 0.55),
    "rdcon_in": (_MHD_X[2], _ROW_STATE, 2.8, 0.55),
    "dcon": (_MHD_X[0], _ROW_METHOD, _MHD_W, 1.45),
    "gpec": (_MHD_X[1], _ROW_METHOD, _MHD_W, 1.45),
    "rdcon": (_MHD_X[2], _ROW_METHOD, _MHD_W, 1.45),
    "ml_dcon": (_MHD_X[0], _ROW_IDS, _MHD_W, 1.05),
    "ml_gpec": (_MHD_X[1], _ROW_IDS, _MHD_W, 1.05),
    "ntms_rdcon": (_MHD_X[2], _ROW_IDS, _MHD_W, 1.05),
    "dcon_native": (_MHD_X[0], _ROW_NATIVE, _MHD_W, 0.75),
    "gpec_native": (_MHD_X[1], _ROW_NATIVE, _MHD_W, 0.75),
    "rdcon_native": (_MHD_X[2], _ROW_NATIVE, _MHD_W, 0.75),
    "sr_geometry": (13.2, _ROW_CHIPS, 1.6, 0.55),
    "sr_closures": (12.85, _ROW_STATE, 1.3, 0.55),
    "tr_profiles": (14.85, _ROW_CHIPS, 1.3, 0.55),
    "tr_closures": (23.6, _ROW_CHIPS, 3.9, 0.55),
    "state": (21.475, _ROW_STATE, 11.95, 0.85),
    "sauter_redl": (_TR_X[0], _ROW_METHOD, _TR_W, 1.45),
    "neo": (_TR_X[1], _ROW_METHOD, _TR_W, 1.45),
    "classical": (_TR_X[2], _ROW_METHOD, _TR_W, 1.45),
    "tglf": (_TR_X[3], _ROW_METHOD, _TR_W, 1.45),
    "cgyro": (_TR_X[4], _ROW_METHOD, _TR_W, 1.45),
    "sr_native": (_TR_X[0], _ROW_NATIVE, _TR_W, 0.75),
    "ct_neo": (_TR_X[1], _ROW_IDS, _TR_W, 1.05),
    "ct_classical": (_TR_X[2], _ROW_IDS, _TR_W, 1.05),
    "ct_tglf": (_TR_X[3], _ROW_IDS, _TR_W, 1.05),
    "gk_cgyro": (_TR_X[4], _ROW_IDS, 2.9, 1.05),  # wider, so its long IDS name is barely shrunk
}
_EQ_TOP = _ROW_EQ + 0.725 + 0.08
#: edges drawn along given points, start and end included: a chip read by two methods leaves from either side
#: of its base so neither arrow passes under its neighbour, and the refinement reads the reconstructed entry
_ROUTES: Dict[Tuple[str, str], Tuple[Tuple[float, float], ...]] = {
    ("eq_machine", "tokamaker"): ((13.0, -0.78), (11.75, _EQ_TOP)),
    ("eq_coils", "tokamaker"): ((13.45, -1.38), (12.25, _EQ_TOP)),
    ("eq_coils", "efit"): ((14.45, -1.38), (15.65, _EQ_TOP)),
    ("eq_machine", "efit"): ((14.9, -0.78), (16.15, _EQ_TOP)),
    ("eq_reconstructed", "chease"): ((_EQ_X[2] + 2.38, _ROW_HUB), (21.35, _ROW_HUB), (21.35, _ROW_EQ),
                                     (_EQ_X[3] - 0.5 * _EQ_W - 0.08, _ROW_EQ)),
    ("hub", "mhd"): ((_MHD_X[1], -5.53), (_MHD_X[1], -6.17)),
    ("eq_reconstructed", "state"): ((_EQ_X[2], _ROW_HUB - 0.38), (_EQ_X[2], _ROW_STATE + 0.505)),
    ("sr_geometry", "sauter_redl"): ((13.65, _ROW_CHIPS - 0.355), (13.65, _ROW_METHOD + 0.805)),
    ("sr_closures", "sauter_redl"): ((12.85, _ROW_STATE - 0.355), (12.85, _ROW_METHOD + 0.805)),
    ("tr_profiles", "sauter_redl"): ((14.35, _ROW_CHIPS - 0.355), (14.35, _ROW_METHOD + 0.805)),
    ("tr_profiles", "state"): ((15.45, _ROW_CHIPS - 0.355), (15.8, _ROW_STATE + 0.505)),
    # the resolved state feeds each model straight down
    **{("state", m): ((x, _ROW_STATE - 0.505), (x, _ROW_METHOD + 0.805))
       for m, x in zip(("neo", "classical", "tglf", "cgyro"), _TR_X[1:])},
}
#: where a labelled edge carries its label: beside it, in the gap between the representation and its consumers
_LABEL_AT: Dict[Tuple[str, str], Tuple[float, float]] = {
    ("hub", "mhd"): (_MHD_X[1] + 0.15, -5.85),
    ("eq_reconstructed", "state"): (_EQ_X[2] + 0.15, -5.85),
}
#: the margin between an enclosure and the boxes it encloses [cm]
_ENCLOSE = 0.13


def _region_box(key: str) -> Box:
    x0, x1, y0, y1 = _REGION_BOUNDS[key]
    return Box(0.5 * (x0 + x1), 0.5 * (y0 + y1), x1 - x0, y0 - y1, ())


def _text(element: Element) -> str:
    """Bold role, the method under it, then its quantities; an IDS result leads with its IDS."""
    detail = f"${element.detail}$" if element.detail else ""
    if element.kind == "standardized" and element.domain == "equilibrium":
        return "{\\small " + escape_latex(element.label) + "}"
    if element.kind == "standardized":  # a long IDS name shrinks to the box rather than overflow it
        fit = f"\\vaftfit{{{_PLACE[element.key][2] - 0.4:.2f}cm}}"
        return (fit + "{\\small\\texttt{" + escape_latex(element.ids_name) + "}}\\\\[1pt]{\\small " + detail + "}")
    if element.method:
        text = "\\textbf{" + escape_latex(element.label) + "}\\\\[1pt]{\\small " + escape_latex(element.method) + "}"
        if element.basis:
            text += "\\\\{\\small\\itshape\\color{black!65}" + escape_latex(element.basis) + "}"
        return text
    if element.key == "state":
        return ("\\textbf{" + escape_latex(element.label) + "}\\ \\ {\\small " + detail
                + "}\\\\{\\small \\texttt{state\\_identity} keys every run below; results are not merged}")
    return "{\\small " + detail + "}"


def _region_items(key: str, title: str, subtitle: str, labels: bool) -> List:
    x0, x1, y0, y1 = _REGION_BOUNDS[key]
    outline = [(x0, y1), (x1, y1), (x1, y0), (x0, y0)]
    if key == "hub":
        style = CONNECTION_STYLE["mapping"].replace("rounded corners=6pt", "rounded corners=8pt")
    elif key in ("imas", "native"):
        style = CONNECTION_STYLE["mapping" if key == "imas" else "native"]
    else:
        style = "concept group"
    items: List = [Polyline.of(outline, style, role=f"region:{key}", closed=True)]
    if labels:
        if key == "hub":  # its title sits under the four entries, clear of the arrows entering them
            at = (x0 + 0.25, y1 + 0.42)
        else:
            at = (x0 + 0.2, y0 - 0.32)
        text = "\\textbf{" + escape_latex(title) + "}"
        if key == "hub":
            text += " (\\texttt{equilibrium})"
        items.append(Label(at, text + "\\quad{\\small\\itshape\\color{black!60}" + escape_latex(subtitle) + "}",
                           "concept group title", anchor="west", role=f"region title:{key}"))
    return items


def _enclosure(keys: Tuple[str, ...], boxes: Dict[str, Box]) -> Polyline:
    """The dash-dot outline around a set of alternative methods."""
    x0 = min(boxes[k].x - 0.5 * boxes[k].width for k in keys) - _ENCLOSE
    x1 = max(boxes[k].x + 0.5 * boxes[k].width for k in keys) + _ENCLOSE
    y0 = max(boxes[k].y + 0.5 * boxes[k].height for k in keys) + _ENCLOSE
    y1 = min(boxes[k].y - 0.5 * boxes[k].height for k in keys) - _ENCLOSE
    return Polyline.of([(x0, y1), (x1, y1), (x1, y0), (x0, y0)], CONNECTION_STYLE["alternative"],
                       role="alternative:" + ",".join(keys), closed=True)


def connection_legend(left: float, right: float, y: float, connections: Tuple[str, ...] = CONNECTIONS,
                      label: Optional[Dict[str, str]] = None, label_style: str = "concept annotation",
                      styles: Optional[Dict[str, str]] = None, arrows: Tuple[str, ...] = ("flow",)) -> List:
    """One sample per connection, centred between ``left`` and ``right`` at ``y``: an arrow for each of
    ``arrows``, a small dash-dot enclosure for alternatives, a band swatch for a mapped or a native result.
    ``styles`` adds or overrides a connection's style (a figure's own arrow kinds)."""
    label = label or CONNECTION_LABEL
    styles = {**CONNECTION_STYLE, **(styles or {})}
    items: List = []
    widths = {c: 1.95 + 0.155 * len(label[c]) for c in connections}  # sample, label, gap
    x = 0.5 * (left + right) - 0.5 * sum(widths.values())
    for c in connections:
        if c in arrows:
            items.append(Arrow((x, y), (x + 1.2, y), styles[c], role=f"legend:{c}"))
        else:
            swatch = [(x, y - 0.22), (x + 1.2, y - 0.22), (x + 1.2, y + 0.22), (x, y + 0.22)]
            items.append(Polyline.of(swatch, styles[c].replace("rounded corners=6pt", "rounded corners=3pt"),
                                     role=f"legend:{c}", closed=True))
        items.append(Label((x + 1.35, y), escape_latex(label[c]), label_style, anchor="west",
                           role=f"legend:{c}"))
        x += widths[c]
    return items


def cross_domain_scientific_workflows(*, labels: bool = True) -> Diagram:
    """Equilibrium, MHD and transport calculations joined through shared representations of a plasma state.

    The technical figure: each method with the inputs it reads beside it.
    Three alternative equilibrium routes -- analytic Grad-Shafranov, a
    free-boundary model and EFIT reconstruction -- and the fixed-boundary
    refinement of a reconstruction write one IMAS ``equilibrium``
    representation whose entries keep their own states. MHD stability and
    response (DCON, GPEC, RDCON) and transport (classical, two neoclassical,
    two turbulent models) are complementary consumers of an equilibrium; the
    transport runs that read the resolved local state are keyed by its
    ``state_identity``. Each result sits under the operation that produced
    it, in the IMAS band when it is mapped to an IDS and in the native band
    when the solver keeps it.

    The figure draws implemented interfaces, not executions; see
    :mod:`vaft.diagram._cross_domain` for what is left out and why, and
    :func:`~vaft.diagram.cross_domain_scientific_architecture` for the
    conceptual summary.
    """
    if not isinstance(labels, bool):
        raise ValueError(f"labels must be True or False, not {labels!r}")
    items: List = []
    for key, title, subtitle in REGIONS:
        items += _region_items(key, title, subtitle, labels)
    boxes: Dict[str, Box] = {key: _region_box(key) for key, *_ in REGIONS}
    for element in ELEMENTS:
        x, y, w, h = _PLACE[element.key]
        b = box(x, y, w, h, _text(element), style=kind_style(element.kind), text_style="concept plain",
                role=f"element:{element.key}", latex=True)
        boxes[element.key] = b
        items += list(b.items)
    for keys in ALTERNATIVES:
        items.append(_enclosure(keys, boxes))
    for src, dst, kind, text in EDGES:
        if kind != "flow":  # a result is placed under its source, in its band: no line to draw
            continue
        style, role = CONNECTION_STYLE[kind], f"edge:{kind}:{src}->{dst}"
        if (src, dst) in _ROUTES:
            path = list(_ROUTES[(src, dst)])
        else:
            a = connector(boxes[src], boxes[dst], role=role)
            path = [a.start, a.end]
        if len(path) > 2:  # a routed arrow keeps its head: the style carries it
            items.append(Polyline.of(path, style, role=role))
        else:
            items.append(Arrow(path[0], path[1], style, role=role))
        if text and labels:
            items.append(Label(_LABEL_AT[(src, dst)], escape_latex(text), "concept annotation", anchor="west",
                               role=f"edge label:{src}->{dst}"))
    if labels:
        mapped = {s for s, _, c, _ in EDGES if c == "mapping"}
        for e in ELEMENTS:  # a method that writes no IDS says so in the band, so the gap is not read as an omission
            if e.kind in ("solver", "derived") and e.method and e.domain != "equilibrium" and e.key not in mapped:
                items.append(Label((_PLACE[e.key][0], _ROW_IDS), "{\\small\\itshape\\color{black!60}no IMAS writer}",
                                   "concept plain", role=f"no writer:{e.key}"))
        used = [k for k in KINDS if any(e.kind == k for e in ELEMENTS)]
        legend, y = kind_legend(used, 0.0, _WIDTH, _REGION_BOUNDS["native"][3] - 0.75)
        items += legend
        items += connection_legend(0.0, _WIDTH, y - 0.1)
    else:
        items = [it for it in items if not isinstance(it, Label)]
    model = {"elements": ELEMENTS, "edges": EDGES, "alternatives": ALTERNATIVES, "join": STATE_IDENTITY_JOIN,
             "boxes": {k: (b.x, b.y, b.width, b.height) for k, b in boxes.items()}}
    return Diagram("cross_domain_scientific_workflows", Scene(tuple(items)), model=model)


def cross_domain_markdown() -> str:
    """The documentation table of the figure, as Diagrams.md carries it (a test keeps the two equal)."""
    def cell(text):
        return text.replace("|", "\\|")

    lines = ["| Element | Domain | Kind | Method | Drawn from | API | IDS leaf | Note |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    from ._workflow import KIND_LABEL

    for e in ELEMENTS:
        sources = ", ".join(f"`{spec}.{node}`" for spec, node in e.sources)
        apis = ", ".join(f"`{a}`" for a in e.apis)
        leaf = f"`{e.dd_path}`" if e.dd_path else ""
        lines.append(f"| {cell(e.label)} | {e.domain} | {KIND_LABEL[e.kind]} | {cell(e.method)} | {sources} | {apis} "
                     f"| {leaf} | {cell(e.note)} |")
    lines += ["", "| From | To | Connection |", "| --- | --- | --- |"]
    for src, dst, kind, _ in EDGES:
        lines.append(f"| `{src}` | `{dst}` | {CONNECTION_LABEL[kind]} |")
    for keys in ALTERNATIVES:
        lines.append(f"| {', '.join(f'`{k}`' for k in keys)} | | {CONNECTION_LABEL['alternative']} |")
    return "\n".join(lines)

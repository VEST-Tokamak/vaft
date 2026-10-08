"""Dataclasses shared by the GPEC-suite orchestration and solver modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence

if TYPE_CHECKING:
    from vaft.machine_mapping.coils_non_axisymmetric_geometry import CoilSet3D

    from ..execution import ExecutionBackend
    from ._coil_input import CoilInputSpec

GPEC_HOME_ENV = "GPECHOME"
DEFAULT_MODULES = ("dcon", "rdcon", "stride", "gpec")
DEFAULT_MODES = (1, 2)
SUPPORTED_MODULES = frozenset(DEFAULT_MODULES)
STABILITY_MODULES = frozenset(("dcon", "rdcon", "stride"))


@dataclass(frozen=True)
class DCONOptions:
    """DCON-specific namelist overrides.

    ``mer_flag``/``bal_flag``/``thmax0`` gate DCON's *local* stability criteria,
    and they matter for more than reproducibility: DCON zero-fills the whole
    local-stability spline before running either scan (``dcon/dcon.F:148-160``),
    so a criterion that was never evaluated is written to the netCDF as exactly
    zero -- which is also its marginal value. Recording what was requested is
    what lets VAFT tell "marginally stable everywhere" from "never computed".

    ``bal_flag`` defaults to the packaged namelist's ``f``. That is deliberately
    unchanged here, but note it makes DCON the odd one out: ``rdcon.in`` and
    ``stride.in`` both ship ``bal_flag=t``.

    ``con_flag`` decides whether ideal GPEC can compute a penetration threshold at
    all.  GPEC computes every resonant-field quantity inside ``IF (singfld_flag)``
    and turns that flag **off** when ``con_flag`` is true (``gpec/gpec.f:560-565``),
    warning ``singfld_flag not supported with con_flag`` and writing no
    rational-surface block -- so a case that asks for ``singthresh_*`` against a
    template shipping ``con_flag=t`` gets zeros, which is also what an unattainable
    threshold looks like.  The packaged ``dcon.in`` ships ``t``, and so does GPEC's
    own ``input/dcon.in``.

    ``None`` leaves whatever the template holds, which is why it is the default:
    ``con_flag`` is not a reporting switch -- it continues the integration through
    the singular layers instead of applying the ideal jump condition at each one --
    and a machine layer with its own ``templates_dir`` may deliberately ship ``f``.
    Writing a default over that would change its eigenvalues silently.  Set it
    explicitly (``con_flag=False``) for a threshold run, and
    :func:`~vaft.code.gpec.validate_threshold_inputs` reads the *prepared file*
    either way, so the check holds whichever template supplied the value.
    """

    sas_flag: bool = False
    qhigh: float = 20.2
    psiedge: float = 1.0
    mer_flag: bool = True
    bal_flag: bool = False
    thmax0: float = 1.0
    con_flag: Optional[bool] = None


@dataclass(frozen=True)
class RDCONOptions:
    """RDCON-specific namelist overrides.

    ``rmatch`` wants one resistivity and one mass density *per rational
    surface*, and the packaged ``rmatch.in`` supplies a single scalar, so it
    stops with ``eta requires N non-zero elements`` before writing
    ``delta.out`` or any inner-layer solution (#716). Delta-prime is unaffected -- RDCON computes it
    and ``rmatch`` does not -- so this is opt-in rather than required.

    Supplying ``t_e``/``n_e`` fills both arrays from the plasma: after RDCON
    reports where its rational surfaces are, the profiles are evaluated there
    and written into ``rmatch.in``. Reading RDCON's own surfaces rather than
    re-deriving them is what makes the arrays line up with the ``msing`` the
    code will clamp to.

    ``None`` throughout preserves the packaged template verbatim.
    """

    #: Electron temperature [eV] and density [m^-3] of the bulk plasma, with
    #: the normalized poloidal flux they are given on. All three or none.
    t_e: Optional[Sequence[float]] = None
    n_e: Optional[Sequence[float]] = None
    psi_norm: Optional[Sequence[float]] = None

    #: Passed through to :func:`vaft.process.equilibrium.resistive_layer_parameters`.
    #: ``z_eff`` and ``ln_lambda`` are an explicit *assumed value* for the NRL
    #: parallel Spitzer resistivity, not a derived one (#1188): they only reach
    #: ``rmatch.in``'s eta/massden and leave Delta-prime untouched. They have
    #: no default: kinetic profiles without both are refused by name in
    #: ``__post_init__`` rather than silently evaluated at Z_eff = 2,
    #: ln Lambda = 17 (Refs #1188). Pass a measured or inferred Z_eff (e.g.
    #: Lane Z's resistive estimate, #1214) when the RMATCH growth rates are
    #: to be read quantitatively.
    ion_mass_amu: float = 1.0
    z_eff: Optional[float] = None
    ln_lambda: Optional[float] = None

    def __post_init__(self) -> None:
        # Checked here, where the caller's mistake is, rather than after RDCON
        # has already run: a malformed triple used to raise out of the suite
        # between the solver and its companion, and a descending coordinate
        # was interpolated as if it were sorted.
        if not self.has_kinetic_profiles:
            return
        missing = [name for name in ("z_eff", "ln_lambda") if getattr(self, name) is None]
        if missing:
            raise ValueError(
                "RDCONOptions with kinetic profiles needs an explicit "
                + " and ".join(missing)
                + " for the Spitzer resistivity written to rmatch.in; there is no "
                "hidden Z_eff = 2 / ln Lambda = 17 fallback (#1188)"
            )
        sizes = {
            name: len(getattr(self, name)) for name in ("psi_norm", "t_e", "n_e")
        }
        if len(set(sizes.values())) != 1:
            raise ValueError(
                "RDCONOptions psi_norm, t_e and n_e must have one length; got "
                + ", ".join(f"{name}: {size}" for name, size in sizes.items())
            )
        coordinate = [float(value) for value in self.psi_norm]
        if len(coordinate) < 2 or any(
            not later > earlier for earlier, later in zip(coordinate, coordinate[1:])
        ):
            raise ValueError(
                "RDCONOptions.psi_norm must be strictly increasing (core to edge); "
                "reverse an outboard-in profile together with its t_e and n_e"
            )

    @property
    def has_kinetic_profiles(self) -> bool:
        return (
            self.t_e is not None
            and self.n_e is not None
            and self.psi_norm is not None
        )


@dataclass(frozen=True)
class STRIDEOptions:
    """STRIDE-specific namelist overrides (none exposed yet -- packaged defaults only)."""


@dataclass(frozen=True)
class IdealGPECOptions:
    """Ideal-GPEC-specific namelist overrides.

    ``coil_specs`` selects and excites 3D coil sets (see
    :class:`vaft.code.gpec.CoilInputSpec`): when set, ``coil.in`` and the
    referenced ``.dat`` files are generated from ``coil_config`` instead of
    copying the packaged template verbatim.  ``None`` preserves the legacy
    template behavior; an explicit ``GPECCaseInputs.coil_in`` always wins
    over both.

    ``machine`` is the GPEC ``machine`` word and the ``<machine>_<set>.dat``
    file prefix.  For ``"vest"`` (default) ``coil_config`` defaults to the
    packaged VEST geometry and the direction words to
    :data:`vaft.machine_mapping.conventions.VEST_GPEC_COIL_DIRECTIONS`.  For
    any other machine ``coil_config`` (name -> ``CoilSet3D``), ``ip_direction``
    and ``bt_direction`` are all required: they are machine facts, never
    inherited from a template.  Derive the direction words from the machine's
    sign contract with
    :func:`vaft.machine_mapping.conventions.gpec_coil_directions` rather than
    writing the pair by hand.

    These four sit here rather than on :class:`GPECSuiteConfig`, next to
    ``coil_data_dir``, because only the ideal-GPEC stage consumes them and
    ``__post_init__`` validates them as one group: DCON, RDCON and STRIDE
    never see a coil.  Move them up only if a stability module starts needing
    the machine word.

    ``ascii_flag`` and ``xclebsch_flag`` are what decide whether PENTRC has an
    input at all, and both default to ``None`` -- "whatever the template holds".
    A machine layer with its own ``templates_dir`` may already ship
    ``ascii_flag=t`` precisely so that its torque runs have something to read, and
    writing a default over that would silently take it away.  PENTRC reads the **ASCII** displacement,
    ``gpec_xclebsch_n<n>.out`` (``pentrc/inputs.f90`` opens ``peq_file`` as a
    text table), and GPEC writes that file only when both flags are on
    (``gpec/gpout.f:5755``).  The packaged template ships ``xclebsch_flag=t``
    and ``ascii_flag=f``, so the default run writes everything in netCDF and
    nothing PENTRC can read -- which made the torque unreachable through this
    API until these two were exposed.

    ``ascii_flag`` is all-or-nothing in GPEC: it writes *every* output in ASCII,
    including the ``*_fun`` (psi, theta) reconstructions when ``fun_flag`` is on,
    which is hundreds of megabytes per run.  Turn it on for a run whose torque
    you want, not as a habit.
    """

    coil_flag: bool = True
    #: Write every output in ASCII as well as netCDF.  ``None`` leaves the
    #: template's value, which is ``f`` in the packaged one; PENTRC's input exists
    #: only with ``t``.
    ascii_flag: Optional[bool] = None
    #: Compute the Clebsch-coordinate displacement PENTRC integrates.  ``None``
    #: leaves the template's value, ``t`` in the packaged one -- on its own it only
    #: reaches the netCDF.
    xclebsch_flag: Optional[bool] = None
    coil_specs: Optional[Sequence["CoilInputSpec"]] = None
    machine: str = "vest"
    coil_config: Optional[Mapping[str, "CoilSet3D"]] = None
    ip_direction: Optional[str] = None
    bt_direction: Optional[str] = None

    #: Both penetration thresholds at once.  GPEC reads this as a shorthand and
    #: forces the two below true (``gpec/gpec.f:274-279``), so it is not a
    #: third, independent switch: ``singthresh_flag=True`` *is* Callen and
    #: SLAYER.  Use the sub-flags to ask for one without the other.
    singthresh_flag: bool = False

    #: Callen's critical island width for island growth, written to
    #: ``gpec_profile_output_n<n>.nc`` as ``w_isl_v_crit``
    #: (``gpec/gpout.f:1757-1776``).  It needs the coil vacuum field, so
    #: ``coil_flag`` must be on; GPEC itself only warns and leaves the column
    #: zero (``gpec/gpec.f:280-291``).
    singthresh_callen_flag: bool = False

    #: SLAYER's critical resonant field, written as ``Phi_res_crit``
    #: (``gpec/gpout.f:1779-1786``).  A different model from Callen's and a
    #: different column, which is why the two flags are separate: a run with
    #: only one of them on leaves the *other* column exactly zero, and zero is
    #: also what an unattainable threshold would look like.
    singthresh_slayer_flag: bool = False

    #: The magnetic Prandtl number ``Pm = nu / eta = tau_R / tau_V`` SLAYER is
    #: run at [-], passed through unchanged: ``Pm = 5`` is written as ``5.0``.
    #:
    #: Not an inverse, despite the ``in`` in GPEC's key: SLAYER builds the
    #: viscous time from it as ``tau_v = tau_r / inpr`` (``slayer/gslayer.f:86``,
    #: ``slayer/params.f:34,40``), so ``inpr`` *is* ``tau_r / tau_v``, the
    #: magnetic Prandtl number, and GPEC's own ``input/gpec.in`` calls it the
    #: "Scalar Prandtl number".  Writing ``1 / Pm`` here runs SLAYER at the wrong
    #: viscosity and moves the critical field by roughly a factor ``Pm``.
    #:
    #: Required whenever a SLAYER threshold is requested, and refused
    #: otherwise.  It is a physics model coefficient -- a critical field scales
    #: with it -- so it is never defaulted here, even though GPEC defaults it
    #: to 5.0 (``gpec/gpec.f:161``).
    #:
    #: Refused when no threshold is requested because GPEC stamps it into
    #: ``gpec_profile_output_n<n>.nc`` as the global attribute ``Pr``
    #: *unconditionally* (``gpec/gpout.f:1851-1852``), flags or no flags; that
    #: attribute records this value, ``Pm`` itself.  A value set on a run that
    #: computed no threshold would therefore sit in the output looking like the
    #: Prandtl number one was computed at.
    singthresh_slayer_inpr: Optional[float] = None
    #: Per-rational-surface magnetic Prandtl numbers ``Pm`` (the same quantity
    #: as the scalar, not its inverse), ascending in ``q``, at most 20 entries.
    #: GPEC falls back to the scalar for any entry ``<= 0``
    #: (``input/gpec.in:62``), so the scalar is required alongside it.
    singthresh_slayer_inpr_prof: Optional[Sequence[float]] = None

    @property
    def writes_pentrc_input(self) -> Optional[bool]:
        """Whether this run will leave the ASCII displacement PENTRC reads.

        Both flags, because either alone writes nothing PENTRC can open: the
        netCDF carries the same displacement and PENTRC does not read it.

        ``None`` when either is ``None``: the answer is then the template's, and
        this object does not know which template it will be prepared against.
        :func:`~vaft.code.gpec.validate_pentrc_inputs` reads the prepared cell,
        which does.
        """
        if self.ascii_flag is None or self.xclebsch_flag is None:
            return None
        return bool(self.ascii_flag and self.xclebsch_flag)

    @property
    def wants_callen_threshold(self) -> bool:
        """Whether this run will compute Callen's critical width [-]."""
        return bool(self.singthresh_flag or self.singthresh_callen_flag)

    @property
    def wants_slayer_threshold(self) -> bool:
        """Whether this run will compute SLAYER's critical resonant field [-]."""
        return bool(self.singthresh_flag or self.singthresh_slayer_flag)

    @property
    def wants_any_threshold(self) -> bool:
        """Whether either threshold is requested, and so kinetic profiles are needed [-]."""
        return self.wants_callen_threshold or self.wants_slayer_threshold

    def __post_init__(self) -> None:
        # A config error is knowable here; refusing at construction keeps it
        # from surfacing after DCON has run and gpec.in / vac.in are staged.
        if self.machine != "vest":
            missing = [
                name
                for name in ("coil_specs", "coil_config", "ip_direction", "bt_direction")
                if getattr(self, name) is None
            ]
            if missing:
                raise ValueError(
                    f"machine {self.machine!r}: {', '.join(missing)} must be given explicitly "
                    "(only 'vest' has packaged coil geometry and direction words); or pass an "
                    "explicit GPECCaseInputs.coil_in"
                )
        # An explicitly empty selection is refused rather than read as "use the
        # packaged template" -- and refused here, for the reason above: it used
        # to surface from inside `prepare`, which is the late failure this
        # method exists to prevent.
        if self.coil_specs is not None and len(self.coil_specs) == 0:
            raise ValueError(
                "coil_specs is empty: at least one coil set name is required; pass "
                "None to use the packaged coil.in template"
            )
        # Callen's threshold is built from the *coil* vacuum field
        # (`gpec/gpout.f:1770-1772` divides by `singflx_mn`, and
        # `gpec/gpec.f:285-291` forces `singfld_flag` on under `coil_flag` and
        # otherwise only warns). Without a coil the column comes back zero,
        # which is indistinguishable from a threshold that was computed and is
        # unattainable -- so the request is refused instead.
        if self.wants_callen_threshold and not self.coil_flag:
            raise ValueError(
                "Callen's penetration threshold needs the coil vacuum field, so "
                "coil_flag must be True; GPEC only warns and leaves w_isl_v_crit zero, "
                "which reads the same as a computed threshold of zero "
                "(gpec/gpec.f:280-291)"
            )
        if self.wants_slayer_threshold and self.singthresh_slayer_inpr is None:
            raise ValueError(
                "a SLAYER penetration threshold needs singthresh_slayer_inpr, the "
                "magnetic Prandtl number Pm = nu/eta (tau_R/tau_V) it is computed "
                "at -- not its inverse, whatever the key's name suggests "
                "(slayer/gslayer.f:86); it scales the critical field, so it is "
                "stated rather than inherited (GPEC's own default is 5.0, "
                "gpec/gpec.f:161)"
            )
        if not self.wants_slayer_threshold and self.singthresh_slayer_inpr is not None:
            raise ValueError(
                "singthresh_slayer_inpr is set but no SLAYER threshold is requested; "
                "GPEC writes it into gpec_profile_output_n<n>.nc as the global "
                "attribute Pr whatever the flags say (gpec/gpout.f:1851-1852), so it "
                "would sit in the output as the Prandtl number a threshold was "
                "computed at. Set singthresh_slayer_flag (or singthresh_flag) too, or "
                "leave it None"
            )
        if self.singthresh_slayer_inpr_prof is not None:
            profile = tuple(self.singthresh_slayer_inpr_prof)
            if not self.wants_slayer_threshold:
                raise ValueError(
                    "singthresh_slayer_inpr_prof is set but no SLAYER threshold is "
                    "requested; set singthresh_slayer_flag (or singthresh_flag) too, "
                    "or leave it None"
                )
            if not 1 <= len(profile) <= 20:
                raise ValueError(
                    "singthresh_slayer_inpr_prof takes 1 to 20 entries, one per "
                    f"rational surface in ascending q; got {len(profile)} "
                    "(gpec/gpec.f reads a fixed 20-element array)"
                )
            if self.singthresh_slayer_inpr is None:
                raise ValueError(
                    "singthresh_slayer_inpr_prof needs singthresh_slayer_inpr beside "
                    "it: GPEC falls back to the scalar for any profile entry <= 0 "
                    "(input/gpec.in:62), and a run with no scalar would fall back to "
                    "whatever the template happened to hold"
                )


#: Main-ion species PENTRC may be run for, as ``(mass number [u], charge [e])``.
#:
#: A named vocabulary rather than two numbers, so a run records *which plasma*
#: its torque belongs to. The NTV torque scales with the ion mass through the
#: bounce and transit frequencies, so "deuterium" is a physics statement about
#: the discharge and is required rather than inherited from a template --
#: GPEC's own ``pentrc.in`` ships ``mi=2 zi=1``, which is right for a great many
#: discharges and is evidence for none of them.
#:
#: Integers, because ``pentrc.in``'s ``mi``/``zi``/``mimp``/``zimp`` are Fortran
#: integers (``pentrc/pentrc_interface.f90:99-103``): a namelist read of ``2.0``
#: into one of them is an error, not a rounding. So these are mass *numbers* --
#: one isotope each -- and a natural-abundance average cannot be expressed here
#: at all.
PENTRC_ION_SPECIES: Mapping[str, tuple[int, int]] = MappingProxyType({
    "hydrogen": (1, 1),
    "deuterium": (2, 1),
    "tritium": (3, 1),
    "helium": (4, 2),
})

#: Impurity species, same form.  PENTRC always carries one (``read_kin`` takes
#: ``zimp``/``mimp`` for the quasineutrality correction), so there is no "none"
#: entry.  Each is one isotope, for the reason above: ``tungsten`` is W-184, the
#: most abundant one, and not tungsten's 183.84 natural-abundance mass.
PENTRC_IMPURITY_SPECIES: Mapping[str, tuple[int, int]] = MappingProxyType({
    "carbon": (12, 6),
    "boron": (11, 5),
    "nitrogen": (14, 7),
    "neon": (20, 10),
    "tungsten": (184, 74),
})

#: PENTRC's collision operators (``pentrc.in``'s ``nutype``), with what each is.
PENTRC_COLLISION_OPERATORS: Mapping[str, str] = MappingProxyType({
    "zero": "collisionless",
    "krook": "a single Krook relaxation rate",
    "harmonic": "an energy-dependent rate, harmonic in the bounce frequency",
})


@dataclass(frozen=True)
class PENTRCOptions:
    """What one PENTRC run computes, and for which plasma.

    PENTRC turns a perturbed field and a kinetic profile into the neoclassical
    toroidal viscous torque.  Which *calculation* it uses is not a detail: its
    eighteen methods are different physics models of the same quantity and
    disagree by design (see :data:`vaft.code.pentrc.TORQUE_METHODS`), so
    ``methods``, ``main_ion``, ``impurity`` and ``collision_operator`` are all
    required.  Nothing here has a physics default, and the packaged
    ``pentrc.in`` ships every method flag off so that a namelist this writes
    states the whole selection rather than inheriting part of it.  (GPEC's own
    template ships ``fgar_flag=.true.``, and its code default is the same, so a
    caller who says nothing would silently get the full calculation.)

    Attributes
    ----------
    methods : sequence of str
        Keys of :data:`vaft.code.pentrc.TORQUE_METHODS`; each becomes
        ``<method>_flag=t`` [-].
    main_ion : str
        A key of :data:`PENTRC_ION_SPECIES` [-].
    impurity : str
        A key of :data:`PENTRC_IMPURITY_SPECIES` [-].
    collision_operator : str
        A key of :data:`PENTRC_COLLISION_OPERATORS`; ``pentrc.in``'s
        ``nutype`` [-].
    bounce_harmonics : int
        ``nl``: the calculation sums bounce harmonics ``-nl..nl``.  A numerical
        truncation rather than a model choice, so it keeps GPEC's own shipped
        value of 6.  The runs this adapter was ported from used 4 [-].
    moment : str
        ``"pressure"`` for torque and particle transport, ``"heat"`` for heat
        transport.  Both write the same variable names and units and differ only
        in ``long_name``, which is what
        :data:`vaft.code.pentrc.TORQUE_LONG_NAME` exists to check [-].
    electron : bool
        Run for electrons instead of ions.  PENTRC does one species per run [-].
    psi_limits : tuple of float, optional
        ``psilims``, the radial range.  ``None`` keeps the template's 0 to 1 [-].
    grids : sequence of str
        Which radial grids each method is written on: ``"dynamic"`` (the
        solver's own adaptive steps, read back as
        :data:`vaft.code.pentrc.TORQUE_GRIDS`' ``lsode``), ``"equil"``,
        ``"input"`` [-].
    artificial_factors : Mapping, optional
        ``wefac``/``wdfac``/``wpfac``/``nufac``/``divxfac``, PENTRC's scans of a
        frequency or rate away from its physical value.  ``None`` leaves every
        one of them at the template's 1, which is the physical case; a value
        other than 1 makes the run a sensitivity study rather than a prediction,
        which is why it is spelled out rather than tucked into a float field [-].
    data_dir : str
        Where the pre-formed ``fnml`` matrices the ``rlar`` and ``clar`` methods
        need live.  ``"default"`` is ``$GPECHOME/pentrc``.  The runs this adapter
        was ported from carried ``"../../../pentrc"``, which is a claim about how
        deep the run tree is rather than about the installation [-].
    threads : int
        ``pentrc_threads``; ``0`` defers to ``$OMP_NUM_THREADS`` [-].
    output_ascii, output_netcdf : bool
        Which of the two output forms PENTRC writes.
        :func:`vaft.code.pentrc.read_pentrc_output` reads the netCDF one [-].
    """

    methods: Sequence[str]
    main_ion: str
    impurity: str
    collision_operator: str
    bounce_harmonics: int = 6
    moment: str = "pressure"
    electron: bool = False
    psi_limits: Optional[tuple[float, float]] = None
    grids: Sequence[str] = ("dynamic",)
    artificial_factors: Optional[Mapping[str, float]] = None
    data_dir: str = "default"
    threads: int = 0
    output_ascii: bool = False
    output_netcdf: bool = True

    #: The three ``pentrc.in`` grid switches, and the ``TORQUE_GRIDS`` key each
    #: produces in the output.
    #:
    #: ``dynamic`` is the one whose names differ on the two sides: the switch is
    #: ``dynamic_grid`` and the output calls the grid ``lsode``, because one names
    #: the method and the other the solver that produced it.
    GRID_FLAGS = MappingProxyType({
        "dynamic": "dynamic_grid",
        "equil": "equil_grid",
        "input": "input_grid",
    })

    #: The five ``&PENT_CONTROL`` scan factors, all 1 in the physical case.
    #:
    #: ``&PENT_CONTROL`` declares two more -- ``nfac`` and ``tfac``, on the density
    #: and the temperature (``pentrc/pentrc_interface.f90:150``) -- which GPEC
    #: ships in no ``pentrc.in`` and the packaged template therefore does not
    #: carry.  They are left out rather than accepted and dropped: the writer
    #: patches keys a template already has, so accepting one would raise at write
    #: time with a message about a template rather than about the request.
    ARTIFICIAL_FACTORS = ("wefac", "wdfac", "wpfac", "nufac", "divxfac")

    def __post_init__(self) -> None:
        from ..pentrc import TORQUE_METHODS

        if not self.methods:
            raise ValueError(
                "PENTRCOptions.methods is empty: PENTRC would run and write no torque. "
                f"Choose from {sorted(TORQUE_METHODS)}"
            )
        unknown = [name for name in self.methods if name not in TORQUE_METHODS]
        if unknown:
            raise ValueError(
                f"unknown PENTRC method(s) {unknown}; the calculations are "
                f"{sorted(TORQUE_METHODS)} (vaft.code.pentrc.TORQUE_METHODS)"
            )
        if self.main_ion not in PENTRC_ION_SPECIES:
            raise ValueError(
                f"main_ion must be one of {sorted(PENTRC_ION_SPECIES)}, got "
                f"{self.main_ion!r}"
            )
        if self.impurity not in PENTRC_IMPURITY_SPECIES:
            raise ValueError(
                f"impurity must be one of {sorted(PENTRC_IMPURITY_SPECIES)}, got "
                f"{self.impurity!r}"
            )
        if self.collision_operator not in PENTRC_COLLISION_OPERATORS:
            raise ValueError(
                f"collision_operator must be one of "
                f"{sorted(PENTRC_COLLISION_OPERATORS)}, got {self.collision_operator!r}"
            )
        if self.moment not in ("pressure", "heat"):
            raise ValueError(f"moment must be 'pressure' or 'heat', got {self.moment!r}")
        if int(self.bounce_harmonics) < 0:
            raise ValueError(
                f"bounce_harmonics is the half-range of a symmetric sum -nl..nl and "
                f"cannot be negative; got {self.bounce_harmonics!r}"
            )
        if not self.grids:
            raise ValueError(
                "PENTRCOptions.grids is empty: every radial grid switched off leaves "
                f"nothing to integrate on. Choose from {sorted(self.GRID_FLAGS)}"
            )
        bad_grids = [name for name in self.grids if name not in self.GRID_FLAGS]
        if bad_grids:
            raise ValueError(
                f"unknown PENTRC grid(s) {bad_grids}; choose from "
                f"{sorted(self.GRID_FLAGS)}"
            )
        if self.artificial_factors is not None:
            bad = sorted(set(self.artificial_factors) - set(self.ARTIFICIAL_FACTORS))
            if bad:
                raise ValueError(
                    f"pentrc.in has no scan factor(s) {bad}; the five are "
                    f"{list(self.ARTIFICIAL_FACTORS)}"
                )
        if self.psi_limits is not None:
            low, high = (float(value) for value in self.psi_limits)
            if not high > low:
                raise ValueError(
                    f"psi_limits must be increasing, got {self.psi_limits!r}"
                )
        if not (self.output_ascii or self.output_netcdf):
            raise ValueError(
                "both output_ascii and output_netcdf are off, so PENTRC would compute "
                "the torque and write nothing"
            )

    @property
    def species_namelist(self) -> dict[str, int]:
        """The four ``pentrc.in`` species keys this selection resolves to [u, e].

        Integers, because the namelist keys are (see
        :data:`PENTRC_ION_SPECIES`).
        """
        main_mass, main_charge = PENTRC_ION_SPECIES[self.main_ion]
        imp_mass, imp_charge = PENTRC_IMPURITY_SPECIES[self.impurity]
        return {
            "mi": int(main_mass),
            "zi": int(main_charge),
            "mimp": int(imp_mass),
            "zimp": int(imp_charge),
        }


@dataclass(frozen=True)
class GPECSuiteConfig:
    """Runtime and VEST-default configuration for the GPEC suite.

    ``gpec_home`` defaults to ``None``, in which case the installation root is
    read from ``$GPECHOME``.  Preparing a case never needs it; only running a
    module does, and a missing installation is reported there.

    Per-solver namelist overrides live on the ``dcon``/``rdcon``/``stride``/
    ``gpec`` sub-options below rather than as flat fields on this dataclass,
    so the field count here stays fixed as solver-specific knobs accumulate.

    ``verify_outputs`` opts into content-level success checks (does the
    produced ``.nc`` actually contain the expected physics variable, not just
    exist) via each solver's ``check_success``. Off by default so tests and
    trivial stub executables -- which produce no real ``.nc`` content -- keep
    working; real production runs should set it.
    """

    gpec_home: Path | str | None = None
    executable_dir: Path | str | None = None
    modules: Sequence[str] = DEFAULT_MODULES
    modes: Sequence[int] = DEFAULT_MODES
    run_mode: str = "run_if_available"
    templates_dir: Path | str | None = None
    coil_data_dir: Path | str | None = None
    psilow: float = 1e-2
    psihigh: float = 0.994
    verify_outputs: bool = False
    timeout: Optional[float] = 1200.0
    env: Mapping[str, str] = field(default_factory=dict)
    dcon: DCONOptions = field(default_factory=DCONOptions)
    rdcon: RDCONOptions = field(default_factory=RDCONOptions)
    stride: STRIDEOptions = field(default_factory=STRIDEOptions)
    gpec: IdealGPECOptions = field(default_factory=IdealGPECOptions)
    backend: Optional["ExecutionBackend"] = None  # None -> LocalBackend (vaft.code.execution)


@dataclass
class GPECCaseInputs:
    """Materialized inputs for one shot/time GPEC-suite case."""

    shot: int
    time_ms: int | str | None
    geqdsk: Path
    workdir: Path
    coil_in: Path | None = None
    # Ideal GPEC consumes the DCON files for the same time/mode.  In the
    # canonical FileDB layout DCON may live in a separate code-specific tree.
    dcon_workdir: Path | None = None


@dataclass
class GPECModuleRun:
    """Status for one module/mode directory."""

    module: str
    mode: int
    workdir: Path
    returncode: Optional[int] = None
    #: ``prepared`` | ``completed`` | ``stable`` | ``skipped`` | ``failed``.
    #:
    #: ``stable`` is a *successful* outcome, kept apart from ``completed``
    #: because a stable equilibrium produces no unstable-mode output and so
    #: cannot be told from a broken run by what it wrote (issue #423).
    status: str = "prepared"
    reason: str = ""
    logs: tuple[Path, ...] = ()
    outputs: tuple[Path, ...] = ()
    commands: tuple[str, ...] = ()
    #: Expected files a companion executable would have written, which this run
    #: directory does not have.  Non-empty is not a failure -- companions are
    #: optional by construction -- but it is the difference between a DCON cell
    #: that has an eigenfunction and one that never will, so it is reported
    #: rather than left for a caller to rediscover by listing the directory.
    missing_optional_outputs: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """Whether this cell produced a usable result.

        ``stable`` counts: the solver ran and found no unstable mode, which is a
        physics result. Excluding it would make every stable discharge read as
        an unusable cell (issue #423).
        """
        return self.status in {"completed", "stable"} and self.returncode == 0


@dataclass
class GPECSuiteResult:
    """Suite status, records, and collected artifacts."""

    returncode: Optional[int]
    workdir: Path
    shot: int | None = None
    time_ms: int | str | None = None
    records: tuple[GPECModuleRun, ...] = ()
    logs: tuple[Path, ...] = ()
    outputs: Mapping[str, tuple[Path, ...]] = field(default_factory=dict)
    stdout: str = ""
    stderr: str = ""
    parsed: Any = None
    #: SHA-256 of the equilibrium this suite consumed.
    #:
    #: The stability end of the provenance chain. Without it, "which CHEASE
    #: equilibrium produced this stability result" is answerable only by
    #: trusting that the file at a recorded path never changed -- which is the
    #: assumption `replication.is_reusable` exists to stop relying on. Empty
    #: when the equilibrium could not be read, which is a fact about this run
    #: rather than a reason to fail it.
    input_equilibrium_sha256: str = ""

    @property
    def ok(self) -> bool:
        return self.returncode == 0

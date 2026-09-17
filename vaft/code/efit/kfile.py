"""EFIT constraints-ODS and k-file generation.

Moved verbatim out of the former monolithic ``efit.py``.
"""

import json
import numpy as np
import os
import re
import warnings
from dataclasses import dataclass, fields
from functools import partial
from numbers import Integral
from pathlib import Path
from typing import Sequence
from omas import ODS, save_omas_json

from vaft.ods_access import path_value

from vaft.machine_mapping.magnetics import (
    INBOARD_FLUX_LOOP_MAX_R,
    INBOARD_PROBE_MAX_R,
    OUTBOARD_FLUX_LOOP_MIN_R,
    OUTBOARD_PROBE_MIN_R,
    SIDE_PROBE_MIN_ABS_Z,
    equilibrium_probe_count,
    vest_equilibrium_magnetics_channel_definitions,
)

from .legacy import (
    vfit_equilibrium_form_constraints,
)
from .recovery import ProbeFamilies, family_of, gaussian_probe_recovery, probe_families
from vaft.validation.channel_decision import MISSING, RECOVERED, ChannelDecisions
from vaft.validation.efit_channels import condemned_channels, decide_efit_channels
from vaft.validation.magnetics import unusable_channels_at

from .efund import table_machine_era
from .magnetic import EFITConfig
from .config import EFITScientificConfig, EFITProfileConfig


#: How a magnetics channel family is named in the EFIT constraint tree. The
#: mapping is EFIT's own vocabulary, so it lives here rather than in the
#: validation layer, which knows only about magnetics channels.
_EFIT_CONSTRAINT_FAMILY = {
    "b_field_pol_probe": "bpol_probe",
    "flux_loop": "flux_loop",
}


def build_efit_coil_currents(destination, source, *, coilset=None, window=None) -> None:
    """The EFIT current groups, built from the machine's own PF circuits.

    VEST energises ten PF circuits.  EFIT's VEST table carries sixteen current
    groups over them: the solenoid split into eight axial segments, and PF5,
    PF6, PF9 and PF10 each split at the midplane.  Splitting a circuit is
    bookkeeping, so every group takes the current of the circuit it belongs
    to, and the eight solenoid segments all carry PF1.

    ``coilset`` is the resolved coilset policy -- which circuits the table
    describes and how each is split -- and defaults to the configured one.  It
    is the same policy the EFUND projection is given, so the table and the
    k-file cannot disagree about which coils exist.  Circuits are matched by
    name, not by position.

    ``window`` is ``(tstart, tend, dt)`` to resample the currents onto, and
    defaults to **not resampling**: the currents are taken on the time base
    they were measured on.  This used to be a fixed 0.26-0.36 s at 40 us
    written into the writer, which is VEST's analysis window and had no
    business here -- and was inert besides.  Every packaged product already
    arrives on exactly that grid (2500 uniform 40 us samples from 0.26 s), so
    the interpolation was a series onto its own abscissa.  A caller that does
    need a different grid resolves one from the machine layer and passes it.
    """
    if coilset is None:
        from vaft.machine_mapping.efit_coilset import vest_efit_coilset_policy

        coilset = vest_efit_coilset_policy()

    by_name: dict[str, int] = {}
    for index in range(len(source["coil"])):
        by_name[str(source[f"coil.{index}.name"])] = index
    missing = sorted(set(coilset.source_circuit.values()) - set(by_name))
    if missing:
        raise ValueError(
            "pf_active is missing the circuits EFIT's groups are driven by: "
            + ", ".join(missing)
            + f" (present: {', '.join(sorted(by_name))})"
        )

    time = np.asarray(source["time"], dtype=float)
    if window is None:
        resampled = time
    else:
        tstart, tend, dt = (float(value) for value in window)
        if dt <= 0:
            raise ValueError(f"the resampling step must be positive, got {dt}")
        resampled = np.arange(max(tstart, time[0]), min(tend, time[-1]), dt)
        if resampled.size == 0:
            raise ValueError(
                f"the window {tstart}-{tend} s does not overlap the record "
                f"{time[0]}-{time[-1]} s"
            )

    destination["ids_properties.comment"] = "PF config from vest_pf_active, grouped for EFIT"
    destination["ids_properties.homogeneous_time"] = 1
    destination["time"] = resampled
    for group, name in enumerate(coilset.group_names):
        circuit = by_name[coilset.source_circuit[name]]
        destination[f"coil.{group}.name"] = name
        destination[f"coil.{group}.identifier"] = name
        destination[f"coil.{group}.current.data"] = np.interp(
            resampled, time, np.asarray(source[f"coil.{circuit}.current.data"], dtype=float)
        )


def _condemned_channels(ods, nbprobe: int) -> set[int]:
    """Legacy-style broken indices for channels with no usable sample at all.

    Kept for callers and tests; the rule lives in
    :func:`vaft.validation.efit_channels.condemned_channels` and the
    constraint builder no longer consults it directly -- it consumes a
    :class:`~vaft.validation.channel_decision.ChannelDecisions` (issue #296).
    """
    return condemned_channels(ods, nbprobe=nbprobe)


def apply_validity_exclusions(ods, EQ, *, min_validity: int = 0) -> dict[tuple[str, int], list[int]]:
    """Zero the weight of every channel unusable at a given reconstruction time.

    Signal quality is time-resolved (issue #189): an integrator that rails at
    0.31 s leaves every earlier slice perfectly constrained, so the exclusion
    is per (slice, channel) rather than per channel.

    This reuses the "weight = 0 means excluded from the fit" contract the
    legacy ``broken`` list and the missing-channel placeholder (#145) already
    rely on, rather than introducing a second exclusion mechanism.  It must run
    *after* the weighting loop, which reassigns every family's nominal weight
    unconditionally and would otherwise overwrite this.

    Only channels the diagnostics stage marked **unusable** gate here.  A
    channel merely flagged with a warning keeps its weight, because those
    thresholds are not yet justified across a representative VEST population
    and must not silently remove data from a reconstruction.

    An ODS carrying no validity excludes nothing, so this is a no-op on data
    produced before the quality layer existed.  Returns the excluded slice
    indices per channel, for reporting.
    """
    excluded: dict[tuple[str, int], list[int]] = {}
    # Only the kinds EFIT actually submits are asked about. The validation
    # layer's model is deliberately open to more of them -- native-rate Mirnov
    # voltage, for one -- and adding one there must not make this raise on a
    # family the constraint tree has no name for.
    for (kind, channel), unusable in unusable_channels_at(
        ods, EQ["time"], min_validity=min_validity, kinds=tuple(_EFIT_CONSTRAINT_FAMILY)
    ).items():
        family = _EFIT_CONSTRAINT_FAMILY[kind]
        for i in np.flatnonzero(unusable):
            # A channel outside the submitted families has no constraint entry
            # to weight, and creating one here would invent a constraint EFIT
            # never saw.
            if f"time_slice.{i}.constraints.{family}.{channel}.measured" not in EQ:
                continue
            EQ[f"time_slice.{i}.constraints.{family}.{channel}.weight"] = 0.0
            excluded.setdefault((kind, channel), []).append(int(i))
    return excluded


PROBE_WEIGHT_INDEX = {"inboard": 3, "side": 4, "outboard": 5}
FLUX_WEIGHT_INDEX = {"inboard": 6, "outboard": 7}


@dataclass(frozen=True)
class ConstraintErrors:
    """The relative error assigned to each constraint family.

    Every value is a fraction of the measurement: the stored
    ``measured_error_upper`` is ``value * measured``.

    These arrived as a nine-element list indexed by position, with the meaning
    of each slot written down nowhere and copied into five workflow scripts and
    a test. Naming them is not cosmetic: ``uncertainty[5]`` and
    ``uncertainty[6]`` are the side probes and the outboard probes, and nothing
    in a call site said so.

    The probe and flux-loop fields still name VEST's three probe families and
    two flux-loop families. That is the next thing to move (#708); a named
    field at least says which family it is.
    """

    pf_current: float
    b_field_tor_vacuum_r: float
    ip: float
    diamagnetic_flux: float
    probe_inboard: float
    probe_side: float
    probe_outboard: float
    flux_loop_inboard: float
    flux_loop_outboard: float

    @classmethod
    def from_sequence(cls, values: Sequence[float]) -> "ConstraintErrors":
        """The legacy positional list, in its documented order."""
        expected = len(fields(cls))
        values = [float(value) for value in values]
        if len(values) != expected:
            raise ValueError(
                f"expected {expected} constraint errors in the legacy order "
                f"({', '.join(field.name for field in fields(cls))}), got {len(values)}"
            )
        return cls(*values)

    @classmethod
    def coerce(cls, values: "ConstraintErrors | Sequence[float]") -> "ConstraintErrors":
        return values if isinstance(values, cls) else cls.from_sequence(values)


@dataclass(frozen=True)
class ConstraintWeights:
    """The fit weight assigned to each constraint family.

    The eight-element positional twin of :class:`ConstraintErrors`, with the
    same complaint: ``weighting[6]`` is the inboard flux loops and only
    ``FLUX_WEIGHT_INDEX`` knew it.
    """

    pf_current: float
    ip: float
    diamagnetic_flux: float
    probe_inboard: float
    probe_side: float
    probe_outboard: float
    flux_loop_inboard: float
    flux_loop_outboard: float

    #: The family names the probe and flux-loop weights are keyed by, so a
    #: caller can look one up without knowing a field name.
    def probe(self, family: str) -> float:
        return float(getattr(self, f"probe_{family}"))

    def flux_loop(self, family: str) -> float:
        return float(getattr(self, f"flux_loop_{family}"))

    @classmethod
    def from_sequence(cls, values: Sequence[float]) -> "ConstraintWeights":
        expected = len(fields(cls))
        values = [float(value) for value in values]
        if len(values) != expected:
            raise ValueError(
                f"expected {expected} constraint weights in the legacy order "
                f"({', '.join(field.name for field in fields(cls))}), got {len(values)}"
            )
        return cls(*values)

    @classmethod
    def coerce(cls, values: "ConstraintWeights | Sequence[float]") -> "ConstraintWeights":
        return values if isinstance(values, cls) else cls.from_sequence(values)


def apply_channel_decisions(
    EQ,
    decisions: ChannelDecisions,
    *,
    weighting,
    families: ProbeFamilies,
    flux_families: tuple,
) -> dict:
    """Translate channel decisions into constraint weights and values (issue #296).

    Per slice and submitted channel whose constraint already exists: a
    ``missing`` decision is skipped (the #145 placeholder stands); every
    other state writes ``weight = nominal * weight_factor``, where the
    nominal weight is the family's entry of ``weighting``, which may be a
    :class:`ConstraintWeights` or the legacy positional list; a ``recovered``
    slice also writes the backend's value
    into ``measured`` and, when it supplied one, its uncertainty into
    ``measured_error_upper``.  A channel in no family gets no weight, as the
    legacy builder left it.  Nothing is invented: a decision for a channel
    the constraint equilibrium never submitted is ignored.

    The decisions themselves are recorded under
    ``code.parameters.channel_decisions`` (channels with a notable state
    only) so the product says why a constraint carries the weight it does.
    """
    weights = ConstraintWeights.coerce(weighting)
    inboard_flux, outboard_flux = (set(int(k) for k in fam) for fam in flux_families)
    written = 0
    skipped: list[tuple[str, int]] = []
    for (kind, index), decision in decisions.entries.items():
        if kind == "b_field_pol_probe":
            family = family_of(index, families)
            nominal = None if family is None else weights.probe(family)
            node = "bpol_probe"
        elif kind == "flux_loop":
            family = "inboard" if index in inboard_flux else "outboard" if index in outboard_flux else None
            nominal = None if family is None else weights.flux_loop(family)
            node = "flux_loop"
        else:
            continue
        for i in range(decisions.times.size):
            base = f"time_slice.{i}.constraints.{node}.{index}"
            if f"{base}.measured" not in EQ:
                skipped.append((kind, index))
                break
            state = decision.state_at(i)
            if state == MISSING or nominal is None:
                continue
            EQ[f"{base}.weight"] = nominal * float(decision.weight_factor[i])
            if state == RECOVERED:
                EQ[f"{base}.measured"] = float(decision.value[i])
                if decision.uncertainty is not None and np.isfinite(decision.uncertainty[i]):
                    EQ[f"{base}.measured_error_upper"] = float(decision.uncertainty[i])
            written += 1
    EQ["code.parameters.channel_decisions"] = json.dumps(decisions.as_dict(only_notable=True), default=float)
    return {"written": written, "skipped_no_constraint": sorted(set(skipped)), **{"states": decisions.summary()}}


def _efit_bpol_probe_count(magnetics) -> int:
    """The B-pol channels EFIT's geometry represents.

    One line, because the rule is not this module's to state: three copies of
    ``min(present, defined)`` had accumulated and the one here disagreed with
    the other two when no channel was defined, returning zero probes where
    they returned every present one.
    """
    return equilibrium_probe_count(magnetics)


def _has_matching_signal(magnetics, path: str) -> bool:
    """Whether a diagnostic leaf is finite-length and aligned to magnetics.time."""
    data, time = path_value(magnetics, path), path_value(magnetics, "time")
    if data is None or time is None:
        return False
    try:
        data = np.asarray(data).reshape(-1)
        time = np.asarray(time).reshape(-1)
    except Exception:
        return False
    return bool(data.size and data.size == time.size)


#: Half-width [s] of the box average each constraint is taken over.  The
#: legacy value; qualifying it against the cadence is issue #468.
DEFAULT_AVERAGE_WINDOW = 0.0005


def generate_constraints_ods(
    ods,
    shotnumber,
    save_dir,
    efit_table_dir,
    time,
    uncertainty,
    weighting,
    broken=None,
    fit=0,
    fl_correct_coeff=None,
    FFCUR=2,
    PPCUR=2,
    *,
    decisions: ChannelDecisions | None = None,
    recovery=None,
    average_window: float = DEFAULT_AVERAGE_WINDOW,
    coil_current_window: tuple[float, float, float] | None = None,
    expected_table_era: str | None = None,
    coilset=None,
) -> ChannelDecisions:
    """Generate the constraints ODS, ``save_dir/{shotnumber}_constraints.json``.

    Channel selection is not decided here (issue #296).  Pass ``decisions``
    from :func:`vaft.validation.efit_channels.decide_efit_channels` and,
    optionally, a ``recovery`` backend ``callable(EQ, decisions) ->
    decisions`` such as :func:`vaft.code.efit.recovery.gaussian_probe_recovery`;
    the builder forms the window-averaged constraints, runs the backend, and
    translates the decisions into weights and values through
    :func:`apply_channel_decisions`.

    ``average_window`` is the half-width of the box average each constraint
    value is taken over, ``[t_i - w, t_i + w]`` (issue #433).  The two
    controls are independent: this one says what a slice measures, the
    decisions say which channels are believed, and the reconstruction
    cadence is a third (issue #468).

    ``broken`` (one-based combined indexes) and ``fit`` (0 none, 1 rejected
    probes, 2 every probe) are the legacy way of saying the same thing and
    are deprecated: when ``decisions`` is not given they are turned into the
    equivalent policy and backend, with a warning.  Returns the decisions the
    product was built from.
    """

    # A Green table is a projection of one machine's conductors and is only
    # valid for the era it was built from. The caller resolves which era this
    # discharge belongs to -- that is a fact about the machine, not about EFIT
    # -- and the table states its own; disagreement is refused here rather
    # than reconstructed against the wrong coils (#805).
    if expected_table_era is not None:
        declared = table_machine_era(efit_table_dir)
        if declared is not None and declared != expected_table_era:
            raise ValueError(
                f"the Green table in {efit_table_dir} was built for machine era "
                f"{declared!r}, but shot {shotnumber} belongs to {expected_table_era!r}. "
                "Coil geometry differs between eras, so this would reconstruct "
                "against conductors that are in the wrong place. Regenerate the "
                "table for this era (workflow/efit_tables/regenerate_legacy_table.py "
                f"--era {expected_table_era}) or analyse a shot from {declared!r}."
            )

    # coilset_opt
    # "16_coils" : PF1 - 8 segments + PF5, 6, 9, 10 Upper and Lower Segments
    # "26_coils" : PF1 - 8 segments + PF2-10 Upper and Lower Segments

    # Default constraints which are used routinely in VEST
    constraints = [
        "pf_current",
        "bpol_probe",
        "flux_loop",
        "ip",
        "diamagnetic_flux",
        "b_field_tor_vacuum_r",
    ]
    # For later need to develop option to add other constraints (e.g. internal magnetic probe, thomson scattering, etc.)

    # Accepts the legacy positional lists; everything below reads names.
    errors = ConstraintErrors.coerce(uncertainty)
    weights = ConstraintWeights.coerce(weighting)

    ods_tmp = ODS()
    PF = ods_tmp["pf_active"]
    PF_orig = ods["pf_active"]

    ## (1) The EFIT current groups, from the measured PF circuits.
    if coilset is None:
        from vaft.machine_mapping.efit_coilset import vest_efit_coilset_policy

        coilset = vest_efit_coilset_policy()
    build_efit_coil_currents(PF, PF_orig, coilset=coilset, window=coil_current_window)

    # Which circuits this discharge energised. Recorded on the product so it
    # says why a coil carries the weight it does, and read back by the writer.
    from vaft.machine_mapping.efit_coilset import detect_energised_circuits

    energisation = detect_energised_circuits(
        PF_orig, coilset, window=(float(time[0]), float(time[-1]))
    )
    if energisation.disagreement:
        warnings.warn(
            "the energised circuits measured for this discharge differ from the "
            f"configured declaration -- {energisation.disagreement}; the measurement is used",
            stacklevel=2,
        )

    for i, _ in enumerate(PF["coil"]):
        PF[f"coil.{i}.current.time"] = PF["time"]
        PF[f"coil.{i}.current.data_error_upper"] = abs(
            errors.pf_current * PF[f"coil.{i}.current.data"]
        )

    #    print('___',PF['coil.0.current.data'])
    #    print('___',PF['coil.0.current.time'])

    ## (2) TF coil
    TF = ods["tf"]
    TF["b_field_tor_vacuum_r.time"] = TF["time"]
    TF["b_field_tor_vacuum_r.data_error_upper"] = abs(
        errors.b_field_tor_vacuum_r * TF["b_field_tor_vacuum_r.data"]
    )

    ## (3) Ip
    MG = ods["magnetics"]
    MG["ip.0.data_error_upper"] = abs(errors.ip * MG["ip.0.data"])

    ## (4) Diamagnetic flux
    MG["diamagnetic_flux.0.time"] = MG["time"]
    MG["diamagnetic_flux.0.data_error_upper"] = abs(
        errors.diamagnetic_flux * MG["diamagnetic_flux.0.data"]
    )

    ## (5) Poloidal magnetic probe
    Index_inBz = np.where(MG["b_field_pol_probe.:.position.r"] < INBOARD_PROBE_MAX_R)
    Index_sideBz = np.where(np.abs(MG["b_field_pol_probe.:.position.z"]) > SIDE_PROBE_MIN_ABS_Z)
    Index_outBz = np.where(MG["b_field_pol_probe.:.position.r"] > OUTBOARD_PROBE_MIN_R)
    efit_bpol_probe_count = _efit_bpol_probe_count(MG)
    # convert tuple to array
    valid_bpol_indices = np.array(
        [
            int(index)
            for index, _ in enumerate(MG["b_field_pol_probe"])
            if index < efit_bpol_probe_count
            and _has_matching_signal(MG, f"b_field_pol_probe.{index}.field.data")
        ],
        dtype=int,
    )
    Index_inBz = np.intersect1d(Index_inBz[0], valid_bpol_indices)
    Index_sideBz = np.intersect1d(Index_sideBz[0], valid_bpol_indices)
    Index_outBz = np.intersect1d(Index_outBz[0], valid_bpol_indices)

    #    print('==!==')
    #    print(broken)
    #    print(Index_inBz)
    #    print(Index_sideBz)
    #    print(Index_outBz)
    #    print('==!==')

    families = probe_families(MG, count=efit_bpol_probe_count)

    for i in Index_inBz:
        MG[f"b_field_pol_probe.{i}.field.time"] = MG["time"]
        MG[f"b_field_pol_probe.{i}.field.data_error_upper"] = abs(
            errors.probe_inboard * MG[f"b_field_pol_probe.{i}.field.data"]
        )
    for i in Index_sideBz:
        MG[f"b_field_pol_probe.{i}.field.time"] = MG["time"]
        MG[f"b_field_pol_probe.{i}.field.data_error_upper"] = abs(
            errors.probe_side * MG[f"b_field_pol_probe.{i}.field.data"]
        )

    for i in Index_outBz:
        MG[f"b_field_pol_probe.{i}.field.time"] = MG["time"]
        MG[f"b_field_pol_probe.{i}.field.data_error_upper"] = abs(
            errors.probe_outboard * MG[f"b_field_pol_probe.{i}.field.data"]
        )

    ## (6) Flux loops
    Index_inFlux = np.where(MG["flux_loop.:.position.0.r"] < INBOARD_FLUX_LOOP_MAX_R)
    Index_OutFlux = np.where(MG["flux_loop.:.position.0.r"] > OUTBOARD_FLUX_LOOP_MIN_R)
    # Same missing-data guard as valid_bpol_indices above: a flux loop
    # without raw data has no `flux.data` to read, so it must be excluded
    # here too, before vfit_equilibrium_form_constraints's placeholder logic
    # is even reached.
    valid_flux_indices = np.array(
        [
            int(index)
            for index, _ in enumerate(MG["flux_loop"])
            if _has_matching_signal(MG, f"flux_loop.{index}.flux.data")
        ],
        dtype=int,
    )

    # conduct ad-hoc correction for the flux loop signal
    if fl_correct_coeff is not None:
        for i in valid_flux_indices:
            MG[f"flux_loop.{i}.flux.data"] = (
                MG[f"flux_loop.{i}.flux.data"] / fl_correct_coeff[i]
            )

    # convert tuple to array
    Index_inFlux = np.intersect1d(Index_inFlux[0].astype(int), valid_flux_indices)
    Index_OutFlux = np.intersect1d(Index_OutFlux[0].astype(int), valid_flux_indices)

    for i in Index_inFlux:
        MG[f"flux_loop.{i}.flux.time"] = MG["time"]
        MG[f"flux_loop.{i}.flux.data_error_upper"] = abs(
            errors.flux_loop_inboard * MG[f"flux_loop.{i}.flux.data"]
        )

    for i in Index_OutFlux:
        MG[f"flux_loop.{i}.flux.time"] = MG["time"]
        MG[f"flux_loop.{i}.flux.data_error_upper"] = abs(
            errors.flux_loop_outboard * MG[f"flux_loop.{i}.flux.data"]
        )

    # Convert diagnostics ODS to equilibrium constraints ODS

    # The constraint-averaging half-window is a control of its own, distinct
    # from the reconstruction cadence (issues #433, #468).
    default_average = float(average_window)

    EQ = ods["equilibrium"]
    EQtime = EQ["time"]

    #    vfit_equilibrium_form_constraints(EQ,PF,MG,TF,EQtime,constraints,default_average)
    vfit_equilibrium_form_constraints(
        EQ,
        PF,
        MG,
        TF,
        time,
        constraints,
        default_average,
        bpol_probe_count=efit_bpol_probe_count,
    )

    PFP = ods["pf_passive"]
    nbloop = len(PFP["loop"])
    # The total probe count, not just those with data: flux_loop's `broken`
    # index offset (`j + nbprobe` below) must stay stable regardless of
    # which probes happen to be missing for a given shot, or the same
    # `broken` config list would silently exclude different physical
    # channels depending on real-time data availability.
    nbprobe = efit_bpol_probe_count
    PM = ods["equilibrium.code.parameters"]

    # The diamagnetic flux keeps its sign (issue #385).  EFIT reads DFLUX as a
    # signed quantity and fits it against cdflux = integral (B_t - B_tv) dA, so
    # a diamagnetic plasma in VEST's positive toroidal field is a *negative*
    # flux.  The donor code compared magnitudes with a magnitude-only
    # reconstruction, which is not what EFIT does.
    for i in range(len(EQ["time"])):
        for j in range(len(EQ[f"time_slice.{i}.constraints.pf_current"])):
            EQ[f"time_slice.{i}.constraints.pf_current.{j}.weight"] = weights.pf_current
        EQ[f"time_slice.{i}.constraints.ip.weight"] = weights.ip
        EQ[f"time_slice.{i}.constraints.diamagnetic_flux.weight"] = weights.diamagnetic_flux

    if decisions is None:
        # The legacy interface: the manual list and the fit option are the
        # policy and the backend, spelled the old way.
        warnings.warn(
            "generate_constraints_ods: broken=/fit= are deprecated; pass decisions= "
            "from vaft.validation.efit_channels.decide_efit_channels and a recovery= backend",
            DeprecationWarning,
            stacklevel=2,
        )
        decisions = decide_efit_channels(
            ods, EQ["time"], nbprobe=nbprobe, manual_rejections=broken or ()
        )
        if int(fit) > 0 and recovery is None:
            recovery = partial(
                gaussian_probe_recovery, mode=int(fit), families=families, uncertainty="legacy"
            )
    if recovery is not None:
        decisions = recovery(EQ, decisions)
    apply_channel_decisions(
        EQ,
        decisions,
        weighting=weighting,
        families=families,
        flux_families=(Index_inFlux, Index_OutFlux),
    )

    # add namelist parameters in the k-file
    for i in range(len(EQ["time"])):
        mytime = EQ["time"][i]
        print(i, mytime)

        PM[f"time_slice.{i}.IN1.INPUT_DIR"] = efit_table_dir
        PM[f"time_slice.{i}.IN1.TABLE_DIR"] = efit_table_dir

        PM[f"time_slice.{i}.IN1.IECURR"] = 0
        PM[f"time_slice.{i}.IN1.KFFCUR"] = FFCUR
        PM[f"time_slice.{i}.IN1.KPPCUR"] = PPCUR
        PM[f"time_slice.{i}.IN1.KFFFNC"] = 0
        PM[f"time_slice.{i}.IN1.KPPFNC"] = 0
        PM[f"time_slice.{i}.IN1.SERROR"] = 0.05
        PM[f"time_slice.{i}.IN1.IVESEL"] = 1
        PM[f"time_slice.{i}.IN1.IFITVS"] = 0
        PM[f"time_slice.{i}.IN1.FCURBD"] = 1
        PM[f"time_slice.{i}.IN1.PCURBD"] = 1
        PM[f"time_slice.{i}.IN1.KCALPA"] = 0
        PM[f"time_slice.{i}.IN1.KCGAMA"] = 0
        # The seed ellipse, the reference radius and the plasma-current floor
        # are NOT written here. They were, as five VEST numbers, and the writer
        # took them from `EFITScientificConfig` regardless -- so they never
        # reached a k-file and simply sat in the stored parameters
        # contradicting the live values. `CUTIP` said 50000 A where the
        # configuration says 5000.

        # Add wall eddy current to the k-file
        Iwall = []
        for j in range(nbloop):
            Iwall.append(np.interp(mytime, PFP["time"], PFP[f"loop.{j}.current"]))
        PM[f"time_slice.{i}.IN1.VCURRT"] = Iwall

        # The coil constraint matrix is not built here. It used to be, from a
        # hard-coded choice between a sixteen- and a twenty-six-coil layout,
        # and none of it ever reached the k-file: the `&INWANT` block is
        # written from `EFITConstraintConfig` (see the writer below), which
        # silently overrode these values. It is gone rather than left to be
        # rediscovered.

    # Which circuits this discharge energised, recorded so the product says why
    # a coil carries the weight it does, and read back by the writer. Written
    # here and not earlier: a leaf placed in `code.parameters` before its
    # `time_slice` sub-tree exists takes the sub-tree's place on the way out.
    EQ["code.parameters.energisation"] = json.dumps(energisation.as_dict(), default=float)

    # Save the constraints ODS
    ods_eq = ODS()
    ods_eq["equilibrium"] = EQ
    fullfilename = os.path.join(save_dir, f"{shotnumber}_constraints.json")
    save_omas_json(ods_eq, fullfilename)
    return decisions


def generate_kfile(
    ods,
    shotnumber,
    npprime=None,
    nffprime=None,
    save_dir="./tmp",
    *,
    config: EFITConfig | EFITScientificConfig | None = None,
    coilset=None,
):
    """
    Generate k-files under ``save_dir/kfile`` for the requested shot.

    ``npprime`` and ``nffprime`` remain supported for legacy callers.  New
    callers should supply an :class:`EFITConfig` or
    :class:`EFITScientificConfig` so every scientific namelist choice is
    explicit and serializable.
    """

    for name, value in (("npprime", npprime), ("nffprime", nffprime)):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, Integral) or value <= 0
        ):
            raise ValueError(f"{name} must be a positive integer")

    if isinstance(config, EFITConfig):
        scientific = config.scientific_config()
    elif isinstance(config, EFITScientificConfig):
        scientific = config
        if npprime is not None and npprime != scientific.profile.kppcur:
            raise ValueError(
                "npprime conflicts with config.profile.kppcur; omit the legacy "
                "argument or make the values equal"
            )
        if nffprime is not None and nffprime != scientific.profile.kffcur:
            raise ValueError(
                "nffprime conflicts with config.profile.kffcur; omit the legacy "
                "argument or make the values equal"
            )
    elif config is None:
        scientific = EFITScientificConfig(
            profile=EFITProfileConfig(
                kppcur=2 if npprime is None else npprime,
                kffcur=2 if nffprime is None else nffprime,
            )
        )
    else:
        raise TypeError("config must be EFITConfig, EFITScientificConfig, or None")
    profile = scientific.profile
    initialization = scientific.initialization
    numerics = scientific.numerics
    constraint_config = scientific.constraints

    # Load the constraints ODS
    EQ = ods["equilibrium"]
    time = EQ["time"]
    PM = ods["equilibrium.code.parameters"]

    # Define the kfile parameters
    vbit = constraint_config.legacy_vbit
    shft = constraint_config.legacy_weight_scale
    TABLE_DIR = "TABLE_DIR = '{}' \n".format(PM["time_slice.0.IN1.INPUT_DIR"])
    INPUT_DIR = "INPUT_DIR = '{}' \n".format(PM["time_slice.0.IN1.INPUT_DIR"])

    def _machine_count(name, default):
        mhdin = Path(PM["time_slice.0.IN1.INPUT_DIR"]).expanduser() / "mhdin.dat"
        if not mhdin.exists():
            return default
        match = re.search(
            rf"\b{name}\s*=\s*(\d+)",
            mhdin.read_text(encoding="utf-8", errors="ignore"),
            re.IGNORECASE,
        )
        return int(match.group(1)) if match else default

    def _namelist_array(name, values, per_line=4, formatter=str):
        chunks = [f"{name}= "]
        for idx, value in enumerate(values, start=1):
            chunks.append(f"{formatter(value)}, ")
            if idx % per_line == 0:
                chunks.append("\n ")
        return "".join(chunks).rstrip(" \n,") + "\n"

    def _weight(cstr, path: str, group: str) -> float:
        original = float(cstr[f"{path}.weight"])
        if original == 0.0:
            return 0.0
        return float(constraint_config.group_weights.get(group, original))

    def _objective_scale(group: str) -> float:
        return float(constraint_config.objective_scales[group])

    def _measurement_error(
        cstr, path: str, fallback: float, *, group: str, unit_scale: float = 1.0
    ) -> float:
        """The sigma submitted for one channel, after the group's scale.

        ``uncertainty_scales`` divides here rather than multiplying ``FWT*``
        because EFIT processes a statistical row as ``FWT/sigma``: narrowing
        the sigma is the only way to raise a row's weight without writing an
        ``FWT*`` outside the range the code was exercised with (#386).
        """
        scale = float(constraint_config.uncertainty_scales[group])
        if constraint_config.uncertainty_mode == "legacy_weight":
            return float(fallback) / scale
        try:
            value = abs(float(cstr[f"{path}.measured_error_upper"]))
        except Exception as exc:
            raise ValueError(
                f"standard_deviation mode requires {path}.measured_error_upper"
            ) from exc
        return value * unit_scale / scale

    def _uncertainty_formatter(group: str):
        """How a submitted sigma is spelled in the namelist.

        The legacy three-decimal spelling is kept exactly wherever it is still
        meaningful, so an unscaled k-file is byte-identical to the one this
        writer has always produced.  It stops being meaningful once a group's
        uncertainty is scaled down: ``f"{0.0001:.3f}"`` is ``"0.000"``, and a
        zero sigma makes EFIT skip its ``fwt/sigma`` division entirely
        (``data_input.F90:2782`` guards on ``> 1e-10``), so the row would
        silently keep the weight the study meant to change.
        """
        if (
            constraint_config.uncertainty_mode == "standard_deviation"
            or float(constraint_config.uncertainty_scales[group]) != 1.0
        ):
            return lambda value: f"{value:.9g}"
        return lambda value: f"{value:.3f}"

    # find the maximum decimal places in the time
    #    for time_idx, _ in enumerate(time):
    #        if time_idx == 0:
    #            digit = get_decimal_places(time[time_idx])
    #        else:
    #            if digit < get_decimal_places(time[time_idx]):
    #                digit = get_decimal_places(time[time_idx])

    for time_idx, _ in enumerate(time):
        # Load the diagnostics constraints data and convert them to kfile parameters
        CSTR = EQ[f"time_slice.{time_idx}.constraints"]

        ## (1) PF Coil currents with weight
        # The constraint tree carries exactly the groups EFIT's table has, in
        # its order, so there is nothing to select -- and a disagreement is an
        # error rather than something to trim. Taking `min()` here silently
        # dropped the trailing groups of a tree longer than the table, which
        # is the shape of every future mistake in this area: the k-file still
        # looks well formed and simply omits coils.
        available = len(CSTR["pf_current"])
        nfsum = _machine_count("nfsum", available)
        if nfsum != available:
            raise ValueError(
                f"the table declares nfsum = {nfsum} but the constraint tree carries "
                f"{available} PF current groups; they describe different coilsets. "
                f"Table directory: {PM['time_slice.0.IN1.INPUT_DIR']}"
            )
        pf_indices = list(range(available))
        nbcoil = available
        # `None` means "the coilset says": splitting a circuit into groups does
        # not split its current, and series-wired circuits carry one, so the
        # equalities follow from the machine description rather than from a
        # table written out by index.
        matrix = constraint_config.coil_constraint_matrix
        targets = constraint_config.coil_constraint_targets
        if matrix is None:
            if coilset is None:
                from vaft.machine_mapping.efit_coilset import vest_efit_coilset_policy

                coilset = vest_efit_coilset_policy()
            matrix = coilset.constraint_matrix()
            if targets is None:
                targets = coilset.constraint_targets()
        elif targets is None:
            targets = (0.0,) * len(matrix[0])
        if constraint_config.use_coil_relation_constraints and len(matrix) != nbcoil:
            raise ValueError(
                "coil_constraint_matrix row count must match the selected "
                f"PF coil count ({len(matrix)} != {nbcoil})"
            )
        column_count = len(matrix[0])
        # Which circuits this discharge energised; an unenergised group is
        # pinned at zero rather than fitted, and its error bar is floored so a
        # zero current does not become a zero uncertainty.
        stored = path_value(EQ, "code.parameters.energisation", None)
        energised = None
        if stored:
            try:
                energised = set(json.loads(stored)["energised"])
            except (ValueError, KeyError, TypeError):
                energised = None
        floor = 0.0 if coilset is None else float(coilset.coil_current_error_floor)

        COILCURRENT = np.zeros(nbcoil)
        BITCURRENT = np.zeros(nbcoil)
        COILWEIGHT = np.zeros(nbcoil)
        for i, source_index in enumerate(pf_indices):
            COILCURRENT[i] = CSTR[
                f"pf_current.{source_index}.measured"
            ]  # Coil current in A
            BITCURRENT[i] = max(
                _measurement_error(
                    CSTR,
                    f"pf_current.{source_index}",
                    CSTR[f"pf_current.{source_index}.measured_error_upper"],
                    group="pf_current",
                ),
                floor,
            )
            if energised is None or coilset is None:
                COILWEIGHT[i] = 1.0
            else:
                circuit = coilset.source_circuit[coilset.group_names[i]]
                COILWEIGHT[i] = 1.0 if circuit in energised else 0.0
        BRSP = _namelist_array("BRSP", COILCURRENT, per_line=3)
        BITFC = _namelist_array("BITFC", BITCURRENT, per_line=3)
        pf_weight = _weight(CSTR, "pf_current.0", "pf_current")
        # Fortran's repeat count while every coil is weighted alike, which is
        # the routine case and keeps the k-file as it was; the array form only
        # when a circuit is switched off. The objective scale multiplies the
        # weight in both forms -- it is a property of the family, not of the
        # spelling.
        pf_objective_scale = _objective_scale("pf_current")
        scaled_pf_weight = pf_weight * pf_objective_scale
        if np.all(COILWEIGHT == 1.0):
            FWTFC = f"FWTFC= {nbcoil}*{scaled_pf_weight}\n"
        else:
            FWTFC = _namelist_array(
                "FWTFC", [scaled_pf_weight * factor for factor in COILWEIGHT], per_line=8
            )

        ## (2) Wall eddy current
        WALLCURRENT = PM[f"time_slice.{time_idx}.IN1.VCURRT"]
        if constraint_config.wall_current_mode == "disabled":
            WALLCURRENT = np.zeros(len(WALLCURRENT))

        VCURRT = _namelist_array("VCURRT", WALLCURRENT, per_line=4)

        ## (3) Toroidal magnetic field (TF coil)
        RCENTR = initialization.rzero
        BTOR = CSTR["b_field_tor_vacuum_r.measured"] / RCENTR

        ## (4) Plasma current with weight
        plasma_weight = _weight(CSTR, "ip", "plasma_current")
        PLASMA = f"PLASMA= {CSTR['ip.measured']}"
        BITIP = f"BITIP= {_measurement_error(CSTR, 'ip', plasma_weight / vbit * shft, group='plasma_current')}"
        FWTCUR = f"FWTCUR= {plasma_weight * _objective_scale('plasma_current')}"

        ## (5) Diamagnetic flux with weight
        flux_scale = (
            1000.0 if constraint_config.diamagnetic_flux_input_units == "Wb" else 1.0
        )
        # "imas" writes the stored, signed value: EFIT's convention (#385).
        VAL = float(CSTR["diamagnetic_flux.measured"]) * flux_scale
        if constraint_config.diamagnetic_flux_sign == "absolute":
            VAL = abs(VAL)
        elif constraint_config.diamagnetic_flux_sign == "negative":
            VAL = -abs(VAL)
        DFLUX = f"DFLUX= {VAL} \n"

        ## Original SIGDLC is written as the standard deviation but we use the fitting weight instead
        diamagnetic_weight = _weight(CSTR, "diamagnetic_flux", "diamagnetic_flux")
        SIGDLC = f"SIGDLC= {_measurement_error(CSTR, 'diamagnetic_flux', diamagnetic_weight * shft * flux_scale, group='diamagnetic_flux', unit_scale=flux_scale)}"
        # SIGDLC=f'SIGDLC= {CSTR["diamagnetic_flux.measured_error_upper"]*1000}' # Standard deviation of diamagnetic flux measurement data in mWb
        # SIGDLC=f'SIGDLC= {VAL*CSTR["diamagnetic_flux.weight"]}' # set sigdlc as measured value * weight

        if (
            not constraint_config.use_diamagnetic_flux
            or diamagnetic_weight == 0
            or _objective_scale("diamagnetic_flux") == 0
        ):  # if the diamagnetic flux weight is 0, the diamagnetic flux is not considered as a constraint
            FWTDLC = "FWTDLC= 0"
        else:
            FWTDLC = f"FWTDLC= {_objective_scale('diamagnetic_flux')}"

        ## (6) Poloidal magnetic probe with weight
        # magpri (dprobe.dat/mhdin.dat) is EFIT's own count of physically
        # fitted probes -- for VEST this is 64, the leading `bpol_probe`
        # entries built from vest_equilibrium_magnetics_channel_definitions(). VAFT's OMAS
        # magnetics IDS additionally carries 4 trailing toroidal-mirnov
        # phase-reference channels (identifier suffix ":phase_reference")
        # that are not part of EFIT's B-pol fitting set; writing all of
        # them into EXPMP2/FWTMP2/BITMPI overflows what EFIT's compiled
        # geometry table expects and is rejected as an invalid namelist
        # line. Same pattern as `nfsum` for PF coils above: read the real
        # count from the table when available, keep every probe otherwise
        # (offline/no-table tests).
        nbprobe = _machine_count("magpri", len(CSTR["bpol_probe"]))
        EXPMP2 = _namelist_array(
            "EXPMP2",
            [CSTR[f"bpol_probe.{i}.measured"] for i in range(nbprobe)],
            per_line=3,
        )
        fwtmp2_values = []
        bitmpi_values = []
        for i in range(nbprobe):
            weight = _weight(CSTR, f"bpol_probe.{i}", "bpol_probe")
            if weight == 0:  # if the probe is broken or the group is disabled
                fwtmp2_values.append(0)
                bitmpi_values.append(0.0)
            else:
                fwtmp2_values.append(_objective_scale("bpol_probe"))
                bitmpi_values.append(
                    _measurement_error(
                        CSTR,
                        f"bpol_probe.{i}",
                        weight / vbit * shft,
                        group="bpol_probe",
                    )
                )
        FWTMP2 = _namelist_array("FWTMP2", fwtmp2_values, per_line=32)
        BITMPI = _namelist_array(
            "BITMPI",
            bitmpi_values,
            per_line=3,
            formatter=_uncertainty_formatter("bpol_probe"),
        )

        ## (4) Flux loops
        nbfl = len(CSTR["flux_loop"])
        COILS = _namelist_array(
            "COILS",
            [
                CSTR[f"flux_loop.{i}.measured"] / 2 / np.pi
                for i in range(len(CSTR["flux_loop"]))
            ],
            per_line=3,
        )

        fwtsi_values = []
        psibit_values = []
        for i in range(nbfl):
            weight = _weight(CSTR, f"flux_loop.{i}", "flux_loop")
            if weight == 0:  # if the flux loop is broken or the group is disabled
                fwtsi_values.append(0)
                psibit_values.append(0.0)
            else:
                fwtsi_values.append(_objective_scale("flux_loop"))
                psibit_values.append(
                    _measurement_error(
                        CSTR,
                        f"flux_loop.{i}",
                        weight / vbit * shft,
                        group="flux_loop",
                        unit_scale=1.0 / (2.0 * np.pi),
                    )
                )
        FWTSI = _namelist_array("FWTSI", fwtsi_values, per_line=32)
        PSIBIT = _namelist_array(
            "PSIBIT",
            psibit_values,
            per_line=3,
            formatter=_uncertainty_formatter("flux_loop"),
        )

        # Named the way EFIT names its outputs: the millisecond, then the
        # exact microsecond remainder when there is one (0.3051 -> 00305_100,
        # 0.30632 -> 00306_320).  The remainder used to be truncated to 0.1 ms,
        # so slices closer than that collided and never matched their a-file.
        slice_us = int(round(time[time_idx] * 1.0e6))
        filename = f"k0{shotnumber}.{slice_us // 1000:05d}"
        if slice_us % 1000:
            filename = f"{filename}_{slice_us % 1000:03d}"

        # Write the kfile
        #        filename=f'k0{shotnumber}.00{time[time_idx]*1e+5:.0f}'
        #        filename=f'k0{shotnumber}.00{time[time_idx]*1e+3:.0f}' # 0.305 -> 305
        # filename=f'k0{shotnumber}.00{time[time_idx]*10**digit:.0f}' # 0.305 -> 305
        print(f"filename: {filename}")
        fullfile = os.path.join(save_dir, "kfile", filename)
        # make the kfile directory if it does not exist
        if not os.path.exists(os.path.join(save_dir, "kfile")):
            os.makedirs(os.path.join(save_dir, "kfile"))
        f = open(fullfile, "w", encoding="utf-8")
        f.write(" &IN1\n")  # the main namelist in the kfile
        f.write(
            " IOUT=4\n"
        )  # write one measurement file for each slice in m0sssss.ttttt
        f.write(f" AELIP = {initialization.minor_radius}\n")
        f.write(f" CUTIP = {initialization.current_threshold}\n")
        f.write(f" EELIP = {initialization.elongation}\n")
        f.write(f" ZELIP = {initialization.zzero}\n")
        f.write(f" FCURBD = {profile.fcurbd}\n")
        f.write(f" FWTBP = {profile.fwtbp}\n")
        if initialization.icinit is not None:
            f.write(f" ICINIT = {initialization.icinit}\n")
        f.write(
            " IECURR = 0\n"
        )  # 0 means that the Ohmic coil flag is ignored (Not classify 5
        passive_flags = {
            "fixed_currents": (1, 0),
            "fit_currents": (1, 1),
            "disabled": (0, 0),
        }
        ivesel, ifitvs = passive_flags[constraint_config.passive_structure_mode]
        f.write(f" IFITVS = {ifitvs}\n")
        # The settings a VEST fit actually terminates on (issue #171). Nothing
        # is written unless the configuration sets it, so the routine k-file is
        # unchanged and EFIT's own defaults still apply -- ERRMIN 1e-2, two
        # orders looser than the ERROR written below, and SAICON 80, which the
        # flat-top passes on the way down. Which is why the fit stops on
        # chi-square at eleven iterations having never reached ERROR.
        for key, value in numerics.termination_keys().items():
            f.write(f" {key} = {value}\n")
        f.write(INPUT_DIR)
        f.write(f" IVESEL = {ivesel}\n")
        f.write(" KCALPA = 0\n")
        f.write(" KCGAMA = 0\n")
        f.write(f" KFFCUR = {profile.kffcur}\n")
        f.write(f" KFFFNC = {profile.kfffnc}\n")
        f.write(f" KPPCUR = {profile.kppcur}\n")
        f.write(f" KPPFNC = {profile.kppfnc}\n")
        f.write(f" PCURBD = {profile.pcurbd}\n")
        # RELIP is the seed ellipse's centre; RZERO is the reference major
        # radius, and RCENTR below sets BTOR with it. They are one field by
        # default and separable when a study needs to move only the seed.
        f.write(f" RELIP = {initialization.seed_rzero}\n")
        f.write(f" RZERO = {initialization.rzero}\n")
        f.write(f" SERROR = {numerics.measurement_error_floor}\n")
        f.write(TABLE_DIR)
        f.write(VCURRT)
        f.write("\n")
        f.write(f" RCENTR = {RCENTR}\n")
        f.write(f" ISHOT = {shotnumber}\n")
        # EFIT reads the slice time as ITIME [ms] + ITIMEU [us] and names its
        # outputs with both, so a sub-millisecond slice needs ITIMEU or it
        # collides with its millisecond neighbour and is mislabelled (#468).
        # Whole-millisecond slices write exactly what they always did.
        f.write(f" ITIME = {slice_us // 1000}\n")
        if slice_us % 1000:
            f.write(f" ITIMEU = {slice_us % 1000}\n")
        f.write(BRSP)
        f.write("\n")
        f.write(BITFC)
        f.write("\n")
        f.write(FWTFC)
        f.write(EXPMP2)
        f.write("\n")
        f.write(BITMPI)
        f.write("\n")
        f.write(COILS)
        f.write("\n")
        f.write(PSIBIT)
        f.write("\n")
        f.write(FWTSI)
        f.write("\n")
        f.write(FWTMP2)
        # f.write('\n')
        f.write(PLASMA)
        f.write("\n")
        f.write(BITIP)
        f.write("\n")
        f.write(FWTCUR)
        f.write("\n")
        f.write(DFLUX)
        f.write(SIGDLC)
        f.write("\n")
        f.write(FWTDLC)
        f.write("\n")
        f.write(f" BTOR = {BTOR}\n")

        f.write(f" RELAX = {numerics.relaxation}\n")
        f.write(f" ERROR = {numerics.error_tolerance}\n")
        f.write(f" MXITER = {-int(numerics.max_iterations)}\n")
        f.write(" NBDRY = 0\n")
        f.write(" /\n")
        f.write(" &INWANT\n")
        if constraint_config.use_coil_relation_constraints:
            for column in range(column_count):
                f.write(
                    _namelist_array(
                        f"CCOILS(1,{column + 1})",
                        [matrix[row][column] for row in range(nbcoil)],
                        per_line=8,
                    )
                )
            f.write(f" KCCOILS = {column_count}\n")
        else:
            f.write(" KCCOILS = 0\n")
        f.write(f" NCCOIL = {constraint_config.nccoil}\n")
        if constraint_config.use_coil_relation_constraints:
            f.write(
                _namelist_array(
                    "XCOILS", targets, per_line=8
                )
            )
        f.write(" /\n")
        f.write("                                            MAG\n")
        f.close()

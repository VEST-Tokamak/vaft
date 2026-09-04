"""EFIT constraints-ODS and k-file generation.

Moved verbatim out of the former monolithic ``efit.py``.
"""

import numpy as np
import os
import re
import statistics
from numbers import Integral
from pathlib import Path
from scipy import optimize
from omas import ODS, save_omas_json

from vaft.ods_access import path_value

from vaft.machine_mapping.magnetics import (
    INBOARD_FLUX_LOOP_MAX_R,
    INBOARD_PROBE_MAX_R,
    OUTBOARD_FLUX_LOOP_MIN_R,
    OUTBOARD_PROBE_MIN_R,
    SIDE_PROBE_MIN_ABS_Z,
    vest_equilibrium_magnetics_channel_definitions,
)

from .legacy import (
    gauss_fit4,
    min_gauss_fit4,
    vfit_equilibrium_form_constraints,
    vfit_pf_active_efit26,
)
from vaft.validation.magnetics import unusable_channels_at

from .magnetic import EFITConfig
from .config import EFITScientificConfig, EFITProfileConfig


#: How a magnetics channel family is named in the EFIT constraint tree. The
#: mapping is EFIT's own vocabulary, so it lives here rather than in the
#: validation layer, which knows only about magnetics channels.
_EFIT_CONSTRAINT_FAMILY = {
    "b_field_pol_probe": "bpol_probe",
    "flux_loop": "flux_loop",
}


def _condemned_channels(ods, nbprobe: int) -> set[int]:
    """Legacy-style broken indices for channels with no usable sample at all.

    Judged on the time-resolved validity, never on the scalar: the scalar is
    "worst state reached", so a channel that merely holds its last value after
    the diagnostics window (every probe on the packaged shot) reads ``-2``
    there while being a perfectly good witness before it.  Only a record the
    quality layer rejected in its entirety -- a physical-ceiling or
    population verdict (#189) -- is broken in the k-file sense.  Probes map to
    their own index, flux loops to ``index + nbprobe``, the offset the rest of
    this module uses for the ``broken`` list.
    """
    from vaft.validation.imas import validity_codes, valid_fraction

    condemned: set[int] = set()
    for kind, quantity, offset in (
        ("b_field_pol_probe", "field", 0),
        ("flux_loop", "flux", nbprobe),
    ):
        count = len(ods[f"magnetics.{kind}"]) if f"magnetics.{kind}" in ods else 0
        for index in range(count):
            base = f"magnetics.{kind}.{index}.{quantity}"
            if validity_codes(ods, base) is None:
                continue  # nothing has assessed this datum
            if valid_fraction(ods, base) == 0.0:
                condemned.add(index + offset)
    return condemned


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


def _efit_bpol_probe_count(magnetics) -> int:
    """Return the B-pol channels represented in VEST EFIT geometry.

    The magnetics IDS also carries trailing toroidal-Mirnov phase-reference
    channels.  They are useful diagnostics, but are not represented in EFIT's
    ``dprobe.dat``/``mhdin.dat`` geometry and must not become constraints.
    The fitted count comes from the canonical MD-channel definition rather
    than from the larger magnetics IDS container.
    """
    return min(
        len(magnetics["b_field_pol_probe"]),
        sum(
            channel["kind"] == "b_field_pol_probe"
            for channel in vest_equilibrium_magnetics_channel_definitions()
        ),
    )


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


def generate_constraints_ods(
    ods,
    shotnumber,
    save_dir,
    efit_table_dir,
    time,
    uncertainty,
    weighting,
    broken=[],
    fit=0,
    fl_correct_coeff=None,
    FFCUR=2,
    PPCUR=2,
):
    """
    Generate Constraints ODS file such that save_dir/{shotnumber}_constraints.json is created
    """
    # fit = 0: no fitting, only exp. data
    # fit = 1: broken data replaced by gaussian fitted data
    # fit = 2: all data replaced by gaussian fitted data

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

    ods_tmp = ODS()
    PF = ods_tmp["pf_active"]
    PF_orig = ods["pf_active"]

    ## (1) PF coil - # 16 coils. It is used for EFIT input, PF1 (CS) Coil is discriesed by 16 coils in EFIT.
    tstart = 0.26
    #    if shotnumber>=43635:
    #        tstart=0.245
    tend = 0.36
    dt = 4e-5

    vfit_pf_active_efit26(PF, PF_orig, shotnumber, tstart, tend, dt)

    for i, _ in enumerate(PF["coil"]):
        PF[f"coil.{i}.current.time"] = PF["time"]
        PF[f"coil.{i}.current.data_error_upper"] = abs(
            uncertainty[0] * PF[f"coil.{i}.current.data"]
        )

    #    print('___',PF['coil.0.current.data'])
    #    print('___',PF['coil.0.current.time'])

    ## (2) TF coil
    TF = ods["tf"]
    TF["b_field_tor_vacuum_r.time"] = TF["time"]
    TF["b_field_tor_vacuum_r.data_error_upper"] = abs(
        uncertainty[1] * TF["b_field_tor_vacuum_r.data"]
    )

    ## (3) Ip
    MG = ods["magnetics"]
    MG["ip.0.data_error_upper"] = abs(uncertainty[2] * MG["ip.0.data"])

    ## (4) Diamagnetic flux
    MG["diamagnetic_flux.0.time"] = MG["time"]
    MG["diamagnetic_flux.0.data_error_upper"] = abs(
        uncertainty[3] * MG["diamagnetic_flux.0.data"]
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

    # Position for In, Out, side
    Bzx = [
        0.54,
        0.5,
        0.46,
        0.42,
        0.38,
        0.34,
        0.3,
        0.26,
        0.22,
        0.16,
        0.12,
        0.08,
        0.04,
        0.0,
        -0.04,
        -0.08,
        -0.12,
        -0.16,
        -0.22,
        -0.26,
        -0.3,
        -0.34,
        -0.38,
        -0.42,
        -0.46,
        -0.5,
        -0.54,
        0.42,
        0.38,
        0.34,
        0.3,
        0.26,
        0.22,
        0.18,
        0.1,
        0.06,
        0.02,
        -0.02,
        -0.06,
        -0.1,
        -0.14,
        -0.18,
        -0.22,
        -0.26,
        -0.3,
        -0.34,
        -0.38,
        -0.42,
        0.8328,
        0.8728,
        0.9128,
        0.9528,
        0.9928,
        1.0328,
        1.0728,
        1.1128,
        -0.8328,
        -0.8728,
        -0.9128,
        -0.9528,
        -0.9928,
        -1.0328,
        -1.0728,
        -1.1128,
    ]

    for i in Index_inBz:
        MG[f"b_field_pol_probe.{i}.field.time"] = MG["time"]
        MG[f"b_field_pol_probe.{i}.field.data_error_upper"] = abs(
            uncertainty[4] * MG[f"b_field_pol_probe.{i}.field.data"]
        )
    for i in Index_sideBz:
        MG[f"b_field_pol_probe.{i}.field.time"] = MG["time"]
        MG[f"b_field_pol_probe.{i}.field.data_error_upper"] = abs(
            uncertainty[5] * MG[f"b_field_pol_probe.{i}.field.data"]
        )

    for i in Index_outBz:
        MG[f"b_field_pol_probe.{i}.field.time"] = MG["time"]
        MG[f"b_field_pol_probe.{i}.field.data_error_upper"] = abs(
            uncertainty[6] * MG[f"b_field_pol_probe.{i}.field.data"]
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
            uncertainty[7] * MG[f"flux_loop.{i}.flux.data"]
        )

    for i in Index_OutFlux:
        MG[f"flux_loop.{i}.flux.time"] = MG["time"]
        MG[f"flux_loop.{i}.flux.data_error_upper"] = abs(
            uncertainty[8] * MG[f"flux_loop.{i}.flux.data"]
        )

    # Convert diagnostics ODS to equilibrium constraints ODS

    default_average = (
        (time[1] - time[0]) / 2
    )  # Diagnostics data of each time point in equilibrium constraints ODS is the average of the +- 0.5*timestep
    #    default_average=0.0002
    default_average = 0.0005

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
    # convert one-based index to zero-based index
    broken = [i - 1 for i in broken]
    # A channel the quality layer condemned for the whole record (#189: a
    # railed or miswired probe) must not be a point in the Gaussian fits
    # below either -- zeroing its weight afterwards does not undo the pull it
    # exerted on every neighbour's fitted value.  It joins the legacy list.
    broken = sorted(set(broken) | _condemned_channels(ods, nbprobe))

    # Add weight and validity (not broken) to the constraints - [pf_coil, tf_coil, pf_passive, ip, dia_flux, inboard_bz, side_bz, outboard_bz, inboard_fl, outboard_fl]
    IPLIM = 45000
    x0 = [0.1, 0.0, 0.2, -0.1]
    x0m = [-0.1, 0.0, 0.2, -0.1]
    x3 = [0.1, 0.2, -0.1]
    x3m = [-0.1, 0.2, -0.1]
    icpt = 0
    for i in range(len(EQ["time"])):
        # PF coil
        for j in range(len(EQ[f"time_slice.{i}.constraints.pf_current"])):
            EQ[f"time_slice.{i}.constraints.pf_current.{j}.weight"] = weighting[0]
        # Ip
        EQ[f"time_slice.{i}.constraints.ip.weight"] = weighting[1]
        IP = EQ[f"time_slice.{i}.constraints.ip.measured"]
        # Diamagnetic flux
        EQ[f"time_slice.{i}.constraints.diamagnetic_flux.weight"] = weighting[2]

        # Inboard Bz
        print(EQ["time"][i], IP)
        if fit > 0 and IP > IPLIM:  # gaussian fitting
            x = []
            y = []
            for j in Index_inBz:
                if j not in broken:
                    x.append(Bzx[j])
                    y.append(EQ[f"time_slice.{i}.constraints.bpol_probe.{j}.measured"])
            res = optimize.minimize(
                min_gauss_fit4,
                x0,
                args=(x, y),
                method="SLSQP",
                tol=1.0e-8,
                options={"maxiter": 1000},
            )
            print("in msg", res.message)
            coef = res.x

            xinBz = []
            yinBz = []
            yinBzg = []
            xinBzb = []
            yinBzb = []
            for j in Index_inBz:
                xinBz.append(Bzx[j])
                yinBz.append(EQ[f"time_slice.{i}.constraints.bpol_probe.{j}.measured"])
                gau = gauss_fit4(coef, Bzx[j])
                yinBzg.append(gau)
                if j in broken:
                    xinBzb.append(Bzx[j])
                    yinBzb.append(
                        EQ[f"time_slice.{i}.constraints.bpol_probe.{j}.measured"]
                    )

            if fit == 1:  # only broken
                for j in Index_inBz:
                    EQ[f"time_slice.{i}.constraints.bpol_probe.{j}.weight"] = weighting[
                        3
                    ]
                    if j in broken:
                        gau = gauss_fit4(coef, Bzx[j])
                        EQ[f"time_slice.{i}.constraints.bpol_probe.{j}.measured"] = gau

            else:  # all data
                for j in Index_inBz:
                    EQ[f"time_slice.{i}.constraints.bpol_probe.{j}.weight"] = weighting[
                        3
                    ]
                    gau = gauss_fit4(coef, Bzx[j])
                    EQ[f"time_slice.{i}.constraints.bpol_probe.{j}.measured"] = gau
        else:  # no fitting
            for j in Index_inBz:
                if j not in broken:
                    EQ[f"time_slice.{i}.constraints.bpol_probe.{j}.weight"] = weighting[
                        3
                    ]
                else:
                    EQ[f"time_slice.{i}.constraints.bpol_probe.{j}.weight"] = 0
        # Side Bz
        if fit > 0 and IP > IPLIM:
            x = []
            y = []
            for j in Index_sideBz:
                if j not in broken:
                    x.append(Bzx[j])
                    y.append(EQ[f"time_slice.{i}.constraints.bpol_probe.{j}.measured"])
            res = optimize.minimize(
                min_gauss_fit4,
                x0m,
                args=(x, y),
                method="SLSQP",
                tol=1.0e-8,
                options={"maxiter": 1000},
            )
            coef1 = res.x
            res = optimize.minimize(
                min_gauss_fit4,
                x0,
                args=(x, y),
                method="SLSQP",
                tol=1.0e-8,
                options={"maxiter": 1000},
            )
            coef2 = res.x
            if min_gauss_fit4(coef1, x, y) < min_gauss_fit4(coef2, x, y):
                coef = coef1
            else:
                coef = coef2

            xsideBz = []
            ysideBz = []
            ysideBzg = []
            xsideBzb = []
            ysideBzb = []
            for j in Index_sideBz:
                xsideBz.append(Bzx[j])
                ysideBz.append(
                    EQ[f"time_slice.{i}.constraints.bpol_probe.{j}.measured"]
                )
                gau = gauss_fit4(coef, Bzx[j])
                ysideBzg.append(gau)
                if j in broken:
                    xsideBzb.append(Bzx[j])
                    ysideBzb.append(
                        EQ[f"time_slice.{i}.constraints.bpol_probe.{j}.measured"]
                    )

            if fit == 1:  # only broken
                for j in Index_sideBz:
                    EQ[f"time_slice.{i}.constraints.bpol_probe.{j}.weight"] = weighting[
                        4
                    ]
                    if j in broken:
                        gau = gauss_fit4(coef, Bzx[j])
                        EQ[f"time_slice.{i}.constraints.bpol_probe.{j}.measured"] = gau

            else:  # all data
                for j in Index_sideBz:
                    EQ[f"time_slice.{i}.constraints.bpol_probe.{j}.weight"] = weighting[
                        4
                    ]
                    gau = gauss_fit4(coef, Bzx[j])
                    EQ[f"time_slice.{i}.constraints.bpol_probe.{j}.measured"] = gau
        else:  # no fitting
            for j in Index_sideBz:
                if j not in broken:
                    EQ[f"time_slice.{i}.constraints.bpol_probe.{j}.weight"] = weighting[
                        4
                    ]
                else:
                    EQ[f"time_slice.{i}.constraints.bpol_probe.{j}.weight"] = 0

        # Outboard Bz
        if fit > 0 and IP > IPLIM:
            x = []
            y = []
            for j in Index_outBz:
                if j not in broken:
                    x.append(Bzx[j])
                    y.append(EQ[f"time_slice.{i}.constraints.bpol_probe.{j}.measured"])
            res = optimize.minimize(
                min_gauss_fit4,
                x0,
                args=(x, y),
                method="SLSQP",
                tol=1.0e-8,
                options={"maxiter": 1000},
            )
            coef = res.x

            xoutBz = []
            youtBz = []
            youtBzg = []
            xoutBzb = []
            youtBzb = []
            for j in Index_outBz:
                xoutBz.append(Bzx[j])
                youtBz.append(EQ[f"time_slice.{i}.constraints.bpol_probe.{j}.measured"])
                gau = gauss_fit4(coef, Bzx[j])
                youtBzg.append(gau)
                if j in broken:
                    xoutBzb.append(Bzx[j])
                    youtBzb.append(
                        EQ[f"time_slice.{i}.constraints.bpol_probe.{j}.measured"]
                    )

            if fit == 1:  # only broken
                for j in Index_outBz:
                    EQ[f"time_slice.{i}.constraints.bpol_probe.{j}.weight"] = weighting[
                        5
                    ]
                    if j in broken:
                        gau = gauss_fit4(coef, Bzx[j])
                        EQ[f"time_slice.{i}.constraints.bpol_probe.{j}.measured"] = gau

            else:  # all data
                for j in Index_outBz:
                    EQ[f"time_slice.{i}.constraints.bpol_probe.{j}.weight"] = weighting[
                        5
                    ]
                    gau = gauss_fit4(coef, Bzx[j])
                    EQ[f"time_slice.{i}.constraints.bpol_probe.{j}.measured"] = gau
        else:  # no fitting
            for j in Index_outBz:
                if j not in broken:
                    EQ[f"time_slice.{i}.constraints.bpol_probe.{j}.weight"] = weighting[
                        5
                    ]
                else:
                    EQ[f"time_slice.{i}.constraints.bpol_probe.{j}.weight"] = 0

        # Inboard flux loop
        for j in Index_inFlux:
            if (j + nbprobe) not in broken:
                EQ[f"time_slice.{i}.constraints.flux_loop.{j}.weight"] = weighting[6]
            else:
                EQ[f"time_slice.{i}.constraints.flux_loop.{j}.weight"] = 0
        # Outboard flux loop
        for j in Index_OutFlux:
            if (j + nbprobe) not in broken:
                EQ[f"time_slice.{i}.constraints.flux_loop.{j}.weight"] = weighting[7]
            else:
                EQ[f"time_slice.{i}.constraints.flux_loop.{j}.weight"] = 0

        # if fit > 0 and IP>IPLIM: # plot all graphs
        #     fig2, axs = plt.subplots(1, 3, layout="constrained")
        #     axs[ 0].scatter(xinBz, yinBz, label='exp.',c='black')
        #     axs[ 0].scatter(xinBz, yinBzg, label='gauss',c='red')
        #     if len(xinBzb) > 0:
        #         axs[ 0].scatter(xinBzb, yinBzb, label='broken',c='blue')
        #     axs[ 0].set_title("Bzin")
        #     for j, txt in enumerate(Index_inBz):
        #         axs[0].text(xinBz[j], yinBz[j], f'{int(txt)+1}', fontsize=8, color='black')
        #     axs[ 1].scatter(xsideBz, ysideBz,c='black')
        #     axs[ 1].scatter(xsideBz, ysideBzg, c='red')
        #     if len(xsideBzb) > 0:
        #         axs[ 1].scatter(xsideBzb, ysideBzb, c='blue')
        #     axs[ 1].set_title("Bzside")
        #     loctime=EQ['time'][i]
        #     plt.xlabel(f'{loctime}')

        #     for j, txt in enumerate(Index_sideBz):
        #         axs[1].text(xsideBz[j], ysideBz[j], f'{int(txt)+1}', fontsize=8, color='black')
        #     axs[ 2].scatter(xoutBz, youtBz, c='black')
        #     axs[ 2].scatter(xoutBz, youtBzg, c='red')
        #     if len(xoutBzb) > 0:
        #         axs[ 2].scatter(xoutBzb, youtBzb, c='blue')
        #     axs[ 2].set_title("Bzout")
        #     for j, txt in enumerate(Index_outBz):
        #         axs[2].text(xoutBz[j], youtBz[j], f'{int(txt)+1}', fontsize=8, color='black')
        #     axs[0].legend()

        #     plt.savefig(os.path.join(save_dir, 'plots', f'{shotnumber}_gfit_{icpt}.png'))
        #     icpt=icpt+1
        #     plt.close(fig2)

    #    make_gif(os.path.join(save_dir, 'plots'), f'{shotnumber}_gfit')

    apply_validity_exclusions(ods, EQ)

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
        PM[f"time_slice.{i}.IN1.CUTIP"] = 50000.0
        PM[f"time_slice.{i}.IN1.RZERO"] = 0.4
        PM[f"time_slice.{i}.IN1.RELIP"] = 0.4
        PM[f"time_slice.{i}.IN1.AELIP"] = 0.3
        PM[f"time_slice.{i}.IN1.EELIP"] = 1.6

        # Add wall eddy current to the k-file
        Iwall = []
        for j in range(nbloop):
            Iwall.append(np.interp(mytime, PFP["time"], PFP[f"loop.{j}.current"]))
        PM[f"time_slice.{i}.IN1.VCURRT"] = Iwall

        # Add coil geometry to the k-file

        nbcoil1 = 8
        coilset_opt = "26_coils"
        if coilset_opt == "16_coils":
            nbcoil = nbcoil1 + 8
            nbcons = nbcoil1 + (nbcoil - nbcoil1) / 2
        elif coilset_opt == "26_coils":
            nbcoil = nbcoil1 + 18
            nbcons = int(nbcoil1 + (nbcoil - nbcoil1) / 2)
            # add extra constraint if needed: PF2U=0, PF2L=0,PF3U=0, PF3L=0,PF4U=0, PF4L=0,PF7U=0, PF7L=0,PF8U=0, PF8L=0
            add0 = []
            for i in range(nbcoil):
                if statistics.mean(PF[f"coil.{i}.current.data"]) == 0.0:
                    add0.append(i)
            nbcons = nbcons + int(len(add0) / 2)

        PM[f"time_slice.{i}.INWANT.NCCOIL"] = 0
        PM[f"time_slice.{i}.INWANT.KCCOILS"] = nbcons
        PM[f"time_slice.{i}.INWANT.XCOILS"] = np.zeros(nbcons)

        CCOILS = np.zeros((nbcoil, nbcons))
        for j in range(nbcoil1 - 1):
            CCOILS[0][j] = 1.0
            CCOILS[j + 1][j] = -1.0

        if coilset_opt == "16_coils":
            # Upper and lower is same
            CCOILS[nbcoil1][nbcoil1 - 1] = 1.0
            CCOILS[nbcoil1 + 1][nbcoil1 - 1] = -1.0
            CCOILS[nbcoil1 + 2][nbcoil1] = 1.0
            CCOILS[nbcoil1 + 3][nbcoil1] = -1.0
            CCOILS[nbcoil1 + 4][nbcoil1 + 1] = 1.0
            CCOILS[nbcoil1 + 5][nbcoil1 + 1] = -1.0
            CCOILS[nbcoil1 + 6][nbcoil1 + 2] = 1.0
            CCOILS[nbcoil1 + 7][nbcoil1 + 2] = -1.0
            # PF 9 = PF 10 is same
            CCOILS[nbcoil1 + 5][nbcoil1 + 3] = 1.0
            CCOILS[nbcoil1 + 6][nbcoil1 + 3] = -1.0
        elif coilset_opt == "26_coils":
            # Upper and lower is same
            CCOILS[nbcoil1][nbcoil1 - 1] = 1.0
            CCOILS[nbcoil1 + 1][nbcoil1 - 1] = -1.0
            CCOILS[nbcoil1 + 2][nbcoil1] = 1.0
            CCOILS[nbcoil1 + 3][nbcoil1] = -1.0
            CCOILS[nbcoil1 + 4][nbcoil1 + 1] = 1.0
            CCOILS[nbcoil1 + 5][nbcoil1 + 1] = -1.0
            CCOILS[nbcoil1 + 6][nbcoil1 + 2] = 1.0
            CCOILS[nbcoil1 + 7][nbcoil1 + 2] = -1.0
            CCOILS[nbcoil1 + 8][nbcoil1 + 3] = 1.0
            CCOILS[nbcoil1 + 9][nbcoil1 + 3] = -1.0
            CCOILS[nbcoil1 + 10][nbcoil1 + 4] = 1.0
            CCOILS[nbcoil1 + 11][nbcoil1 + 4] = -1.0
            CCOILS[nbcoil1 + 12][nbcoil1 + 5] = 1.0
            CCOILS[nbcoil1 + 13][nbcoil1 + 5] = -1.0
            CCOILS[nbcoil1 + 14][nbcoil1 + 6] = 1.0
            CCOILS[nbcoil1 + 15][nbcoil1 + 6] = -1.0
            CCOILS[nbcoil1 + 16][nbcoil1 + 7] = 1.0
            CCOILS[nbcoil1 + 17][nbcoil1 + 7] = -1.0
            # PF 9 = PF 10 is same
            CCOILS[nbcoil1 + 15][nbcoil1 + 8] = 1.0
            CCOILS[nbcoil1 + 16][nbcoil1 + 8] = -1.0

            # force 0 for unused coils
            for j in range(int(len(add0) / 2)):
                CCOILS[add0[2 * j]][nbcoil1 + 9 + j] = 1.0
        PM[f"time_slice.{i}.INWANT.CCOILS"] = CCOILS

    # Save the constraints ODS
    ods_eq = ODS()
    ods_eq["equilibrium"] = EQ
    fullfilename = os.path.join(save_dir, f"{shotnumber}_constraints.json")
    save_omas_json(ods_eq, fullfilename)


def generate_kfile(
    ods,
    shotnumber,
    npprime=None,
    nffprime=None,
    save_dir="./tmp",
    *,
    config: EFITConfig | EFITScientificConfig | None = None,
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

    def _efit16_indices(count: int, target: int) -> list[int]:
        if target == 16 and count >= 26:
            return [0, 1, 2, 3, 4, 5, 6, 7, 14, 15, 16, 17, 22, 23, 24, 25]
        return list(range(min(count, target)))

    def _weight(cstr, path: str, group: str) -> float:
        original = float(cstr[f"{path}.weight"])
        if original == 0.0:
            return 0.0
        return float(constraint_config.group_weights.get(group, original))

    def _measurement_error(
        cstr, path: str, fallback: float, *, unit_scale: float = 1.0
    ) -> float:
        if constraint_config.uncertainty_mode == "legacy_weight":
            return float(fallback)
        try:
            value = abs(float(cstr[f"{path}.measured_error_upper"]))
        except Exception as exc:
            raise ValueError(
                f"standard_deviation mode requires {path}.measured_error_upper"
            ) from exc
        return value * unit_scale

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
        nfsum = _machine_count("nfsum", len(CSTR["pf_current"]))
        pf_indices = _efit16_indices(len(CSTR["pf_current"]), nfsum)
        nbcoil = len(pf_indices)
        matrix = constraint_config.coil_constraint_matrix
        if len(matrix) != nbcoil:
            raise ValueError(
                "coil_constraint_matrix row count must match the selected "
                f"PF coil count ({len(matrix)} != {nbcoil})"
            )
        column_count = len(matrix[0])
        COILCURRENT = np.zeros(nbcoil)
        BITCURRENT = np.zeros(nbcoil)
        for i, source_index in enumerate(pf_indices):
            COILCURRENT[i] = CSTR[
                f"pf_current.{source_index}.measured"
            ]  # Coil current in A
            BITCURRENT[i] = _measurement_error(
                CSTR,
                f"pf_current.{source_index}",
                CSTR[f"pf_current.{source_index}.measured_error_upper"],
            )
        BRSP = _namelist_array("BRSP", COILCURRENT, per_line=3)
        BITFC = _namelist_array("BITFC", BITCURRENT, per_line=3)
        pf_weight = _weight(CSTR, "pf_current.0", "pf_current")
        FWTFC = f"FWTFC= {nbcoil}*{pf_weight}\n"

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
        BITIP = f"BITIP= {_measurement_error(CSTR, 'ip', plasma_weight / vbit * shft)}"
        FWTCUR = f"FWTCUR= {plasma_weight}"

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
        SIGDLC = f"SIGDLC= {_measurement_error(CSTR, 'diamagnetic_flux', diamagnetic_weight * shft * flux_scale, unit_scale=flux_scale)}"
        # SIGDLC=f'SIGDLC= {CSTR["diamagnetic_flux.measured_error_upper"]*1000}' # Standard deviation of diamagnetic flux measurement data in mWb
        # SIGDLC=f'SIGDLC= {VAL*CSTR["diamagnetic_flux.weight"]}' # set sigdlc as measured value * weight

        if (
            not constraint_config.use_diamagnetic_flux or diamagnetic_weight == 0
        ):  # if the diamagnetic flux weight is 0, the diamagnetic flux is not considered as a constraint
            FWTDLC = "FWTDLC= 0"
        else:
            FWTDLC = "FWTDLC= 1"

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
                fwtmp2_values.append(1)
                bitmpi_values.append(
                    _measurement_error(
                        CSTR,
                        f"bpol_probe.{i}",
                        weight / vbit * shft,
                    )
                )
        FWTMP2 = _namelist_array("FWTMP2", fwtmp2_values, per_line=32)
        uncertainty_formatter = (
            (lambda value: f"{value:.9g}")
            if constraint_config.uncertainty_mode == "standard_deviation"
            else (lambda value: f"{value:.3f}")
        )
        BITMPI = _namelist_array(
            "BITMPI", bitmpi_values, per_line=3, formatter=uncertainty_formatter
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
                fwtsi_values.append(1)
                psibit_values.append(
                    _measurement_error(
                        CSTR,
                        f"flux_loop.{i}",
                        weight / vbit * shft,
                        unit_scale=1.0 / (2.0 * np.pi),
                    )
                )
        FWTSI = _namelist_array("FWTSI", fwtsi_values, per_line=32)
        PSIBIT = _namelist_array(
            "PSIBIT", psibit_values, per_line=3, formatter=uncertainty_formatter
        )

        TTIME = str(int(np.round(time[time_idx], 4) * 1e6))
        #        print('TTIME=',TTIME)
        ITIME = TTIME[0:3]
        UTIME = TTIME[3:]
        #        print(ITIME,UTIME)

        filename = f"k0{shotnumber}.00{time[time_idx] * 1000.0:.0f}"  # 0.305 -> 305
        if UTIME != "000":
            filename = f"k0{shotnumber}.00{ITIME}_{UTIME}"  # 0.3051 -> 305_100

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
        # f.write(' ERRMIN = 1.0e-3\n') # Minimum relative error for the fitting (Default is 1.0e-2 - fitted < 1 min, if 1.0e-3 - fitted < 3 min)
        # f.write(' SAICON = 60.0\n') # Minimum chi2 error for the fitting (Default is 80.0)
        f.write(INPUT_DIR)
        f.write(f" IVESEL = {ivesel}\n")
        f.write(" KCALPA = 0\n")
        f.write(" KCGAMA = 0\n")
        f.write(f" KFFCUR = {profile.kffcur}\n")
        f.write(f" KFFFNC = {profile.kfffnc}\n")
        f.write(f" KPPCUR = {profile.kppcur}\n")
        f.write(f" KPPFNC = {profile.kppfnc}\n")
        f.write(f" PCURBD = {profile.pcurbd}\n")
        f.write(f" RELIP = {initialization.rzero}\n")
        f.write(f" RZERO = {initialization.rzero}\n")
        f.write(f" SERROR = {numerics.measurement_error_floor}\n")
        f.write(TABLE_DIR)
        f.write(VCURRT)
        f.write("\n")
        f.write(f" RCENTR = {RCENTR}\n")
        f.write(f" ISHOT = {shotnumber}\n")
        f.write(f" ITIME = {int(round(time[time_idx] * 1000.0))}\n")
        # if digit > 3: # Add ITIMEU (microsecond) if digit is greater than 3
        #     f.write(f' ITIMEU = {(time[time_idx]*1000-int(time[time_idx]*1000))*1000}\n')
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
        for column in range(column_count):
            f.write(
                _namelist_array(
                    f"CCOILS(1,{column + 1})",
                    [matrix[row][column] for row in range(nbcoil)],
                    per_line=8,
                )
            )
        f.write(f" KCCOILS = {column_count}\n")
        f.write(f" NCCOIL = {constraint_config.nccoil}\n")
        f.write(
            _namelist_array(
                "XCOILS", constraint_config.coil_constraint_targets, per_line=8
            )
        )
        f.write(" /\n")
        f.write("                                            MAG\n")
        f.close()

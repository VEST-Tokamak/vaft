"""Legacy VEST/OMFIT signal-processing and constraint-shaping helpers.

These functions back :func:`vaft.code.efit.generate_constraints_ods` (and, for
``correct_flux_loop``, the routine-pipeline
``workflow/automatic_pipeline_1_routine_data_processing/generate_constraints_ods.py``
script directly). Moved verbatim out of the former monolithic ``efit.py``.
"""

from vaft.formula import green_br_bz, green_r, calculate_distance
import numpy as np

from vaft.ods_access import path_value

import math
import warnings
from scipy.signal import savgol_filter


def gauss_fit4(coef, x):
    return coef[0] * np.exp(-((x - coef[1]) ** 2) / 2 / coef[2] / coef[2]) + coef[3]


def min_gauss_fit4(coef, x, y):
    res = 0.0
    for i in range(len(x)):
        res = res + (y[i] - gauss_fit4(coef, x[i])) ** 2
    return np.sqrt(res)


def _signal_matches_time(container, path, time) -> bool:
    """Whether ``path`` holds a waveform the length of ``time``.

    Probed through the shared non-mutating accessor (issue #118): asking about
    a channel with no raw data must not leave a placeholder behind, which is
    what corrupted a constraints ODS here in the first place.
    """
    values = path_value(container, path)
    if values is None:
        return False
    try:
        values = np.asarray(values).reshape(-1)
        time_values = np.asarray(time).reshape(-1)
    except Exception:
        return False
    return bool(values.size and values.size == time_values.size)


CONSTRAINT_EQUILIBRIUM_COMMENT = "constraint equilibrium"


def annotate_constraint_equilibrium(EQ, times):
    """Stamp the constraint equilibrium's identity without erasing its provenance.

    A caller may already have written where the constraint times came from
    (the plasma window and its source, issue #409) into
    ``ids_properties.comment``; that line is kept and the identity appended.
    """
    existing = EQ["ids_properties.comment"] if "ids_properties.comment" in EQ else ""
    if existing and CONSTRAINT_EQUILIBRIUM_COMMENT not in str(existing):
        EQ["ids_properties.comment"] = f"{existing}; {CONSTRAINT_EQUILIBRIUM_COMMENT}"
    elif not existing:
        EQ["ids_properties.comment"] = CONSTRAINT_EQUILIBRIUM_COMMENT
    EQ["ids_properties.homogeneous_time"] = 1
    EQ["time"] = times


def box_average(time, data, center: float, half_width: float, *, what: str = "signal") -> float:
    """The mean of the samples inside ``[center - half_width, center + half_width]``.

    A true box average: every sample the window contains counts once, none
    is interpolated onto a grid of its own (issue #433).  The window is
    closed on both ends.  A window that contains no sample is an error named
    after the constraint -- averaging an interpolated fiction there would
    hide that the diagnostics grid does not cover the reconstruction time.
    """
    t = np.asarray(time, dtype=float).reshape(-1)
    y = np.asarray(data, dtype=float).reshape(-1)
    if t.size != y.size:
        raise ValueError(f"{what}: {y.size} samples against {t.size} time instants")
    inside = (t >= center - half_width) & (t <= center + half_width)
    if not inside.any():
        raise ValueError(
            f"{what}: no sample inside [{center - half_width:.6g}, {center + half_width:.6g}] s "
            f"(grid {t[0]:.6g}..{t[-1]:.6g} s, {t.size} samples)"
        )
    return float(np.mean(y[inside]))


def _window_pair(time, data, error, center, half_width, *, what):
    return (
        box_average(time, data, center, half_width, what=what),
        box_average(time, error, center, half_width, what=f"{what} uncertainty"),
    )


def vfit_equilibrium_form_constraints(
    EQ,
    PF,
    MG,
    TF,
    times,
    constraints,
    average,
    *,
    bpol_probe_count=None,
):
    # difference with omas version: pf_current not multiplied by nbturn
    # flux_loop divided by 2Pi

    #    EQ=ods['equilibrium']
    #    PF=ods['pf_active']
    #    MG=ods['magnetics']
    #    TF=ods['tf']

    annotate_constraint_equilibrium(EQ, times)
    nbt = len(times)
    # Each constraint is the box average of the diagnostic samples inside
    # [t_i - average, t_i + average]: every sample once, equal weights, no
    # interpolation grid of its own (issue #433).
    times = [float(t) for t in times]

    if "pf_current" in constraints:
        for channel in PF["coil"]:
            label = PF[f"coil.{channel}.name"]
            # Deliberately not read: the constraint is the measured circuit
            # current, not an ampere-turn. Reading it would only materialize
            # the path on an ODS that does not carry it.
            time = PF[f"coil.{channel}.current.time"]
            data = PF[f"coil.{channel}.current.data"]
            error = PF[f"coil.{channel}.current.data_error_upper"]

            for i in range(nbt):
                const, const_error = _window_pair(
                    time, data, error, times[i], average, what=f"pf_current[{channel}] {label}"
                )

                EQ[f"time_slice.{i}.constraints.pf_current.{channel}.measured"] = const
                EQ[
                    f"time_slice.{i}.constraints.pf_current.{channel}.measured_error_upper"
                ] = const_error
                EQ[f"time_slice.{i}.constraints.pf_current.{channel}.source"] = label

    if "bpol_probe" in constraints:
        available_bpol_probes = len(MG["b_field_pol_probe"])
        if bpol_probe_count is not None:
            available_bpol_probes = min(available_bpol_probes, int(bpol_probe_count))
        for channel in range(available_bpol_probes):
            label = MG[f"b_field_pol_probe.{channel}.identifier"]
            time = MG["time"]
            if not _signal_matches_time(
                MG,
                f"b_field_pol_probe.{channel}.field.data",
                time,
            ):
                # No raw data for this channel. OMAS array-of-structures grow
                # contiguously from index 0, so silently omitting an index
                # breaks every later index (and, via `nbprobe`, the
                # flux_loop `broken`-index offset). Preserve the position
                # and identity with a finite, explicitly zero-weighted
                # placeholder instead: weight=0 already means "excluded
                # from fitting" to the k-file writer, the same as a
                # legacy-listed broken channel.
                for i in range(nbt):
                    EQ[f"time_slice.{i}.constraints.bpol_probe.{channel}.measured"] = (
                        0.0
                    )
                    EQ[
                        f"time_slice.{i}.constraints.bpol_probe.{channel}.measured_error_upper"
                    ] = 0.0
                    EQ[f"time_slice.{i}.constraints.bpol_probe.{channel}.source"] = (
                        label
                    )
                    EQ[f"time_slice.{i}.constraints.bpol_probe.{channel}.weight"] = 0.0
                continue
            data = MG[f"b_field_pol_probe.{channel}.field.data"]
            error = MG[f"b_field_pol_probe.{channel}.field.data_error_upper"]
            for i in range(nbt):
                const, const_error = _window_pair(
                    time, data, error, times[i], average, what=f"bpol_probe[{channel}] {label}"
                )
                EQ[f"time_slice.{i}.constraints.bpol_probe.{channel}.measured"] = const
                EQ[
                    f"time_slice.{i}.constraints.bpol_probe.{channel}.measured_error_upper"
                ] = const_error
                EQ[f"time_slice.{i}.constraints.bpol_probe.{channel}.source"] = label

    if "flux_loop" in constraints:
        for channel in MG["flux_loop"]:
            label = MG[f"flux_loop.{channel}.identifier"]
            time = MG["time"]
            if not _signal_matches_time(MG, f"flux_loop.{channel}.flux.data", time):
                # Same missing-channel placeholder as bpol_probe above.
                for i in range(nbt):
                    EQ[f"time_slice.{i}.constraints.flux_loop.{channel}.measured"] = 0.0
                    EQ[
                        f"time_slice.{i}.constraints.flux_loop.{channel}.measured_error_upper"
                    ] = 0.0
                    EQ[f"time_slice.{i}.constraints.flux_loop.{channel}.source"] = label
                    EQ[f"time_slice.{i}.constraints.flux_loop.{channel}.weight"] = 0.0
                continue
            data = MG[f"flux_loop.{channel}.flux.data"]
            error = MG[f"flux_loop.{channel}.flux.data_error_upper"]
            for i in range(nbt):
                # Stored in Wb per the IDS convention; the division by 2 pi
                # happens in the k-file writer.
                const, const_error = _window_pair(
                    time, data, error, times[i], average, what=f"flux_loop[{channel}] {label}"
                )
                EQ[f"time_slice.{i}.constraints.flux_loop.{channel}.measured"] = const
                EQ[
                    f"time_slice.{i}.constraints.flux_loop.{channel}.measured_error_upper"
                ] = const_error
                EQ[f"time_slice.{i}.constraints.flux_loop.{channel}.source"] = label

    if "ip" in constraints:
        time = MG[f"ip.0.time"]
        data = MG[f"ip.0.data"]
        error = MG[f"ip.0.data_error_upper"]

        for i in range(nbt):
            const, const_error = _window_pair(time, data, error, times[i], average, what="ip")

            EQ[f"time_slice.{i}.constraints.ip.measured"] = const
            EQ[f"time_slice.{i}.constraints.ip.measured_error_upper"] = const_error

    if "diamagnetic_flux" in constraints:
        time = MG[f"diamagnetic_flux.0.time"]
        data = MG[f"diamagnetic_flux.0.data"]
        error = MG[f"diamagnetic_flux.0.data_error_upper"]

        for i in range(nbt):
            const, const_error = _window_pair(
                time, data, error, times[i], average, what="diamagnetic_flux"
            )

            EQ[f"time_slice.{i}.constraints.diamagnetic_flux.measured"] = const
            EQ[f"time_slice.{i}.constraints.diamagnetic_flux.measured_error_upper"] = (
                const_error
            )

    if "b_field_tor_vacuum_r" in constraints:
        time = TF[f"b_field_tor_vacuum_r.time"]
        data = TF[f"b_field_tor_vacuum_r.data"]
        error = TF[f"b_field_tor_vacuum_r.data_error_upper"]
        for i in range(nbt):
            const, const_error = _window_pair(
                time, data, error, times[i], average, what="b_field_tor_vacuum_r"
            )

            EQ[f"time_slice.{i}.constraints.b_field_tor_vacuum_r.measured"] = const
            EQ[
                f"time_slice.{i}.constraints.b_field_tor_vacuum_r.measured_error_upper"
            ] = const_error


def correct_flux_loop(ods, *, window=None):
    """
    In the inboard flux loop near the central solenoid,
    there is an issue where the uncertainty in the vacuum component of the total signal becomes larger than the plasma signal.
    To address this, a scaling factor vector is calculated by fitting the ratio between the measured values and the calculated values before plasma onset,
    aligning the measured data with the calculated values.

    Caclulate
    """
    #    xrange = [0.3, 0.36] # ms
    # Print the code
    print("Run Inboard Flux Loop Correction Script")

    # Find the loop voltage (flux loop) voltage onset time
    MG = ods["magnetics"]
    nbflux = len(MG["flux_loop"])
    fl_data = MG[f"flux_loop.{nbflux - 1}.flux.data"]
    fl_time = MG["time"]
    #    (fl_time, fl_data) = vest_load(shotnumber, 26)
    tstart = max(0.24, fl_time[0])
    tend = min(0.305, fl_time[-1])
    (fl_onset, _, _) = vest_signal_onoffsetpeak(
        fl_time, fl_data, tstart=tstart, tend=tend, threshold=0.01
    )
    print("Flux Loop Onset Time: ", fl_onset)

    # Find the plasma onset and offset time
    (onset, offset) = _plasma_window(ods, window)
    fl_onset = fl_onset + 0.003
    onset = onset - 0.003

    # Calculate response matrix
    (psi_total, psi_coil, psi_eddy, _, _, _, _, _) = calculate_md_by_ods(
        ods, method="vectorized"
    )

    # Extract the measured and calculated flux loop data
    measured_flux_loop = MG["flux_loop.:.flux.data"]
    calculated_flux_loop_temp = psi_coil + psi_eddy
    calculated_flux_loop = np.zeros((nbflux, len(MG["time"])))
    pf_time = np.asarray(ods["pf_active.time"], dtype=float)
    for i in range(nbflux):
        calculated_flux_loop[i, :] = np.interp(
            MG["time"], pf_time, calculated_flux_loop_temp[i, :]
        )

    # Filter the data between the flux loop onset time and plasma onset time
    fl_onset_idx = (np.abs(MG["time"] - fl_onset)).argmin()
    plas_onset_idx = (np.abs(MG["time"] - onset)).argmin()
    measured_flux_loop = measured_flux_loop[:, fl_onset_idx:plas_onset_idx]
    calculated_flux_loop = calculated_flux_loop[:, fl_onset_idx:plas_onset_idx]

    # Calculate the scaling factor for each flux loop based on least square fitting
    scaling_factor = np.ones(nbflux)
    for i in range(nbflux):
        measured = measured_flux_loop[i]
        calculated = calculated_flux_loop[i]

        denominator = np.dot(measured, calculated)
        if denominator != 0.0:
            scaling_factor[i] = np.dot(measured, measured) / denominator

    return scaling_factor


def vfit_signal_startend(time, data):
    threshold = 0.06  # minimum value
    nbt = len(time)

    # index of maximum value
    indxm = min(range(len(data)), key=lambda i: abs(data[i] - max(data)))
    indxs = -1
    indxe = -1

    # We are looking for windows that constain continue data above threshold.
    # The window we are looking for, must contain the maximum value
    for i in range(nbt):
        if data[i] >= threshold:
            if indxs == -1:
                indxs = i  # start of the window
        else:
            indxe = i - 1  # end of the window
            if indxs < indxm and indxm < indxe:
                break  # if the window contains the maximum, we stop
    #            indxs=-1
    tstart = time[indxs]
    tend = time[indxe]

    return (tstart, tend)


def smooth(array, span):
    if span % 2 == 0:
        span = span - 1

    nbv = len(array)
    out = np.zeros(nbv)
    span2 = int((span - 1) / 2)
    for i in range(span2):
        div = 2 * i + 1
        win = [j for j in range(div)]
        out[i] = np.sum(array[win]) / div
        win2 = [nbv - 1 - j for j in range(div)]
        out[nbv - 1 - i] = np.sum(array[win2]) / div

    endl = nbv - span2
    for i in range(span2, endl):
        div = span
        win = [i - span2 + j for j in range(div)]
        out[i] = np.sum(array[win]) / div

    return out


def vest_rspv1(ods, plasma, rz):
    PF = ods["pf_active"]
    PFp = ods["pf_passive"]

    nbcoil = len(PF["coil"])
    nbloop = len(PFp["loop"])
    plasma = plasma if plasma is not None else []
    nbplas = len(plasma)
    tot = len(rz)
    Br = np.zeros((tot, nbcoil + nbloop + nbplas))
    Bz = np.zeros((tot, nbcoil + nbloop + nbplas))
    Psi = np.zeros((tot, nbcoil + nbloop + nbplas))

    shft = 0.01
    for i in range(tot):
        r1 = rz[i][0]
        z1 = rz[i][1]

        # From coils
        for ii in range(nbcoil):
            nbelti = len(PF["coil.{}.element".format(ii)])
            sumr = 0.0
            sumz = 0.0
            sump = 0.0
            for jj in range(nbelti):
                nbturnl = PF["coil.{}.element.{}.turns_with_sign".format(ii, jj)]
                gtype = PF["coil.{}.element.{}.geometry.geometry_type".format(ii, jj)]
                if gtype == 1:
                    myr = PF["coil.{}.element.{}.geometry.outline.r".format(ii, jj)]
                    myz = PF["coil.{}.element.{}.geometry.outline.z".format(ii, jj)]
                    r2 = sum(myr) / len(myr)
                    z2 = sum(myz) / len(myz)
                elif gtype == 2:
                    r2 = PF["coil.{}.element.{}.geometry.rectangle.r".format(ii, jj)]
                    z2 = PF["coil.{}.element.{}.geometry.rectangle.z".format(ii, jj)]
                elif gtype == 3:
                    r2 = PF["coil.{}.element.{}.geometry.oblique.r".format(ii, jj)]
                    z2 = PF["coil.{}.element.{}.geometry.oblique.z".format(ii, jj)]
                elif gtype == 5:
                    r2 = PF["coil.{}.element.{}.geometry.annulus.r".format(ii, jj)]
                    z2 = PF["coil.{}.element.{}.geometry.annulus.z".format(ii, jj)]
                elif gtype == 6:
                    r21 = PF[
                        "coil.{}.element.{}.geometry.thick_line.first_point.r".format(
                            ii, jj
                        )
                    ]
                    r22 = PF[
                        "coil.{}.element.{}.geometry.thick_line.second_point.r".format(
                            ii, jj
                        )
                    ]
                    r2 = (r21 + r22) / 2
                    z21 = PF[
                        "coil.{}.element.{}.geometry.thick_line.first_point.z".format(
                            ii, jj
                        )
                    ]
                    z22 = PF[
                        "coil.{}.element.{}.geometry.thick_line.second_point.z".format(
                            ii, jj
                        )
                    ]
                    z2 = (z21 + z22) / 2

                if calculate_distance(r1, r2, z1, z2) < shft / 3.0:
                    print(1)
                    (myBr1, myBz1) = green_br_bz(r1 + shft, z1, r2, z2)
                    (myBr2, myBz2) = green_br_bz(r1 - shft, z1, r2, z2)
                    myP1 = green_r(r1 + shft, z1, r2, z2)
                    myP2 = green_r(r1 - shft, z1, r2, z2)
                    myBr = (myBr1 + myBr2) / 2.0
                    myBz = (myBz1 + myBz2) / 2.0
                    myP = (myP1 + myP2) / 2.0
                else:
                    (myBr, myBz) = green_br_bz(r1, z1, r2, z2)
                    myP = green_r(r1, z1, r2, z2)
                sumr = sumr + myBr * nbturnl
                sumz = sumz + myBz * nbturnl
                sump = sump + myP * nbturnl

            Br[i][ii] = sumr
            Bz[i][ii] = sumz
            Psi[i][ii] = sump

        # From wall
        for ii in range(nbloop):
            nbelti = len(PFp["loop.{}.element".format(ii)])
            sumr = 0.0
            sumz = 0.0
            sump = 0.0
            for jj in range(nbelti):
                gtype = PFp["loop.{}.element.{}.geometry.geometry_type".format(ii, jj)]
                if gtype == 1:
                    myr = PFp["loop.{}.element.{}.geometry.outline.r".format(ii, jj)]
                    myz = PFp["loop.{}.element.{}.geometry.outline.z".format(ii, jj)]
                    r2 = sum(myr) / len(myr)
                    z2 = sum(myz) / len(myz)
                elif gtype == 2:
                    r2 = PFp["loop.{}.element.{}.geometry.rectangle.r".format(ii, jj)]
                    z2 = PFp["loop.{}.element.{}.geometry.rectangle.z".format(ii, jj)]
                elif gtype == 3:
                    r2 = PFp["loop.{}.element.{}.geometry.oblique.r".format(ii, jj)]
                    z2 = PFp["loop.{}.element.{}.geometry.oblique.z".format(ii, jj)]

                if calculate_distance(r1, r2, z1, z2) < shft / 3.0:
                    print(2)
                    (myBr1, myBz1) = green_br_bz(r1 + shft, z1, r2, z2)
                    (myBr2, myBz2) = green_br_bz(r1 - shft, z1, r2, z2)
                    myP1 = green_r(r1 + shft, z1, r2, z2)
                    myP2 = green_r(r1 - shft, z1, r2, z2)
                    myBr = (myBr1 + myBr2) / 2.0
                    myBz = (myBz1 + myBz2) / 2.0
                    myP = (myP1 + myP2) / 2.0
                else:
                    (myBr, myBz) = green_br_bz(r1, z1, r2, z2)
                    myP = green_r(r1, z1, r2, z2)
                sumr = sumr + myBr
                sumz = sumz + myBz
                sump = sump + myP

            Br[i][nbcoil + ii] = sumr
            Bz[i][nbcoil + ii] = sumz
            Psi[i][nbcoil + ii] = sump

        # From plasma (if any)
        for ii in range(nbplas):
            r2 = plasma[ii][0]
            z2 = plasma[ii][1]

            if calculate_distance(r1, r2, z1, z2) < shft / 3.0:
                print(3)
                (myBr1, myBz1) = green_br_bz(r1 + shft, z1, r2, z2)
                (myBr2, myBz2) = green_br_bz(r1 - shft, z1, r2, z2)
                myP1 = green_r(r1 + shft, z1, r2, z2)
                myP2 = green_r(r1 - shft, z1, r2, z2)
                myBr = (myBr1 + myBr2) / 2.0
                myBz = (myBz1 + myBz2) / 2.0
                myP = (myP1 + myP2) / 2.0
            else:
                (myBr, myBz) = green_br_bz(r1, z1, r2, z2)
                myP = green_r(r1, z1, r2, z2)

            Br[i][nbcoil + nbloop + ii] = myBr
            Bz[i][nbcoil + nbloop + ii] = myBz
            Psi[i][nbcoil + nbloop + ii] = myP

    return (Psi, Bz, Br)


def calculate_md_by_ods(
    ods, filament_position=[], filament_fraction=[], method="vectorized"
):
    # Method
    ## nested_loop : calculate by nested loop
    ## vectorized : calculate by vectorized method (default, faster)

    # Load the magnetics, pf_active, and pf_passive ODS
    MG = ods["magnetics"]
    PFP = ods["pf_passive"]
    PF = ods["pf_active"]

    # Load the Magnetics Position
    probe_rz = []  # (r, z) for each Bz probe points
    nbprobe = len(MG["b_field_pol_probe"])

    for i in range(nbprobe):
        r = MG["b_field_pol_probe.{}.position.r".format(i)]
        z = MG["b_field_pol_probe.{}.position.z".format(i)]
        probe_rz.append([r, z])

    fl_rz = []  # (r, z) for each flux loop points
    nbfl = len(MG["flux_loop"])

    for i in range(nbfl):
        r = MG["flux_loop.{}.position.0.r".format(i)]
        z = MG["flux_loop.{}.position.0.z".format(i)]
        fl_rz.append([r, z])

    # Calculate response matrix
    #    if filament_position != []:
    (cpsi, _, _) = vest_rspv1(
        ods, filament_position, fl_rz
    )  # flux loop response matrix
    (_, cbz, _) = vest_rspv1(
        ods, filament_position, probe_rz
    )  # Bz probe response matrix

    # Load and Interpolate Ip to the time of PF
    if filament_position != []:
        Ip_total = np.interp(PF["time"], MG["ip.0.time"], MG["ip.0.data"])
        Ip = np.array([Ip_total * fraction for fraction in filament_fraction])

    # Initialize the variables
    nbtime = len(PF["time"])
    nbcoil = len(PF["coil"])
    nbloop = len(PFP["loop"])
    if filament_position != []:
        nbplas = len(Ip)
    else:
        nbplas = 0

    bz_total = np.zeros((nbprobe, nbtime))
    bz_coil = np.zeros((nbprobe, nbtime))
    bz_eddy = np.zeros((nbprobe, nbtime))
    if filament_position != []:
        bz_plas = np.zeros((nbprobe, nbtime))

    psi_total = np.zeros((nbfl, nbtime))
    psi_coil = np.zeros((nbfl, nbtime))
    psi_eddy = np.zeros((nbfl, nbtime))
    if filament_position != []:
        psi_plas = np.zeros((nbfl, nbtime))

    # Calculate the induced magnetic field and poloidal flux quantities by the PF, eddy currents, and filamentry plasma
    if method == "nested_loop":
        I_coil = np.zeros(nbcoil + nbloop + nbplas)
        I_plas = np.zeros(nbcoil + nbloop + nbplas)
        I_eddy = np.zeros(nbcoil + nbloop + nbplas)

        for k in range(nbtime):
            for i in range(nbcoil):
                I_coil[i] = PF["coil.{}.current.data".format(i)][k]
            for i in range(nbloop):
                I_eddy[nbcoil + i] = PFP["loop.{}.current".format(i)][k]
            for i in range(nbplas):
                I_plas[nbcoil + nbloop + i] = Ip[i][k]

            for i in range(nbprobe):
                bz_coil[i][k] = np.matmul(cbz[i], I_coil)
                bz_eddy[i][k] = np.matmul(cbz[i], I_eddy)
                bz_plas[i][k] = np.matmul(cbz[i], I_plas)
                bz_total[i][k] = bz_coil[i][k] + bz_eddy[i][k] + bz_plas[i][k]
        for i in range(nbfl):
            psi_coil[i][k] = np.matmul(cpsi[i], I_coil)
            psi_eddy[i][k] = np.matmul(cpsi[i], I_eddy)
            psi_plas[i][k] = np.matmul(cpsi[i], I_plas)
            psi_total[i][k] = psi_coil[i][k] + psi_eddy[i][k] + psi_plas[i][k]

    elif method == "vectorized":
        # Initialize current arrays
        I_coil = np.zeros((nbcoil + nbloop + nbplas, nbtime))
        I_eddy = np.zeros((nbcoil + nbloop + nbplas, nbtime))
        I_plas = np.zeros((nbcoil + nbloop + nbplas, nbtime))

        # Assign current values
        print(nbtime, len(PFP["time"]))
        I_coil[:nbcoil] = PF["coil.:.current.data"]
        I_eddy[nbcoil : nbcoil + nbloop] = PFP["loop.:.current"]
        if filament_position != []:
            I_plas[nbcoil + nbloop :] = Ip

        # Compute magnetic field contributions
        bz_coil = np.dot(cbz, I_coil)
        bz_eddy = np.dot(cbz, I_eddy)
        if filament_position != []:
            bz_plas = np.dot(cbz, I_plas)
        else:
            bz_plas = 0.0
        bz_total = bz_coil + bz_eddy + bz_plas

        # Compute poloidal flux contributions
        psi_coil = np.dot(cpsi, I_coil)
        psi_eddy = np.dot(cpsi, I_eddy)
        if filament_position != []:
            psi_plas = np.dot(cpsi, I_plas)
        else:
            psi_plas = 0.0
        psi_total = psi_coil + psi_eddy + psi_plas

    return psi_total, psi_coil, psi_eddy, psi_plas, bz_total, bz_coil, bz_eddy, bz_plas


def vest_signal_onoffsetpeak(time, data, tstart, tend, threshold):
    """
    Finds the onset, peak, and offset times of the signal within the specified time range
    by normalizing the signal and identifying the time window where the signal exceeds a threshold.

    Parameters:
        time (array): Array of time values corresponding to the data points.
        data (array): Array of data values to analyze.
        tstart (float): Start time of the interval to analyze.
        tend (float): End time of the interval to analyze.
        threshold (float): Minimum value to consider as signal onset (0 ~ 1).

    Returns:
        t_onset (float): Time of signal onset.
        t_peak (float): Time of signal peak.
        t_offset (float): Time of signal offset.
    """
    # Parameters
    smooth_factor = 50

    # Smooth the data using Savitzky-Golay filter
    data_smoothed = savgol_filter(
        data, smooth_factor, 3
    )  # window size 50, polynomial order 3
    nbt = len(time)

    # Find indices for the start and end times
    ind_start = np.argmin(np.abs(time - tstart))
    ind_end = np.argmin(np.abs(time - tend))

    # Ensure the specified range is within the data length
    if ind_start < 0 or ind_end > nbt or ind_end <= ind_start:
        raise ValueError("Specified time range is out of bounds.", ind_start, ind_end)

    mini = np.min(data_smoothed[ind_start:ind_end])
    maxi = np.max(data_smoothed[ind_start:ind_end])

    # Normalize the data using the larger absolute value of the minimum or maximum
    if abs(mini) > abs(maxi):
        data_norm = data_smoothed / mini
    else:
        data_norm = data_smoothed / maxi

    # Find the index of the maximum value within the specified range
    indxm = np.argmax(data_norm[ind_start:ind_end]) + ind_start

    indxs = -1
    indxe = -1

    # Look for windows that contain continuous data above the threshold
    # The window must contain the maximum value
    for i in range(ind_start, ind_end):
        if data_norm[i] >= threshold:
            if indxs == -1:
                indxs = i  # Start of the window
        else:
            if indxs != -1:
                indxe = i - 1  # End of the window
                if indxs < indxm < indxe:
                    break  # Stop if the window contains the maximum
                indxs = -1

    # Determine onset, peak, and offset times
    t_onset = time[indxs] if indxs != -1 else 0
    t_peak = time[indxm]
    t_offset = time[indxe] if indxe != -1 else 0

    return t_onset, t_peak, t_offset


def _plasma_window(ods, window=None):
    """``(onset, offset)`` for this module's own callers.

    The ``window`` a policy-aware caller already resolved (the constraints
    script hands over the one it chose) wins; otherwise the shared timing
    through the same helper the ``vaft.omas`` finders use, which raises
    ``ValueError`` naming the reason when no source shows a plasma.
    """
    if window is not None:
        onset, offset = window
        return float(onset), float(offset)
    from vaft.omas.general import _plasma_timing_or_raise

    return _plasma_timing_or_raise(ods).window


def vest_Halpha_tstart_tend(ods):
    """Deprecated H-alpha plasma window (0.3-0.36 s, min-normalised, legacy detector).

    Superseded by :func:`vaft.omas.plasma_timing.plasma_timing`, which finds
    the H-alpha line by label, validates it and returns the window with its
    provenance inside the configured ``plasma_analysis`` range (issue #409).
    """
    warnings.warn(
        "vest_Halpha_tstart_tend() is deprecated and will be removed in the next release; use "
        "vaft.omas.plasma_timing.plasma_timing(ods).window instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Load the data
    SP = ods["spectrometer_uv"]
    data = SP["channel.0.processed_line.0.intensity.data"]
    time = SP["time"]
    #    (time, data) = vest_load(int(shot), 101)
    data_smoothed = smooth(data, 10)

    # Look for the minimum (Halpha signal negative) between 0.3 and 0.6 s
    indx1 = min(range(len(time)), key=lambda i: abs(time[i] - 0.3))
    indx2 = min(range(len(time)), key=lambda i: abs(time[i] - 0.36))
    mini = min(data_smoothed[indx1:indx2])

    # Normalize data
    if mini != 0:
        data_smoothed = data_smoothed / mini

    #    plot(time[indx1:indx2], data_smoothed[indx1:indx2])

    # Find start and end time
    (tstart, tend) = vfit_signal_startend(time[indx1:indx2], data_smoothed[indx1:indx2])

    return (tstart, tend)


def set_discharge_index(ods):
    """
    Set the time range higher than 20 kA and between 0.3 ~ 0.36 sec with manual tstep.
    If no Ip > 20 kA is found, return time range and 'vacuum'; otherwise return time range and 'plasma'.

    Deprecated: the EFIT constraint script now takes the configured
    ``plasma_analysis`` range (``resolve_plasma_timing_policy().window``)
    intersected with the window ``vaft.omas.plasma_timing.plasma_timing``
    detects, and flags the range fallback instead of a silent 'vacuum' status
    (issue #409).
    """
    warnings.warn(
        "set_discharge_index() is deprecated and will be removed in the next release; use the plasma_analysis window from "
        "vaft.machine_mapping.utils.resolve_plasma_timing_policy() intersected with "
        "vaft.omas.plasma_timing.plasma_timing(ods).window instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    Ip = ods["magnetics.ip.0.data"]
    time = ods["magnetics.ip.0.time"]

    # Define base time range (0.280.38 s)
    base_index = (time >= 0.28) & (time <= 0.38)

    # Filter by both time and current threshold
    valid_index = base_index & (Ip > 20e3)

    # Check if plasma current > 20 kA exists
    if np.any(valid_index):
        status = "plasma"
        selected_index = valid_index
    else:
        status = "vacuum"
        selected_index = base_index

    print("status", status)
    time = time[selected_index]
    Ip = Ip[selected_index]

    return time

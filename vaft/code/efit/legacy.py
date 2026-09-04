"""Legacy VEST/OMFIT signal-processing and constraint-shaping helpers.

These functions back :func:`vaft.code.efit.generate_constraints_ods` (and, for
``correct_flux_loop``, the routine-pipeline
``workflow/automatic_pipeline_1_routine_data_processing/generate_constraints_ods.py``
script directly). Moved verbatim out of the former monolithic ``efit.py``.
"""

from vaft.formula import green_br_bz, green_r, calculate_distance
import numpy as np

from vaft.ods_access import path_value

import statistics
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

    EQ["ids_properties.comment"] = "constraint equilibrium"
    EQ["ids_properties.homogeneous_time"] = 1
    EQ["time"] = times
    nbt = len(times)

    ave_time = []
    for i in range(nbt):
        ave_time.append(np.arange(times[i] - average, times[i] + average, 0.0001))

    if "pf_current" in constraints:
        for channel in PF["coil"]:
            label = PF[f"coil.{channel}.name"]
            turns = PF[f"coil.{channel}.element.0.turns_with_sign"]
            time = PF[f"coil.{channel}.current.time"]
            data = PF[f"coil.{channel}.current.data"]
            error = PF[f"coil.{channel}.current.data_error_upper"]

            for i in range(nbt):
                const = statistics.mean(np.interp(ave_time[i], time, data))
                const_error = statistics.mean(np.interp(ave_time[i], time, error))

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
                const = statistics.mean(np.interp(ave_time[i], time, data))
                const_error = statistics.mean(np.interp(ave_time[i], time, error))
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
                # const=statistics.mean(np.interp(ave_time[i],time,data/2/math.pi)) # Origianl version
                # const_error=statistics.mean(np.interp(ave_time[i],time,error/2/math.pi))
                const = statistics.mean(
                    np.interp(ave_time[i], time, data)
                )  # Modified Version for ods/ids convention (division by 2Pi are conducted in the k-file generation)
                const_error = statistics.mean(np.interp(ave_time[i], time, error))  # Wb
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
            const = statistics.mean(np.interp(ave_time[i], time, data))
            const_error = statistics.mean(np.interp(ave_time[i], time, error))

            EQ[f"time_slice.{i}.constraints.ip.measured"] = const
            EQ[f"time_slice.{i}.constraints.ip.measured_error_upper"] = const_error

    if "diamagnetic_flux" in constraints:
        time = MG[f"diamagnetic_flux.0.time"]
        data = MG[f"diamagnetic_flux.0.data"]
        error = MG[f"diamagnetic_flux.0.data_error_upper"]

        for i in range(nbt):
            const = statistics.mean(np.interp(ave_time[i], time, data))
            const_error = statistics.mean(np.interp(ave_time[i], time, error))
            #            print(i,const*2*math.pi)

            EQ[f"time_slice.{i}.constraints.diamagnetic_flux.measured"] = const
            EQ[f"time_slice.{i}.constraints.diamagnetic_flux.measured_error_upper"] = (
                const_error
            )

    if "b_field_tor_vacuum_r" in constraints:
        time = TF[f"b_field_tor_vacuum_r.time"]
        data = TF[f"b_field_tor_vacuum_r.data"]
        error = TF[f"b_field_tor_vacuum_r.data_error_upper"]
        for i in range(nbt):
            const = statistics.mean(np.interp(ave_time[i], time, data))
            const_error = statistics.mean(np.interp(ave_time[i], time, error))

            EQ[f"time_slice.{i}.constraints.b_field_tor_vacuum_r.measured"] = const
            EQ[
                f"time_slice.{i}.constraints.b_field_tor_vacuum_r.measured_error_upper"
            ] = const_error


def vfit_pf_active_efit26(PF, PF_orig, shot, tstart, tend, dt):
    nbcoil = 26  # number of coil
    PFname = [
        "PF1-1",
        "PF1-2",
        "PF1-3",
        "PF1-4",
        "PF1-5",
        "PF1-6",
        "PF1-7",
        "PF1-8",
        "PF2U",
        "PF2L",
        "PF3U",
        "PF3L",
        "PF4U",
        "PF4L",
        "PF5U",
        "PF5L",
        "PF6U",
        "PF6L",
        "PF7U",
        "PF7L",
        "PF8U",
        "PF8L",
        "PF9U",
        "PF9L",
        "PF10U",
        "PF10L",
    ]
    Rcoil = 1.68e-8  # Copper
    # For resistance calculation
    if shot <= 45957:
        nbELT = [
            79,
            79,
            79,
            79,
            79,
            79,
            79,
            79,
            250,
            250,
            8,
            8,
            8,
            8,
            12,
            12,
            12,
            12,
            24,
            24,
            24,
            24,
            24,
            24,
            24,
            24,
        ]  # number of turn of each coil
        RPF = [
            0.053,
            0.053,
            0.053,
            0.053,
            0.053,
            0.053,
            0.053,
            0.053,
            0.104,
            0.104,
            0.29,
            0.29,
            0.57,
            0.57,
            0.71,
            0.71,
            0.71,
            0.71,
            0.71,
            0.71,
            0.71,
            0.71,
            0.93,
            0.93,
            0.93,
            0.93,
        ]  # Radius
        ZPF = [
            0.15,
            0.45,
            0.75,
            1.05,
            -0.15,
            -0.45,
            -0.75,
            -1.05,
            0.98,
            -0.98,
            1.25,
            -1.25,
            1.25,
            -1.25,
            1.15,
            -1.15,
            0.875,
            -0.875,
            0.72,
            -0.72,
            0.68,
            -0.68,
            0.5223,
            -0.5223,
            0.4807,
            -0.4807,
        ]
        dRPF = [
            0.0172,
            0.0172,
            0.0172,
            0.0172,
            0.0172,
            0.0172,
            0.0172,
            0.0172,
            0.04,
            0.04,
            0.028,
            0.028,
            0.028,
            0.028,
            0.042,
            0.042,
            0.042,
            0.042,
            0.042,
            0.042,
            0.042,
            0.042,
            0.042,
            0.042,
            0.042,
            0.042,
        ]
        dZPF = [
            0.3,
            0.3,
            0.3,
            0.3,
            0.3,
            0.3,
            0.3,
            0.3,
            0.38,
            0.38,
            0.0145,
            0.0145,
            0.0145,
            0.0145,
            0.0145,
            0.0145,
            0.0145,
            0.0145,
            0.0324,
            0.0324,
            0.0324,
            0.0324,
            0.0324,
            0.0324,
            0.0324,
            0.0324,
        ]
    else:
        nbELT = [
            79,
            79,
            79,
            79,
            79,
            79,
            79,
            79,
            250,
            250,
            8,
            8,
            8,
            8,
            12,
            12,
            24,
            24,
            12,
            12,
            24,
            24,
            24,
            24,
            24,
            24,
        ]  # number of turn of each coil
        RPF = [
            0.053,
            0.053,
            0.053,
            0.053,
            0.053,
            0.053,
            0.053,
            0.053,
            0.104,
            0.104,
            0.29,
            0.29,
            0.57,
            0.57,
            0.71,
            0.71,
            0.71,
            0.71,
            0.71,
            0.71,
            0.71,
            0.71,
            0.93,
            0.93,
            0.93,
            0.93,
        ]  # Radius
        ZPF = [
            0.15,
            0.45,
            0.75,
            1.05,
            -0.15,
            -0.45,
            -0.75,
            -1.05,
            0.98,
            -0.98,
            1.25,
            -1.25,
            1.25,
            -1.25,
            1.15,
            -1.15,
            0.8827,
            -0.8827,
            0.7119,
            -0.7119,
            0.68,
            -0.68,
            0.5223,
            -0.5223,
            0.4807,
            -0.4807,
        ]
        dRPF = [
            0.0172,
            0.0172,
            0.0172,
            0.0172,
            0.0172,
            0.0172,
            0.0172,
            0.0172,
            0.04,
            0.04,
            0.028,
            0.028,
            0.028,
            0.028,
            0.042,
            0.042,
            0.042,
            0.042,
            0.042,
            0.042,
            0.042,
            0.042,
            0.042,
            0.042,
            0.042,
            0.042,
        ]
        dZPF = [
            0.3,
            0.3,
            0.3,
            0.3,
            0.3,
            0.3,
            0.3,
            0.3,
            0.38,
            0.38,
            0.0145,
            0.0145,
            0.0145,
            0.0145,
            0.0145,
            0.0145,
            0.0308,
            0.0308,
            0.0162,
            0.0162,
            0.0324,
            0.0324,
            0.0324,
            0.0324,
            0.0324,
            0.0324,
        ]

    #    (time,data)=vest_loadn(shot,'PF1 Current')
    time = PF_orig["time"]
    if dt > 0:
        tstart = max(tstart, time[0])
        tend = min(tend, time[-1])
        time_1 = np.arange(tstart, tend, dt)
    else:
        time_1 = time

    PF["ids_properties.comment"] = "PF config from vfit_pf_active_efit16"
    PF["ids_properties.homogeneous_time"] = 1

    PF["time"] = time_1
    nbt = len(time_1)

    for i in range(nbcoil):
        PF["coil.{}.name".format(i)] = PFname[i]
        PF["coil.{}.identifier".format(i)] = PFname[i]

    for i in range(nbcoil):
        APF = dRPF[i] * dZPF[i]
        PF["coil.{}.resistance".format(i)] = 2.0 * math.pi * Rcoil * RPF[i] / APF
        PF["coil.{}.element.0.turns_with_sign".format(i)] = nbELT[i]
        PF["coil.{}.element.0.geometry.geometry_type".format(i)] = 2
        PF["coil.{}.element.0.geometry.rectangle.r".format(i)] = RPF[i]
        PF["coil.{}.element.0.geometry.rectangle.z".format(i)] = ZPF[i]
        PF["coil.{}.element.0.geometry.rectangle.width".format(i)] = dRPF[i]
        PF["coil.{}.element.0.geometry.rectangle.height".format(i)] = dZPF[i]
        PF["coil.{}.element.0.area".format(i)] = APF

    # Take current from VEST DB
    PFdata = []
    nbcoil2 = len(PF_orig["coil"])
    for i in range(nbcoil2):
        PFdata.append(PF_orig[f"coil.{i}.current.data"])

    conver = [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        2,
        2,
        3,
        3,
        4,
        4,
        5,
        5,
        6,
        6,
        7,
        7,
        8,
        8,
        9,
        9,
    ]
    for i in range(nbcoil):
        j = conver[i]
        PF[f"coil.{i}.current.data"] = np.interp(time_1, time, PFdata[j])


#        PF[f'coil.{i}.current.time']=time_1


def correct_flux_loop(ods):
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
    (onset, offset) = vest_Halpha_tstart_tend(ods)
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
    ods, filament_position=[], filament_fraction=[], method="vecterized"
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


def brokenFinder(ods, option=2):
    # Calculate response matrix
    (psi_total, psi_coil, psi_eddy, psi_plas, bz_total, bz_coil, bz_eddy, bz_plas) = (
        calculate_md_by_ods(ods, method="vectorized")
    )

    # Find the plasma onset and offset time
    (onset, offset) = vest_Halpha_tstart_tend(ods)
    bz_calc_time = ods["pf_active.time"]
    bz_exp_time = ods["magnetics.time"]

    min_time = max(bz_calc_time[0], bz_exp_time[0])
    max_time = min(bz_calc_time[-1], bz_exp_time[-1])

    # phase1: time before plasma
    time1 = np.linspace(min_time, onset - 0.015, 11)
    # phase2: time after plasma
    time2 = np.linspace(offset + 0.001, max_time, 11)

    #    print(time1)
    #    print(time2)

    # list of probes:
    broken = []

    if option == 1:
        # brokenThreshold
        brokenThreshold = 0.95
        # Bz probes
        for index in range(64):
            bz_calc = bz_coil[index] + bz_eddy[index]
            bz_exp = ods["magnetics"][f"b_field_pol_probe.{index}.field.data"]

            # phase 1:
            bz_calc1 = np.interp(time1, bz_calc_time, bz_calc)
            bz_exp1 = np.interp(time1, bz_exp_time, bz_exp)

            cxy = np.corrcoef(bz_calc1, bz_exp1)
            if cxy[0][1] < brokenThreshold:
                broken.append(index + 1)
            #                print('1',index+1,cxy[0][1])

            else:
                # phase 2:
                bz_calc2 = np.interp(time2, bz_calc_time, bz_calc)
                bz_exp2 = np.interp(time2, bz_exp_time, bz_exp)

                cxy = np.corrcoef(bz_calc2, bz_exp2)
                if cxy[0][1] < brokenThreshold:
                    broken.append(index + 1)
        #                    print('2',index+1,cxy[0][1])

        # Flux loop
        for index in range(11):
            psi_calc = psi_coil[index] + psi_eddy[index]
            psi_exp = ods["magnetics"][f"flux_loop.{index}.flux.data"]

            psi_calc1 = np.interp(time1, bz_calc_time, psi_calc)
            psi_exp1 = np.interp(time1, bz_exp_time, psi_exp)

            cxy = np.corrcoef(psi_calc1, psi_exp1)
            if cxy[0][1] < brokenThreshold:
                broken.append(64 + index + 1)
            #                print('1',index+65,cxy[0][1])

            else:
                # phase 2:
                psi_calc2 = np.interp(time2, bz_calc_time, psi_calc)
                psi_exp2 = np.interp(time2, bz_exp_time, psi_exp)

                cxy = np.corrcoef(psi_calc2, psi_exp2)
                if cxy[0][1] < brokenThreshold:
                    broken.append(64 + index + 1)
    #                    print('2',index+65,cxy[0][1])

    else:
        # brokenThreshold
        brokenThreshold = 0.005  # mV
        # Bz probes
        for index in range(64):
            if index < 27:
                brokenThreshold = 0.009
            elif index < 48:
                brokenThreshold = 0.005
            else:
                brokenThreshold = 0.007

            bz_calc = bz_coil[index] + bz_eddy[index]
            bz_exp = ods["magnetics"][f"b_field_pol_probe.{index}.field.data"]

            # phase 1:
            bz_calc1 = np.interp(time1, bz_calc_time, bz_calc)
            bz_exp1 = np.interp(time1, bz_exp_time, bz_exp)

            # max1=max(abs(bz_calc1))
            # max2=max(abs(bz_exp1))
            # max3=max(max1,max2)

            cxy = abs(bz_calc1 - bz_exp1)
            max4 = max(cxy)

            if max4 > brokenThreshold:
                broken.append(index + 1)
            #                print('1',index+1,max4)

            else:
                # phase 2:
                bz_calc2 = np.interp(time2, bz_calc_time, bz_calc)
                bz_exp2 = np.interp(time2, bz_exp_time, bz_exp)

                # max1=max(abs(bz_calc2))
                # max2=max(abs(bz_exp2))
                # max3=max(max1,max2)

                cxy = abs(bz_calc2 - bz_exp2)
                max4 = max(cxy)

                if max4 > brokenThreshold:
                    broken.append(index + 1)
        #                    print('2',index+1,max4)

        # Flux loop
        for index in range(11):
            psi_calc = psi_coil[index] + psi_eddy[index]
            psi_exp = ods["magnetics"][f"flux_loop.{index}.flux.data"]

            psi_calc1 = np.interp(time1, bz_calc_time, psi_calc)
            psi_exp1 = np.interp(time1, bz_exp_time, psi_exp)

            # max1=max(abs(psi_calc1))
            # max2=max(abs(psi_exp1))
            # max3=max(max1,max2)

            cxy = abs(psi_calc1 - psi_exp1)
            max4 = max(cxy)

            if max4 > brokenThreshold:
                broken.append(64 + index + 1)
            #                print('1',index+65,max4)

            else:
                # phase 2:
                psi_calc2 = np.interp(time2, bz_calc_time, psi_calc)
                psi_exp2 = np.interp(time2, bz_exp_time, psi_exp)

                # max1=max(abs(psi_calc2))
                # max2=max(abs(psi_exp2))
                # max3=max(max1,max2)

                cxy = abs(psi_calc2 - psi_exp2)
                max4 = max(cxy)
                if max4 > brokenThreshold:
                    broken.append(64 + index + 1)
    #                    print('2',index+65,max4)

    return broken


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


def vest_Halpha_tstart_tend(ods):
    """Deprecated H-alpha plasma window (0.3-0.36 s, min-normalised, legacy detector).

    Superseded by :func:`vaft.omas.plasma_timing.plasma_timing`, which finds
    the H-alpha line by label, validates it and returns the window with its
    provenance inside the configured ``plasma_analysis`` range (issue #409).
    """
    warnings.warn(
        "vest_Halpha_tstart_tend() is deprecated; use "
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
        "set_discharge_index() is deprecated; use the plasma_analysis window from "
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

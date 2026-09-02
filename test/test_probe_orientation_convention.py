"""The stored probe orientation must mean, to a DD reader, what the probes measure.

Issue #288. VEST's poloidal probes measure ``+Bz``. The IMAS DD defines
``poloidal_angle`` as a **clockwise** angle from ``+R``, so the sensitive axis is
``(cos, -sin)`` and ``+Bz`` must be stored as ``3*pi/2``.

VAFT stored ``pi/2`` and projected with ``(cos, +sin)``. Those are two errors
that cancel: every VAFT-internal result was right, while the value written into
the IDS told any DD-conformant reader the probes measure ``-Bz``. The tests here
pin both halves, because moving either one alone re-introduces the inversion.
"""

from __future__ import annotations

import math

import numpy as np
import pytest


def test_the_data_dictionary_defines_the_angle_as_clockwise():
    """The whole fix rests on this, so read it from the DD rather than assume."""
    from omas.omas_utils import omas_info_node

    doc = omas_info_node("magnetics.b_field_pol_probe.:.poloidal_angle")["documentation"]
    assert "clockwise" in doc.lower()
    # Zero points along +R, which is what makes (cos, -sin) the axis.
    assert "increasing major radius" in doc.lower()


def test_stored_angle_declares_plus_bz_to_a_dd_conformant_reader():
    from vaft.machine_mapping.magnetics import POLOIDAL_ANGLE

    axis = (math.cos(POLOIDAL_ANGLE), -math.sin(POLOIDAL_ANGLE))
    assert axis[0] == pytest.approx(0.0, abs=1e-12)
    assert axis[1] == pytest.approx(+1.0), "the DD reading of the stored angle must be +Bz"
    # pi/2 is the value that was wrong; guard against a silent revert.
    assert POLOIDAL_ANGLE != pytest.approx(math.pi / 2)


def test_the_impa_vertical_sensors_use_the_same_convention():
    from vaft.machine_mapping.impa import IMPA_POLOIDAL_ANGLE
    from vaft.machine_mapping.magnetics import POLOIDAL_ANGLE

    assert IMPA_POLOIDAL_ANGLE == pytest.approx(POLOIDAL_ANGLE)
    assert -math.sin(IMPA_POLOIDAL_ANGLE) == pytest.approx(+1.0)


def test_the_consumer_projects_with_the_clockwise_axis():
    """The projection and the stored angle must move together.

    Fixing the stored angle while leaving the consumer on ``(cos, +sin)`` would
    invert every VAFT-internal result, which is the failure this pins.
    """
    import inspect

    from vaft.formula.magnetics import probe_axis, project_poloidal_field
    from vaft.machine_mapping.magnetics import POLOIDAL_ANGLE
    from vaft.omas import vacuum_magnetics

    # The consumer no longer spells the projection out: it goes through the
    # one shared helper, so the rule cannot be re-derived wrongly here ...
    source = inspect.getsource(vacuum_magnetics)
    assert "project_poloidal_field(" in source
    assert "np.sin(angle)" not in source, "no private copy of the projection rule"
    # ... and the helper is the clockwise reading: the stored +Bz angle
    # projects to +Bz, and a pure Bz field comes back with its own sign.
    c_r, c_z = probe_axis(POLOIDAL_ANGLE)
    assert c_r == pytest.approx(0.0, abs=1e-12)
    assert c_z == pytest.approx(+1.0)
    assert project_poloidal_field(0.0, 0.37, POLOIDAL_ANGLE) == pytest.approx(+0.37)


def test_the_projection_recovers_plus_bz_from_the_stored_angle():
    """End to end: a pure +Bz field through the stored angle and the projection."""
    from vaft.machine_mapping.magnetics import POLOIDAL_ANGLE

    b_r, b_z = 0.0, 0.37
    response = b_r * math.cos(POLOIDAL_ANGLE) - b_z * math.sin(POLOIDAL_ANGLE)
    assert response == pytest.approx(+0.37), "a +Bz field must give a positive response"


def test_the_impa_crosstalk_offset_preserves_the_physical_axis():
    """The offset is subtracted because the angle now runs the other way.

    An axis written ``pi/2 + delta`` counter-clockwise is the same axis as
    ``3*pi/2 - delta`` clockwise; that identity is why the sign flipped.
    """
    from vaft.machine_mapping.impa import IMPA_POLOIDAL_ANGLE

    for delta in (0.0, 0.05, -0.03, 0.2):
        counter_clockwise = (math.cos(math.pi / 2 + delta), math.sin(math.pi / 2 + delta))
        clockwise = (
            math.cos(IMPA_POLOIDAL_ANGLE - delta),
            -math.sin(IMPA_POLOIDAL_ANGLE - delta),
        )
        assert clockwise[0] == pytest.approx(counter_clockwise[0], abs=1e-12), delta
        assert clockwise[1] == pytest.approx(counter_clockwise[1], abs=1e-12), delta


def test_probe_orientation_is_established_from_the_coil_forward_model():
    """The empirical basis for +Bz, rerun rather than cited.

    Forward-models the PF coil response at each probe over the packaged
    ``pf_active`` geometry and correlates it against the mapped probe signals in
    the pre-plasma vacuum window, where the plasma contributes nothing. A probe
    measuring +Bz correlates positively with the modelled +Bz.

    Coils only, no eddy model, so the correlation saturates around 0.85 rather
    than 1.0 -- the residual is the vessel response. The sign is what this
    establishes, and it is unambiguous: 62 of 63 probes.
    """
    from vaft.data.resources import data_path, require_repository_sample
    from vaft.formula.green import green_br_bz_exact
    import vaft

    sample = require_repository_sample(data_path("samples/39915/omas.json.gz"))
    ods = vaft.omas.load(str(sample))

    coil_count = len(ods["pf_active.coil"])
    r_src, z_src, turns, owner = [], [], [], []
    for coil in range(coil_count):
        base = f"pf_active.coil.{coil}"
        for element in range(len(ods[f"{base}.element"])):
            geometry = f"{base}.element.{element}.geometry.rectangle"
            r_src.append(float(ods[f"{geometry}.r"]))
            z_src.append(float(ods[f"{geometry}.z"]))
            turns.append(float(ods[f"{base}.element.{element}.turns_with_sign"]))
            owner.append(coil)
    r_src = np.asarray(r_src); z_src = np.asarray(z_src)
    turns = np.asarray(turns); owner = np.asarray(owner)

    time = np.asarray(ods["pf_active.coil.0.current.time"], dtype=float)
    currents = np.stack([
        np.asarray(ods[f"pf_active.coil.{coil}.current.data"], dtype=float)
        for coil in range(coil_count)
    ])
    plasma = np.interp(
        time,
        np.asarray(ods["magnetics.ip.0.time"], dtype=float),
        np.asarray(ods["magnetics.ip.0.data"], dtype=float),
    )
    onset = time[np.argmax(np.abs(plasma) > 0.02 * np.nanmax(np.abs(plasma)))]
    vacuum = time < onset
    assert vacuum.sum() > 100, "need a usable pre-plasma window"

    correlations = []
    for index in range(len(ods["magnetics.b_field_pol_probe"])):
        base = f"magnetics.b_field_pol_probe.{index}"
        try:
            r = float(ods[f"{base}.position.r"])
            z = float(ods[f"{base}.position.z"])
            measured = np.asarray(ods[f"{base}.field.data"], dtype=float)
            stamps = np.asarray(ods[f"{base}.field.time"], dtype=float)
        except Exception:
            continue
        if stamps.size < 2 or measured.size != stamps.size:
            continue
        _, b_z = green_br_bz_exact(r, z, r_src, z_src)
        response = np.array([
            float(np.sum(b_z[owner == coil] * turns[owner == coil])) for coil in range(coil_count)
        ])
        modelled = response @ currents
        sampled = np.interp(time, stamps, measured)
        if np.std(modelled[vacuum]) < 1e-14 or np.std(sampled[vacuum]) < 1e-14:
            continue
        correlations.append(float(np.corrcoef(sampled[vacuum], modelled[vacuum])[0, 1]))

    correlations = np.asarray(correlations)
    assert correlations.size > 50, "expected most probes to be usable"
    positive = int((correlations > 0).sum())
    assert positive >= correlations.size - 2, (
        f"only {positive}/{correlations.size} probes correlate positively with +Bz"
    )
    assert np.median(correlations) > 0.7

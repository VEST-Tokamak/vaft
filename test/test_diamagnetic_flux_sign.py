"""The diamagnetic-flux sign convention, end to end (issue #385).

One convention, EFIT's own: DFLUX is signed, and the reconstruction is fitted
against ``cdflux = integral (B_t - B_tv) dA`` signed with B_t.  In VEST's
positive toroidal field a diamagnetic plasma is therefore a *negative* flux.
Three things must agree with that and are pinned here on the packaged
shot-39915 sample:

* the k-file writer passes the stored, signed measurement through (it used to
  take the absolute value, inherited from a magnitude-only donor fitter);
* the reconstructed-flux kernel is EFIT's definition, so its sign follows the
  F profile it is given;
* the measured loop carries the diamagnetic sign in that same convention --
  shown by the virial energy balance closing with the measured flux and
  breaking with its negation.

The sample's own reconstruction is paramagnetic (its F profile rises toward
the axis) and so its computed flux is positive against a negative
measurement.  That is a disagreement between reconstruction and measurement,
which the #72 report flags, not a convention error in the computed path; the
last test records it so that the two are not confused again.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
from omas import ODS

import vaft
from vaft.code.efit.config import EFITConstraintConfig, EFITScientificConfig
from vaft.code.efit.kfile import generate_kfile
from vaft.omas.sample import sample_ods

TABLES = Path(vaft.__file__).parent / "data" / "efit"
SHOT, SLICE = 39915, 0


@pytest.fixture(scope="module")
def sample():
    return sample_ods()


def _constraints_from(sample, index=SLICE):
    """One slice of the sample's constraints, as the k-file writer expects them."""
    ods = ODS(consistency_check=False)
    time = float(sample["equilibrium.time"][index])
    ods["equilibrium.time"] = np.array([time])
    ods["equilibrium.time_slice.0.time"] = time
    ods["equilibrium.time_slice.0.constraints"] = copy.deepcopy(
        sample[f"equilibrium.time_slice.{index}.constraints"]
    )
    ods["equilibrium.code.parameters.time_slice.0.IN1.INPUT_DIR"] = f"{TABLES}/"
    ods["equilibrium.code.parameters.time_slice.0.IN1.VCURRT"] = np.zeros(950)
    return ods


def _dflux_line(tmp_path, ods, constraints=None):
    generate_kfile(
        ods, SHOT, save_dir=str(tmp_path),
        config=EFITScientificConfig(constraints=constraints or EFITConstraintConfig()),
    )
    text = next((tmp_path / "kfile").iterdir()).read_text(encoding="utf-8")
    return next(line for line in text.splitlines() if line.startswith("DFLUX"))


# --- the k-file writer ------------------------------------------------------

def test_the_default_writes_the_signed_measurement(sample, tmp_path):
    assert EFITConstraintConfig().diamagnetic_flux_sign == "imas"
    ods = _constraints_from(sample)
    measured = float(ods["equilibrium.time_slice.0.constraints.diamagnetic_flux.measured"])
    assert measured < 0, "the sample's loop measures a diamagnetic plasma"
    line = _dflux_line(tmp_path, ods)
    assert float(line.split("=")[1]) == pytest.approx(1000.0 * measured)
    assert float(line.split("=")[1]) < 0


def test_forcing_a_sign_stays_an_explicit_opt_in(sample, tmp_path):
    ods = _constraints_from(sample)
    measured = float(ods["equilibrium.time_slice.0.constraints.diamagnetic_flux.measured"])
    forced = float(_dflux_line(tmp_path / "abs", ods, EFITConstraintConfig(diamagnetic_flux_sign="absolute")).split("=")[1])
    assert forced == pytest.approx(abs(1000.0 * measured)) and forced > 0
    negative = float(_dflux_line(tmp_path / "neg", ods, EFITConstraintConfig(diamagnetic_flux_sign="negative")).split("=")[1])
    assert negative == pytest.approx(-abs(1000.0 * measured))
    with pytest.raises(ValueError, match="diamagnetic_flux_sign"):
        EFITConstraintConfig(diamagnetic_flux_sign="legacy")


def test_the_ad_hoc_absolute_value_is_gone_from_the_constraint_builder():
    """`generate_constraints_ods` used to overwrite the measurement with its
    magnitude before any sign option could see it.  The writer's option is the
    only place a sign may be forced."""
    import ast

    from vaft.code.efit import kfile

    tree = ast.parse(Path(kfile.__file__).read_text(encoding="utf-8"))
    builder = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "generate_constraints_ods")
    abs_targets = [
        ast.unparse(node.targets[0])
        for node in ast.walk(builder)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and getattr(node.value.func, "id", None) == "abs"
        and "diamagnetic_flux.measured" in ast.unparse(node.targets[0])
    ]
    assert abs_targets == [], abs_targets


# --- the kernel --------------------------------------------------------------

def test_the_kernel_is_efit_cdflux_and_follows_the_f_profile(sample):
    """``integral (B_t - B_tv) dA``: linear in F, so flipping the profile about
    its edge value flips the flux exactly.  The sample's F rises toward the
    axis (paramagnetic) in a positive field, so its flux is positive; the
    mirrored, diamagnetic profile is negative -- the measured loop's sign."""
    from vaft.omas.process_wrapper import compute_reconstructed_diamagnetic_flux

    root = f"equilibrium.time_slice.{SLICE}"
    f = np.asarray(sample[f"{root}.profiles_1d.f"], dtype=float)
    assert f[0] > f[-1] > 0 and float(sample["equilibrium.vacuum_toroidal_field.b0"][SLICE]) > 0
    paramagnetic = compute_reconstructed_diamagnetic_flux(sample, SLICE)
    assert paramagnetic > 0

    mirrored = copy.deepcopy(sample)
    mirrored[f"{root}.profiles_1d.f"] = 2.0 * f[-1] - f
    diamagnetic = compute_reconstructed_diamagnetic_flux(mirrored, SLICE)
    assert diamagnetic == pytest.approx(-paramagnetic, rel=1e-9)
    assert np.sign(diamagnetic) == np.sign(np.interp(
        float(sample["equilibrium.time"][SLICE]),
        np.asarray(sample["magnetics.time"], float),
        np.asarray(sample["magnetics.diamagnetic_flux.0.data"], float),
    ))


# --- the measured loop and the virial closure ---------------------------------

def _virial_closure(sample, flux_sign):
    from vaft.formula.equilibrium import virial_beta_pd_from_S_mu_rt
    from vaft.omas.process_wrapper import compute_virial_equilibrium_quantities_ods
    from vaft.process.equilibrium import as_equilibrium, computed_diamagnetism_from_phi, derive_global_descriptors

    virial = compute_virial_equilibrium_quantities_ods(copy.deepcopy(sample), time_slice=SLICE)[SLICE]
    measured = float(np.interp(
        float(sample["equilibrium.time"][SLICE]),
        np.asarray(sample["magnetics.time"], float),
        np.asarray(sample["magnetics.diamagnetic_flux.0.data"], float),
    ))
    r_0 = float(derive_global_descriptors(as_equilibrium(sample, time_index=SLICE)).values["major_radius"].value)
    b_t0 = float(sample["equilibrium.vacuum_toroidal_field.b0"][SLICE])
    mui = computed_diamagnetism_from_phi(flux_sign * measured, b_t0, r_0, virial["V_p"], virial["B_pa"])
    beta_pd = virial_beta_pd_from_S_mu_rt(virial["s_1"], virial["s_2"], mui, virial["rt"] / r_0)
    return measured, mui, beta_pd, virial["beta_p"]


def test_the_measured_loop_closes_the_virial_balance_in_the_shared_convention(sample):
    """Same signed convention on both sides: the measured flux, taken as it is
    stored, reproduces the virial beta_p; negating it does not.  This is what
    makes the #72 diamagnetic-energy check a test of the data rather than of
    an accidental sign agreement."""
    measured, mui, beta_pd, beta_p = _virial_closure(sample, +1.0)
    assert measured < 0 and mui < 0
    assert beta_pd == pytest.approx(beta_p, rel=0.05)
    _, _, beta_pd_negated, _ = _virial_closure(sample, -1.0)
    assert not beta_pd_negated == pytest.approx(beta_p, rel=0.5)


def test_the_sample_reconstruction_disagrees_with_its_loop_and_that_is_a_fit_failure(sample):
    """Recorded so the two are not confused: the packaged reconstruction was
    made with the diamagnetic constraint written as a magnitude, and under the
    legacy weighting that constraint was effectively weightless -- so EFIT
    returned a paramagnetic plasma.  Its computed flux is honestly positive.
    The #72 report must keep flagging this as a `diagnostic_fit`/`physical`
    disagreement, never make it pass by negating the kernel."""
    from vaft.data.aeqdsk import read_aeqdsk
    from vaft.omas.process_wrapper import compute_reconstructed_diamagnetic_flux

    assert compute_reconstructed_diamagnetic_flux(sample, SLICE) > 0
    afile = read_aeqdsk(TABLES / "a039915.00319").scalars
    assert afile["diamagnetic_flux_vs"] > 0, "EFIT was fed |DFLUX| when this a-file was produced"
    assert afile["cdflux"] > 0

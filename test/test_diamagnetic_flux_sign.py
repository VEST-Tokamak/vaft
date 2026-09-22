"""The diamagnetic-flux sign convention, end to end (issues #385, #1196).

One convention, EFIT's own: DFLUX is signed, and the reconstruction is fitted
against ``cdflux = integral (B_t - B_tv) dA`` signed with B_t.  In VEST's
positive toroidal field a *paramagnetic* plasma is therefore a positive flux,
and a diamagnetic one negative.  Pinned here:

* the mapper keeps the donor's sign (`VEST_DiamagneticFlux.m` returns
  ``DiaFlux - Baseline2``); the port once negated it, and #385 read the
  negated value as a diamagnetic plasma.  The packaged samples were corrected
  in place (#1196) and must equal a fresh mapping of their raw source;
* the k-file writer passes the stored, signed measurement through (it used to
  take the absolute value, inherited from a magnitude-only donor fitter);
* the reconstructed-flux kernel is EFIT's definition, so its sign follows the
  F profile it is given;
* the measured loop and the reconstruction then agree in sign -- both
  paramagnetic -- and the loop implies an ordinary ohmic beta_p.
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


def _measured_at_slice(sample, index=SLICE):
    return float(np.interp(
        float(sample["equilibrium.time"][index]),
        np.asarray(sample["magnetics.time"], float),
        np.asarray(sample["magnetics.diamagnetic_flux.0.data"], float),
    ))


# --- the mapper ----------------------------------------------------------------

def test_the_mapper_keeps_the_donor_sign():
    """No negation between the triple integration and the stored flux (#1196),
    and the product says which convention it was mapped under."""
    from vaft.machine_mapping import magnetics

    assert magnetics.DIAMAGNETIC_FLUX_SCALE > 0
    assert "paramagnetic-positive" in magnetics.DIAMAGNETIC_FLUX_SIGN_CONVENTION


@pytest.mark.parametrize("shot", [39915, 41524, 41672])
def test_the_corrected_samples_equal_a_fresh_mapping(shot):
    """The in-place correction of the packaged samples is exactly what the
    corrected mapper produces from their raw source -- data and method_name."""
    from vaft.machine_mapping import magnetics

    stored = sample_ods() if shot == SHOT else sample_ods(shot)
    ods = {}
    magnetics.diamagnetic_flux_rogowski_coil_from_raw_database(
        ods, shot, raw_source=f"vaft/data/samples/{shot}/source/vest_{shot}_daq_raw.json.gz"
    )
    fresh = ods["magnetics"]["diamagnetic_flux"][0]
    np.testing.assert_array_equal(np.asarray(fresh["time"], float),
                                  np.asarray(stored["magnetics.diamagnetic_flux.0.time"], float))
    np.testing.assert_array_equal(np.asarray(fresh["data"], float),
                                  np.asarray(stored["magnetics.diamagnetic_flux.0.data"], float))
    assert fresh["method_name"] == stored["magnetics.diamagnetic_flux.0.method_name"]
    assert magnetics.DIAMAGNETIC_FLUX_SIGN_CONVENTION in fresh["method_name"]
    # Ohmic and paramagnetic: the flux is positive at its extreme.
    data = np.asarray(fresh["data"], float)
    assert data[np.argmax(np.abs(data))] > 0
    measured = [
        float(stored[f"equilibrium.time_slice.{i}.constraints.diamagnetic_flux.measured"])
        for i in range(len(stored["equilibrium.time"]))
    ]
    assert min(measured) > 0


# --- the k-file writer ------------------------------------------------------

def test_the_default_writes_the_signed_measurement(sample, tmp_path):
    assert EFITConstraintConfig().diamagnetic_flux_sign == "imas"
    ods = _constraints_from(sample)
    measured = float(ods["equilibrium.time_slice.0.constraints.diamagnetic_flux.measured"])
    assert measured > 0, "the sample's loop measures a paramagnetic plasma"
    line = _dflux_line(tmp_path, ods)
    assert float(line.split("=")[1]) == pytest.approx(1000.0 * measured)
    assert float(line.split("=")[1]) > 0


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
    axis (paramagnetic) in a positive field, so its flux is positive -- the
    measured loop's sign; the mirrored, diamagnetic profile is negative."""
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
    assert np.sign(paramagnetic) == np.sign(_measured_at_slice(sample))


# --- the measured loop and the virial closure ---------------------------------

def test_the_measured_loop_and_the_reconstruction_agree_in_sign(sample):
    """Both paramagnetic, within 20 % of each other in size, and the loop
    implies an ordinary ohmic beta_p.

    Under the negated flux #385/#386 read the loop as a beta_p ~ 1.44 plasma
    the reconstruction did not reproduce -- a '~90x pressure deficit'.  With
    the sign corrected (#1196) the loop's diamagnetic beta_p is ~0.18 against
    the reconstruction's 0.034: a factor of a few, not two orders.
    """
    from vaft.formula.equilibrium import virial_beta_pd_from_S_mu_rt
    from vaft.omas.process_wrapper import (
        compute_reconstructed_diamagnetic_flux, compute_virial_equilibrium_quantities_ods,
    )
    from vaft.process.equilibrium import (
        as_equilibrium, computed_diamagnetism_from_phi, derive_global_descriptors,
    )

    virial = compute_virial_equilibrium_quantities_ods(copy.deepcopy(sample), time_slice=SLICE)[SLICE]
    measured = _measured_at_slice(sample)
    reconstructed = compute_reconstructed_diamagnetic_flux(sample, SLICE)
    r_0 = float(derive_global_descriptors(as_equilibrium(sample, time_index=SLICE)).values["major_radius"].value)
    b_t0 = float(sample["equilibrium.vacuum_toroidal_field.b0"][SLICE])
    # computed_diamagnetism_from_phi is the flux convention, which is what
    # virial_beta_pd_from_S_mu_rt takes -- the pairing is internally consistent.
    mui_hat = computed_diamagnetism_from_phi(measured, b_t0, r_0, virial["V_p"], virial["B_pa"])
    beta_pd = virial_beta_pd_from_S_mu_rt(virial["s_1"], virial["s_2"], mui_hat, virial["rt"] / r_0)

    assert measured > 0 and reconstructed > 0
    assert reconstructed / measured == pytest.approx(1.19, abs=0.01)
    assert virial["mui"] < 0, "the reconstruction is paramagnetic on the volume convention"
    assert beta_pd == pytest.approx(0.176, abs=0.005)
    assert abs(virial["beta_p"]) < 0.1


def test_the_packaged_a_file_is_paramagnetic_like_its_loop(sample):
    """The packaged a-file was produced with |DFLUX| submitted, weighted out
    under legacy weighting; its cdflux is positive, the loop's sign."""
    from vaft.data.aeqdsk import read_aeqdsk

    afile = read_aeqdsk(TABLES / "a039915.00319").scalars
    assert afile["diamagnetic_flux_vs"] > 0, "EFIT was fed |DFLUX| when this a-file was produced"
    assert afile["cdflux"] > 0
    assert np.sign(afile["cdflux"]) == np.sign(_measured_at_slice(sample))

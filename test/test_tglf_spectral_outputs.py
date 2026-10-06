"""TGLF spectral diagnostics: parsed in the layouts TGLF's writers use (#1591).

Fixtures are real TGLF runs from the #1482 sensitivity campaign (39915 @ 0.317 s,
magnetics lineage, r/a 0.7, GACODE b493397): SAT0 electrostatic and SAT2 with A_parallel.
Each check ties a parsed array back to a property of the Fortran writer it came from
(``tglf/src/tglf_inout.f90``) rather than to the parser's own assumptions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vaft.code.gacode.tglf.outputs import (
    FIELD_SPECTRUM_COLUMNS,
    INTENSITY_MOMENTS,
    SCHEMA_VERSION,
    TglfOutputs,
    collect_tglf_outputs,
)

DATA = Path(__file__).parent / "data" / "gacode"
SAT0 = DATA / "tglf_vest_39915_r0.70_sat0-es"
SAT2 = DATA / "tglf_vest_39915_r0.70_sat2-em-bper"


@pytest.fixture(scope="module")
def sat0():
    return collect_tglf_outputs(SAT0)


@pytest.fixture(scope="module")
def sat2():
    return collect_tglf_outputs(SAT2)


def test_every_spectral_output_parses_with_its_writer_shape(sat0, sat2):
    nky, nmodes, ns = 21, 2, 3
    for run, nfield in ((sat0, 1), (sat2, 2)):
        assert run.field_spectrum.shape == (nky, nmodes, len(FIELD_SPECTRUM_COLUMNS))
        assert run.intensity_spectrum.shape == (ns, nky, nmodes, len(INTENSITY_MOMENTS))
        assert run.density_spectrum.shape == (nky, ns)
        assert run.temperature_spectrum.shape == (nky, ns)
        assert run.nete_crossphase_spectrum.shape == (nky, nmodes)
        assert run.nsts_crossphase_spectrum.shape == (ns, nky, nmodes)
        assert run.ql_flux_spectrum.shape == (ns, nfield, nmodes, nky, 5)
        for name in ("width_spectrum", "spectral_shift_spectrum", "ave_p0_spectrum"):
            assert getattr(run, name).shape == (nky,)
        assert run.spectral_species == (1, 2, 3)


def test_amplitudes_are_the_root_of_the_mode_summed_intensity(sat0, sat2):
    """write_tglf_density_spectrum: density(is) = sqrt(sum_n intensity(1, is, ky, n))."""
    for run in (sat0, sat2):
        intensity = run.intensity_spectrum
        assert np.allclose(run.density_spectrum.T, np.sqrt(intensity[..., 0].sum(-1)), rtol=1e-6)
        assert np.allclose(run.temperature_spectrum.T, np.sqrt(intensity[..., 1].sum(-1)), rtol=1e-6)


def test_the_electron_cross_phase_is_the_first_species_cross_phase(sat0):
    assert np.allclose(sat0.nete_crossphase_spectrum, sat0.nsts_crossphase_spectrum[0])


def test_an_absent_field_is_nan_not_the_zero_tglf_writes(sat0, sat2):
    """field_spectrum headers say a_par_no / b_par_no; those columns must not read as nulls."""
    assert np.all(np.isnan(sat0.field_spectrum[..., 2]))          # ES: no A_par
    assert np.all(np.isnan(sat0.field_spectrum[..., 3]))
    assert np.all(np.isfinite(sat2.field_spectrum[..., 2]))       # EM: A_par present
    assert np.all(np.isnan(sat2.field_spectrum[..., 3]))          # no B_par


def test_saturation_parameters_record_the_preset_that_changes_the_linear_model(sat0, sat2):
    assert sat0.saturation_parameters["SAT_RULE"] == 0
    assert sat0.saturation_parameters["UNITS"] == "GYRO"
    assert sat0.saturation_parameters["XNU_MODEL"] == 2
    assert sat2.saturation_parameters["UNITS"] == "CGYRO"
    assert sat2.saturation_parameters["XNU_MODEL"] == 3


def test_units_cgyro_moves_the_ky_grid_by_grad_r0_not_the_units(sat0, sat2):
    """tglf_kygrid.f90: ky_factor = grad_r0_out when UNITS != GYRO, so the grid points
    shift while ky stays in the same normalisation."""
    factor = sat2.saturation_parameters["grad_r0_out"]
    assert np.allclose(sat2.ky_spectrum, sat0.ky_spectrum * factor, rtol=1e-9)


def test_ql_weights_are_per_mode_weights_not_the_saturated_flux(sat0):
    """Saturated flux (sum_flux) != weight; they differ by the intensity, so a plot must
    never label QL weights as fluxes."""
    weight = sat0.ql_flux_spectrum[0, 0, 0, :, 1]      # electrons, phi, mode 1, energy
    flux = sat0.sum_flux_spectrum[0, 0, :, 1]
    assert not np.allclose(weight, flux)


def test_the_container_round_trips_and_older_payloads_still_read(sat2, tmp_path):
    back = TglfOutputs.read_json(sat2.write_json(tmp_path / "out.json"))
    assert np.allclose(back.ql_flux_spectrum, sat2.ql_flux_spectrum, equal_nan=True)
    assert back.saturation_parameters == sat2.saturation_parameters
    assert back.spectral_species == (1, 2, 3)
    old = sat2.to_dict()
    old["schema_version"] = 2
    for key in ("field_spectrum", "intensity_spectrum", "saturation_parameters"):
        old.pop(key)
    legacy = TglfOutputs.from_dict(old)
    assert legacy.field_spectrum is None and legacy.growth_rate is not None
    assert SCHEMA_VERSION == 3


def test_a_file_that_does_not_match_its_writer_is_refused(tmp_path):
    target = tmp_path / "run"
    target.mkdir()
    for path in SAT0.iterdir():
        (target / path.name).write_bytes(path.read_bytes())
    lines = (target / "out.tglf.field_spectrum").read_text().splitlines()
    (target / "out.tglf.field_spectrum").write_text("\n".join(lines[:-3]) + "\n")
    run = collect_tglf_outputs(target)
    assert run.field_spectrum is None and run.intensity_spectrum is not None


def test_ql_weights_share_the_absolute_field_axis(sat2):
    """The QL writer counts fields compactly, sum_flux by position; both are stored on
    (phi, a_par, b_par) so index 1 is A_parallel in both."""
    assert sat2.ql_flux_spectrum.shape[1] == sat2.sum_flux_spectrum.shape[1] == 2


def test_a_b_par_only_run_does_not_label_b_par_as_a_par(tmp_path):
    """USE_BPER=F, USE_BPAR=T: QL has 2 compact fields (phi, b_par); they must land in
    slots 0 and 2, and sum_flux's unwritten A_par slot must be NaN."""
    target = tmp_path / "run"
    target.mkdir()
    for path in SAT2.iterdir():
        (target / path.name).write_bytes(path.read_bytes())
    text = (target / "out.tglf.field_spectrum").read_text()
    (target / "out.tglf.field_spectrum").write_text(
        text.replace("a_par_yes", "a_par_no").replace("b_par_no", "b_par_yes"))
    run = collect_tglf_outputs(target)
    assert run.ql_flux_spectrum.shape[1] == 3
    assert np.all(np.isnan(run.ql_flux_spectrum[:, 1]))
    assert np.all(np.isfinite(run.ql_flux_spectrum[:, 2]))


def test_json_is_strict_and_nan_survives_as_absent(sat0, tmp_path):
    import json

    path = sat0.write_json(tmp_path / "out.json")
    json.loads(path.read_text(), parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
    back = TglfOutputs.read_json(path)
    assert np.all(np.isnan(back.field_spectrum[..., 2]))


def test_infinities_and_non_finite_scalars_round_trip_through_strict_json(tmp_path):
    """A blown-up flux (Fortran ``Infinity``) or a NaN ``precision`` is a solver verdict.

    ``write_json`` must still produce strict JSON for it (cold review 0.8.0 F3): the
    transport atlas archives ``outputs.json`` for every surface, and a writer that raised
    turned TGLF's own non-finite result into a VAFT error with no record to resume from.
    """
    import json
    import math

    run = TglfOutputs(
        directory=str(tmp_path), precision=float("nan"),
        gbflux={"particle": np.array([0.1, np.inf]), "energy": np.array([-np.inf, np.nan])},
        saturation_parameters={"SAT_geo0_out": float("nan"), "B_unit": 1.5, "note": "inf"},
    )
    path = run.write_json(tmp_path / "out.json")
    text = path.read_text(encoding="utf-8")
    json.loads(text, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
    assert run.write_json(tmp_path / "again.json").read_text(encoding="utf-8") == text
    back = TglfOutputs.read_json(path)
    assert np.array_equal(back.gbflux["particle"], [0.1, np.inf])
    assert np.isneginf(back.gbflux["energy"][0]) and np.isnan(back.gbflux["energy"][1])
    assert math.isnan(back.precision)
    assert math.isnan(back.saturation_parameters["SAT_geo0_out"])
    assert back.saturation_parameters["B_unit"] == 1.5
    assert back.saturation_parameters["note"] == "inf"   # a genuine string stays one
    assert not back.solved

"""The passive wall's fitted band factors are a versioned calibration (#308).

The shipped static geometry carries fitted effective resistances on two of
its eleven regions.  These tests pin that the separation into nominal
geometry times band factors is exact -- bitwise against all 950 shipped
values -- and that the benchmark seam applying a calibration is a no-op for
the vintage the asset was built from.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

from vaft.machine_mapping.pf_passive import DEFAULT_REFERENCE_ODS
from vaft.machine_mapping.static_geometry import load_static_ods
from vaft.machine_mapping.wall_resistance import (
    LEGACY_CALIBRATIONS,
    WallResistanceCalibration,
    band_factors,
    band_layout,
    calibrated_resistance,
    identify_calibration,
    nominal_resistance,
)


@pytest.fixture(scope="module")
def static_ods():
    return load_static_ods(DEFAULT_REFERENCE_ODS)


def _shipped(ods) -> np.ndarray:
    return np.array(
        [float(ods[f"pf_passive.loop.{i}.resistance"]) for i in range(len(ods["pf_passive.loop"]))]
    )


def test_band_layout_matches_the_donor_indexing(static_ods):
    layout = band_layout(static_ods)
    outboard, inboard = layout["W1"], layout["W11"]
    assert [b.size for b in outboard] == [20] * 12
    assert [b.size for b in inboard] == [12] * 18 + [14]
    assert outboard[0][0] == 0 and inboard[0][0] == 720
    assert np.concatenate(inboard)[-1] == 949


def test_vintage_2303_reproduces_every_shipped_resistance_bitwise(static_ods):
    rebuilt = calibrated_resistance(static_ods, LEGACY_CALIBRATIONS["2303"])
    np.testing.assert_array_equal(rebuilt, _shipped(static_ods))


def test_unfitted_regions_carry_the_nominal_hoop_resistance_exactly(static_ods):
    shipped = _shipped(static_ods)
    nominal = nominal_resistance(static_ods)
    fitted = np.concatenate(sum(band_layout(static_ods).values(), []))
    free = np.setdiff1d(np.arange(shipped.size), fitted)
    assert free.size == 480
    np.testing.assert_array_equal(shipped[free], nominal[free])


def test_the_shipped_asset_is_identified_as_vintage_2303(static_ods):
    found = identify_calibration(static_ods)
    assert found["key"] == "2303"
    assert found["max_relative_deviation"] == 0.0
    assert found["unfitted_loops_nominal"] is True
    assert len(found["measured_factor_digest"]) == 12


def test_a_perturbed_asset_is_not_misidentified_but_still_fingerprinted(static_ods):
    perturbed = copy.deepcopy(static_ods)
    perturbed["pf_passive.loop.5.resistance"] = float(perturbed["pf_passive.loop.5.resistance"]) * 1.01
    found = identify_calibration(perturbed)
    assert found["key"] is None
    assert found["nearest_key"] == "2303"
    assert found["measured_factor_digest"] != identify_calibration(static_ods)["measured_factor_digest"]


def test_a_factor_changes_only_its_own_band(static_ods):
    base = LEGACY_CALIBRATIONS["2303"]
    outboard = list(base.outboard)
    outboard[3] *= 2.0
    changed = base.replace(key="probe", outboard=tuple(outboard))
    ratio = band_factors(static_ods, changed) / band_factors(static_ods, base)
    band = band_layout(static_ods)["W1"][3]
    assert np.all(ratio[band] == 2.0)
    assert np.all(np.delete(ratio, band) == 1.0)
    assert changed.digest() != base.digest()


def test_calibration_shapes_and_values_are_validated():
    with pytest.raises(ValueError, match="outboard needs 12"):
        WallResistanceCalibration(key="x", outboard=(1.0,) * 11, inboard=(1.0,) * 19)
    with pytest.raises(ValueError, match="finite and positive"):
        WallResistanceCalibration(key="x", outboard=(1.0,) * 11 + (0.0,), inboard=(1.0,) * 19)


def test_the_static_model_fingerprint_names_the_vintage(static_ods):
    from vaft.validation.vacuum_benchmark import _static_model

    fingerprint = _static_model(static_ods)
    assert fingerprint["wall_calibration"]["key"] == "2303"


def _real_shot(shot: int):
    import gzip, shutil, tempfile, warnings
    from pathlib import Path

    from omas import load_omas_json

    import vaft.machine_mapping.em_coupling as em
    from vaft.data import resources

    try:
        source = resources.data_path(f"samples/{shot}/source/pipeline-until-efit.json.gz")
    except Exception:  # pragma: no cover
        pytest.skip("packaged pipeline sample unavailable")
    if not Path(source).is_file():
        pytest.skip("packaged pipeline sample is repository-only")
    with gzip.open(source, "rt") as handle, tempfile.NamedTemporaryFile(
        "w", suffix=".json", delete=False
    ) as plain:
        shutil.copyfileobj(handle, plain)
        plain_path = plain.name
    try:
        ods = load_omas_json(plain_path, consistency_check=False)
    finally:
        Path(plain_path).unlink(missing_ok=True)
    del ods["em_coupling"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        em.em_coupling(ods, shot=shot)
    return ods


def test_applying_the_shipped_vintage_leaves_the_real_wall_solve_byte_identical():
    """The seam exists so a fit can vary the factors; for the vintage the asset
    was built from it must change nothing, down to the last bit of every
    passive-loop current on a real shot."""
    from vaft.validation.vacuum_benchmark import benchmark_wall_currents, run_benchmark_case

    ods = _real_shot(39915)
    plain = benchmark_wall_currents(ods)
    explicit = benchmark_wall_currents(ods, calibration=LEGACY_CALIBRATIONS["2303"])
    n = len(plain["pf_passive.loop"])
    for i in range(n):
        np.testing.assert_array_equal(
            plain[f"pf_passive.loop.{i}.current"], explicit[f"pf_passive.loop.{i}.current"]
        )
    case = run_benchmark_case(ods, shot=39915, calibration=LEGACY_CALIBRATIONS["2303"])
    assert case["static_model"]["wall_calibration"]["key"] == "2303"
    assert case["static_model"]["applied_calibration"]["key"] == "2303"
    assert case["static_model"]["applied_calibration"]["digest"] == LEGACY_CALIBRATIONS["2303"].digest()


def test_a_foreign_passive_model_is_fingerprinted_unidentified_without_side_effects():
    """A wall that is not the banded VEST model (the benchmark's synthetic
    machine, say) must fingerprint as unidentified -- and reading it must not
    materialize any missing OMAS path, which a bare read would."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from test_vacuum_benchmark import _machine
    from vaft.validation.vacuum_benchmark import _static_model

    ods = _machine()
    before = set(ods.flat().keys())
    fingerprint = _static_model(ods)
    assert fingerprint["wall_calibration"]["key"] is None
    assert "error" in fingerprint["wall_calibration"]
    assert set(ods.flat().keys()) == before

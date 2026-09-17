"""Reciprocity of the passive-passive coupling matrix (issues #347, #373).

A mutual-inductance matrix must satisfy M_ij == M_ji exactly. The packaged
asset did not: 35% of its entries were asymmetric, up to 4% on the worst pair
-- the donor applied its SUS material factor from one side only -- and it was
loaded verbatim into every eddy-current solve. #347 made `em_coupling()`
symmetrize on load and record how far off the input was; #373 repaired the
asset itself, with provenance, so the fold is now a silent no-op on the
packaged matrix and still guards a caller-supplied one.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from omas import ODS

import vaft.machine_mapping.em_coupling as em
from vaft.machine_mapping.pf_active import vfit_pf_active_static
from vaft.machine_mapping.pf_passive import pf_passive

SHOT = 41672


def _machine_ods() -> ODS:
    ods = ODS(consistency_check=False)
    vfit_pf_active_static(ods, SHOT)
    pf_passive(ods)
    return ods


def test_the_loaded_passive_coupling_is_exactly_symmetric():
    ods = _machine_ods()
    em.em_coupling(ods, shot=SHOT)
    M = np.asarray(ods["em_coupling.mutual_passive_passive"], dtype=float)
    assert M.shape == (950, 950)
    assert np.array_equal(M, M.T)


def test_the_packaged_asset_is_symmetric_and_carries_provenance():
    """The #373 repair: every reciprocity-bound matrix in the asset is exact,
    the asset says who repaired it and how, and loading it is silent."""
    with np.load(em.DEFAULT_VERSIONED_COUPLING, allow_pickle=False) as versioned:
        keys = set(versioned.files)
        for key in ("mutual_passive_passive", "mutual_active_active_1906", "mutual_active_active_2507"):
            raw = np.asarray(versioned[key], dtype=float)
            assert float(np.max(np.abs(raw - raw.T)) / np.max(np.abs(raw))) <= 1e-13, key
    assert em.COUPLING_PROVENANCE_KEY in keys
    provenance = em.load_versioned_coupling_provenance()
    assert provenance["passive_material_factor"] == 1.04
    assert provenance["convention"] == "sus_side"
    assert provenance["generator"].endswith("regenerate_passive_coupling.py")
    assert len(provenance["source_sha256"]) == 64
    assert provenance["input_asymmetry"] == pytest.approx(1.27e-3, rel=0.05)
    assert provenance["output_asymmetry"] == 0.0

    ods = _machine_ods()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        em.em_coupling(ods, shot=SHOT)
    params = ods["em_coupling.code.parameters"]
    assert "passive_passive_symmetrized=false" in params
    assert "passive_passive_input_asymmetry=0.000000e+00" in params
    assert "coupling_asset_passive_material_factor=1.04" in params
    assert "coupling_asset_source_sha256=" in params
    comment = ods["em_coupling.ids_properties.comment"]
    assert "reciprocity exact" in comment and "issue #373" in comment
    assert "symmetrized to" not in comment


def test_the_loaded_matrix_is_the_asset_bit_for_bit():
    """A symmetric input is its own average: the fold must change nothing."""
    with np.load(em.DEFAULT_VERSIONED_COUPLING, allow_pickle=False) as versioned:
        raw = np.asarray(versioned["mutual_passive_passive"], dtype=float)
    ods = _machine_ods()
    em.em_coupling(ods, shot=SHOT)
    loaded = np.asarray(ods["em_coupling.mutual_passive_passive"], dtype=float)
    np.testing.assert_array_equal(loaded, raw)
    np.testing.assert_array_equal(loaded, (raw + raw.T) / 2.0)


def test_a_legacy_asset_without_provenance_still_loads_and_says_symmetrized(tmp_path, monkeypatch):
    """An asset written before #373 has no provenance key and a lopsided
    matrix; it loads with the #347 warning and an honest record."""
    with np.load(em.DEFAULT_VERSIONED_COUPLING, allow_pickle=False) as versioned:
        arrays = {k: np.asarray(versioned[k], dtype=float) for k in versioned.files if k != em.COUPLING_PROVENANCE_KEY}
    lopsided = arrays["mutual_passive_passive"].copy()
    lopsided[720:, :720] /= 1.04  # the donor defect, re-created
    arrays["mutual_passive_passive"] = lopsided
    legacy = tmp_path / "legacy.npz"
    np.savez_compressed(legacy, **arrays)
    monkeypatch.setattr(em, "DEFAULT_VERSIONED_COUPLING", legacy)

    ods = _machine_ods()
    with pytest.warns(RuntimeWarning, match="violates reciprocity"):
        em.em_coupling(ods, shot=SHOT)
    params = ods["em_coupling.code.parameters"]
    assert "passive_passive_symmetrized=true" in params
    assert "coupling_asset_provenance=absent" in params
    recorded = float(params.split("passive_passive_input_asymmetry=")[1].split()[0])
    assert recorded == pytest.approx(1.27e-3, rel=0.05)
    assert "symmetrized to (M + M^T)/2" in ods["em_coupling.ids_properties.comment"]
    assert em.load_versioned_coupling_provenance(legacy) is None


@pytest.mark.parametrize(
    ("asymmetry", "expect"),
    [
        (0.0, "silent"),
        (1.0e-9, "silent"),   # float64 round-off territory
        (1.0e-3, "warn"),     # the packaged asset's regime
        (5.0e-2, "warn"),
        (2.0e-1, "reject"),   # not a mutual-inductance matrix
    ],
)
def test_thresholds_separate_round_off_from_defect_from_nonsense(asymmetry, expect):
    n = 6
    rng = np.random.default_rng(347)
    base = rng.uniform(1.0, 2.0, size=(n, n))
    sym = (base + base.T) / 2.0
    skew = np.zeros((n, n))
    skew[0, 1] = asymmetry * np.max(np.abs(sym))
    matrix = sym + skew
    # The injected scalar is only the asymmetry if [0, 1] is not the matrix's
    # maximum, which depends on the seed. Assert against what the matrix IS.
    expected = float(np.max(np.abs(matrix - matrix.T)) / np.max(np.abs(matrix)))

    if expect == "reject":
        with pytest.raises(ValueError, match="will not be symmetrized"):
            em._symmetrize_passive_coupling(matrix, source="test")
        return

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out, measured = em._symmetrize_passive_coupling(matrix, source="test")

    assert np.array_equal(out, out.T)
    assert measured == pytest.approx(expected, rel=1e-12, abs=1e-15)
    raised = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert (len(raised) == 1) == (expect == "warn")


def test_a_caller_supplied_reference_matrix_is_also_held_to_reciprocity(tmp_path):
    """Reciprocity is physics, not provenance: an override is symmetrized too."""
    from omas import save_omas_json

    reference = _machine_ods()
    em.em_coupling(reference, shot=SHOT)
    M = np.asarray(reference["em_coupling.mutual_passive_passive"], dtype=float)
    lopsided = M.copy()
    lopsided[0, 1] *= 1.02  # 2% on one pair: warn regime, not reject
    reference["em_coupling.mutual_passive_passive"] = lopsided
    path = tmp_path / "reference.json"
    save_omas_json(reference, str(path))

    ods = _machine_ods()
    with pytest.warns(RuntimeWarning, match="reference ODS"):
        em.em_coupling(ods, source=str(path), shot=SHOT)
    out = np.asarray(ods["em_coupling.mutual_passive_passive"], dtype=float)
    assert np.array_equal(out, out.T)
    assert out[0, 1] == pytest.approx((lopsided[0, 1] + lopsided[1, 0]) / 2.0)


def test_a_non_finite_matrix_is_refused_rather_than_recorded_as_symmetric():
    """NaN compares false against every threshold, so without a guard a NaN
    matrix would be reported as asymmetry 0.0 and stored with a provenance
    record claiming it was clean. Refuse it before measuring anything.
    """
    matrix = np.full((4, 4), 1.0e-6)
    matrix[1, 2] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        em._symmetrize_passive_coupling(matrix, source="test")

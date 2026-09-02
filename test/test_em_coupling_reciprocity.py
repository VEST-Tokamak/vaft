"""Reciprocity of the passive-passive coupling matrix (issue #347).

A mutual-inductance matrix must satisfy M_ij == M_ji exactly. The packaged
asset does not: 35% of its entries are asymmetric, up to 4% on the worst pair,
and it was loaded verbatim into every eddy-current solve. `em_coupling()` now
symmetrizes on load and records how far off the input was, rather than
applying the correction silently.
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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        em.em_coupling(ods, shot=SHOT)
    M = np.asarray(ods["em_coupling.mutual_passive_passive"], dtype=float)
    assert M.shape == (950, 950)
    assert np.array_equal(M, M.T)


def test_the_packaged_asset_is_known_to_violate_reciprocity_and_says_so():
    """This pins the defect, so a regenerated asset that fixes it is noticed.

    If the asset is ever replaced by a symmetric one, this test fails on the
    warning count -- which is the right outcome: delete the expectation, and
    the warn path stays covered by the synthetic test below.
    """
    with np.load(em.DEFAULT_VERSIONED_COUPLING, allow_pickle=False) as versioned:
        raw = np.asarray(versioned["mutual_passive_passive"], dtype=float)
    asymmetry = float(np.max(np.abs(raw - raw.T)) / np.max(np.abs(raw)))
    assert asymmetry == pytest.approx(1.27e-3, rel=0.05)

    ods = _machine_ods()
    with pytest.warns(RuntimeWarning, match="violates reciprocity"):
        em.em_coupling(ods, shot=SHOT)

    # Provenance carries the measured number, in both DD-sanctioned homes.
    params = ods["em_coupling.code.parameters"]
    assert "passive_passive_symmetrized=true" in params
    recorded = float(params.split("passive_passive_input_asymmetry=")[1].split()[0])
    assert recorded == pytest.approx(asymmetry, rel=1e-6)
    assert "symmetrized to (M + M^T)/2" in ods["em_coupling.ids_properties.comment"]


def test_symmetrization_is_the_exact_average_not_something_else():
    """(M + M^T)/2, bit for bit -- no rounding, scaling or reordering."""
    with np.load(em.DEFAULT_VERSIONED_COUPLING, allow_pickle=False) as versioned:
        raw = np.asarray(versioned["mutual_passive_passive"], dtype=float)
    ods = _machine_ods()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        em.em_coupling(ods, shot=SHOT)
    np.testing.assert_array_equal(
        np.asarray(ods["em_coupling.mutual_passive_passive"], dtype=float),
        (raw + raw.T) / 2.0,
    )


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
    matrix = sym + skew  # exactly the requested max relative asymmetry

    if expect == "reject":
        with pytest.raises(ValueError, match="will not be symmetrized"):
            em._symmetrize_passive_coupling(matrix, source="test")
        return

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out, measured = em._symmetrize_passive_coupling(matrix, source="test")

    assert np.array_equal(out, out.T)
    assert measured == pytest.approx(asymmetry, rel=1e-9, abs=1e-15)
    raised = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert (len(raised) == 1) == (expect == "warn")


def test_a_caller_supplied_reference_matrix_is_also_held_to_reciprocity(tmp_path):
    """Reciprocity is physics, not provenance: an override is symmetrized too."""
    from omas import save_omas_json

    reference = _machine_ods()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
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

"""IMPA source validity in the validation layer.

The grading itself is covered by ``test_impa_processing.py`` and
``test_impa_mapping.py``; this file pins what the move promised: one status
vocabulary, the DD validity codes taken from ``vaft.validation.imas``, the
stage-manifest spelling kept, the old names delegating, and a success path
through the mapper -- which the packaged shots cannot give, since all three
reject calibration, so the processed shot is synthesized.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from omas import ODS

import vaft.machine_mapping.impa as impa_mapping
from vaft.process.impa import (
    IMPA_CHANNEL_COUNT,
    ImpaGeometryFit,
    ImpaProcessingConfig,
    process_impa,
    toroidal_field,
)
from vaft.validation import imas
from vaft.validation.impa import (
    MANIFEST_LABELS,
    ImpaAssessment,
    bz_channel_validity,
    channel_validity,
    manifest_checks,
    manifest_label,
    validate_impa,
)
from vaft.validation.model import ValidationStatus

SAMPLE_RATE = 25_000.0


def _clean_poloidal_shot():
    """A synthetic poloidal array that ``process_impa`` grades PASS."""
    time = np.arange(25_000, dtype=float) / SAMPLE_RATE
    i_tf = np.interp(time, [0.0, 0.11, 0.16, 0.21, 0.60, 0.90], [0.0, 0.0, 4_800.0, 12_000.0, 10_800.0, 0.0])
    radii = 0.4 + np.arange(8) * 0.05
    alpha = np.full(8, 0.05)
    bz = 3.0e-5 * np.sin(2 * np.pi * 3 * time)
    bt = toroidal_field(radii[:, None], i_tf[None, :])
    measured = alpha[:, None] * bt + np.sqrt(1.0 - alpha[:, None] ** 2) * bz
    measured = measured + np.random.default_rng(0).normal(0.0, 1e-6, measured.shape)
    raw = measured / (-2.0 / 15.0)
    result = process_impa(time, raw, i_tf, r=radii, config=ImpaProcessingConfig(orientation="poloidal"))
    return result, raw


# ---------------------------------------------------------------------------
# vocabulary
# ---------------------------------------------------------------------------
def test_the_assessment_speaks_validation_status():
    result, _ = _clean_poloidal_shot()

    assert isinstance(result.quality, ImpaAssessment)
    assert result.quality.status is ValidationStatus.PASS
    assert all(isinstance(state, ValidationStatus) for state in result.quality.checks.values())
    assert result.quality.is_usable


def test_legacy_strings_are_converted_on_construction():
    """The deprecated ``ImpaQuality`` is this class; its old strings still build one."""
    legacy = ImpaAssessment(status="warning", checks={"geometry": "invalid", "time_axis": "valid"})

    assert legacy.status is ValidationStatus.WARN
    assert legacy.checks == {"geometry": ValidationStatus.FAIL, "time_axis": ValidationStatus.PASS}
    assert legacy.is_usable
    assert not ImpaAssessment(status=ValidationStatus.FAIL, checks={}).is_usable


@pytest.mark.parametrize(
    "state", ["usable", ValidationStatus.INDETERMINATE, ValidationStatus.NOT_AVAILABLE]
)
def test_a_state_the_grading_cannot_reach_is_refused(state):
    """An undecided assessment would otherwise project to a valid channel code."""
    with pytest.raises(ValueError):
        ImpaAssessment(status=state, checks={})


def test_the_default_channel_count_matches_the_kernels():
    """Restated in the validation module so it never imports the kernels."""
    import inspect

    default = inspect.signature(validate_impa).parameters["expected_channels"].default
    assert default == IMPA_CHANNEL_COUNT


# ---------------------------------------------------------------------------
# projections
# ---------------------------------------------------------------------------
def test_the_manifest_keeps_its_spelling():
    """The IMPA product reads ``invalid`` as rejected; the strings are a serialization only."""
    assert MANIFEST_LABELS == {
        ValidationStatus.PASS: "valid",
        ValidationStatus.WARN: "warning",
        ValidationStatus.FAIL: "invalid",
    }
    assessment = ImpaAssessment(
        status=ValidationStatus.FAIL,
        checks={"geometry": ValidationStatus.WARN, "tf_coupling": ValidationStatus.FAIL},
    )
    assert manifest_label(assessment.status) == "invalid"
    assert manifest_checks(assessment) == {"geometry": "warning", "tf_coupling": "invalid"}


@pytest.mark.parametrize(
    ("status", "kwargs", "code"),
    [
        (ValidationStatus.PASS, {"wired": False, "coupled": True, "finite": True}, imas.VALIDITY_INVALID),
        (ValidationStatus.PASS, {"wired": True, "coupled": False, "finite": True}, imas.VALIDITY_SUSPECT),
        (ValidationStatus.PASS, {"wired": True, "coupled": True, "finite": False}, imas.VALIDITY_SUSPECT),
        (ValidationStatus.FAIL, {"wired": True, "coupled": True, "finite": True}, imas.VALIDITY_SUSPECT),
        (ValidationStatus.WARN, {"wired": True, "coupled": True, "finite": True}, imas.VALIDITY_CERTIFIED),
        (ValidationStatus.PASS, {"wired": True, "coupled": True, "finite": True}, imas.VALIDITY_VALID),
    ],
)
def test_channel_validity_keeps_the_codes_products_carry(status, kwargs, code):
    """-2 unwired, -1 uncalibrated or rejected, 1 with warnings, 0 clean -- unchanged by the move."""
    validity, reason = channel_validity(ImpaAssessment(status=status, checks={}), **kwargs)
    assert validity == code
    assert reason


@pytest.mark.parametrize(
    ("bz_state", "kwargs", "code"),
    [
        (ValidationStatus.PASS, {"wired": False, "calibrated": True}, imas.VALIDITY_INVALID),
        (ValidationStatus.PASS, {"wired": True, "calibrated": False}, imas.VALIDITY_SUSPECT),
        (ValidationStatus.FAIL, {"wired": True, "calibrated": True}, imas.VALIDITY_SUSPECT),
        (ValidationStatus.WARN, {"wired": True, "calibrated": True}, imas.VALIDITY_CERTIFIED),
        (ValidationStatus.PASS, {"wired": True, "calibrated": True}, imas.VALIDITY_VALID),
    ],
)
def test_bz_validity_follows_the_bz_check_not_the_shot(bz_state, kwargs, code):
    # The shot failed on its Hall geometry; that must not condemn a Bz sensor.
    assessment = ImpaAssessment(status=ValidationStatus.FAIL, checks={"bz_sensors": bz_state})
    validity, _ = bz_channel_validity(assessment, **kwargs)
    assert validity == code


# ---------------------------------------------------------------------------
# the mapper's success path, which no packaged shot reaches
# ---------------------------------------------------------------------------
def test_a_clean_shot_maps_valid_channels_and_a_valid_manifest(monkeypatch):
    result, raw = _clean_poloidal_shot()
    specs = [{"field": 100 + index, "label": f"IMPA {index + 1:02d}"} for index in range(raw.shape[0])]
    inputs = {"specs": specs, "raw": raw, "notes": [], "bz_specs": [], "bz_raw": None}
    monkeypatch.setattr(impa_mapping, "process_impa_shot", lambda *args, **kwargs: (result, inputs))

    ods = ODS()
    status = impa_mapping.impa(ods, 99999, tstart=0.3, tend=0.4, dt=1.0e-3)

    assert status["status"] == "valid"
    assert set(status["checks"].values()) == {"valid"}
    node = status["ids_node"]
    for channel in status["channels"].values():
        assert channel["validity"] == imas.VALIDITY_VALID
        assert ods[f"{node}.{channel['probe_index']}.field.validity"] == imas.VALIDITY_VALID
        assert ods[f"{node}.{channel['probe_index']}.voltage.validity"] == imas.VALIDITY_VALID


# ---------------------------------------------------------------------------
# the old names
# ---------------------------------------------------------------------------
def _minimal_args():
    time = np.linspace(0.0, 1.0, 50)
    raw = np.vstack([np.sin(time * (index + 1)) for index in range(2)])
    geometry = ImpaGeometryFit(r=np.array([0.4, 0.45]), z=np.zeros(2), method="configured")
    return (time, raw, raw, raw, np.ones(2, dtype=bool), None, geometry, None), {"expected_channels": 2}


@pytest.mark.parametrize("name", ["grade_impa_quality", "validate_impa"])
def test_the_process_names_warn_and_answer_identically(name):
    import vaft.process.impa as kernels

    args, kwargs = _minimal_args()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        legacy = getattr(kernels, name)(*args, **kwargs)

    assert any(item.category is DeprecationWarning for item in caught)
    assert "vaft.validation.impa.validate_impa" in str(caught[0].message)
    assert legacy == validate_impa(*args, **kwargs)


def test_impa_quality_is_an_alias_of_the_assessment():
    from vaft.process.impa import ImpaQuality

    assert ImpaQuality is ImpaAssessment

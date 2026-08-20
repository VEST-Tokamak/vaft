import importlib

import numpy as np
import pytest

from vaft.database.raw import RawSignalUnavailableError, require_signal
from vaft.machine_mapping import utils as mapping_utils
from vaft.process.magnetics import VestMagneticsProcessingConfig


def test_required_signal_error_identifies_the_missing_waveform():
    with pytest.raises(RawSignalUnavailableError) as caught:
        require_signal(None, shot=39915, field=13, signal_name="main gauge")

    assert caught.value.shot == 39915
    assert caught.value.field == 13
    assert "main gauge" in str(caught.value)
    assert "raw archive" in str(caught.value)


@pytest.mark.parametrize(
    ("module_name", "function_name", "arguments", "expected_field"),
    [
        ("vaft.machine_mapping.barometry", "vfit_barometry_dynamic", ({}, 39915, 0.2, 0.4, 4e-5), 13),
        ("vaft.machine_mapping.tf", "vfit_tf_dynamic", ({}, 39915, 0.2, 0.4, 4e-5), 1),
        ("vaft.machine_mapping.spectrometer_uv", "vfit_filterscope", ({}, 39915, 0.2, 0.4, 4e-5), 101),
    ],
)
def test_diagnostic_mappers_do_not_replace_missing_raw_data_with_zeros(
    monkeypatch,
    module_name,
    function_name,
    arguments,
    expected_field,
):
    module = importlib.import_module(module_name)
    monkeypatch.setattr(module, "_safe_vest_load", lambda *args, **kwargs: None)

    with pytest.raises(RawSignalUnavailableError) as caught:
        getattr(module, function_name)(*arguments)

    assert caught.value.shot == 39915
    assert caught.value.field == expected_field


def test_required_signal_rejects_placeholder_and_misaligned_waveforms():
    with pytest.raises(RawSignalUnavailableError, match="at least 2"):
        require_signal(
            (np.array([0.0]), np.array([0.0])),
            shot=39915,
            field=1,
        )

    with pytest.raises(RawSignalUnavailableError, match="lengths differ"):
        require_signal(
            (np.array([0.0, 1.0]), np.array([0.0, 1.0, 2.0])),
            shot=39915,
            field=1,
        )


def test_generic_shot_loader_does_not_synthesize_a_zero_signal(monkeypatch):
    monkeypatch.setattr(mapping_utils.raw_db, "load", lambda *args, **kwargs: None)

    with pytest.raises(RawSignalUnavailableError, match="shot 39915, field 13"):
        mapping_utils.load_raw_data("39915", 13)


@pytest.mark.parametrize(
    ("shot", "expected"),
    [
        (41445, (6000, 8500, 8500)),
        (41446, (6500, 9000, 5000)),
        (41451, (6500, 9000, 5000)),
        (41452, (6000, 8500, 8500)),
        (41659, (6000, 8500, 8500)),
        (41660, (6500, 9000, 5000)),
    ],
)
def test_magnetics_processing_era_boundaries_are_explicit(shot, expected):
    assert VestMagneticsProcessingConfig().window_for_shot(shot) == expected

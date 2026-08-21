import importlib.util
import warnings

import numpy as np


def test_detect_active_window_selects_peak_containing_region():
    from vaft.process import detect_active_window

    time = np.arange(8, dtype=float)
    values = np.array([0.0, 0.2, 0.3, 0.0, 0.4, 0.9, 0.4, 0.0])

    assert detect_active_window(time, values, threshold=0.1) == (4.0, 6.0)


def test_detect_active_window_uses_full_range_when_threshold_is_not_reached():
    from vaft.process import detect_active_window

    assert detect_active_window([0.0, 1.0], [0.0, 0.0], threshold=0.1) == (0.0, 1.0)


def test_legacy_process_alias_warns_and_preserves_result():
    from vaft.process.signal_processing import vfit_signal_start_end

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = vfit_signal_start_end([0.0, 1.0, 2.0], [0.0, 1.0, 0.0])

    assert result == (1.0, 1.0)
    assert any(item.category is DeprecationWarning for item in caught)
    assert "detect_active_window" in str(caught[0].message)


def test_machine_mapping_public_surface_is_canonical_but_legacy_imports_work():
    from vaft import machine_mapping

    assert "magnetics" in machine_mapping.__all__
    assert not any(name.startswith("vfit_") for name in machine_mapping.__all__)
    assert "VEST_DiamagneticFlux" not in machine_mapping.__all__
    assert "pf_plasma" not in machine_mapping.__all__

    # Package-level lazy exports are cached after first access. Remove a value
    # that an earlier test may have resolved so this test remains order-independent.
    machine_mapping.__dict__.pop("vfit_barometry_static", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        legacy_builder = machine_mapping.vfit_barometry_static

    payload = {}
    legacy_builder(payload)
    assert payload["barometry"]["gauge"][0]["name"] == "PKR-251 Main Gauge"
    assert any(item.category is DeprecationWarning for item in caught)
    assert "diagnostic module" in str(caught[0].message)


def test_unimplemented_pf_plasma_module_is_not_shipped():
    assert importlib.util.find_spec("vaft.machine_mapping.pf_plasma") is None


def test_top_level_namespace_has_no_duplicate_exports():
    import vaft

    assert len(vaft.__all__) == len(set(vaft.__all__))

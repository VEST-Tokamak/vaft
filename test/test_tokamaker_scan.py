"""Unit tests for TokaMaker parameter-scan argument validation.

``scan_tokamaker`` feeds ``param`` straight into ``dataclasses.replace``, so a
typo would surface as an opaque ``TypeError`` from deep inside the stdlib. It
is validated against ``TokaMakerConfig`` up front instead — before the ODS (or
the solver) is touched.
"""

import pytest

from vaft.code.tokamaker import TokaMakerConfig, scan_tokamaker


def test_scan_rejects_unknown_param():
    with pytest.raises(ValueError, match="Unknown TokaMakerConfig field"):
        scan_tokamaker(
            ods=None,
            base_config=TokaMakerConfig(),
            values=[50.0e3],
            param="Ip",  # correct spelling is ip
        )


def test_scan_error_lists_valid_fields():
    with pytest.raises(ValueError) as excinfo:
        scan_tokamaker(ods=None, base_config=TokaMakerConfig(), values=[1.0], param="nonsense")

    message = str(excinfo.value)
    assert "ip" in message
    assert "alpha_p_a" in message
    assert "mesh_file" in message


def test_scan_validates_before_touching_the_ods():
    # ods=None would fail loudly downstream; the ValueError proves validation
    # happens before any prepare/solve is attempted.
    with pytest.raises(ValueError):
        scan_tokamaker(ods=None, base_config=TokaMakerConfig(), values=[], param="bogus")

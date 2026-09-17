"""Composing a shot back out of the per-stage subtrees each stage owns.

`vaft.database.replication._project` splits a shot into owned subtrees on the
way out to HSDS; `compose_stage_products` unions them on the way back in. The
tests here are about the guards, because the composition itself is a copy: what
matters is that a pairing from two different runs is refused rather than
silently producing wall currents for the wrong instants.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from omas import ODS

from vaft.database.composition import StageCompositionError, compose_stage_products
from vaft.database.sources import STAGE_REPLICATION
from vaft.omas import save as save_ods


def _diagnostics(tmp_path, time, name="diagnostics.json"):
    ods = ODS(consistency_check=False)
    ods["dataset_description.data_entry.pulse"] = 39915
    ods["pf_active.time"] = time
    ods["pf_active.coil.0.current.data"] = np.zeros_like(time)
    ods["magnetics.ip.0.time"] = time
    ods["magnetics.ip.0.data"] = np.zeros_like(time)
    path = tmp_path / name
    save_ods(ods, path)
    return path


def _eddy(tmp_path, time, name="eddy.json"):
    ods = ODS(consistency_check=False)
    ods["dataset_description.data_entry.pulse"] = 39915
    ods["pf_passive.time"] = time
    ods["pf_passive.loop.0.current"] = np.ones_like(time)
    path = tmp_path / name
    save_ods(ods, path)
    return path


def _manifest(tmp_path, diagnostics_sha256, name="eddy-manifest.json"):
    path = tmp_path / name
    path.write_text(json.dumps({"input": {"diagnostics_sha256": diagnostics_sha256}}))
    return path


def test_composing_the_two_products_yields_one_shot(tmp_path):
    time = np.linspace(0.0, 0.01, 50)
    diagnostics = _diagnostics(tmp_path, time)
    eddy = _eddy(tmp_path, time)

    composed, provenance = compose_stage_products(diagnostics=diagnostics, eddy=eddy)

    assert "pf_active" in composed and "magnetics" in composed
    assert "pf_passive" in composed
    assert provenance["eddy"]["contributed"] == STAGE_REPLICATION["eddy"].ids
    assert provenance["diagnostics"]["path"] == str(diagnostics)
    assert provenance["eddy"]["path"] == str(eddy)


def test_a_pairing_whose_time_grids_disagree_is_refused(tmp_path):
    """The failure this guard exists for is silent, not loud.

    The EFIT constraint builder interpolates each slice onto `pf_passive.time`.
    A mismatched pairing does not raise there -- it returns wall currents for
    instants the diagnostics never measured.
    """
    diagnostics = _diagnostics(tmp_path, np.linspace(0.0, 0.01, 50))
    eddy = _eddy(tmp_path, np.linspace(0.0, 0.02, 50))

    with pytest.raises(StageCompositionError, match="not this diagnostics"):
        compose_stage_products(diagnostics=diagnostics, eddy=eddy)


def test_a_grid_of_the_right_length_but_the_wrong_values_is_still_refused(tmp_path):
    """Same length, different instants -- what a length check would wave through.

    This is the shape a shot reprocessed with a different magnetics
    configuration takes, which is exactly the pairing most likely to occur.
    """
    time = np.linspace(0.0, 0.01, 50)
    diagnostics = _diagnostics(tmp_path, time)
    eddy = _eddy(tmp_path, time + 1e-6)

    with pytest.raises(StageCompositionError):
        compose_stage_products(diagnostics=diagnostics, eddy=eddy)


def test_a_diagnostics_file_the_eddy_manifest_does_not_name_is_refused(tmp_path):
    time = np.linspace(0.0, 0.01, 50)
    diagnostics = _diagnostics(tmp_path, time)
    eddy = _eddy(tmp_path, time)
    manifest = _manifest(tmp_path, "0" * 64)

    with pytest.raises(StageCompositionError, match="different diagnostics file"):
        compose_stage_products(
            diagnostics=diagnostics, eddy=eddy, eddy_manifest=manifest
        )


def test_strict_false_downgrades_the_hash_check_but_not_the_grid_check(tmp_path):
    """Harnesses waive provenance; nothing waives the interpolation invariant."""
    time = np.linspace(0.0, 0.01, 50)
    diagnostics = _diagnostics(tmp_path, time)
    eddy = _eddy(tmp_path, time)
    manifest = _manifest(tmp_path, "0" * 64)

    with pytest.warns(RuntimeWarning, match="different diagnostics file"):
        composed, _ = compose_stage_products(
            diagnostics=diagnostics, eddy=eddy, eddy_manifest=manifest, strict=False
        )
    assert "pf_passive" in composed

    mismatched = _eddy(tmp_path, np.linspace(0.0, 0.02, 50), name="eddy2.json")
    with pytest.raises(StageCompositionError):
        compose_stage_products(
            diagnostics=diagnostics, eddy=mismatched, strict=False
        )


def test_a_product_without_pf_passive_is_named_rather_than_silently_composed(tmp_path):
    time = np.linspace(0.0, 0.01, 50)
    diagnostics = _diagnostics(tmp_path, time)
    not_eddy = _diagnostics(tmp_path, time, name="other.json")

    with pytest.raises(StageCompositionError, match="carries no pf_passive"):
        compose_stage_products(diagnostics=diagnostics, eddy=not_eddy)


def test_a_missing_time_grid_is_refused_rather_than_skipping_the_check(tmp_path):
    """The guard must not switch itself off for the input that needs it.

    An eddy product always has `pf_passive.time` and its diagnostics always has
    `pf_active.time`, so an absent one means these are not the pair they claim
    to be -- treating it as "nothing to compare" would wave through exactly the
    case the check exists for.
    """
    time = np.linspace(0.0, 0.01, 50)
    diagnostics = _diagnostics(tmp_path, time)

    ods = ODS(consistency_check=False)
    ods["pf_passive.loop.0.current"] = np.ones_like(time)
    gridless = tmp_path / "gridless.json"
    save_ods(ods, gridless)

    with pytest.raises(StageCompositionError, match="carries no pf_passive.time"):
        compose_stage_products(diagnostics=diagnostics, eddy=gridless)


def test_composition_never_reads_a_manifest_in_the_locale_encoding():
    """Cold review data F17: one `read_text()` had no encoding, so a manifest
    with a non-ASCII path or note decoded differently on a cp949/cp1252 host."""
    import inspect
    import re

    from vaft.database import composition

    source = inspect.getsource(composition)
    assert not re.search(r"\.read_text\(\s*\)", source)

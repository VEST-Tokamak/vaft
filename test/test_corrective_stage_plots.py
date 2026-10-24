"""The corrective stages' validation figures (#1842): declared once, optional, and drawn from the stage product."""

from __future__ import annotations

import numpy as np
import pytest
from omas import ODS

from vaft.database import production_qa as qa
from vaft.database.corrective_lineage import CORRECTIVE_LINEAGE


def test_every_corrective_stage_declares_a_plot_set_and_none_is_required():
    for stage in CORRECTIVE_LINEAGE:
        entries = qa.validation_plots(stage)
        assert entries, stage
        assert all(not entry.required for entry in entries), (
            f"{stage}: most shots have no upload, so a figure can never be a stage failure"
        )
        assert qa.stage_plot_filenames(stage, required_only=True) == ()


def test_the_filenames_are_unique_within_a_stage_and_png():
    for stage in CORRECTIVE_LINEAGE:
        names = qa.stage_plot_filenames(stage)
        assert len(set(names)) == len(names) and all(n.endswith(".png") for n in names), stage


def test_every_declared_recipe_is_a_registered_plot():
    from vaft.plot import canonical_names

    known = set(canonical_names())
    for stage in CORRECTIVE_LINEAGE:
        for entry in qa.validation_plots(stage):
            assert entry.plot in known, (stage, entry.plot)


def test_a_stage_with_no_data_records_every_figure_as_skipped_not_failed(tmp_path):
    """The normal result for most shots: the product is a bare shell and nothing can be drawn."""
    shell = ODS(consistency_check=False)
    shell["dataset_description.data_entry.pulse"] = 48226
    manifest = qa.render_stage_plots("thomson", shell, tmp_path)
    assert [p["status"] for p in manifest["plots"]] == ["skipped", "skipped"]
    assert all("does not carry the data" in p["reason"] for p in manifest["plots"])
    assert list(tmp_path.glob("*.png")) == []


def test_a_composed_profile_product_draws_its_figures(tmp_path):
    pytest.importorskip("matplotlib")
    from omas import ODS

    rho = np.linspace(0.0, 0.99, 30)
    ods = ODS(consistency_check=False)
    ods["core_profiles.time"] = np.array([0.3])
    base = "core_profiles.profiles_1d.0"
    ods[f"{base}.time"] = 0.3
    ods[f"{base}.grid.rho_tor_norm"] = rho
    ods[f"{base}.electrons.temperature"] = 10.0 + 150.0 * (1 - rho**2)
    ods[f"{base}.electrons.density"] = 0.4e19 + 1.4e19 * (1 - rho**2)
    ods[f"{base}.zeff"] = 1.0 + 1.0 * (1 - rho**2)
    ods[f"{base}.zeff_fit.parameters"] = "origin=derived; method=openadas_coronal; target_zeff=2.0"
    manifest = qa.render_stage_plots("resistive_zeff", ods, tmp_path)
    statuses = {p["name"]: p["status"] for p in manifest["plots"]}
    assert statuses == {"core_profiles_profile_zeff": "generated"}
    assert (tmp_path / "resistive_zeff_profile.png").stat().st_size > 1000

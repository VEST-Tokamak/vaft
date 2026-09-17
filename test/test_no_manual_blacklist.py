"""The routine pipeline depends on no hard-coded flux-loop blacklist (issue #295 §4).

Channel selection is the diagnostics-stage assessment's verdict alone.  The
list that used to exclude six flux loops on every shot is retired; it
survives only by name in the back-test that measured the evidence against
it.  This test fails if the pipeline grows a list again.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).parents[1]
PIPELINE = ROOT / "workflow/automatic_pipeline_1_routine_data_processing"


def test_the_routine_config_and_snakefile_carry_no_channel_list():
    config = yaml.safe_load((PIPELINE / "config.yaml").read_text())
    assert "broken" not in config["constraints"]
    assert config["constraints"]["detect_broken"] is True
    snakefile = (PIPELINE / "Snakefile").read_text()
    assert "--broken" not in snakefile and '"broken"' not in snakefile


def test_the_constraint_script_forms_decisions_from_the_assessment_alone():
    source = (PIPELINE / "generate_constraints_ods.py").read_text()
    assert "manual_rejections" not in source
    assert "--broken" not in source


def test_the_packaged_shot_is_selected_by_the_assessment_alone():
    """Default policy on the packaged product: only the probe the quality
    layer condemned is rejected; every flux loop is a constraint."""
    from vaft.omas.sample import sample_ods
    from vaft.validation.channel_decision import REJECTED
    from vaft.validation.efit_channels import decide_efit_channels

    ods = sample_ods()
    decisions = decide_efit_channels(ods, np.asarray(ods["equilibrium.time"], dtype=float))
    rejected = {key for key, d in decisions.entries.items() if np.any(d.state == REJECTED)}
    assert rejected == {("b_field_pol_probe", 25)}
    assert all(decisions.get("flux_loop", i).all_usable for i in range(11))

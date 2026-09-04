"""Pipeline 1's HSDS replication rules (issue #94).

These pin the workflow-side contract: replication is opt-in, refuses the
shot-first tree, is a state distinct from local completion, and never restates
where a stage's product goes.
"""

import re
from pathlib import Path

import pytest


WORKFLOW = (
    Path(__file__).resolve().parents[1]
    / "workflow"
    / "automatic_pipeline_1_routine_data_processing"
)

pytestmark = pytest.mark.skipif(
    not WORKFLOW.exists(), reason="workflow scripts are not part of the distribution"
)


@pytest.fixture(scope="module")
def snakefile() -> str:
    return (WORKFLOW / "Snakefile").read_text(encoding="utf-8")


def test_every_replicable_stage_has_a_rule(snakefile):
    from vaft.database.sources import replicable_stages

    for stage in replicable_stages():
        assert f"rule replicate_{stage}_to_hsds:" in snakefile, stage


def test_a_deferred_stage_has_no_rule(snakefile):
    """ideal-GPEC's destination is mapped, but its rule belongs to #95."""
    assert "rule replicate_gpec_ideal_to_hsds:" not in snakefile


def test_the_workflow_does_not_restate_where_a_stage_goes(snakefile):
    """The registry is the only authority for a destination.

    A source name appearing in a rule is a second mapping that will drift from
    the first.
    """
    from vaft.database.filedb import OMASStage
    from vaft.database.sources import known_sources

    stage_names = {member.value for member in OMASStage}
    for source in known_sources():
        if source.name == "main":
            # Too common a word to match on; the hyphenated lineages are the
            # ones a hand-written mapping would spell out.
            continue
        if source.name in stage_names:
            # `impa` names both a stage and the source it goes to. The workflow
            # spells the *stage* (`--stage impa`), which is what every rule
            # does; the destination still comes from the registry alone, so the
            # string is not a second mapping.
            continue
        assert source.name not in snakefile, source.name


def test_replication_is_opt_in(snakefile):
    assert 'HSDS_CONFIG.get("replicate", False)' in snakefile
    config = (WORKFLOW / "config.yaml").read_text(encoding="utf-8")
    assert re.search(r"^hsds:\s*$", config, re.MULTILINE)
    assert re.search(r"^\s+replicate:\s*false\s*$", config, re.MULTILINE)


def test_replication_refuses_the_shot_first_tree(snakefile):
    assert "hsds.replicate requires layout: filedb" in snakefile

    import sys

    sys.path.insert(0, str(WORKFLOW))
    try:
        from paths import PipelinePaths
    finally:
        sys.path.remove(str(WORKFLOW))

    legacy = PipelinePaths("/srv/vest.filedb", "shot_first")
    with pytest.raises(ValueError, match="requires layout: filedb"):
        legacy.replication_record(39915, "efit")


def test_the_replication_record_is_its_own_workflow_output(snakefile):
    """A product on disk must never imply it reached HSDS."""
    assert 'PATHS.shot_pattern("replication_record"' in snakefile
    assert "outputs += replication_records(shots)" in snakefile


def test_each_stage_replicates_on_its_own(snakefile):
    """One rule per stage, so a failure downstream cannot invalidate a product
    that already reached its source."""
    rules = re.findall(r"rule (replicate_\w+_to_hsds):", snakefile)
    assert len(rules) == len(set(rules)) >= 5
    # No rule takes another stage's product as input, which would couple them.
    for stage in ("efit", "chease", "mhd_linear"):
        block = snakefile.split(f"rule replicate_{stage}_to_hsds:")[1].split("rule ")[0]
        assert f'"{stage}_ods"' in block
        others = {"efit", "chease", "mhd_linear", "diagnostics", "eddy"} - {stage}
        for other in others:
            assert f'"{other}_ods"' not in block, (stage, other)


def test_efit_and_chease_publish_a_stage_manifest():
    """Replication reads the manifest to decide eligibility, so the two stages
    that had none now write one (absorbed from #137)."""
    for stage in ("efit", "chease"):
        script = (WORKFLOW / f"generate_{stage}_ods.py").read_text(encoding="utf-8")
        assert "write_manifest(" in script, stage
        assert '"--metadata"' in script, stage


def test_replication_is_throttled_independently_of_compute_cores(snakefile):
    """Uploads are bounded by the HSDS deployment, not by local cores.

    30 parallel uploads against a 4-node server produced transient failures on
    almost every shot. The `hsds` resource caps concurrent replication while the
    compute stages still use every core.
    """
    from vaft.database.sources import replicable_stages

    assert "hsds=1" in snakefile.replace(" ", "").replace("\n", "")
    for stage in replicable_stages():
        block = snakefile.split(f"rule replicate_{stage}_to_hsds:")[1].split("rule ")[0]
        assert "resources:" in block and "hsds=1" in block.replace(" ", ""), stage
    assert 'HSDS_CONFIG.get("concurrency"' in snakefile


# --------------------------------------------------------------------------- #
# the optional IMPA branch (issue #305)
# --------------------------------------------------------------------------- #


def test_the_optional_branch_is_not_part_of_the_baseline_target_set(snakefile):
    """A shot that never produced IMPA must not read as a missing output.

    The baseline products are demanded for every eligible shot; IMPA is demanded
    only for the sparse subset its own checkpoint selects.
    """
    baseline = snakefile.split("def eligible_pipeline_outputs")[1].split("\nrule ")[0]
    assert "impa" not in baseline
    assert "def impa_pipeline_outputs" in snakefile
    assert "checkpoint select_impa_shots:" in snakefile


def test_which_stages_are_optional_is_the_registry_answer(snakefile):
    """Restating it here is how the two would drift apart."""
    assert "STAGE_REPLICATION[stage].optional" in snakefile
    from vaft.database.sources import STAGE_REPLICATION

    assert STAGE_REPLICATION["impa"].optional
    assert not STAGE_REPLICATION["diagnostics"].optional


def test_the_impa_branch_is_opt_in(snakefile):
    assert 'IMPA_CONFIG.get("enable", False)' in snakefile
    config = (WORKFLOW / "config.yaml").read_text()
    assert re.search(r"^impa:\s*$", config, re.MULTILINE)
    assert re.search(r"^\s+enable:\s*false\s*$", config, re.MULTILINE)


def test_the_impa_stage_does_not_depend_on_the_baseline_product(snakefile):
    """A failure in either direction has to stay on its own side of the split."""
    block = snakefile.split("rule generate_impa_ods:")[1].split("\n############")[0]
    assert '"raw_dump"' in block
    assert "diagnostics" not in block


def test_the_impa_stage_records_a_failure_instead_of_raising_one():
    script = (WORKFLOW / "generate_impa_ods.py").read_text()
    assert "except Exception" in script
    assert '"status": "failed"' in script
    assert "return 0" in script

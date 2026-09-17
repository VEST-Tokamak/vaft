"""Pipeline 2's Snakefile builds the DAG it claims to (#130).

Nothing else in the suite loads this file. It is generated in a `for` loop with
computed rule names, and Snakemake's preprocessor mangles f-strings inside a
Snakefile -- `f"{stage}_ods"` once reached `getattr` as `" thomson _ods "` --
so "does it parse, and does it produce the rules" is worth asserting rather
than discovering on a server.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

WORKFLOW = (
    Path(__file__).resolve().parents[1]
    / "workflow"
    / "automatic_pipeline_2_corrective_data_update"
)

def _snakemake_importable() -> bool:
    try:
        import snakemake  # noqa: F401
    except Exception:
        return False
    return True


pytestmark = [
    pytest.mark.skipif(
        not WORKFLOW.exists(), reason="workflow scripts are not part of the distribution"
    ),
    pytest.mark.skipif(
        shutil.which("snakemake") is None and not _snakemake_importable(),
        reason="snakemake is not installed",
    ),
]


def _dry_run(tmp_path: Path, shot: int = 48226, extra: list[str] | None = None):
    config = tmp_path / "config.yaml"
    config.write_text(
        json.dumps(
            {
                "base_dir": str(tmp_path / "filedb"),
                "layout": "filedb",
                "shots": [shot],
                "external": {"data_root": str(tmp_path / "incoming"), "ces_options": "ids"},
                "kinetic": {"encoding": "raw6", "executable": ""},
                "hsds": {"replicate": False},
                "conda": None,
            }
        ),
        encoding="utf-8",
    )
    return subprocess.run(
        [
            sys.executable, "-m", "snakemake",
            "--snakefile", str(WORKFLOW / "Snakefile"),
            "--configfile", str(config),
            "--directory", str(tmp_path),
            "--cores", "1", "-n",
            *(extra or []),
        ],
        capture_output=True,
        text=True,
    )


def test_the_snakefile_parses_and_builds_every_stage(tmp_path):
    result = _dry_run(tmp_path)
    assert result.returncode == 0, result.stderr[-3000:]
    for rule in (
        "generate_thomson_ods",
        "generate_ces_ods",
        "generate_core_profiles_ods",
        "generate_electron_efit_ods",
        "generate_kinetic_efit_ods",
    ):
        assert rule in result.stdout, f"{rule} missing from the DAG\n{result.stdout[-2000:]}"


def test_both_kinetic_lineages_become_their_own_rule(tmp_path):
    """The loop that generates them is where a closure mistake would collapse them."""
    result = _dry_run(tmp_path)
    assert "generate_electron_efit_ods" in result.stdout
    assert "generate_kinetic_efit_ods" in result.stdout


def test_replication_adds_one_rule_per_stage(tmp_path):
    result = _dry_run(tmp_path, extra=["--config", 'hsds={"replicate": true}'])
    assert result.returncode == 0, result.stderr[-3000:]
    for stage in ("thomson", "ces", "core_profiles", "electron_efit", "kinetic_efit"):
        assert f"replicate_{stage}_to_hsds" in result.stdout, result.stdout[-2000:]


def test_the_shot_first_layout_is_refused(tmp_path):
    """These stages have no legacy path; the run must stop rather than invent one."""
    result = _dry_run(tmp_path, extra=["--config", "layout=shot_first"])
    assert result.returncode != 0
    assert "filedb" in (result.stderr + result.stdout)


def test_the_lineage_is_checked_against_the_registry():
    """A stage renamed in the registry must fail the run, not vanish from it."""
    from vaft.database.sources import replicable_stages

    corrective = set(replicable_stages(produced_by="corrective"))
    lineage = {"thomson", "ces", "core_profiles", "electron_efit", "kinetic_efit"}
    assert lineage <= corrective, sorted(lineage - corrective)

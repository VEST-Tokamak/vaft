"""``generate_chease_ods.py`` embeds `chease_runs.json`'s comparison metrics
onto the refined ODS (issue #172), so the ODS itself -- not only the g-files
-- carries what the ``chease`` validation stage needs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from omas import load_omas_json

from vaft.data.resources import data_path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "workflow/automatic_pipeline_1_routine_data_processing/generate_chease_ods.py"
SAMPLE_GFILE = data_path("efit/g039915.00319")


def _run(args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(REPO), "MPLBACKEND": "Agg"},
    )


def test_runs_summary_is_embedded_onto_equilibrium_code_parameters(tmp_path):
    refined = tmp_path / "g039915.00319"
    refined.write_bytes(Path(SAMPLE_GFILE).read_bytes())
    manifest = tmp_path / "refined_gfiles_generated.txt"
    manifest.write_text(str(refined) + "\n", encoding="utf-8")
    status = tmp_path / "chease_status.txt"
    status.write_text("completed: refined_gfiles=1\n", encoding="utf-8")
    runs = tmp_path / "chease_runs.json"
    runs.write_text(
        json.dumps(
            {
                "shot": 39915,
                "records": [
                    {
                        "input": "g039915.00319",
                        "staged": str(refined),
                        "status": "completed",
                        "comparison": {"q_rms_rel": 0.01, "current_rel_diff": 0.002},
                    },
                    {"input": "g039915.00299", "status": "missing_input"},
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "chease.json"

    result = _run(
        [
            "--shot", "39915",
            "--refined-gfile-manifest", str(manifest),
            "--status", str(status),
            "--runs-summary", str(runs),
            "--output", str(output),
            "--metadata", str(tmp_path / "manifest.json"),
        ]
    )
    assert result.returncode == 0, result.stderr

    ods = load_omas_json(str(output), consistency_check=False)
    assert ods["equilibrium.code.name"] == "chease"
    assert ods["equilibrium.code.library.0.name"] == "chease"
    parameters = json.loads(ods["equilibrium.code.parameters"])
    assert parameters["comparison_metrics"]["0"] == {
        "q_rms_rel": 0.01,
        "current_rel_diff": 0.002,
    }
    assert parameters["records_summary"] == [
        {"input": "g039915.00319", "status": "completed"},
        {"input": "g039915.00299", "status": "missing_input"},
    ]


def test_records_summary_survives_the_all_failed_minimal_ods(tmp_path):
    manifest = tmp_path / "refined_gfiles_generated.txt"
    manifest.write_text("", encoding="utf-8")
    status = tmp_path / "chease_status.txt"
    status.write_text("failed: refined_gfiles=0; failed=1\n", encoding="utf-8")
    runs = tmp_path / "chease_runs.json"
    runs.write_text(
        json.dumps(
            {
                "shot": 39915,
                "records": [{"input": "g039915.00319", "status": "failed"}],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "chease.json"

    result = _run(
        [
            "--shot", "39915",
            "--refined-gfile-manifest", str(manifest),
            "--status", str(status),
            "--runs-summary", str(runs),
            "--output", str(output),
            "--metadata", str(tmp_path / "manifest.json"),
        ]
    )
    assert result.returncode == 0, result.stderr

    ods = load_omas_json(str(output), consistency_check=False)
    assert ods["equilibrium.code.library.0.name"] == "chease"
    parameters = json.loads(ods["equilibrium.code.parameters"])
    assert parameters["comparison_metrics"] == {}
    assert parameters["records_summary"] == [{"input": "g039915.00319", "status": "failed"}]
    assert "equilibrium.time_slice" not in ods or len(ods["equilibrium.time_slice"]) == 0


def test_missing_runs_summary_is_tolerated(tmp_path):
    refined = tmp_path / "g039915.00319"
    refined.write_bytes(Path(SAMPLE_GFILE).read_bytes())
    manifest = tmp_path / "refined_gfiles_generated.txt"
    manifest.write_text(str(refined) + "\n", encoding="utf-8")
    status = tmp_path / "chease_status.txt"
    status.write_text("completed: refined_gfiles=1\n", encoding="utf-8")
    output = tmp_path / "chease.json"

    result = _run(
        [
            "--shot", "39915",
            "--refined-gfile-manifest", str(manifest),
            "--status", str(status),
            "--output", str(output),
            "--metadata", str(tmp_path / "manifest.json"),
        ]
    )
    assert result.returncode == 0, result.stderr

    ods = load_omas_json(str(output), consistency_check=False)
    parameters = json.loads(ods["equilibrium.code.parameters"])
    assert parameters == {"comparison_metrics": {}, "records_summary": []}


def test_a_refinement_that_parsed_nothing_says_so_in_its_manifest(tmp_path):
    """The placeholder ODS stays on disk; the manifest is what refuses it.

    Without this, a hollow equilibrium is indistinguishable from a refined one
    to anything downstream -- including HSDS replication.
    """
    import json

    manifest = tmp_path / "refined_gfiles_generated.txt"
    manifest.write_text("", encoding="utf-8")
    status = tmp_path / "chease_status.txt"
    status.write_text("failed", encoding="utf-8")
    output = tmp_path / "chease.json"
    metadata = tmp_path / "manifest.json"

    result = _run(
        [
            "--shot", "39915",
            "--refined-gfile-manifest", str(manifest),
            "--status", str(status),
            "--output", str(output),
            "--metadata", str(metadata),
        ]
    )

    assert result.returncode == 0, result.stderr
    recorded = json.loads(metadata.read_text(encoding="utf-8"))
    assert recorded["stage"] == "chease"
    assert recorded["status"] == "no_output"

    from vaft.database.replication import REPLICABLE_STATUSES

    assert recorded["status"] not in REPLICABLE_STATUSES

"""Workflow coverage for split EFIT reliability materialization."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "workflow"
    / "automatic_pipeline_3_data_summary"
    / "gen_efit_reliability.py"
)
SPEC = importlib.util.spec_from_file_location("gen_efit_reliability", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_workflow_materializes_independent_magnetic_and_kinetic_outputs(
    tmp_path,
    monkeypatch,
):
    summary_calls = []
    export_calls = []

    def summary(shot_range, *, preset, source):
        summary_calls.append((shot_range, preset, source))
        return pd.DataFrame(
            [
                {
                    "shot": 42,
                    "eq_index": 0,
                    "measurement_type": preset,
                    "measurement_index": 0,
                }
            ]
        )

    def export(frame, path, **kwargs):
        export_calls.append((frame, Path(path), kwargs))
        return frame

    monkeypatch.setattr(MODULE.database, "summary", summary)
    monkeypatch.setattr(MODULE.database, "export_summary", export)

    outputs = MODULE.generate_efit_reliability_history(
        (40, 42),
        source="private",
        magnetic_output_path=str(tmp_path / "magnetic.csv"),
        kinetic_output_path=str(tmp_path / "kinetic.xlsx"),
    )

    assert [call[1] for call in summary_calls] == [
        "efit_magnetic_reliability",
        "efit_kinetic_reliability",
    ]
    assert all(call[2] == "private" for call in summary_calls)
    assert [call[1].name for call in export_calls] == [
        "magnetic.csv",
        "kinetic.xlsx",
    ]
    assert all(call[2]["mode"] == "upsert" for call in export_calls)
    assert set(outputs) == {"magnetic", "kinetic"}

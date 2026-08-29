"""Execution contract for the completed Session 01 tutorial notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorial" / "01_getting_started_with_vaft.ipynb"


def _source_text(cell) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def test_session_01_executes_offline_and_keeps_source_clean(monkeypatch, tmp_path):
    """The offline lesson runs from a transient notebook without credentials."""
    source_book = nbformat.read(NOTEBOOK, as_version=4)
    executed_book = nbformat.from_dict(source_book)

    monkeypatch.setenv("VAFT_TUTORIAL_MODE", "offline")
    monkeypatch.setenv("VAFT_TUTORIAL_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("MPLBACKEND", "Agg")

    client = NotebookClient(
        executed_book,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    )
    client.execute()

    output_text = "\n".join(
        output.get("text", "")
        for cell in executed_book.cells
        for output in cell.get("outputs", [])
        if output.output_type == "stream"
    )
    assert "Mode: offline; source: VAFT packaged sample; shot: 39915" in output_text
    assert "The packaged sample supports" in output_text
    assert "SESSION_01_OFFLINE_READY" in output_text
    assert (tmp_path / "session01_plasma_current.png").is_file()
    assert (tmp_path / "session01_magnetic_overview.png").is_file()
    assert (tmp_path / "session01_uv_intensity.png").is_file()
    assert (tmp_path / "session01_pf_active_time_current.png").is_file()
    assert (
        tmp_path / "session01_exercise_magnetics_time_diamagnetic_flux.png"
    ).is_file()

    committed_book = nbformat.read(NOTEBOOK, as_version=4)
    for cell in committed_book.cells:
        if cell.cell_type == "code":
            assert cell.execution_count is None
            assert cell.outputs == []

    assert "VAFT_TUTORIAL_MODE" in "\n".join(
        _source_text(cell)
        for cell in committed_book.cells
        if cell.cell_type == "code"
    )

"""Execution contract for the completed Session 01 tutorial notebook."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import nbformat
from nbclient import NotebookClient
import numpy as np
import pytest


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
    monkeypatch.delenv("MPLBACKEND", raising=False)

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

    figure_cell_ids = {
        "session01-current-plot",
        "session01-magnetics-plot",
        "session01-uv-plot",
        "session01-integrated-plot",
        "session01-exercise-plot",
    }
    for cell in executed_book.cells:
        if cell.get("id") in figure_cell_ids:
            assert [output.output_type for output in cell.outputs] == ["display_data"]

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


def test_session_01_default_output_is_stable_from_tutorial_directory(
    monkeypatch,
):
    """Launching Jupyter from tutorial/ still uses the ignored output tree."""
    book = nbformat.read(NOTEBOOK, as_version=4)
    setup_cell = next(cell for cell in book.cells if cell.get("id") == "session01-setup")

    monkeypatch.chdir(ROOT / "tutorial")
    monkeypatch.delenv("VAFT_TUTORIAL_OUTPUT_DIR", raising=False)
    monkeypatch.setenv("VAFT_TUTORIAL_MODE", "offline")

    namespace = {}
    exec(compile(setup_cell.source, f"{NOTEBOOK.name}:session01-setup", "exec"), namespace)

    assert namespace["OUTPUT_DIR"] == ROOT / "tutorial" / "outputs" / "01"


def test_session_01_lab_branch_propagates_unexpected_api_errors():
    """The lab gate handles access failures without hiding programming defects."""
    book = nbformat.read(NOTEBOOK, as_version=4)
    lab_cell = next(cell for cell in book.cells if cell.get("id") == "session01-lab-read")
    plot_cell = next(
        cell for cell in book.cells if cell.get("id") == "session01-current-plot"
    )

    def broken_open(*args, **kwargs):
        raise RuntimeError("simulated API regression")

    namespace = {
        "MODE": "lab",
        "SHOT": 39915,
        "np": np,
        "vaft": SimpleNamespace(database=SimpleNamespace(open=broken_open)),
    }
    with pytest.raises(RuntimeError, match="simulated API regression"):
        exec(compile(lab_cell.source, f"{NOTEBOOK.name}:session01-lab-read", "exec"), namespace)

    assert "lab_time_s" in plot_cell.source
    assert "lab_ip_a" in plot_cell.source

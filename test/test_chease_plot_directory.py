"""Regression: the CHEASE plot directory must not silently become the CWD."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "workflow/automatic_pipeline_1_routine_data_processing/run_chease_refinement.py"


def _run(tmp_path, plot_dir_args):
    output = tmp_path / "out" / "refined_gfiles_generated.txt"
    output.parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "gfiles.txt").write_text("", encoding="utf-8")
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--shot", "39915",
         "--gfile-manifest", str(tmp_path / "gfiles.txt"),
         "--output", str(output), "--status", str(tmp_path / "out" / "status.txt"),
         "--run", "false", *plot_dir_args],
        capture_output=True, text=True, cwd=str(cwd),
        env={**__import__("os").environ, "PYTHONPATH": str(REPO), "MPLBACKEND": "Agg"},
    )
    assert result.returncode == 0, result.stderr
    return cwd, output.parent


def test_empty_plot_dir_falls_back_beside_the_output_not_the_cwd(tmp_path):
    cwd, output_dir = _run(tmp_path, ["--plot-dir", ""])
    assert not (cwd / "plot_refined_gfiles_generated.txt").exists(), \
        "an empty --plot-dir must not write into the working directory"
    assert (output_dir / "plots" / "plot_refined_gfiles_generated.txt").exists()


def test_absent_plot_dir_falls_back_beside_the_output_not_the_cwd(tmp_path):
    cwd, output_dir = _run(tmp_path, [])
    assert not (cwd / "plot_refined_gfiles_generated.txt").exists()
    assert (output_dir / "plots" / "plot_refined_gfiles_generated.txt").exists()


def test_an_explicit_plot_dir_is_honoured(tmp_path):
    target = tmp_path / "canonical" / "plot"
    cwd, output_dir = _run(tmp_path, ["--plot-dir", str(target)])
    assert (target / "plot_refined_gfiles_generated.txt").exists()
    assert not (output_dir / "plots").exists()


def test_empty_gfile_manifest_skips_even_when_chease_is_enabled(tmp_path):
    output = tmp_path / "out" / "refined_gfiles_generated.txt"
    gfiles = tmp_path / "gfiles.txt"
    gfiles.write_text("", encoding="utf-8")
    status = tmp_path / "out" / "status.txt"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--shot",
            "39915",
            "--gfile-manifest",
            str(gfiles),
            "--output",
            str(output),
            "--status",
            str(status),
            "--run",
            "true",
        ],
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "PYTHONPATH": str(REPO), "MPLBACKEND": "Agg"},
    )

    assert result.returncode == 0, result.stderr
    assert status.read_text(encoding="utf-8").strip() == "skipped: no EFIT gfiles; input_gfiles=0"

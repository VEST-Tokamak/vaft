from __future__ import annotations

import json

import pandas as pd

from vaft.cli._main import main as cli_main
from vaft.cli.filedb import main as filedb_main
from vaft.cli.summary import main as summary_main


def test_filedb_cli_delegates_to_read_only_audit(tmp_path, capsys):
    legacy = tmp_path / "public"
    (legacy / "39915/omas").mkdir(parents=True)
    (legacy / "39915/omas/39915_efit.json").write_text("reference", encoding="utf-8")

    exit_code = filedb_main(["audit", str(legacy)])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["dry_run"] is True
    assert payload["summary"]["files"] == 1


def test_top_level_cli_dispatches_filedb_workflow(tmp_path, capsys):
    legacy = tmp_path / "public"
    (legacy / "39915/omas").mkdir(parents=True)
    (legacy / "39915/omas/39915_efit.json").write_text("reference", encoding="utf-8")

    exit_code = cli_main(["filedb", "audit", str(legacy)])

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out)["dry_run"] is True


def test_summary_cli_composes_query_and_export(monkeypatch, tmp_path):
    calls = {}
    frame = pd.DataFrame({"shot": [10], "eq_index": [0], "time_s": [0.1]})

    def fake_summary(shot_range, **kwargs):
        calls["summary"] = (shot_range, kwargs)
        return frame

    def fake_export(value, path, **kwargs):
        calls["export"] = (value, path, kwargs)
        return value

    monkeypatch.setattr("vaft.cli.summary.database.summary", fake_summary)
    monkeypatch.setattr("vaft.cli.summary.database.export_summary", fake_export)
    output = tmp_path / "history.xlsx"

    assert summary_main([
        "export", "--shot-range", "10:12", "--output", str(output), "--upsert"
    ]) == 0

    # No --source means the VAFT-native default; the registry resolves None.
    assert calls["summary"] == ((10, 12), {"preset": "equilibrium_global", "source": None})
    assert calls["export"][0] is frame
    assert calls["export"][1] == str(output)
    assert calls["export"][2]["mode"] == "upsert"
    assert calls["export"][2]["replace_groups"] == ("shot",)


def test_summary_sources_lists_the_catalog(capsys):
    from vaft.cli.summary import main as summary_main

    assert summary_main(["sources"]) == 0

    printed = capsys.readouterr().out
    assert "main" in printed
    assert "chease-mhd-stability" in printed
    # The legacy namespace has to be visibly read-only in the listing.
    assert "public" in printed and "read-only" in printed
    # A sparse source holds only the shots its product was produced for, so the
    # listing has to say that rather than let a missing shot read as a gap.
    impa = next(line for line in printed.splitlines() if line.startswith("impa"))
    assert "sparse" in impa
    assert all(
        "complete" in line
        for line in printed.splitlines()
        if line.startswith(("main", "public"))
    )


def test_export_cli_forwards_to_the_database_api(monkeypatch, tmp_path, capsys):
    calls = {}

    def fake_export(shot, source, **kwargs):
        calls["args"] = (shot, source, kwargs)
        return {name: tmp_path / name for name in kwargs["backend"]}

    monkeypatch.setattr("vaft.database.export", fake_export)
    exit_code = cli_main([
        "export", "--shot", "41672", "--source", "public",
        "--backend", "imas-nc", "omas-json", "geqdsk", "--output", str(tmp_path),
    ])

    assert exit_code == 0
    assert calls["args"] == (41672, "public", {
        "backend": ["imas-nc", "omas-json", "geqdsk"], "output": str(tmp_path),
        "overwrite": False, "cache": "auto", "transport": "auto",
    })
    assert capsys.readouterr().out.splitlines() == [
        f"{name}: {tmp_path / name}" for name in ("imas-nc", "omas-json", "geqdsk")
    ]


def test_export_cli_reports_errors_and_explains_every_backend(monkeypatch, capsys):
    import pytest

    from vaft.cli import export as export_cli
    from vaft.database._export import BACKENDS

    def refuse(*_args, **_kwargs):
        raise FileExistsError("omas_1.json already exists; pass overwrite=True")

    monkeypatch.setattr("vaft.database.export", refuse)
    assert export_cli.main(["--shot", "1", "--backend", "omas-json"]) == 1
    assert "already exists" in capsys.readouterr().err

    assert list(export_cli.BACKEND_HELP) == list(BACKENDS)
    with pytest.raises(SystemExit):
        export_cli.main(["--help"])
    help_text = capsys.readouterr().out
    for name in BACKENDS:
        assert name in help_text
    assert "NOT an IMAS Data Entry" in help_text and "NOT the IMAS netCDF convention" in help_text
    with pytest.raises(SystemExit) as raised:
        export_cli.main(["--shot", "1", "--backend", "nc"])
    assert raised.value.code == 2


def test_export_cli_imports_nothing_heavy_before_parsing():
    import subprocess
    import sys

    code = (
        "import sys, vaft.cli.export; "
        "heavy = sorted(m for m in sys.modules if m.startswith(('vaft.omas', 'vaft.database', 'imas', 'omas'))); "
        "assert not heavy, heavy"
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=300)

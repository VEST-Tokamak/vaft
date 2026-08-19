from __future__ import annotations

import json

from vaft.cli._main import main as cli_main
from vaft.cli.filedb import main as filedb_main


def test_filedb_cli_delegates_to_read_only_audit(tmp_path, capsys):
    legacy = tmp_path / "public"
    (legacy / "39915/omas").mkdir(parents=True)
    (legacy / "39915/omas/39915_efit.json").write_text("reference")

    exit_code = filedb_main(["audit", str(legacy)])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["dry_run"] is True
    assert payload["summary"]["files"] == 1


def test_top_level_cli_dispatches_filedb_workflow(tmp_path, capsys):
    legacy = tmp_path / "public"
    (legacy / "39915/omas").mkdir(parents=True)
    (legacy / "39915/omas/39915_efit.json").write_text("reference")

    exit_code = cli_main(["filedb", "audit", str(legacy)])

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out)["dry_run"] is True

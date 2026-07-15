"""Unit tests for TES ``.RESULT`` parsing and output discovery.

These cover the pure-Python half of the TES adapter, so they run without the
``rtes`` binary: scalar/coil parsing of a ``.RESULT`` file, where the scalar
header stops, and which files ``collect_tes_outputs`` picks out of a workdir.
"""

import pytest

from vaft.code.tes import (
    collect_tes_outputs,
    parse_result_coils,
    parse_result_scalars,
)


RESULT_TEXT = """\
 TES EQUILIBRIUM RESULT
 IP[kA]     125.300   BT0[T]      0.180
 KAPPA        1.850   DELTA_U     0.310
 Q95          4.200   RESIDUAL    1.5e-7
 BETAP        0.450   WMHD[MJ]    0.0123

 EXT. COIL CURRENT UPDATED
 [  1]    97.0 -->   99.5 :    2.5
 [  2]    26.0 -->   25.1 :   -0.9
 [  3]   -71.0 -->  -70.2 :    0.8

 FLUX-DIFFERENCE   0.001
"""


@pytest.fixture()
def result_file(tmp_path):
    path = tmp_path / "vest.RESULT"
    path.write_text(RESULT_TEXT)
    return path


def test_parse_scalars_with_and_without_units(result_file):
    scalars = parse_result_scalars(result_file)

    # Labelled with a unit: raw label kept, friendly alias added alongside.
    assert scalars["IP[kA]"] == pytest.approx(125.3)
    assert scalars["ip_kA"] == pytest.approx(125.3)
    assert scalars["WMHD[MJ]"] == pytest.approx(0.0123)
    assert scalars["wmhd_MJ"] == pytest.approx(0.0123)

    # Unitless label.
    assert scalars["KAPPA"] == pytest.approx(1.85)
    assert scalars["kappa"] == pytest.approx(1.85)
    assert scalars["Q95"] == pytest.approx(4.2)
    assert scalars["q95"] == pytest.approx(4.2)

    # Scientific notation, and an unaliased label still captured raw.
    assert scalars["RESIDUAL"] == pytest.approx(1.5e-7)
    assert "residual" not in scalars


def test_parse_scalars_stops_at_coil_block(result_file):
    scalars = parse_result_scalars(result_file)

    # Everything past the coil-current header belongs to a list block.
    assert "FLUX-DIFFERENCE" not in scalars
    assert not any(key.startswith("[") for key in scalars)


def test_parse_scalars_stops_at_bracketed_list_block(tmp_path):
    path = tmp_path / "iso.RESULT"
    path.write_text(
        " IP[kA]     125.300\n"
        " [ISO-FLUX POINTS]\n"
        " AFTERBLOCK    9.999\n"
    )

    scalars = parse_result_scalars(path)

    assert scalars["ip_kA"] == pytest.approx(125.3)
    assert "AFTERBLOCK" not in scalars


def test_parse_coils(result_file):
    coils = parse_result_coils(result_file)

    assert len(coils) == 3
    assert coils[0] == {
        "index": 1,
        "base_kA": pytest.approx(97.0),
        "updated_kA": pytest.approx(99.5),
        "delta_kA": pytest.approx(2.5),
    }
    # Negative currents and deltas round-trip.
    assert coils[1]["delta_kA"] == pytest.approx(-0.9)
    assert coils[2]["base_kA"] == pytest.approx(-71.0)

    # ``index`` is the coil number, not a current: it stays an int.
    assert all(isinstance(coil["index"], int) for coil in coils)


def test_parse_coils_without_indentation(tmp_path):
    # Same block flush against column 0 rather than padded.
    path = tmp_path / "flush.RESULT"
    path.write_text(
        "EXT. COIL CURRENT UPDATED\n"
        "[  1]    97.0 -->   99.5 :    2.5\n"
        "[  2]    26.0 -->   25.1 :   -0.9\n"
        "\n"
    )

    coils = parse_result_coils(path)

    assert [coil["index"] for coil in coils] == [1, 2]
    assert coils[1]["updated_kA"] == pytest.approx(25.1)


def test_parse_coils_absent_block_returns_empty(tmp_path):
    path = tmp_path / "nocoils.RESULT"
    path.write_text(" IP[kA]  125.3\n")

    assert parse_result_coils(path) == []


def test_collect_prefers_shot_time_gfile_over_bare_candidate(tmp_path):
    (tmp_path / "g012345.00500").write_text("not a real geqdsk")
    (tmp_path / "g999").write_text("also not a geqdsk")
    (tmp_path / "a012345.00500").write_text("not a real aeqdsk")
    (tmp_path / "vest.RESULT").write_text(RESULT_TEXT)

    result = collect_tes_outputs(tmp_path)

    assert result.gfile is not None and result.gfile.name == "g012345.00500"
    assert result.afile is not None and result.afile.name == "a012345.00500"
    assert result.result_file is not None and result.result_file.name == "vest.RESULT"
    assert result.scalars["ip_kA"] == pytest.approx(125.3)


def test_collect_ignores_unrelated_gpec_outputs(tmp_path):
    # A reused workdir holding GPEC outputs but no EFIT g-file/a-file: the broad
    # ``g*``/``a*`` fallbacks used to match these.
    (tmp_path / "gpec_control_output_n1.nc").write_text("gpec")
    (tmp_path / "gpec_profile_output_n1.nc").write_text("gpec")
    (tmp_path / "ahg2msc.out").write_text("vacuum")

    result = collect_tes_outputs(tmp_path)

    assert result.gfile is None
    assert result.afile is None
    assert result.ods is None
    assert "_geqdsk_error" not in result.scalars


def test_collect_records_unparsable_gfile_instead_of_raising(tmp_path):
    (tmp_path / "g012345.00500").write_text("this is not a geqdsk")

    result = collect_tes_outputs(tmp_path)

    # Discovery still succeeds; the parse failure is reported, not swallowed.
    assert result.gfile is not None and result.gfile.name == "g012345.00500"
    assert result.ods is None
    assert result.geqdsk == ()
    assert "g012345.00500" in result.scalars["_geqdsk_error"]

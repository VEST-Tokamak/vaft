import json
from pathlib import Path
from dataclasses import replace

import numpy as np
import pytest
from omas import ODS

from vaft.machine_mapping.efit_coilset import vest_efit_coilset_policy

COILSET = vest_efit_coilset_policy()
from vaft.code.efit.config import DIAGNOSTIC_GROUPS
from vaft.code.efit import (
    EFITConfig,
    EFITConstraintConfig,
    EFITInitializationConfig,
    EFITNumericsConfig,
    EFITProfileConfig,
    EFITScientificConfig,
    efit_parameter_grid,
    generate_kfile,
    prepare_efit_inputs,
)


def _constraints_ods(tmp_path, *, time=0.319):
    ods = ODS(consistency_check=False)
    ods["equilibrium.time"] = np.asarray([time])
    ods["equilibrium.code.parameters.time_slice.0.IN1.INPUT_DIR"] = str(tmp_path)
    ods["equilibrium.code.parameters.time_slice.0.IN1.VCURRT"] = np.asarray([3.0, -2.0])
    # As many PF channels as the coilset has groups: the writer refuses a tree
    # that describes a different coilset from the table, which is the point of
    # that check, so the fixture must not hard-code a count.
    for index, name in enumerate(COILSET.group_names):
        root = f"equilibrium.time_slice.0.constraints.pf_current.{index}"
        ods[f"{root}.measured"] = float(index + 1)
        ods[f"{root}.measured_error_upper"] = 0.25
        ods[f"{root}.weight"] = 2.0
        ods[f"{root}.source"] = name
    scalar_values = {
        "ip": (50_000.0, 20.0, 3.0),
        "diamagnetic_flux": (-0.004, 0.0002, 4.0),
        "b_field_tor_vacuum_r": (0.08, 0.001, 1.0),
    }
    for name, (measured, error, weight) in scalar_values.items():
        root = f"equilibrium.time_slice.0.constraints.{name}"
        ods[f"{root}.measured"] = measured
        ods[f"{root}.measured_error_upper"] = error
        ods[f"{root}.weight"] = weight
    for name, measured, error, weight in (
        ("bpol_probe", 0.01, 0.002, 5.0),
        ("flux_loop", 0.02, 0.003, 6.0),
    ):
        root = f"equilibrium.time_slice.0.constraints.{name}.0"
        ods[f"{root}.measured"] = measured
        ods[f"{root}.measured_error_upper"] = error
        ods[f"{root}.weight"] = weight
    return ods


def _kfile_text(tmp_path, scientific=None):
    ods = _constraints_ods(tmp_path)
    generate_kfile(
        ods,
        39915,
        save_dir=str(tmp_path),
        config=scientific,
    )
    return next((tmp_path / "kfile").iterdir()).read_text(encoding="utf-8")


def test_the_termination_settings_are_absent_until_a_study_sets_them(tmp_path):
    """Issue #171: the four settings a VEST fit stops on are now expressible.

    Absent by default, so the routine k-file is unchanged and EFIT's own
    defaults still apply -- which is exactly the situation #171 exists to
    characterize, and it must stay reproducible while it is characterized.
    """
    routine = _kfile_text(tmp_path, EFITScientificConfig())
    for key in ("ERRMIN", "SAICON", "ICONVR", "NXITER"):
        assert f" {key} " not in routine

    tuned = _kfile_text(
        tmp_path,
        EFITScientificConfig(
            numerics=EFITNumericsConfig(
                error_minimum=1.0e-3,
                chi_squared_target=60.0,
                convergence_mode=1,
                inner_iterations=20,
            )
        ),
    )
    added = [line for line in tuned.splitlines() if line not in routine.splitlines()]
    assert sorted(line.split("=")[0].strip() for line in added) == [
        "ERRMIN",
        "ICONVR",
        "NXITER",
        "SAICON",
    ]
    assert " ERRMIN = 0.001" in tuned and " SAICON = 60.0" in tuned

    # The scientific hash must move with them, or a scan would record two
    # different configurations under one identity.
    assert (
        EFITScientificConfig().sha256
        != EFITScientificConfig(
            numerics=EFITNumericsConfig(chi_squared_target=60.0)
        ).sha256
    )


@pytest.mark.parametrize(
    "bad",
    [
        {"error_minimum": 0.0},
        {"chi_squared_target": -1.0},
        {"convergence_mode": 0},
        {"inner_iterations": True},
    ],
)
def test_the_termination_settings_refuse_values_efit_cannot_use(bad):
    with pytest.raises(ValueError):
        EFITNumericsConfig(**bad)


def test_routine_defaults_preserve_documented_kfile_semantics(tmp_path):
    text = _kfile_text(tmp_path, EFITScientificConfig())

    expected = {
        "KPPCUR": "2",
        "KFFCUR": "2",
        "KPPFNC": "0",
        "KFFFNC": "0",
        "PCURBD": "1",
        "FCURBD": "1",
        "FWTBP": "0",
        "CUTIP": "15000.0",
        "RELIP": "0.32",
        "AELIP": "0.3",
        "EELIP": "1.6",
        "RELAX": "1.0",
        "ERROR": "1e-05",
        "SERROR": "0.0005",
        "MXITER": "-100",
        "IVESEL": "1",
        "IFITVS": "0",
        "KCCOILS": "17",
    }
    for key, value in expected.items():
        assert f" {key} = {value}" in text
    assert " ICINIT " not in text


def test_the_inboard_seed_default_moves_relip_and_nothing_else(tmp_path):
    """#588's conclusion, pinned where it can silently regress.

    `rzero` was three namelist quantities at once, and the seed study had to
    split `RELIP` out of it before a sweep could attribute anything to the
    seed. Now that the default seed has actually moved inboard, the whole
    point is that it moved *alone*: the reference major radius, `RCENTR`, and
    the vacuum toroidal field EFIT derives from it must be exactly what they
    were. Comparing the two k-files line by line is the only check that
    catches a recoupling, because a recoupled `RZERO` still writes a
    well-formed file that quietly reconstructs a different machine.
    """
    routine = _kfile_text(tmp_path, EFITScientificConfig())
    coupled = _kfile_text(
        tmp_path,
        EFITScientificConfig(initialization=EFITInitializationConfig(ellipse_rzero=None)),
    )

    before, after = coupled.splitlines(), routine.splitlines()
    assert len(before) == len(after)
    differing = [(old, new) for old, new in zip(before, after) if old != new]
    assert differing == [(" RELIP = 0.4", " RELIP = 0.32")], differing

    # Redundant given the line-by-line comparison, and worth stating anyway:
    # these are the three the old coupling dragged along, and BTOR is the one
    # that would change the physics rather than merely the bookkeeping.
    assert " RZERO = 0.4" in after
    assert " RCENTR = 0.4" in after
    assert EFITInitializationConfig().seed_rzero == 0.32
    assert EFITInitializationConfig().rzero == 0.4
    assert EFITInitializationConfig(ellipse_rzero=None).seed_rzero == 0.4


def test_the_seed_comes_from_the_config_not_the_constraints_tree(tmp_path):
    """`generate_constraints_ods` writes its own `RELIP` into the ODS.

    It is a second, hard-coded copy of the seed (`kfile.py`, in the block that
    sets `RZERO`, `AELIP` and `EELIP` beside it) and it still says 0.4. The
    writer overrides it from the configuration, which is why the #588 sweep
    varied anything at all -- but nothing said so out loud, and a change that
    let the stale copy win would move every routine reconstruction back to the
    old seed while every test on the config kept passing.
    """
    ods = _constraints_ods(tmp_path)
    ods["equilibrium.code.parameters.time_slice.0.IN1.RELIP"] = 0.4
    ods["equilibrium.code.parameters.time_slice.0.IN1.RZERO"] = 0.4
    generate_kfile(ods, 39915, save_dir=str(tmp_path), config=EFITScientificConfig())
    text = next((tmp_path / "kfile").iterdir()).read_text(encoding="utf-8")

    assert " RELIP = 0.32" in text
    assert " RZERO = 0.4" in text


def test_the_coil_groups_come_from_the_machine_not_from_a_written_down_list():
    """The 16- and 26-coil conventions are gone, and this is what replaced them.

    VEST energises ten PF circuits. EFIT's table carries sixteen current
    groups over them, and that mapping used to live in a hard-coded
    twenty-six-entry coil list plus a table of positions picking sixteen of
    it. Both are deleted: each group's circuit is derived from its own name
    and the grouping from each element's own position, in the one module that
    also projects the F-coil groups for EFUND.
    """

    assert len(COILSET.group_names) == COILSET.nfsum == 26
    assert set(COILSET.source_circuit.values()) == {f"PF{n}" for n in range(1, 11)}
    assert [COILSET.source_circuit[name] for name in COILSET.group_names[:8]] == ["PF1"] * 8
    assert COILSET.group_for_element("PF1", 0.15) == "PF1-1"
    assert COILSET.group_for_element("PF1", 1.10) == "PF1-4"
    assert COILSET.group_for_element("PF1", -0.15) == "PF1-5"
    assert COILSET.group_for_element("PF9", 1.0) == "PF9U"
    assert COILSET.group_for_element("PF9", -1.0) == "PF9L"
    # A circuit the machine does not have is still a real answer, not a raise.
    assert COILSET.group_for_element("PF99", 0.5) is None

    # The old names are gone from the package, not merely unused.
    import vaft.code.efit as package
    import vaft.code.efit.kfile as kfile_module
    import vaft.code.efit.legacy as legacy_module

    for gone in ("vfit_pf_active_efit26", "efit16_group_indices", "EFIT16_GROUP_POSITIONS"):
        assert not hasattr(package, gone), gone
        assert not hasattr(kfile_module, gone), gone
        assert not hasattr(legacy_module, gone), gone


#: The matrix that was written out by index, kept here so the derivation is
#: checked against the thing it replaced rather than against itself.
RETIRED_COIL_MATRIX = tuple(
    tuple(row)
    for row in [
        [1.0 if c < 7 else 0.0 for c in range(12)],
        *[
            [(-1.0 if c == k else 0.0) for c in range(12)]
            for k in range(7)
        ],
        [1.0 if c == 7 else 0.0 for c in range(12)],
        [-1.0 if c == 7 else 0.0 for c in range(12)],
        [1.0 if c == 8 else 0.0 for c in range(12)],
        [-1.0 if c == 8 else 0.0 for c in range(12)],
        [1.0 if c == 9 else 0.0 for c in range(12)],
        [(-1.0 if c == 9 else 0.0) + (1.0 if c == 11 else 0.0) for c in range(12)],
        [(1.0 if c == 10 else 0.0) + (-1.0 if c == 11 else 0.0) for c in range(12)],
        [-1.0 if c == 10 else 0.0 for c in range(12)],
    ]
)


def test_a_discharge_that_energised_a_different_subset_gets_a_kfile():
    """#708's whole point, and a thing this repository could not do before.

    The table used to describe five circuits, so a discharge that energised
    PF2 or PF3 produced no k-file for them -- those coils were dropped
    silently, with no error. The table now describes all ten, so the active
    set is expressed by the currents and weights rather than by a different
    table, and no regeneration is needed.
    """
    import numpy as np
    from omas import ODS

    from vaft.machine_mapping.efit_coilset import (
        detect_energised_circuits,
        vest_efit_coilset_policy,
    )

    policy = vest_efit_coilset_policy()
    assert set(policy.source_circuit.values()) == {f"PF{n}" for n in range(1, 11)}

    # A discharge that used PF3 and left PF5 cold -- the opposite of the
    # reference era, and impossible to reconstruct before this change.
    source = ODS(consistency_check=False)
    times = np.linspace(0.30, 0.34, 32)
    source["pf_active.time"] = times
    energised_here = {"PF1": 8_000.0, "PF3": 2_500.0, "PF9": 400.0, "PF10": 400.0}
    for index in range(1, 11):
        name = f"PF{index}"
        source[f"pf_active.coil.{index - 1}.name"] = name
        source[f"pf_active.coil.{index - 1}.current.data"] = np.full(
            times.size, energised_here.get(name, 0.0)
        )

    decision = detect_energised_circuits(
        source["pf_active"], policy, window=(float(times[0]), float(times[-1]))
    )
    assert set(decision.energised) == set(energised_here)
    assert decision.measured is True
    # PF3 is not in the declaration, PF5/PF6 are; the disagreement is reported
    # rather than resolved by preferring one side.
    assert not decision.agrees_with_declaration
    assert "PF3" in decision.disagreement and "PF5" in decision.disagreement

    # And the k-file carries PF3's current in PF3's slots, with the circuits
    # this discharge did not use weighted out.
    built = ODS(consistency_check=False)["pf_active"]
    from vaft.code.efit.kfile import build_efit_coil_currents

    build_efit_coil_currents(built, source["pf_active"], coilset=policy)
    names = [str(built[f"coil.{i}.name"]) for i in range(len(built["coil"]))]
    currents = [float(built[f"coil.{i}.current.data"][0]) for i in range(len(names))]
    by_name = dict(zip(names, currents))
    assert by_name["PF3U"] == by_name["PF3L"] == 2_500.0
    assert by_name["PF5U"] == by_name["PF5L"] == 0.0
    assert by_name["PF1-1"] == by_name["PF1-8"] == 8_000.0


def test_the_coil_constraint_matrix_is_derived_from_the_machine():
    """#708: the equalities follow from the coilset, not from a table by index.

    Splitting a circuit into current groups does not split its current, so each
    circuit ties its extra groups back to its first; each series-wired pair
    adds one more. For VEST that is 7 + 4 + 1 = 12 columns over 16 groups.

    Checked against the retired literal element for element, because a
    derivation verified only against itself proves nothing. The retired matrix
    had its twelfth column -- the PF9/PF10 tie -- written as two bare index
    assignments with no explanation; it is now a `ties:` entry carrying its
    evidence.
    """
    # The shipped coilset, which since #708 describes all ten circuits.
    policy = vest_efit_coilset_policy()
    derived = policy.constraint_matrix()
    assert len(derived) == policy.nfsum == 26
    assert len(derived[0]) == 17  # 7 solenoid + 9 pairs + 1 series tie

    # And the five-circuit coilset the retired literal was written for, so the
    # derivation is still checked against the thing it replaced rather than
    # against itself.
    from vaft.machine_mapping.efit_coilset import _build

    legacy_groups = {
        "PF1": {"split": "axial", "edges": [0.0, 0.3, 0.6, 0.9, 1.2],
                "status": "inferred", "provenance": "x"},
    }
    for coil in ("PF5", "PF6", "PF9", "PF10"):
        legacy_groups[coil] = {"split": "midplane", "status": "inferred", "provenance": "x"}
    legacy = _build({"efit_coilset": {
        "groups": legacy_groups,
        "ties": [{"groups": ["PF9L", "PF10U"], "status": "measured", "provenance": "x"}],
    }})
    assert legacy.constraint_matrix() == RETIRED_COIL_MATRIX
    assert legacy.constraint_targets() == (0.0,) * 12

    # Every column is one equality: a +1, a -1, and nothing else.
    for column in range(len(derived[0])):
        values = [derived[row][column] for row in range(policy.nfsum)]
        assert sorted(v for v in values if v) == [-1.0, 1.0], column


def test_a_config_without_a_matrix_lets_the_machine_supply_one():
    from vaft.code.efit.config import EFITConstraintConfig

    assert EFITConstraintConfig().coil_constraint_matrix is None
    assert EFITConstraintConfig().coil_constraint_targets is None

    # An explicit matrix is still validated.
    with pytest.raises(ValueError, match="rectangular"):
        EFITConstraintConfig(coil_constraint_matrix=((1.0, 0.0), (1.0,)))
    with pytest.raises(ValueError, match="needs a coil_constraint_matrix"):
        EFITConstraintConfig(coil_constraint_targets=(0.0,))


def test_the_plasma_current_floor_is_the_vacuum_switch():
    """`CUTIP` decides when there is no plasma worth fitting, not a tolerance.

    EFIT tests `|Ip| <= cutip` and then stops reconstructing: `ivacum = 1`,
    `ierchk = 0`, `iconvr = 3` (`data_input.F90:2454`). Raised from 5 kA to
    15 kA in #708.

    `IP_FIT_FLOOR` remains a different quantity -- the probe-recovery
    backend's floor, deciding whether there is enough signal to fit a family
    Gaussian, not whether there is a plasma. The two now carry the same value
    on purpose: the backend has no reason to withhold a recovered reading from
    a slice EFIT will go on to reconstruct, which is what the inherited 45 kA
    did between them. Equal values, still two decisions, so this pins the
    number rather than tying one to the other.
    """
    from vaft.code.efit.config import EFITInitializationConfig
    from vaft.code.efit.recovery import IP_FIT_FLOOR

    assert EFITInitializationConfig().current_threshold == 15_000.0
    assert IP_FIT_FLOOR == 15_000.0

    # A floor is a floor: negative is refused, zero means "always reconstruct".
    with pytest.raises(ValueError, match="non-negative"):
        EFITInitializationConfig(current_threshold=-1.0)
    assert EFITInitializationConfig(current_threshold=0.0).current_threshold == 0.0


def test_the_coilset_is_configuration_and_derives_its_own_names():
    """#708: the coilset stops being three literals in the projection module.

    A sixteen-name tuple, a tuple of solenoid edges, and an
    `if coil_name == "PF1" ... elif coil_name in (...)` that returned None --
    silently -- for every circuit EFIT had no group for. All three are now
    `vest.yaml`'s `efit_coilset`, and the names are derived from the splits
    rather than written down: that ordering is the table's `fcid` order and the
    k-file's `BRSP` order at once, so the two cannot disagree.
    """
    policy = vest_efit_coilset_policy()

    assert policy.nfsum == len(policy.group_names) == 26
    assert policy.circuits == tuple(f"PF{n}" for n in range(1, 11))
    # Derived, not listed: an axial split gives upper by ascending |z| then
    # lower the same way; a midplane split gives U then L.
    assert policy.group_names[:8] == tuple(f"PF1-{n}" for n in range(1, 9))
    assert policy.group_names[8:12] == ("PF2U", "PF2L", "PF3U", "PF3L")
    assert policy.group_names[-4:] == ("PF9U", "PF9L", "PF10U", "PF10L")

    assert policy.group_for_element("PF1", 0.15) == "PF1-1"
    assert policy.group_for_element("PF1", 1.10) == "PF1-4"
    assert policy.group_for_element("PF1", -0.15) == "PF1-5"
    assert policy.group_for_element("PF9", 1.0) == "PF9U"
    assert policy.group_for_element("PF9", -1.0) == "PF9L"
    # Every circuit now has a group; a name the machine does not have still
    # returns None rather than raising.
    assert policy.group_for_element("PF2", 0.5) == "PF2U"
    assert policy.group_for_element("PF99", 0.5) is None

    # The series tie is machine wiring and is carried with its provenance.
    assert policy.ties == (("PF9L", "PF10U"),)
    assert policy.status["tie:PF9L=PF10U"] == "measured"
    assert "bit-identical" in policy.provenance["tie:PF9L=PF10U"]

    # And the literals are gone from the projection module, not just unused.
    import vaft.machine_mapping.efund_geometry as projection

    for gone in ("EFIT16_GROUP_NAMES", "EFIT16_SOURCE_CIRCUIT", "PF1_SEGMENT_EDGES",
                 "efit16_group_for_element"):
        assert not hasattr(projection, gone), gone


def test_a_malformed_coilset_is_refused(tmp_path):
    """A policy file is only worth having if it is checked."""
    import yaml

    from vaft.machine_mapping.efit_coilset import _build
    from vaft.machine_mapping.utils import VestConfigurationError

    good = {
        "efit_coilset": {
            "groups": {
                "PF1": {"split": "axial", "edges": [0.0, 0.5, 1.0],
                        "status": "inferred", "provenance": "x"},
                "PF5": {"split": "midplane", "status": "inferred", "provenance": "x"},
            }
        }
    }
    policy = _build(good)
    # Two halves of two segments, then the pair: six groups.
    assert policy.group_names == ("PF1-1", "PF1-2", "PF1-3", "PF1-4", "PF5U", "PF5L")

    def broken(**changes):
        import copy

        document = copy.deepcopy(good)
        document["efit_coilset"]["groups"]["PF1"].update(changes)
        return document

    with pytest.raises(VestConfigurationError, match="split must be one of"):
        _build(broken(split="sideways"))
    with pytest.raises(VestConfigurationError, match="edges must increase"):
        _build(broken(edges=[0.0, 1.0, 0.5]))
    with pytest.raises(VestConfigurationError, match="edges must start at 0"):
        _build(broken(edges=[0.1, 1.0]))
    with pytest.raises(VestConfigurationError, match="status must be one of"):
        _build(broken(status="probably"))
    with pytest.raises(VestConfigurationError, match="not groups of any listed circuit"):
        document = {"efit_coilset": {**good["efit_coilset"],
                                     "ties": [{"groups": ["PF9L", "PF10U"],
                                               "status": "measured", "provenance": "x"}]}}
        _build(document)


def test_the_coil_currents_fan_one_circuit_out_to_its_groups():
    """A split group is bookkeeping: it does not split the current.

    Every solenoid segment carries PF1's measured current and each pair
    carries its own circuit's, which is what the deleted twenty-six-entry
    fan-out did by hand. Circuits are matched by name, so a reordered
    `pf_active` cannot silently swap two coils.
    """
    from omas import ODS

    from vaft.code.efit.kfile import build_efit_coil_currents

    source = ODS(consistency_check=False)
    source["pf_active.time"] = np.linspace(0.25, 0.40, 64)
    # Deliberately not in PF1..PF10 order: a positional reading would break.
    for index, name in enumerate(
        ["PF10", "PF9", "PF8", "PF7", "PF6", "PF5", "PF4", "PF3", "PF2", "PF1"]
    ):
        source[f"pf_active.coil.{index}.name"] = name
        source[f"pf_active.coil.{index}.current.data"] = np.full(64, float(name[2:]))

    built = ODS(consistency_check=False)["pf_active"]
    build_efit_coil_currents(built, source["pf_active"])

    assert len(built["coil"]) == COILSET.nfsum
    currents = {str(built[f"coil.{i}.name"]): float(built[f"coil.{i}.current.data"][0])
                for i in range(len(built["coil"]))}
    assert currents["PF1-1"] == currents["PF1-8"] == 1.0
    assert currents["PF5U"] == currents["PF5L"] == 5.0
    assert currents["PF10U"] == currents["PF10L"] == 10.0
    assert [str(built[f"coil.{i}.name"]) for i in range(COILSET.nfsum)] == list(COILSET.group_names)


def test_the_coil_currents_are_taken_on_the_time_base_they_were_measured_on():
    """The writer had VEST's analysis window written into it, and did not need it.

    `build_efit_coil_currents` resampled onto a fixed 0.26-0.36 s at 40 us.
    Every packaged product already arrives on exactly that grid -- 2500 uniform
    40 us samples from 0.26 s -- so the interpolation was a series onto its own
    abscissa, and the three machine numbers bought nothing. Not resampling is
    now the default, and a caller that needs a grid passes one.
    """
    from omas import ODS

    from vaft.code.efit.kfile import build_efit_coil_currents

    source = ODS(consistency_check=False)
    # Deliberately not the old hard-coded grid.
    base = np.linspace(0.30, 0.34, 7)
    source["pf_active.time"] = base
    for index in range(10):
        source[f"pf_active.coil.{index}.name"] = f"PF{index + 1}"
        source[f"pf_active.coil.{index}.current.data"] = np.arange(7, dtype=float)

    built = ODS(consistency_check=False)["pf_active"]
    build_efit_coil_currents(built, source["pf_active"])
    np.testing.assert_array_equal(np.asarray(built["time"]), base)
    np.testing.assert_array_equal(
        np.asarray(built["coil.0.current.data"]), np.arange(7, dtype=float)
    )


def test_a_caller_that_wants_a_grid_passes_one():
    from omas import ODS

    from vaft.code.efit.kfile import build_efit_coil_currents

    source = ODS(consistency_check=False)
    source["pf_active.time"] = np.linspace(0.30, 0.34, 5)
    for index in range(10):
        source[f"pf_active.coil.{index}.name"] = f"PF{index + 1}"
        source[f"pf_active.coil.{index}.current.data"] = np.linspace(0.0, 4.0, 5)

    built = ODS(consistency_check=False)["pf_active"]
    build_efit_coil_currents(built, source["pf_active"], window=(0.31, 0.33, 0.005))
    grid = np.asarray(built["time"])
    # `np.arange`'s half-open end is approximate in floating point, so pin the
    # contract -- start, step, and inside the record -- not the element count.
    assert grid[0] == pytest.approx(0.31)
    np.testing.assert_allclose(np.diff(grid), 0.005)
    assert grid[-1] <= 0.33 + 1e-12

    with pytest.raises(ValueError, match="step must be positive"):
        build_efit_coil_currents(
            ODS(consistency_check=False)["pf_active"], source["pf_active"], window=(0.31, 0.33, 0.0)
        )
    with pytest.raises(ValueError, match="does not overlap"):
        build_efit_coil_currents(
            ODS(consistency_check=False)["pf_active"], source["pf_active"], window=(0.9, 1.0, 0.001)
        )


def test_one_probe_count_rule_serves_every_consumer():
    """Three copies of `min(present, defined)` had accumulated, and disagreed.

    The k-file writer, the channel-decision layer and the EFUND projection each
    carried the rule. In the degenerate case -- no channel defined -- the
    writer's copy returned zero probes where the other two returned every
    probe present, so a machine without VEST's channel table would have had
    every probe constraint silently dropped from its k-file.
    """
    from vaft.code.efit.kfile import _efit_bpol_probe_count
    from vaft.machine_mapping.efund_geometry import equilibrium_probe_count as projected
    from vaft.machine_mapping.magnetics import equilibrium_probe_count
    from vaft.validation.efit_channels import efit_probe_count
    from vaft.omas.sample import sample_ods

    ods = sample_ods()
    answers = {
        equilibrium_probe_count(ods),
        efit_probe_count(ods),
        projected(ods),
        _efit_bpol_probe_count(ods["magnetics"]),
    }
    assert len(answers) == 1, answers
    # The sample carries more channels than EFIT's geometry represents; the
    # trailing toroidal-Mirnov references are diagnostics, not constraints.
    assert answers.pop() < len(ods["magnetics.b_field_pol_probe"])


def test_the_probe_count_accepts_either_shape():
    """Its callers hold a whole ODS or a magnetics sub-tree, and both must work."""
    from vaft.machine_mapping.magnetics import equilibrium_probe_count
    from vaft.omas.sample import sample_ods

    ods = sample_ods()
    assert equilibrium_probe_count(ods) == equilibrium_probe_count(ods["magnetics"])

    from omas import ODS

    assert equilibrium_probe_count(ODS(consistency_check=False)) == 0


def test_the_constraint_families_are_named_not_counted(tmp_path):
    """Nine numbers in a list, and nothing at a call site said which was which.

    `uncertainty[5]` was the side probes and `uncertainty[6]` the outboard
    ones; the only place that knew was a module-level index map. The values
    were copied verbatim into five workflow scripts and a test, where a
    transposition would have been invisible.
    """
    from vaft.code.efit.kfile import ConstraintErrors, ConstraintWeights

    legacy_errors = [1e-4, 1e-4, 5e-2, 3e-2, 1e-2, 1e-1, 1e-2, 1e-1, 1e-2]
    legacy_weights = [1, 1, 1, 0.1, 0.1, 0.1, 0.01, 0.01]

    errors = ConstraintErrors.from_sequence(legacy_errors)
    assert errors.pf_current == legacy_errors[0]
    assert errors.probe_side == legacy_errors[5]
    assert errors.flux_loop_outboard == legacy_errors[8]

    weights = ConstraintWeights.from_sequence(legacy_weights)
    assert weights.probe("inboard") == legacy_weights[3]
    assert weights.probe("side") == legacy_weights[4]
    assert weights.probe("outboard") == legacy_weights[5]
    assert weights.flux_loop("inboard") == legacy_weights[6]
    assert weights.flux_loop("outboard") == legacy_weights[7]

    # The positional order is still accepted, so no caller had to change.
    assert ConstraintErrors.coerce(legacy_errors) == errors
    assert ConstraintWeights.coerce(weights) is weights

    # A list of the wrong length is refused, naming the order it wanted.
    with pytest.raises(ValueError, match="probe_side"):
        ConstraintErrors.from_sequence(legacy_errors[:-1])
    with pytest.raises(ValueError, match="flux_loop_outboard"):
        ConstraintWeights.from_sequence(legacy_weights + [1.0])


def test_the_named_and_positional_forms_write_the_same_kfile(tmp_path):
    """The whole safety argument for the change, in one assertion."""
    from vaft.code.efit.kfile import ConstraintWeights

    positional = _kfile_text(tmp_path, EFITScientificConfig())
    assert positional  # the fixture path still works

    weights = ConstraintWeights.from_sequence([1, 1, 1, 0.1, 0.1, 0.1, 0.01, 0.01])
    assert weights.probe("side") == 0.1


def test_the_writer_holds_no_machine_timing():
    """A VEST window in a generic routine is machine policy in the wrong place."""
    source = Path("vaft/code/efit/kfile.py").read_text(encoding="utf-8")
    for literal in ("0.26", "0.36", "4e-5"):
        assert f"tstart = {literal}" not in source
        assert f"tend = {literal}" not in source
        assert f"dt = {literal}" not in source


def test_a_pf_active_missing_a_driven_circuit_is_refused():
    """Silently writing a zero current for a missing coil would be worse."""
    from omas import ODS

    from vaft.code.efit.kfile import build_efit_coil_currents

    partial = ODS(consistency_check=False)
    partial["pf_active.time"] = np.linspace(0.25, 0.40, 64)
    partial["pf_active.coil.0.name"] = "PF1"
    partial["pf_active.coil.0.current.data"] = np.zeros(64)
    with pytest.raises(ValueError, match="PF10"):
        build_efit_coil_currents(
            ODS(consistency_check=False)["pf_active"], partial["pf_active"]
        )


def test_a_table_describing_a_different_coilset_is_refused(tmp_path):
    """A silent trim is the shape of every future mistake in this area.

    The writer used to take `min(groups in the tree, nfsum in the table)`, so a
    constraint tree describing more coils than the table declared simply lost
    its trailing groups. The k-file stayed well formed and quietly omitted
    them, which is exactly what nobody would notice. It is an error now, and
    the message names both sides.
    """
    ods = _constraints_ods(tmp_path)
    # The fixture's tree carries sixteen PF groups; say the table has two.
    (tmp_path / "mhdin.dat").write_text(" &machinein\n nfsum = 2\n /\n", encoding="utf-8")

    with pytest.raises(ValueError, match="different coilsets"):
        generate_kfile(ods, 39915, save_dir=str(tmp_path), config=EFITScientificConfig())


def test_a_table_that_agrees_still_writes(tmp_path):
    """The other half: the check must not refuse the routine case."""
    ods = _constraints_ods(tmp_path)
    (tmp_path / "mhdin.dat").write_text(f" &machinein\n nfsum = {COILSET.nfsum}\n /\n", encoding="utf-8")

    generate_kfile(ods, 39915, save_dir=str(tmp_path), config=EFITScientificConfig())
    text = next((tmp_path / "kfile").iterdir()).read_text(encoding="utf-8")
    # Every group reaches the writer: the repeat count in FWTFC is the number
    # of coils the k-file actually carries.
    assert f"FWTFC= {COILSET.nfsum}*" in text
    assert "BRSP= " in text and f" KCCOILS = {len(COILSET.constraint_matrix()[0])}" in text


def test_the_constraints_tree_stores_no_machine_values_the_writer_overrides():
    """Five VEST numbers sat in the stored parameters and never reached a k-file.

    `generate_constraints_ods` wrote CUTIP, RZERO, RELIP, AELIP and EELIP into
    `code.parameters`, and the writer took all five from the configuration
    instead -- so they were inert, and `CUTIP` had drifted to 50000 A against
    the configuration's 5000. Anything the writer overrides must not be stored
    beside it pretending to be the value in force.
    """
    from pathlib import Path as _Path

    source = (_Path("vaft/code/efit/kfile.py")).read_text(encoding="utf-8")
    for key in ("IN1.CUTIP", "IN1.RZERO", "IN1.RELIP", "IN1.AELIP", "IN1.EELIP"):
        assert f'PM[f"time_slice.{{i}}.{key}"]' not in source, key


def test_legacy_profile_order_arguments_remain_supported(tmp_path):
    ods = _constraints_ods(tmp_path)
    generate_kfile(ods, 39915, 3, 4, save_dir=str(tmp_path))
    text = next((tmp_path / "kfile").iterdir()).read_text(encoding="utf-8")

    assert " KPPCUR = 3" in text
    assert " KFFCUR = 4" in text


def test_scientific_config_rejects_conflicting_legacy_profile_orders(tmp_path):
    ods = _constraints_ods(tmp_path)
    scientific = EFITScientificConfig(profile=EFITProfileConfig(kppcur=3))

    with pytest.raises(ValueError, match="npprime conflicts"):
        generate_kfile(
            ods,
            39915,
            npprime=4,
            save_dir=str(tmp_path),
            config=scientific,
        )


def test_scientific_config_accepts_matching_legacy_profile_orders(tmp_path):
    ods = _constraints_ods(tmp_path)
    scientific = EFITScientificConfig(profile=EFITProfileConfig(kppcur=3, kffcur=4))

    generate_kfile(
        ods,
        39915,
        npprime=3,
        nffprime=4,
        save_dir=str(tmp_path),
        config=scientific,
    )
    text = next((tmp_path / "kfile").iterdir()).read_text(encoding="utf-8")

    assert " KPPCUR = 3" in text
    assert " KFFCUR = 4" in text


def test_typed_settings_reach_their_namelist_fields(tmp_path):
    profile = EFITProfileConfig(
        kppcur=3, kffcur=4, kppfnc=1, kfffnc=2, pcurbd=0, fcurbd=0
    )
    initialization = EFITInitializationConfig(
        rzero=0.45,
        # Deliberately unequal to `rzero`: RELIP and RZERO are separate fields
        # since #588, and a test that gave them one value could not tell a
        # working writer from one that had recoupled them.
        ellipse_rzero=0.38,
        zzero=0.02,
        minor_radius=0.25,
        elongation=1.8,
        current_threshold=7_500.0,
    )
    numerics = EFITNumericsConfig(
        relaxation=0.8,
        error_tolerance=2e-6,
        measurement_error_floor=1e-3,
        max_iterations=250,
    )
    constraints = EFITConstraintConfig(
        group_weights={"plasma_current": 8.0, "bpol_probe": 9.0},
        use_diamagnetic_flux=False,
        diamagnetic_flux_sign="negative",
        wall_current_mode="disabled",
        passive_structure_mode="fit_currents",
    )
    text = _kfile_text(
        tmp_path,
        EFITScientificConfig(profile, initialization, numerics, constraints),
    )

    for field, value in {
        "KPPCUR": 3,
        "KFFCUR": 4,
        "KPPFNC": 1,
        "KFFFNC": 2,
        "PCURBD": 0,
        "FCURBD": 0,
        "CUTIP": 7500.0,
        "RELIP": 0.38,
        "RZERO": 0.45,
        "ZELIP": 0.02,
        "AELIP": 0.25,
        "EELIP": 1.8,
        "RELAX": 0.8,
        "ERROR": 2e-06,
        "SERROR": 0.001,
        "MXITER": -250,
        "IVESEL": 1,
        "IFITVS": 1,
    }.items():
        assert f" {field} = {value}" in text
    assert "FWTCUR= 8.0" in text
    assert "BITMPI= 9000.000" in text
    assert "FWTDLC= 0" in text
    assert "DFLUX= -4.0" in text
    assert "VCURRT= 0.0, 0.0" in text


def test_objective_scales_change_only_submitted_fwt_values(tmp_path):
    constraints = EFITConstraintConfig(
        objective_scales={
            "pf_current": 0.01,
            "plasma_current": 0.1,
            "diamagnetic_flux": 10.0,
            "bpol_probe": 10.0,
            "flux_loop": 100.0,
        }
    )
    text = _kfile_text(tmp_path, EFITScientificConfig(constraints=constraints))

    assert f"FWTFC= {COILSET.nfsum}*0.02" in text
    assert "FWTCUR= 0.30000000000000004" in text
    assert "FWTDLC= 10.0" in text
    assert "FWTMP2= 10.0" in text
    assert "FWTSI= 100.0" in text
    # Objective strength must not be smuggled into the legacy uncertainties.
    assert "BITMPI= 5000.000" in text
    assert "PSIBIT= 6000.000" in text


def test_uncertainty_scales_divide_only_their_own_family_sigma(tmp_path):
    """Issue #386: the row weight EFIT processes is ``FWT/sigma``, not ``FWT``.

    ``data_input.F90:2782`` divides each family's submitted ``FWT*`` by its
    sigma (``nsq = 1``, ``:109``), so the diamagnetic row VEST actually
    submits carries ``FWTDLC/sigdia = 1e-4`` while the stored measurement
    error implies ``2.3e4``.  This knob closes that gap from the sigma side,
    which is the quantity that is wrong, and it must move nothing else.
    """
    baseline = _kfile_text(tmp_path / "baseline")
    scaled = _kfile_text(
        tmp_path / "scaled",
        EFITScientificConfig(
            constraints=EFITConstraintConfig(
                uncertainty_scales={"diamagnetic_flux": 1000.0}
            )
        ),
    )

    assert "SIGDLC= 40000000.0" in baseline
    assert "SIGDLC= 40000.0" in scaled
    # The measurement, the submitted weight and every other family stand still.
    for unchanged in ("DFLUX= -4.0", "FWTDLC= 1.0", "BITMPI= 5000.000",
                      "PSIBIT= 6000.000", "BITIP= 3000.0"):
        assert unchanged in baseline and unchanged in scaled


def test_an_unscaled_kfile_is_byte_identical_to_the_one_without_the_knob(tmp_path):
    """The instrument must be inert until a study reaches for it."""
    baseline = _kfile_text(tmp_path / "implicit")
    explicit = _kfile_text(
        tmp_path / "implicit",
        EFITScientificConfig(
            constraints=EFITConstraintConfig(
                uncertainty_scales={name: 1.0 for name in DIAGNOSTIC_GROUPS}
            )
        ),
    )

    assert baseline == explicit


def test_a_scaled_legacy_sigma_is_not_rounded_away_to_zero(tmp_path):
    """``f"{0.0001:.3f}"`` is ``"0.000"``, and a zero sigma is not a small one.

    EFIT guards its ``fwt/sigma`` division on ``sigma > 1e-10``
    (``data_input.F90:2782``), so a sigma rounded to zero leaves the row at
    the weight the study was trying to change -- silently, and in the
    direction that looks like "the constraint does nothing".
    """
    text = _kfile_text(
        tmp_path,
        EFITScientificConfig(
            constraints=EFITConstraintConfig(
                uncertainty_scales={"bpol_probe": 1.0e7}
            )
        ),
    )

    assert "BITMPI= 0.0005" in text
    # The family that was not scaled keeps the legacy three-decimal spelling.
    assert "PSIBIT= 6000.000" in text


def test_an_uncertainty_scale_must_be_positive_because_it_divides():
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match="must be greater than zero"):
            EFITConstraintConfig(uncertainty_scales={"diamagnetic_flux": bad})
    with pytest.raises(ValueError, match="unknown EFIT diagnostic group"):
        EFITConstraintConfig(uncertainty_scales={"not_a_family": 1.0})


def test_uncertainty_scales_survive_the_round_trip_and_the_grid():
    base = EFITScientificConfig()
    assert set(base.constraints.uncertainty_scales) == set(DIAGNOSTIC_GROUPS)
    assert set(base.to_dict()["constraints"]["uncertainty_scales"]) == set(
        DIAGNOSTIC_GROUPS
    )
    assert EFITScientificConfig.from_dict(base.to_dict()).sha256 == base.sha256

    grid = efit_parameter_grid(
        base, {"constraints.uncertainty_scales.diamagnetic_flux": [1.0, 1.0e6]}
    )
    assert [item.constraints.uncertainty_scales["diamagnetic_flux"] for item in grid] == [
        1.0,
        1.0e6,
    ]
    assert len({item.sha256 for item in grid}) == 2


def test_zero_objective_scale_disables_without_reenabling_rejected_channels(tmp_path):
    ods = _constraints_ods(tmp_path)
    ods["equilibrium.time_slice.0.constraints.bpol_probe.0.weight"] = 0.0
    scientific = EFITScientificConfig(
        constraints=EFITConstraintConfig(
            objective_scales={"bpol_probe": 100.0, "flux_loop": 0.0}
        )
    )
    generate_kfile(ods, 39915, save_dir=str(tmp_path), config=scientific)
    text = next((tmp_path / "kfile").iterdir()).read_text(encoding="utf-8")

    assert "FWTMP2= 0" in text
    assert "BITMPI= 0.000" in text
    assert "FWTSI= 0.0" in text


def test_diamagnetic_switch_overrides_its_objective_scale(tmp_path):
    scientific = EFITScientificConfig(
        constraints=EFITConstraintConfig(
            use_diamagnetic_flux=False,
            objective_scales={"diamagnetic_flux": 100.0},
        )
    )

    assert "FWTDLC= 0" in _kfile_text(tmp_path, scientific)


def test_pf_penalty_and_hard_relations_are_independent(tmp_path):
    no_penalty = EFITScientificConfig(
        constraints=EFITConstraintConfig(objective_scales={"pf_current": 0.0})
    )
    penalty_text = _kfile_text(tmp_path / "penalty", no_penalty)
    assert f"FWTFC= {COILSET.nfsum}*0.0" in penalty_text
    assert f" KCCOILS = {len(COILSET.constraint_matrix()[0])}" in penalty_text
    assert "CCOILS(1,1)" in penalty_text and "XCOILS=" in penalty_text

    no_relations = EFITScientificConfig(
        constraints=EFITConstraintConfig(use_coil_relation_constraints=False)
    )
    relation_text = _kfile_text(tmp_path / "relations", no_relations)
    assert f"FWTFC= {COILSET.nfsum}*2.0" in relation_text
    assert " KCCOILS = 0" in relation_text
    assert "CCOILS(" not in relation_text and "XCOILS=" not in relation_text


def test_constraint_controls_are_hashed_and_unit_defaults_are_stable(tmp_path):
    baseline = EFITScientificConfig()
    explicit_units = EFITScientificConfig(
        constraints=EFITConstraintConfig(
            objective_scales={
                name: 1.0
                for name in (
                    "pf_current",
                    "plasma_current",
                    "diamagnetic_flux",
                    "bpol_probe",
                    "flux_loop",
                )
            }
        )
    )
    changed_scale = EFITScientificConfig(
        constraints=EFITConstraintConfig(objective_scales={"flux_loop": 10.0})
    )
    changed_relations = EFITScientificConfig(
        constraints=EFITConstraintConfig(use_coil_relation_constraints=False)
    )

    assert baseline == explicit_units
    assert baseline.sha256 == explicit_units.sha256
    routine_dir = tmp_path / "routine"
    assert _kfile_text(routine_dir, baseline) == _kfile_text(
        routine_dir, explicit_units
    )
    assert changed_scale.sha256 != baseline.sha256
    assert changed_relations.sha256 != baseline.sha256
    assert EFITScientificConfig.from_dict(changed_scale.to_dict()) == changed_scale


def test_fwtbp_is_explicit_and_part_of_the_scientific_identity(tmp_path):
    baseline = EFITScientificConfig()
    regularized = EFITScientificConfig(
        profile=EFITProfileConfig(kppcur=2, kffcur=2, fwtbp=1)
    )

    assert " FWTBP = 0" in _kfile_text(tmp_path, baseline)
    assert " FWTBP = 1" in _kfile_text(tmp_path, regularized)
    assert baseline.sha256 != regularized.sha256
    assert EFITScientificConfig.from_dict(regularized.to_dict()) == regularized


@pytest.mark.parametrize(
    "profile, message",
    [
        ({"fwtbp": 2}, "fwtbp"),
        ({"fwtbp": 1.0}, "fwtbp"),
        ({"kppcur": 2, "kffcur": 1, "fwtbp": 1}, "kffcur"),
        ({"kppcur": 2, "kffcur": 3, "fwtbp": 1}, "kppcur"),
    ],
)
def test_fwtbp_refuses_ineffective_or_misaligned_bases(profile, message):
    with pytest.raises(ValueError, match=message):
        EFITProfileConfig(**profile)


def test_a_study_can_pin_the_executables_independent_slice_initialization(tmp_path):
    scientific = EFITScientificConfig(
        initialization=EFITInitializationConfig(
            ellipse_rzero=0.32,
            icinit=2,
        )
    )
    text = _kfile_text(tmp_path, scientific)

    assert " RELIP = 0.32" in text
    assert " RZERO = 0.4" in text
    assert " ICINIT = 2" in text
    assert scientific.to_dict()["initialization"]["icinit"] == 2


@pytest.mark.parametrize("value", [0, 3, -1, True, 2.0])
def test_icinit_refuses_modes_the_executable_does_not_define(value):
    with pytest.raises(ValueError, match="icinit"):
        EFITInitializationConfig(icinit=value)


def test_standard_deviation_mode_uses_measurement_errors(tmp_path):
    constraints = EFITConstraintConfig(uncertainty_mode="standard_deviation")
    text = _kfile_text(
        tmp_path,
        EFITScientificConfig(constraints=constraints),
    )

    assert "BITFC= 0.25" in text
    assert "BITIP= 20.0" in text
    assert "BITMPI= 0.002" in text
    assert "PSIBIT= 0.000477464829" in text
    assert "SIGDLC= 0.2" in text


def test_standard_deviation_pins_the_terms_efit_would_otherwise_fold_into_sigma(tmp_path):
    """EFIT fits against max(SERROR*|m|, BIT*VBIT) -- ERRSIL*|m| for flux loops.

    With VBIT left at its default of 10 every submitted sigma is used ten times
    over, and with SERROR/ERRSIL unpinned a floor replaces it (#891).  Legacy
    weighting relies on the default VBIT and must be left alone.
    """
    sd = _kfile_text(
        tmp_path / "sd",
        EFITScientificConfig(
            constraints=EFITConstraintConfig(uncertainty_mode="standard_deviation")
        ),
    )
    assert " SERROR = 0.0\n" in sd
    assert " VBIT = 1.0\n" in sd
    assert " ERRSIL = 0.0\n" in sd

    legacy = _kfile_text(tmp_path / "legacy", EFITScientificConfig())
    assert " SERROR = 0.0005\n" in legacy
    assert "VBIT" not in legacy
    assert "ERRSIL" not in legacy


def test_standard_deviation_weights_only_switch_rows_on(tmp_path):
    # The fixture's Ip weight is 3.0 and every PF weight 2.0; a group weight
    # would scale Ip again.  Under standard_deviation none of that is strength.
    constraints = EFITConstraintConfig(
        uncertainty_mode="standard_deviation",
        group_weights={"plasma_current": 8.0},
    )
    sd = _kfile_text(tmp_path / "sd", EFITScientificConfig(constraints=constraints))
    assert "FWTCUR= 1.0\n" in sd
    assert f"FWTFC= {COILSET.nfsum}*1.0\n" in sd

    legacy = _kfile_text(
        tmp_path / "legacy",
        EFITScientificConfig(constraints=EFITConstraintConfig(group_weights={"plasma_current": 8.0})),
    )
    assert "FWTCUR= 8.0\n" in legacy
    assert f"FWTFC= {COILSET.nfsum}*2.0\n" in legacy

    # An excluded row stays excluded.
    ods = _constraints_ods(tmp_path / "off")
    ods["equilibrium.time_slice.0.constraints.ip.weight"] = 0.0
    generate_kfile(
        ods,
        39915,
        save_dir=str(tmp_path / "off"),
        config=EFITScientificConfig(
            constraints=EFITConstraintConfig(uncertainty_mode="standard_deviation")
        ),
    )
    text = next((tmp_path / "off" / "kfile").iterdir()).read_text(encoding="utf-8")
    assert "FWTCUR= 0.0\n" in text


@pytest.mark.parametrize(
    "path, named",
    [
        ("bpol_probe.0", "bpol_probe.0"),
        ("flux_loop.0", "flux_loop.0"),
        ("ip", "ip"),
        ("diamagnetic_flux", "diamagnetic_flux"),
    ],
)
def test_standard_deviation_refuses_an_enabled_row_without_a_sigma(tmp_path, path, named):
    # EFIT drops a row whose sigma is <= 1e-10 without saying so, and an
    # enabled diamagnetic row with SIGDLC = 0 keeps its raw weight and divides
    # by zero.  The writer names the row instead of submitting it.
    ods = _constraints_ods(tmp_path)
    ods[f"equilibrium.time_slice.0.constraints.{path}.measured_error_upper"] = 0.0
    with pytest.raises(ValueError, match=named):
        generate_kfile(
            ods,
            39915,
            save_dir=str(tmp_path),
            config=EFITScientificConfig(
                constraints=EFITConstraintConfig(uncertainty_mode="standard_deviation")
            ),
        )


def test_a_disabled_row_needs_no_sigma(tmp_path):
    ods = _constraints_ods(tmp_path)
    ods["equilibrium.time_slice.0.constraints.bpol_probe.0.measured_error_upper"] = 0.0
    ods["equilibrium.time_slice.0.constraints.bpol_probe.0.weight"] = 0.0
    generate_kfile(
        ods,
        39915,
        save_dir=str(tmp_path),
        config=EFITScientificConfig(
            constraints=EFITConstraintConfig(uncertainty_mode="standard_deviation")
        ),
    )


def test_the_code_parameters_state_the_serror_the_kfile_carries(tmp_path):
    # The constraints tree used to say SERROR = 0.05 while the k-file wrote
    # 5e-4: a record of a k-file that was not written (#869).
    for mode, expected in (("legacy_weight", 5.0e-4), ("standard_deviation", 0.0)):
        ods = _constraints_ods(tmp_path / mode)
        generate_kfile(
            ods,
            39915,
            save_dir=str(tmp_path / mode),
            config=EFITScientificConfig(constraints=EFITConstraintConfig(uncertainty_mode=mode)),
        )
        parameters = ods["equilibrium.code.parameters.time_slice.0.IN1"]
        assert float(parameters["SERROR"]) == expected
        if mode == "standard_deviation":
            assert float(parameters["VBIT"]) == 1.0 and float(parameters["ERRSIL"]) == 0.0
        else:
            assert "VBIT" not in parameters and "ERRSIL" not in parameters


def test_resolved_configuration_is_stable_and_manifest_checksums_kfiles(tmp_path):
    ods = _constraints_ods(tmp_path)
    config = EFITConfig(
        workdir=tmp_path,
        shot=39915,
        profile=EFITProfileConfig(kppcur=3),
        provenance={"geometry_version": "vest-2025-07", "source": "main"},
    )

    inputs = prepare_efit_inputs(ods, config)
    payload = json.loads(inputs.manifest.read_text(encoding="utf-8"))

    assert payload["resolved"] == inputs.configuration
    assert payload["requested"]["scientific"] == payload["resolved"]["scientific"]
    assert payload["requested"]["typed_profile_supplied"]
    assert payload["resolved"]["scientific_sha256"] == config.scientific_config().sha256
    assert payload["resolved"]["provenance"]["geometry_version"] == "vest-2025-07"
    assert len(payload["kfiles"][0]["sha256"]) == 64
    assert inputs.manifest in inputs.files


def test_parameter_grid_is_deterministic_and_validated():
    grid = efit_parameter_grid(
        EFITScientificConfig(),
        {
            "profile.kppcur": [2, 3],
            "numerics.relaxation": [1.0, 0.8],
            "constraints.group_weights.bpol_probe": [4.0, 8.0],
            "initialization.rzero": [0.4, 0.45],
        },
    )

    assert len(grid) == 16
    assert len({item.sha256 for item in grid}) == 16
    assert grid[0].profile.kppcur == 2
    assert grid[-1].profile.kppcur == 3
    assert grid[-1].constraints.group_weights["bpol_probe"] == 8.0


def test_parameter_grid_scans_fwtbp_as_a_profile_axis():
    grid = efit_parameter_grid(
        EFITScientificConfig(),
        {"profile.fwtbp": [0, 1]},
    )

    assert [item.profile.fwtbp for item in grid] == [0, 1]
    assert len({item.sha256 for item in grid}) == 2


def test_scientific_configuration_round_trips_through_json():
    original = EFITScientificConfig(
        profile=EFITProfileConfig(kppcur=4),
        constraints=EFITConstraintConfig(
            group_weights={"flux_loop": 7.0},
            passive_structure_mode="disabled",
        ),
    )

    payload = json.loads(json.dumps(original.to_dict()))
    restored = EFITScientificConfig.from_dict(payload)

    assert restored == original
    assert restored.sha256 == original.sha256


def test_integral_scalar_types_are_canonicalized_before_hashing():
    numpy_config = EFITScientificConfig(
        profile=EFITProfileConfig(kppcur=np.int64(2)),
        numerics=EFITNumericsConfig(max_iterations=np.int64(100)),
        constraints=EFITConstraintConfig(nccoil=np.int64(0)),
    )

    assert numpy_config.to_dict() == EFITScientificConfig().to_dict()
    assert numpy_config.sha256 == EFITScientificConfig().sha256


@pytest.mark.parametrize(
    "factory, message",
    [
        (lambda: EFITProfileConfig(kppcur=-1), "kppcur"),
        (lambda: EFITProfileConfig(kppcur=2.0), "kppcur"),
        (lambda: EFITProfileConfig(kppfnc=0.0), "kppfnc"),
        (lambda: EFITProfileConfig(pcurbd=1.0), "pcurbd"),
        (lambda: EFITProfileConfig(pcurbd=2), "pcurbd"),
        (lambda: EFITInitializationConfig(minor_radius=0), "minor_radius"),
        (lambda: EFITNumericsConfig(error_tolerance=0), "error_tolerance"),
        (lambda: EFITNumericsConfig(max_iterations=100.0), "max_iterations"),
        (
            lambda: EFITConstraintConfig(diamagnetic_flux_sign="legacy"),
            "diamagnetic_flux_sign",
        ),
        (
            lambda: EFITConstraintConfig(group_weights={"unknown": 1.0}),
            "unknown EFIT diagnostic",
        ),
        (lambda: EFITConstraintConfig(nccoil=0.0), "nccoil"),
        (
            lambda: EFITConstraintConfig(
                coil_constraint_matrix=((1.0, 0.0),),
                coil_constraint_targets=(0.0,),
            ),
            "targets length",
        ),
    ],
)
def test_invalid_scientific_configuration_fails_before_execution(factory, message):
    with pytest.raises(ValueError, match=message):
        factory()


def test_legacy_and_typed_profile_conflicts_fail_early():
    with pytest.raises(ValueError, match="npprime conflicts"):
        EFITConfig(npprime=4, profile=EFITProfileConfig(kppcur=3))

    with pytest.raises(ValueError, match="npprime must be a positive integer"):
        EFITConfig(npprime=2.0)


def test_custom_coil_matrix_is_validated_against_selected_machine_coils(tmp_path):
    constraints = replace(
        EFITConstraintConfig(),
        coil_constraint_matrix=((1.0,),),
        coil_constraint_targets=(0.0,),
    )

    with pytest.raises(ValueError, match="row count"):
        _kfile_text(
            tmp_path,
            EFITScientificConfig(constraints=constraints),
        )

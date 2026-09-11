"""input.gacode read/write, and the ODS projection that feeds it (issue #550).

The fixtures under `test/data/gacode/neo_reg18/` are a real GACODE artifact:
`input.gacode` is the file shipped with NEO's reg18 regression case, and the
`out.neo.*` files are a run of it that reproduced the shipped `out.neo.prec`
exactly. So "VAFT reads and rewrites this file without changing it" is a
statement about the real format, not about a fixture VAFT invented.

None of this needs GACODE installed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vaft.code.gacode._input_gacode import (
    HEADER_KEYS,
    PROFILE_TAGS,
    read_input_gacode,
    write_input_gacode,
)
from vaft.code.gacode._profiles import GACODEProfile
from vaft.code.gacode.inputs import (
    ProfileConversionError,
    prepare_gacode_inputs,
    prepare_gacode_profile,
)

REG18 = Path(__file__).parent / "data" / "gacode" / "neo_reg18" / "input.gacode"

#: `expro` relabelled these two between the release that wrote reg18's file and
#: the current source. The reader keys on the tag and ignores the unit; the
#: writer emits the current one, so a round trip changes exactly these lines.
RELABELLED_TAGS = ("qpar_beam", "qpar_wall", "qmom")

SAMPLE = None
try:  # pragma: no cover - depends on whether the repository sample is present
    from vaft.data.resources import data_path

    _candidate = Path(data_path("kineticEfit/ods_48224_300ms.json"))
    SAMPLE = _candidate if _candidate.exists() else None
except Exception:
    SAMPLE = None

requires_sample = pytest.mark.skipif(
    SAMPLE is None, reason="the packaged 48224 kinetic sample is a repository-only asset"
)


@pytest.fixture(scope="module")
def reg18_profile() -> GACODEProfile:
    return read_input_gacode(REG18)


@pytest.fixture(scope="module")
def ods_48224():
    from omas import load_omas_json

    # consistency_check=False: the committed sample carries a handful of leaves
    # that the installed IMAS version no longer recognises (`profiles_1d.centroid`
    # among them), which is a property of the sample, not of this conversion.
    return load_omas_json(str(SAMPLE), consistency_check=False)


# --------------------------------------------------------------------------
# Reading a real input.gacode
# --------------------------------------------------------------------------


def test_reads_the_shipped_reg18_file(reg18_profile):
    assert reg18_profile.n_exp == 51
    assert reg18_profile.n_ion == 2
    assert reg18_profile.shot == 141459
    assert reg18_profile.time == 3890
    assert reg18_profile.name == ("D", "C")
    assert reg18_profile.type == ("[therm]", "[therm]")
    np.testing.assert_allclose(reg18_profile.z, [1.0, 6.0])
    np.testing.assert_allclose(reg18_profile.mass, [2.0, 12.0])


def test_per_ion_sections_are_shaped_species_by_radius(reg18_profile):
    assert reg18_profile.ni.shape == (2, 51)
    assert reg18_profile.ti.shape == (2, 51)
    # The carbon density is an order of magnitude below the deuterium one, so a
    # transposed read would be obvious here.
    assert reg18_profile.ni[0, 0] > 10.0 * reg18_profile.ni[1, 0]


def test_the_radial_coordinate_runs_from_zero_to_one(reg18_profile):
    assert reg18_profile.rho[0] == pytest.approx(0.0)
    assert reg18_profile.rho[-1] == pytest.approx(1.0, abs=1e-6)


def test_scalars_and_header_are_read(reg18_profile):
    assert reg18_profile.torfluxa == pytest.approx(5.6625370e-01)
    assert reg18_profile.rcentr == pytest.approx(1.6955000e00)
    assert reg18_profile.bcentr == pytest.approx(1.8316507e00)
    assert reg18_profile.current == pytest.approx(-1.2579084e00)
    assert reg18_profile.header["statefile"] == "iterdb141459.03890"


def test_shape_harmonics_and_sources_are_kept_apart(reg18_profile):
    assert set(reg18_profile.shape) == {
        "shape_cos0", "shape_cos1", "shape_cos2", "shape_cos3", "shape_sin3",
    }
    assert "qohme" in reg18_profile.sources
    assert reg18_profile.extra == {}


def test_reg18_has_everything_neo_needs(reg18_profile):
    assert reg18_profile.check_neo_requirements() == ()


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------


def _differing_sections(left: Path, right: Path) -> set[str]:
    """Tags whose section text differs between two input.gacode files."""

    def sections(path: Path) -> dict[str, list[str]]:
        found: dict[str, list[str]] = {}
        current = None
        for line in path.read_text().splitlines():
            if line.startswith("#") and ":" not in line and line.strip() != "#":
                current = line[1:].split("|")[0].strip()
                found[current] = [line]
            elif current is not None:
                found[current].append(line)
        return found

    first, second = sections(left), sections(right)
    return {
        tag
        for tag in set(first) | set(second)
        if first.get(tag) != second.get(tag)
    }


def test_the_round_trip_changes_only_two_stale_unit_labels(reg18_profile, tmp_path):
    """VAFT rewrites GACODE's own file essentially byte for byte.

    The two tags that do differ differ only in the unit string in their header,
    which `expro` itself renamed; the numbers are untouched.
    """
    target = write_input_gacode(reg18_profile, tmp_path / "input.gacode")
    original = REG18.read_text().splitlines()
    rewritten = target.read_text().splitlines()
    assert len(original) == len(rewritten)

    differing = [
        (a, b) for a, b in zip(original, rewritten) if a != b
    ]
    assert len(differing) == len(RELABELLED_TAGS)
    for before, after in differing:
        assert any(tag in before and tag in after for tag in RELABELLED_TAGS)
        assert before.split("|")[0] == after.split("|")[0]


def test_writing_is_idempotent(reg18_profile, tmp_path):
    first = write_input_gacode(reg18_profile, tmp_path / "a.gacode")
    second = write_input_gacode(read_input_gacode(first), tmp_path / "b.gacode")
    assert first.read_text() == second.read_text()


def test_every_numeric_field_survives_the_round_trip(reg18_profile, tmp_path):
    target = write_input_gacode(reg18_profile, tmp_path / "input.gacode")
    reloaded = read_input_gacode(target)
    def lookup(profile, tag):
        # Explicit, not an `or` chain: these are arrays, and truthiness on an
        # array is ambiguous.
        if tag in profile.shape:
            return profile.shape[tag]
        if tag in profile.sources:
            return profile.sources[tag]
        return getattr(profile, tag, None)

    for tag, _unit, _per_ion in PROFILE_TAGS:
        before = lookup(reg18_profile, tag)
        after = lookup(reloaded, tag)
        if before is None:
            assert after is None, f"{tag} appeared from nowhere"
            continue
        np.testing.assert_allclose(np.asarray(after), np.asarray(before), err_msg=tag)


def test_an_identically_zero_profile_is_omitted_not_written_as_zeros(tmp_path):
    """`expro_writev` skips a zero vector, so writing zeros would be a lie.

    A tag's absence means "not set, or identically zero"; writing an explicit
    zero array would claim a measurement that was never made.
    """
    profile = GACODEProfile(
        rho=np.linspace(0.0, 1.0, 5),
        z=np.array([1.0]),
        mass=np.array([2.0]),
        ne=np.ones(5),
        te=np.ones(5),
        jbs=np.zeros(5),
    )
    text = write_input_gacode(profile, tmp_path / "input.gacode").read_text()
    assert "# ne | 10^19/m^3" in text
    assert "jbs" not in text


def test_the_header_keeps_expros_fixed_six_line_order(tmp_path):
    profile = GACODEProfile(
        rho=np.linspace(0.0, 1.0, 5), z=np.array([1.0]), mass=np.array([2.0]),
        ne=np.ones(5), te=np.ones(5),
    )
    lines = write_input_gacode(profile, tmp_path / "input.gacode").read_text().splitlines()
    for index, key in enumerate(HEADER_KEYS):
        assert f"*{key}" in lines[index]
    assert lines[len(HEADER_KEYS)].strip() == "#"


# --------------------------------------------------------------------------
# Malformed input
# --------------------------------------------------------------------------


def test_a_file_without_rho_is_refused(tmp_path):
    path = tmp_path / "input.gacode"
    path.write_text("#\n# nexp\n3\n# z\n 1.0000000E+00\n")
    with pytest.raises(ValueError, match="no 'rho' section"):
        read_input_gacode(path)


def test_a_declared_ion_count_that_disagrees_with_z_is_refused(tmp_path):
    original = REG18.read_text().replace("# nion\n2\n", "# nion\n3\n", 1)
    path = tmp_path / "input.gacode"
    path.write_text(original)
    with pytest.raises(ValueError, match="nion is 3 but 'z' lists 2"):
        read_input_gacode(path)


def test_a_declared_point_count_that_disagrees_with_rho_is_refused(tmp_path):
    original = REG18.read_text().replace("# nexp\n51\n", "# nexp\n50\n", 1)
    path = tmp_path / "input.gacode"
    path.write_text(original)
    with pytest.raises(ValueError, match="nexp is 50 but 'rho' has 51"):
        read_input_gacode(path)


def test_ragged_rows_are_refused(tmp_path):
    path = tmp_path / "input.gacode"
    path.write_text(
        "#\n# z\n 1.0000000E+00\n# rho | -\n"
        "  1  0.0000000E+00\n  2  5.0000000E-01  1.0000000E+00\n"
    )
    with pytest.raises(ValueError, match="differing width"):
        read_input_gacode(path)


def test_a_per_ion_section_whose_width_disagrees_with_the_species_count(tmp_path):
    """Two ion species declared, one column of ion temperature written."""
    path = tmp_path / "input.gacode"
    path.write_text(
        "#\n"
        "# z\n 1.0000000E+00 6.0000000E+00\n"
        "# rho | -\n  1  0.0000000E+00\n  2  1.0000000E+00\n"
        "# ti | keV\n  1  1.0000000E+00\n  2  5.0000000E-01\n"
    )
    with pytest.raises(ValueError, match="'ti' has 1 columns but there are 2"):
        read_input_gacode(path)


def test_a_profile_needs_at_least_two_radial_points():
    with pytest.raises(ValueError, match="at least two points"):
        GACODEProfile(rho=np.array([0.0]), z=np.array([1.0]))


def test_mass_and_charge_must_describe_the_same_species():
    with pytest.raises(ValueError, match="mass has 1 entries but z has 2"):
        GACODEProfile(
            rho=np.linspace(0, 1, 4), z=np.array([1.0, 6.0]), mass=np.array([2.0])
        )


# --------------------------------------------------------------------------
# The ODS projection
# --------------------------------------------------------------------------


@requires_sample
def test_the_packaged_48224_state_is_refused_without_an_explicit_truncation(ods_48224):
    """Its fitted profiles reach exactly zero at the boundary.

    GACODE takes logarithmic gradients, so a zero is not usable. Clipping it
    quietly would fabricate an edge; the conversion stops and names the grid
    point instead, leaving the decision to the caller.
    """
    with pytest.raises(ProfileConversionError, match="electron density is not positive"):
        prepare_gacode_profile(ods_48224)


@requires_sample
def test_the_48224_state_converts_once_the_caller_truncates(ods_48224):
    profile = prepare_gacode_profile(ods_48224, rho_max=0.95, z_eff=2.0)
    assert profile.n_ion == 1
    assert profile.name == ("H+",)
    assert profile.check_neo_requirements() == ()
    assert profile.n_exp < 129, "the truncation must actually drop points"
    assert np.all(profile.ne > 0.0) and np.all(profile.te > 0.0)
    assert profile.shot == 48224
    assert profile.time == 300


@requires_sample
def test_truncation_does_not_rescale_the_radial_coordinate(ods_48224):
    """`torfluxa` stays the plasma-boundary flux when the grid is cut short.

    `rho` is normalised to the boundary, so taking `phi` after truncation would
    silently rescale every radius in the file.
    """
    full = prepare_gacode_profile(ods_48224, rho_max=0.999, z_eff=2.0)
    cut = prepare_gacode_profile(ods_48224, rho_max=0.80, z_eff=2.0)
    assert cut.torfluxa == pytest.approx(full.torfluxa, rel=1e-12)
    assert cut.rho[-1] < 0.81
    assert cut.n_exp < full.n_exp


@requires_sample
def test_the_conversion_records_all_three_times(ods_48224):
    """Requested, equilibrium and core_profiles times are recorded separately."""
    profile = prepare_gacode_profile(ods_48224, rho_max=0.95, z_eff=2.0)
    times = profile.provenance["time"]
    assert times["requested_time"] == pytest.approx(0.3)
    assert times["equilibrium_time"] == pytest.approx(0.3)
    assert times["core_profiles_time"] == pytest.approx(0.3)
    assert times["tolerance"] > 0.0


@requires_sample
def test_a_time_beyond_the_tolerance_is_refused(ods_48224):
    with pytest.raises(ProfileConversionError, match="beyond the .* tolerance"):
        prepare_gacode_profile(ods_48224, time=0.9, rho_max=0.95)


@requires_sample
def test_provenance_distinguishes_measured_derived_assumed_and_absent(ods_48224):
    profile = prepare_gacode_profile(ods_48224, rho_max=0.95, z_eff=2.0)
    kinds = {name: record["kind"] for name, record in profile.provenance.items()}
    assert kinds["ne"] == "measured"
    assert kinds["te"] == "measured"
    assert kinds["rmin"] == "derived"
    assert kinds["z_eff"] == "caller_supplied"
    assert kinds["zeta"] == "unavailable"
    assert "zeta" in profile.missing()


@requires_sample
def test_a_single_ion_species_with_no_zeff_is_reported_not_invented(ods_48224):
    """Zeff is a real gap in this state, and the conversion says so."""
    profile = prepare_gacode_profile(ods_48224, rho_max=0.95)
    assert profile.z_eff is None
    assert "z_eff" in profile.missing()


@requires_sample
def test_prepare_writes_a_readable_file_in_the_callers_directory(ods_48224, tmp_path):
    staged = prepare_gacode_inputs(ods_48224, tmp_path / "case", rho_max=0.95, z_eff=2.0)
    assert staged.input_gacode == tmp_path / "case" / "input.gacode"
    assert staged.input_gacode.is_file()
    reloaded = read_input_gacode(staged.input_gacode)
    np.testing.assert_allclose(reloaded.rho, staged.profile.rho, rtol=1e-6)
    np.testing.assert_allclose(reloaded.ne, staged.profile.ne, rtol=1e-6)


def test_an_ods_without_core_profiles_is_refused():
    from omas import ODS

    ods = ODS()
    ods["equilibrium.time"] = np.array([0.3])
    with pytest.raises(ProfileConversionError, match="no core_profiles.time"):
        prepare_gacode_profile(ods)


def test_an_ods_without_an_equilibrium_is_refused():
    from omas import ODS

    with pytest.raises(ProfileConversionError, match="no equilibrium.time"):
        prepare_gacode_profile(ODS())


@requires_sample
def test_ion_rotation_is_carried_when_every_species_has_it(ods_48224):
    """48224 carries a toroidal velocity, so it reaches the file rather than being read and dropped."""
    profile = prepare_gacode_profile(ods_48224, rho_max=0.95, z_eff=2.0)
    assert profile.vtor is not None
    assert profile.vtor.shape == (profile.n_ion, profile.n_exp)
    assert profile.provenance["vtor"]["kind"] == "measured"


def test_rotation_is_left_absent_when_a_species_lacks_it():
    """A per-ion array with one species zeroed would claim a stationary impurity."""
    from omas import ODS

    rho = np.linspace(0.0, 1.0, 9)
    ods = ODS(consistency_check=False)
    ods["equilibrium.time"] = np.array([0.3])
    ods["core_profiles.time"] = np.array([0.3])
    eq = "equilibrium.time_slice.0.profiles_1d"
    ods[f"{eq}.rho_tor_norm"] = rho
    ods[f"{eq}.phi"] = rho**2
    ods[f"{eq}.psi"] = np.linspace(0.0, 0.05, rho.size)
    ods[f"{eq}.q"] = np.linspace(1.0, 3.0, rho.size)
    ods[f"{eq}.r_inboard"] = np.linspace(0.4, 0.1, rho.size)
    ods[f"{eq}.r_outboard"] = np.linspace(0.4, 0.7, rho.size)
    ods["equilibrium.time_slice.0.global_quantities.ip"] = 1.0e5
    cp = "core_profiles.profiles_1d.0"
    ods[f"{cp}.grid.rho_tor_norm"] = rho
    ods[f"{cp}.electrons.density_thermal"] = np.linspace(1e19, 1e18, rho.size)
    ods[f"{cp}.electrons.temperature"] = np.linspace(100.0, 10.0, rho.size)
    for index, charge in enumerate((1.0, 6.0)):
        ods[f"{cp}.ion.{index}.label"] = "H+" if index == 0 else "C6+"
        ods[f"{cp}.ion.{index}.z_ion"] = charge
        ods[f"{cp}.ion.{index}.density_thermal"] = np.linspace(1e19, 1e18, rho.size)
        ods[f"{cp}.ion.{index}.temperature"] = np.linspace(80.0, 8.0, rho.size)
    # Only the main ion is measured.
    ods[f"{cp}.ion.0.velocity.toroidal"] = np.linspace(1e4, 0.0, rho.size)

    profile = prepare_gacode_profile(ods)
    assert profile.vtor is None
    assert profile.provenance["vtor"]["kind"] == "unavailable"
    # Two ion species and no zeff profile: Zeff is derivable and is derived.
    assert profile.z_eff is not None
    assert profile.provenance["z_eff"]["kind"] == "derived"
    assert profile.z_eff[0] == pytest.approx(
        (1.0 * 1.0**2 + 1.0 * 6.0**2) / 1.0, rel=1e-9
    )


# --------------------------------------------------------------------------
# Review findings: signs, the toroidal flux, and the field's time
# --------------------------------------------------------------------------


def _directions(profile: GACODEProfile) -> tuple[int, int]:
    """(btccw, ipccw) exactly as expro derives them (expro_locsim.f90:202-203)."""
    signb = int(np.sign(profile.torfluxa))
    signq = int(np.sign(profile.q[0]))
    return -signb, -signq * signb


def test_the_gacode_convention_is_registered_and_marked_as_inferred():
    from vaft.data.cocos import convention_for

    convention = convention_for("gacode")
    assert convention.cocos == 2
    assert convention.confirmed is False


def test_reg18_is_self_consistent_with_that_convention(reg18_profile):
    """DIII-D in the normal orientation: Bt clockwise, Ip counter-clockwise."""
    assert _directions(reg18_profile) == (-1, +1)


@requires_sample
def test_the_converted_file_gives_neo_the_imas_field_directions(ods_48224):
    """48224 has b0 > 0 and ip > 0: both counter-clockwise under COCOS 11.

    Written without the COCOS 11 -> 2 transform, expro would read both as
    clockwise -- the device mirrored. <j.B> survives that (the helicity is
    preserved), which is why the scalar cross-checks could not catch it.
    """
    profile = prepare_gacode_profile(ods_48224, rho_max=0.95, z_eff=2.0)
    b0 = float(np.ravel(ods_48224["equilibrium.vacuum_toroidal_field.b0"])[0])
    ip = float(ods_48224["equilibrium.time_slice.0.global_quantities.ip"])
    assert _directions(profile) == (int(np.sign(b0)), int(np.sign(ip)))
    assert profile.provenance["cocos"]["to"] == 2


@requires_sample
def test_toroidal_components_change_sign_and_q_does_not(ods_48224):
    profile = prepare_gacode_profile(ods_48224, rho_max=0.95, z_eff=2.0)
    eq = "equilibrium.time_slice.0.profiles_1d"
    assert np.sign(profile.bcentr) == -np.sign(
        float(np.ravel(ods_48224["equilibrium.vacuum_toroidal_field.b0"])[0])
    )
    assert np.sign(profile.current) == -np.sign(
        float(ods_48224["equilibrium.time_slice.0.global_quantities.ip"])
    )
    np.testing.assert_allclose(
        profile.fpol, -np.asarray(ods_48224[f"{eq}.f"])[: profile.n_exp]
    )
    np.testing.assert_allclose(profile.q, np.asarray(ods_48224[f"{eq}.q"])[: profile.n_exp])


@requires_sample
def test_torfluxa_is_phi_over_two_pi_whatever_the_psi_convention(ods_48224, monkeypatch):
    """phi is written in weber by every VAFT producer, however psi is stored.

    Forcing the psi-storage probe to report Wb/rad must not change torfluxa;
    it used to, by a factor of 2*pi.
    """
    import vaft.data.eqdsk as eqdsk

    expected = -float(ods_48224["equilibrium.time_slice.0.profiles_1d.phi"][-1]) / (2 * np.pi)
    stored_in_weber = prepare_gacode_profile(ods_48224, rho_max=0.95, z_eff=2.0)
    monkeypatch.setattr(eqdsk, "ods_psi_to_wb_per_radian_factor", lambda *a, **k: 1.0)
    stored_per_radian = prepare_gacode_profile(ods_48224, rho_max=0.95, z_eff=2.0)
    assert stored_in_weber.torfluxa == pytest.approx(expected, rel=1e-12)
    assert stored_per_radian.torfluxa == pytest.approx(expected, rel=1e-12)


def test_bcentr_is_read_at_the_converted_slice():
    """b0 lives on the equilibrium time base; index 0 is the wrong instant."""
    from omas import ODS

    rho = np.linspace(0.0, 1.0, 9)
    ods = ODS(consistency_check=False)
    ods["equilibrium.time"] = np.array([0.2, 0.3])
    ods["core_profiles.time"] = np.array([0.2, 0.3])
    ods["equilibrium.vacuum_toroidal_field.r0"] = 0.4
    ods["equilibrium.vacuum_toroidal_field.b0"] = np.array([0.10, 0.25])
    for index in (0, 1):
        eq = f"equilibrium.time_slice.{index}.profiles_1d"
        ods[f"{eq}.rho_tor_norm"] = rho
        ods[f"{eq}.phi"] = rho**2
        ods[f"{eq}.psi"] = np.linspace(0.0, 0.05, rho.size)
        ods[f"{eq}.q"] = np.linspace(1.0, 3.0, rho.size)
        ods[f"equilibrium.time_slice.{index}.global_quantities.ip"] = 1.0e5
        cp = f"core_profiles.profiles_1d.{index}"
        ods[f"{cp}.grid.rho_tor_norm"] = rho
        ods[f"{cp}.electrons.density_thermal"] = np.linspace(1e19, 1e18, rho.size)
        ods[f"{cp}.electrons.temperature"] = np.linspace(100.0, 10.0, rho.size)
        ods[f"{cp}.ion.0.label"] = "H+"
        ods[f"{cp}.ion.0.z_ion"] = 1.0
        ods[f"{cp}.ion.0.density_thermal"] = np.linspace(1e19, 1e18, rho.size)
        ods[f"{cp}.ion.0.temperature"] = np.linspace(80.0, 8.0, rho.size)

    profile = prepare_gacode_profile(ods, time_index=1)
    assert abs(profile.bcentr) == pytest.approx(0.25)

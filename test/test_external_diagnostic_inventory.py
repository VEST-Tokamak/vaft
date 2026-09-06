"""Classification tests for the external soft X-ray / camera consolidation.

The consolidation decides, from names and header text alone, which of ~2000
directories a machine mapping can read, which are derived output, and which are
fluctuation-grade and therefore reserved for issue #161. Those decisions move
250 GB and are hard to inspect afterwards, so they are tested here against the
shapes that actually occur in the export tree rather than against the happy
path only.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

from vaft.machine_mapping.soft_x_rays import _daq_label_from_filename


def _load(name: str):
    script = (
        Path(__file__).resolve().parents[1]
        / "workflow"
        / "automatic_pipeline_2_corrective_data_update"
        / f"{name}.py"
    )
    spec = importlib.util.spec_from_file_location(name, script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # Registered before execution because the module defines dataclasses, whose
    # decorator resolves annotations through ``sys.modules``.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


INVENTORY = _load("inventory_external_diagnostics")
CONSOLIDATE = _load("consolidate_external_diagnostics")


# A GX-8 header, trimmed to the fields the consolidation reads. Real headers
# are 150 CRLF-terminated lines; the carriage returns are kept because the
# files on disk have them and a parser that trips over them is useless.
HEADER_50KHZ = (
    "Frame_Rate: 50000\r\n"
    "Frame_Size: 208x208\r\n"
    "Type: GX-8\r\n"
    "TopFrame: 15000\r\n"
    "BottomFrame: 17000\r\n"
    "Frames: 2001\r\n"
    "Rec_Time: 20/07/06 14:29:05.190273\r\n"
    "ShutterSpeed: OPEN(19.1us)\r\n"
)
HEADER_25KHZ = (
    "Frame_Rate: 2500\r\n"
    "Frame_Size: 1024x1280\r\n"
    "Frames: 151\r\n"
    "TopFrame: 750\r\n"
    "BottomFrame: 900\r\n"
)


def _indexed(stem: str, count: int) -> list[str]:
    return [f"{stem}_{index:08d}.bmp" for index in range(count)] + [f"{stem}_bmp.txt"]


class TestHeaderParsing:
    def test_reads_the_fields_consolidation_keys_on(self):
        header = INVENTORY.parse_camera_header(HEADER_50KHZ)
        assert header.frame_rate_hz == 50000
        assert header.frame_size == "208x208"
        assert header.frame_count == 2001
        assert header.top_frame == 15000
        assert header.bottom_frame == 17000

    def test_missing_fields_are_none_not_guessed(self):
        header = INVENTORY.parse_camera_header("Frame_Rate: 2500\r\n")
        assert header.frame_rate_hz == 2500
        assert header.frame_size is None
        assert header.frame_count is None

    def test_binary_garbage_yields_no_frame_rate(self):
        """Seven headers in the export tree are filesystem index records.

        They must not parse into a plausible-looking number; the directory has
        to fall out to review instead.
        """
        header = INVENTORY.parse_camera_header("INDX(\x00\x09\x00\xa0i);\x00 4 0 A 3 B 4 ~ 1 . B M P")
        assert header.frame_rate_hz is None


class TestCameraClassification:
    def test_fifty_kilohertz_raw_frames_are_reserved_for_fluctuation(self):
        info = INVENTORY.classify_camera_directory(
            "27134_50kHz", _indexed("27134_50kHz", 3), HEADER_50KHZ
        )
        assert info.diagnostic == "camera_visible_fluctuation"
        assert info.shot == 27134
        assert info.suffix == "_50kHz"
        assert info.family == "raw_indexed"

    def test_wide_view_raw_frames_go_to_routine_ingest(self):
        info = INVENTORY.classify_camera_directory(
            "36976-N001", _indexed("36976-N001", 3), HEADER_25KHZ
        )
        assert info.diagnostic == "camera_visible"
        assert info.shot == 36976
        assert info.suffix == "-N001"

    def test_arranged_frames_are_never_mapper_input(self):
        """`bmp_arranger` output is derived, and issue #161 forbids it as a fixture."""
        info = INVENTORY.classify_camera_directory(
            "39915",
            ["39915_305.2_ms.bmp", "39915_305.6_ms.bmp", "39915_bmp.txt", "Thumbs.db"],
            HEADER_25KHZ,
        )
        assert info.diagnostic == "camera_visible_arranged"
        assert info.family == "arranged"
        assert info.junk == ("Thumbs.db",)

    def test_raw_frames_without_a_header_are_not_dated_by_guesswork(self):
        info = INVENTORY.classify_camera_directory(
            "27140_50kHz", [f"27140_50kHz_{i:08d}.bmp" for i in range(3)], None
        )
        assert info.diagnostic is None
        assert info.needs_review
        assert "header" in info.reason

    def test_empty_directory_is_flagged_not_silently_dropped(self):
        info = INVENTORY.classify_camera_directory("32345", ["32345_bmp.txt"], HEADER_25KHZ)
        assert info.diagnostic is None
        assert info.family == "empty"

    def test_applestuff_is_ignored_when_deciding_the_family(self):
        members = _indexed("27134_50kHz", 2) + ["._27134_50kHz_00000000.bmp", ".DS_Store"]
        info = INVENTORY.classify_camera_directory("27134_50kHz", members, HEADER_50KHZ)
        assert info.family == "raw_indexed"
        assert info.frame_count == 2


class TestSuffixNormalisation:
    def test_acquisition_suffix_comes_off_directory_and_members(self):
        renames = INVENTORY.normalised_camera_names(
            "27134_50kHz", 27134, _indexed("27134_50kHz", 2) + ["Thumbs.db"]
        )
        assert renames["27134_50kHz_00000000.bmp"] == "27134_00000000.bmp"
        assert renames["27134_50kHz_bmp.txt"] == "27134_bmp.txt"
        assert "Thumbs.db" not in renames

    def test_unsuffixed_directories_need_no_renames(self):
        renames = INVENTORY.normalised_camera_names("40595", 40595, _indexed("40595", 2))
        assert all(old == new for old, new in renames.items())

    def test_result_matches_what_camera_visible_looks_for(self):
        """The whole point of the rename is this filename shape."""
        renames = INVENTORY.normalised_camera_names(
            "36976-N001", 36976, _indexed("36976-N001", 1)
        )
        assert set(renames.values()) == {"36976_00000000.bmp", "36976_bmp.txt"}


class TestDigitizerClassification:
    def test_soft_and_hard_x_ray_files_are_told_apart(self):
        soft = INVENTORY.classify_digitizer_filename("digitizer_17592_39107.csv")
        assert (soft.diagnostic, soft.shot, soft.daq_label) == ("soft_x_rays", 39107, "17592")

        hard = INVENTORY.classify_digitizer_filename("digitizer_hxr_Eflux_17592_40140.csv")
        assert (hard.diagnostic, hard.shot, hard.variant) == ("hard_x_rays", 40140, "Eflux")

    def test_hard_x_ray_files_never_leak_into_the_soft_set(self):
        """`digitizer_hxr_raw_17592_40140.csv` also matches a loose soft pattern.

        If the hard pattern were checked second, the shot would come out as the
        DAQ label and the file would be archived as soft X-ray data.
        """
        for name in (
            "digitizer_hxr_raw_17592_40140.csv",
            "digitizer_hxr_Eflux_17592_40140.csv",
        ):
            assert INVENTORY.classify_digitizer_filename(name).diagnostic == "hard_x_rays"

    @pytest.mark.parametrize("name", ["ratio.csv", "digitizer_17592_45531.png", "notes.txt"])
    def test_unrelated_files_are_not_claimed(self, name):
        assert INVENTORY.classify_digitizer_filename(name) is None

    def test_agrees_with_the_mapping_that_will_read_the_archive(self):
        """The inventory must infer the same DAQ label `soft_x_rays` resolves.

        `vaft` parses the label knowing the shot; the inventory has to infer
        both at once. A disagreement here would file a CSV under a shot the
        mapping then cannot find.
        """
        name = "digitizer_22577_45531.csv"
        info = INVENTORY.classify_digitizer_filename(name)
        assert info.daq_label == _daq_label_from_filename(name, info.shot)


class TestOtherArtifacts:
    def test_vendor_container_is_recognised_by_shot(self):
        assert INVENTORY.classify_mcf_filename("41656.mcf") == 41656
        assert INVENTORY.classify_mcf_filename("notes.mcf") is None

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("shot #3619", (3619, "")),
            ("shot # 4838", (4838, "")),
            ("Shot #5568", (5568, "")),
            ("shot#3881", (3881, "")),
            ("shot #5378_ECHplasma", (5378, "ECHplasma")),
            ("shot #3589 O2filter 500-1800", (3589, "O2filter 500-1800")),
        ],
    )
    def test_ccd_directories_keep_the_condition_label(self, name, expected):
        """Two exports of shot 5378 differ only by label; dropping it collides them."""
        assert INVENTORY.classify_ccd_directory(name) == expected

    @pytest.mark.parametrize("name", ["camera back up", "20130816_FastCamera", "Camera"])
    def test_grouping_directories_are_not_mistaken_for_shots(self, name):
        assert INVENTORY.classify_ccd_directory(name) is None


class TestDuplicateDetection:
    def test_repeated_content_points_at_the_first_copy(self):
        entries = [
            INVENTORY.Entry(source="/a/x.csv", kind="digitizer_csv", diagnostic="soft_x_rays",
                            shot=1, size_bytes=10, sha256="abc"),
            INVENTORY.Entry(source="/b/x.csv", kind="digitizer_csv", diagnostic="soft_x_rays",
                            shot=1, size_bytes=10, sha256="abc"),
            INVENTORY.Entry(source="/c/y.csv", kind="digitizer_csv", diagnostic="soft_x_rays",
                            shot=2, size_bytes=10, sha256="def"),
        ]
        INVENTORY.mark_duplicates(entries)
        assert entries[0].duplicate_of is None
        assert entries[1].duplicate_of == "/a/x.csv"
        assert entries[2].duplicate_of is None


class TestTargetPaths:
    ROOT = Path("/filedb")

    def _entry(self, **kwargs):
        base = {
            "source": "/src/thing",
            "kind": "camera_directory",
            "diagnostic": "camera_visible",
            "shot": 40595,
            "size_bytes": 0,
            "file_count": 1,
            "detail": {},
            "sha256": None,
            "duplicate_of": None,
            "reason": None,
        }
        base.update(kwargs)
        return base

    def test_mapped_camera_lands_where_camera_visible_looks(self):
        target = CONSOLIDATE.target_for(self._entry(), self.ROOT)
        assert target == self.ROOT / "legacy" / "camera_visible" / "40595"

    def test_fluctuation_set_lands_in_its_own_tree(self):
        target = CONSOLIDATE.target_for(
            self._entry(diagnostic="camera_visible_fluctuation", shot=27134), self.ROOT
        )
        assert target == self.ROOT / "legacy" / "camera_visible_fluctuation" / "27134"

    def test_soft_x_ray_csv_lands_in_a_per_shot_directory(self):
        target = CONSOLIDATE.target_for(
            self._entry(
                source="/src/digitizer_17592_39107.csv",
                kind="digitizer_csv",
                diagnostic="soft_x_rays",
                shot=39107,
            ),
            self.ROOT,
        )
        assert target == self.ROOT / "legacy" / "soft_x_rays" / "39107" / "digitizer_17592_39107.csv"

    def test_unreadable_data_stays_out_of_the_legacy_domain(self):
        for diagnostic in ("camera_visible_arranged", "camera_visible_mcf", "hard_x_rays"):
            target = CONSOLIDATE.target_for(self._entry(diagnostic=diagnostic), self.ROOT)
            assert target is not None
            assert target.parts[:3] == ("/", "filedb", "unmapped"), diagnostic

    def test_ccd_condition_label_keeps_two_exports_apart(self):
        first = CONSOLIDATE.target_for(
            self._entry(kind="ccd_directory", diagnostic="camera_ccd_2013", shot=5378,
                        detail={"note": "ECHplasma"}),
            self.ROOT,
        )
        second = CONSOLIDATE.target_for(
            self._entry(kind="ccd_directory", diagnostic="camera_ccd_2013", shot=5378,
                        detail={"note": "Swing-down"}),
            self.ROOT,
        )
        assert first != second

    def test_duplicates_and_unclassified_entries_are_left_alone(self):
        assert CONSOLIDATE.target_for(self._entry(duplicate_of="/a/x"), self.ROOT) is None
        assert CONSOLIDATE.target_for(self._entry(diagnostic=None), self.ROOT) is None

    def test_two_sources_claiming_one_target_is_refused(self):
        inventory = {
            "entries": [
                self._entry(source="/one"),
                self._entry(source="/two"),
            ]
        }
        with pytest.raises(CONSOLIDATE.ConsolidationError, match="Two sources claim"):
            CONSOLIDATE.plan_operations(inventory, self.ROOT)


class TestGitSafeguard:
    """Data inside someone's checkout is copied, never moved out of it."""

    def test_a_path_under_a_git_checkout_is_detected(self, tmp_path):
        repo = tmp_path / "repo"
        (repo / ".git").mkdir(parents=True)
        (repo / "data" / "raw").mkdir(parents=True)
        target = repo / "data" / "raw" / "digitizer_17592_45531.csv"
        target.write_text("x")
        assert CONSOLIDATE.inside_git_worktree(target)

    def test_a_path_outside_any_checkout_is_not(self, tmp_path):
        plain = tmp_path / "acq" / "digitizer_17592_45531.csv"
        plain.parent.mkdir(parents=True)
        plain.write_text("x")
        assert not CONSOLIDATE.inside_git_worktree(plain)

    def test_copy_leaves_the_original_and_revert_removes_the_copy(self, tmp_path):
        repo = tmp_path / "repo"
        (repo / ".git").mkdir(parents=True)
        source = repo / "digitizer_17592_45531.csv"
        source.write_text("payload")
        target = tmp_path / "filedb" / "legacy" / "soft_x_rays" / "45531" / source.name
        manifest_path = tmp_path / "move_manifest.jsonl"

        with CONSOLIDATE.Manifest(manifest_path) as manifest:
            mode = CONSOLIDATE._move_path(
                source, target, manifest, kind="move_file", keep_source=True
            )
        assert mode == "copy_keep"
        assert source.read_text() == "payload"
        assert target.read_text() == "payload"

        CONSOLIDATE.revert(manifest_path, execute_changes=True)
        assert source.read_text() == "payload", "the original must survive a revert"
        assert not target.exists(), "the copy must be removed"


class TestReviewRegressions:
    """Cases a review found, each of which silently produced wrong data."""

    def test_a_date_named_directory_is_not_a_shot(self):
        """`20130816_FastCamera` used to classify as shot 201308.

        The greedy `\\d{3,6}` took the first six digits of the date and left
        `16_FastCamera` as the acquisition suffix, so a directory that never
        held a discharge would be consolidated under a shot VEST never fired.
        """
        info = INVENTORY.classify_camera_directory(
            "20130816_FastCamera",
            ["20130816_FastCamera_00000000.bmp", "20130816_FastCamera_bmp.txt"],
            "Frame_Rate: 2500\r\nFrames: 1\r\n",
        )
        assert info.shot is None
        assert info.diagnostic is None
        assert "does not start with a shot number" in info.reason

    @pytest.mark.parametrize(
        ("name", "shot", "suffix"),
        [("41451", 41451, ""), ("27134_50kHz", 27134, "_50kHz"), ("36976-N001", 36976, "-N001")],
    )
    def test_real_shot_directories_still_parse(self, name, shot, suffix):
        info = INVENTORY.classify_camera_directory(
            name, [f"{name}_00000000.bmp", f"{name}_bmp.txt"], "Frame_Rate: 2500\r\nFrames: 1\r\n"
        )
        assert (info.shot, info.suffix) == (shot, suffix)

    def test_an_already_moved_target_is_skipped_not_fatal(self, tmp_path):
        """`os.rename` onto a non-empty directory raises ENOTEMPTY.

        The guard covered only files, so a resumed run died at the first
        already-moved shot and abandoned every operation after it.
        """
        source = tmp_path / "src" / "41451"
        source.mkdir(parents=True)
        (source / "41451_00000000.bmp").write_text("x")
        target = tmp_path / "dst" / "camera_visible" / "41451"
        target.mkdir(parents=True)
        (target / "41451_00000000.bmp").write_text("x")

        operation = CONSOLIDATE.Operation(
            kind="move_directory", source=str(source), target=str(target),
            diagnostic="camera_visible", shot=41451,
        )
        with CONSOLIDATE.Manifest(tmp_path / "m.jsonl") as manifest:
            counts = CONSOLIDATE.execute([operation], manifest)
        assert counts.get("already_present") == 1
        assert source.exists(), "the source must be left alone, not half-moved"

    def test_a_file_under_both_names_is_refused(self, tmp_path):
        """A half-renamed directory still looks complete to camera_visible.

        Its header resolves and `_load_raw_frame` simply returns None for the
        frames still carrying the suffix, so the mapping would build an IDS
        from whichever subset it could open.
        """
        directory = tmp_path / "27134"
        directory.mkdir()
        (directory / "27134_50kHz_00000000.bmp").write_text("a")
        (directory / "27134_00000000.bmp").write_text("b")
        with pytest.raises(CONSOLIDATE.ConsolidationError, match="under both"):
            CONSOLIDATE._apply_member_renames(
                directory, {"27134_50kHz_00000000.bmp": "27134_00000000.bmp"}
            )

    def test_an_already_renamed_file_is_not_an_error(self, tmp_path):
        """Only genuine ambiguity is refused; a converged rename is fine."""
        directory = tmp_path / "27134"
        directory.mkdir()
        (directory / "27134_00000000.bmp").write_text("b")
        applied = CONSOLIDATE._apply_member_renames(
            directory, {"27134_50kHz_00000000.bmp": "27134_00000000.bmp"}
        )
        assert applied == {}

    def test_a_truncated_copy_is_caught_before_the_source_is_deleted(self, tmp_path):
        """`copytree` not raising says no call failed, not that bytes arrived."""
        source, target = tmp_path / "src", tmp_path / "dst"
        source.mkdir(); target.mkdir()
        (source / "a.bmp").write_text("xxxxx")
        (target / "a.bmp").write_text("xx")
        with pytest.raises(CONSOLIDATE.ConsolidationError, match="refusing to delete"):
            CONSOLIDATE._verify_directory_copy(source, target)

    def test_a_missing_file_is_caught_too(self, tmp_path):
        source, target = tmp_path / "src", tmp_path / "dst"
        source.mkdir(); target.mkdir()
        (source / "a.bmp").write_text("x")
        with pytest.raises(CONSOLIDATE.ConsolidationError, match="Missing"):
            CONSOLIDATE._verify_directory_copy(source, target)

    def test_export_residue_does_not_fail_a_matching_copy(self, tmp_path):
        source, target = tmp_path / "src", tmp_path / "dst"
        source.mkdir(); target.mkdir()
        (source / "a.bmp").write_text("x")
        (source / "Thumbs.db").write_text("junk")
        (target / "a.bmp").write_text("x")
        CONSOLIDATE._verify_directory_copy(source, target)

    def test_every_move_records_its_intent_before_acting(self, tmp_path):
        """A crash between the rename and its record left an unrevertable move."""
        source = tmp_path / "src" / "digitizer_17592_41451.csv"
        source.parent.mkdir(parents=True)
        source.write_text("1,2\n")
        target = tmp_path / "dst" / "41451" / source.name
        manifest_path = tmp_path / "m.jsonl"
        with CONSOLIDATE.Manifest(manifest_path) as manifest:
            CONSOLIDATE._move_path(source, target, manifest, kind="move_file")

        kinds = [json.loads(line)["kind"] for line in
                 manifest_path.read_text().splitlines() if line.strip()]
        assert kinds == ["move_file.intent", "move_file"]

    def test_revert_ignores_intent_records(self, tmp_path):
        source = tmp_path / "src" / "digitizer_17592_41451.csv"
        source.parent.mkdir(parents=True)
        source.write_text("payload")
        target = tmp_path / "dst" / "41451" / source.name
        manifest_path = tmp_path / "m.jsonl"
        with CONSOLIDATE.Manifest(manifest_path) as manifest:
            CONSOLIDATE._move_path(source, target, manifest, kind="move_file")
        assert not source.exists() and target.exists()

        CONSOLIDATE.revert(manifest_path, execute_changes=True)
        assert source.read_text() == "payload"
        assert not target.exists()

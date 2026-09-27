"""Discover every ShotLog file, choose the records, and convert them.

The ShotLog folder holds the monthly workbooks and, beside them, Excel lock
files, copies (``복사본``/``사본``), autosaves, forms, in-progress workbooks,
and a few real logs that follow no naming rule (``ERC_ShotLog``,
``KSTAR_Conference_ShotLog``, ``conditioning``). The ShotLog is kept because
it is complete -- every discharge from the first one logged -- so discovery
loses nothing: each file gets a role, and only lock files and filesystem
residue are left out, each with a reason.

Roles:

``monthly_record``       the chosen workbook of a month; converted
``supplementary_log``    a real log outside the monthly naming; converted, ranked below monthly
``superseded_open_ended``  an in-progress ``#first-`` copy of a closed month; kept, not converted
``copy`` / ``autosave`` / ``template`` / ``unrecognised_workbook`` / ``other_file``
                         kept byte for byte, not converted
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, replace
import json
from pathlib import Path
import re
from typing import Any, Iterable, Iterator
import unicodedata

from openpyxl import load_workbook

from .converter import convert_sheet, marker_shots, session_path, sha256_file, validate_dataset
from .schema import packaged_registry, schema_versions

MONTHLY_RE = re.compile(
    r"^ShotLog_(?P<year>(?:19|20)\d{2})_(?P<month>0?[1-9]|1[0-2])(?:\s|#|$)", re.IGNORECASE
)
SHOT_SPAN_RE = re.compile(r"#\s*(?P<first>\d+)\s*[-~]\s*(?P<last>\d+)?\s*$")
#: Name tokens and the role they mark, matched in NFC and case-folded against
#: the file name and the folders above it. The Korean tokens are distinctive
#: as substrings; the English ones must be whole words, or "information",
#: "folder" and "plasma formation" would hide a real month.
ROLE_TOKENS = (
    (re.compile(r"복사본|사본"), "copy"),
    (re.compile(r"(?<![a-z])old(?![a-z])"), "copy"),
    (re.compile(r"자동\s?저장"), "autosave"),
    (re.compile(r"(?<![a-z])temp(?![a-z])"), "autosave"),
    (re.compile(r"(?<![a-z])(?:form|plan)(?![a-z])"), "template"),
)
MONTHLY = "monthly_record"
SUPPLEMENTARY = "supplementary_log"
CONVERTED_ROLES = frozenset({MONTHLY, SUPPLEMENTARY})
#: Archive-only roles still read as a last resort: a copy or autosave can hold
#: shots the chosen record lost (the 2015-10 autosave has seven), and a shot
#: with no other source takes its record from one. Never used otherwise.
FALLBACK_ROLES = frozenset({"copy", "autosave", "template", "superseded_open_ended"})
#: How far outside a workbook's ``#first-last`` a column-A number may sit and
#: still be one of its shots. Beyond it, the number is a reference to an older
#: shot (2025 sheets open with 43960) or not a shot at all (400, 1000 in 2013).
SPAN_MARGIN = 200
#: Derived output of the standalone tool, never data.
DERIVED_DIRECTORIES = frozenset({"ShotLog_dataset", ".index"})
JUNK_FILES = frozenset({".DS_Store", "Thumbs.db", "desktop.ini"})
#: Present in a FileDB archive root, and only there.
ARCHIVE_MANIFEST = "manifest.json"


@dataclass(frozen=True)
class SourceWorkbook:
    """One converted ShotLog file: a month's record, a supplementary log, or a fallback."""

    path: Path
    year: int | None
    month: int | None
    first_shot: int | None
    last_shot: int | None
    role: str = MONTHLY
    #: Upper bound for an open-ended workbook: the next month's first shot.
    span_limit: int | None = None
    #: Where the file lives under ``legacy/shotlog/input``.
    archive_path: str = ""
    #: For a log whose name carries no ``#first-last``: the span its own
    #: column A implies (see :func:`_content_span`).
    content_span: tuple[int, int] | None = None

    @property
    def open_ended(self) -> bool:
        return self.first_shot is not None and self.last_shot is None

    @property
    def fallback(self) -> bool:
        return self.role in FALLBACK_ROLES

    def covers(self, shot: int) -> bool:
        if self.first_shot is None:
            return False
        return self.first_shot <= shot and (self.last_shot is None or shot <= self.last_shot)

    def span(self) -> tuple[int, int | None] | None:
        """Shots this file can plausibly hold, with :data:`SPAN_MARGIN`."""
        if self.first_shot is None:
            return self.content_span
        upper = self.last_shot if self.last_shot is not None else self.span_limit
        return self.first_shot - SPAN_MARGIN, None if upper is None else upper + SPAN_MARGIN


@dataclass(frozen=True)
class ArchivedFile:
    """A file kept byte for byte but not converted as a record."""

    path: Path
    role: str
    archive_path: str
    note: str | None = None


@dataclass
class Discovery:
    included: list[SourceWorkbook] = field(default_factory=list)
    archived_only: list[ArchivedFile] = field(default_factory=list)
    #: Archive-only workbooks that log shots: read only for shots nothing else has.
    fallback: list[SourceWorkbook] = field(default_factory=list)
    #: Not archived at all, each with a reason.
    excluded: list[dict[str, Any]] = field(default_factory=list)


def display_name(path: Path) -> str:
    return unicodedata.normalize("NFC", path.name)


def _describe(path: Path) -> SourceWorkbook | None:
    match = MONTHLY_RE.match(unicodedata.normalize("NFC", path.stem))
    if not match:
        return None
    first, last = _name_span(path)
    return SourceWorkbook(path, int(match.group("year")), int(match.group("month")), first, last)


def _name_span(path: Path) -> tuple[int | None, int | None]:
    span = SHOT_SPAN_RE.search(unicodedata.normalize("NFC", path.stem))
    if not span:
        return None, None
    first = int(span.group("first"))
    last = int(span.group("last")) if span.group("last") else None
    # "#47000-4730" is a typo, not an empty month: read it as open-ended.
    return first, last if last is None or last >= first else None


def _token_role(text: str) -> str | None:
    lowered = unicodedata.normalize("NFC", text).casefold()
    return next((role for pattern, role in ROLE_TOKENS if pattern.search(lowered)), None)


def classify_source(path: Path) -> tuple[SourceWorkbook | None, str | None]:
    """``(workbook, None)`` for a monthly ShotLog by name, else ``(None, reason)``.

    Names are compared in NFC. macOS hands back Korean file names decomposed
    (NFD), in which ``자동 저장`` never matches the composed token -- which is
    how two ``(자동 저장됨)`` autosaves were once taken for their months' record.
    """
    name = unicodedata.normalize("NFC", path.name)
    if name.startswith("._"):
        return None, "appledouble_sidecar"
    if name.startswith("~$"):
        return None, "excel_lock_file"
    role = _token_role(name)
    if role is not None:
        return None, role
    described = _describe(path)
    if described is None:
        return None, "not_monthly_shotlog"
    return described, None


def _column_a_shots(path: Path, rows: int | None = None) -> list[int]:
    try:
        workbook = load_workbook(path, read_only=True, data_only=True)
    except Exception:  # noqa: BLE001 - an unreadable file is kept, just not converted
        return []
    shots: list[int] = []
    try:
        for ws in workbook.worksheets:
            for (value,) in ws.iter_rows(min_row=1, max_row=rows, max_col=1, values_only=True):
                shots.extend(marker_shots(value))
    finally:
        workbook.close()
    return shots


def has_shot_markers(path: Path, minimum: int = 2, rows: int = 60) -> bool:
    """Whether any sheet's column A starts with shot numbers -- i.e. it is a log."""
    return len(_column_a_shots(path, rows)) >= minimum


def _content_span(path: Path) -> tuple[int, int] | None:
    """The span a log without ``#first-last`` implies: its middle 80 %, widened.

    ``ERC_ShotLog`` or ``conditioning`` name no range, and without one every
    number in column A would be a shot -- including the 400s and 1000s that
    turned out not to be. The bulk of the file's own numbers says where its
    shots are; a stray far from them is a reference or a value, not a shot.
    """
    shots = sorted(_column_a_shots(path))
    if not shots:
        return None
    low = shots[len(shots) // 10]
    high = shots[(len(shots) * 9) // 10]
    margin = max(SPAN_MARGIN, (high - low) // 2)
    return low - margin, high + margin


def _relative(path: Path, root: Path | None) -> str:
    try:
        parts = path.relative_to(root).parts if root is not None else (path.name,)
    except ValueError:
        parts = (path.name,)
    return "/".join(unicodedata.normalize("NFC", part) for part in parts)


def _is_archive(root: Path) -> bool:
    """A FileDB ``legacy/shotlog/input`` root, recognised by its manifest."""
    manifest = root / ARCHIVE_MANIFEST
    if not manifest.is_file():
        return False
    try:
        document = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(document, dict) and ("files" in document or "workbooks" in document)


def discover_sources(
    input_dir: str | Path,
    extra: Iterable[str | Path] = (),
    exclude: Iterable[str | Path] = (),
) -> Discovery:
    """Give every file under ``input_dir`` (and every ``extra`` file) a role.

    ``input_dir`` is either the operators' ShotLog folder or the FileDB
    archive (``legacy/shotlog/input``, recognised by its manifest), which files
    months by year, real non-monthly logs under ``supplementary/`` and
    everything else under ``other/``. Nothing found is dropped silently: a file
    is converted, archived, used as a fallback, or listed in ``excluded`` with
    the reason.

    An open-ended workbook (``#47752-``) is the in-progress copy of a month.
    It is superseded when the same month also has a closed one and is
    otherwise the only record of that month, so it is converted.
    """
    root = Path(input_dir)
    archive = _is_archive(root)
    excluded_roots = [Path(item).resolve() for item in exclude]
    discovery = Discovery()
    by_month: dict[tuple[int, int], list[SourceWorkbook]] = defaultdict(list)
    supplementary: list[SourceWorkbook] = []

    paths: list[tuple[Path, Path | None]] = []
    if root.is_dir():
        paths.extend((path, root) for path in root.rglob("*") if path.is_file())
    paths.extend((Path(item), None) for item in extra)
    skipped_under: dict[str, int] = defaultdict(int)
    for path, base in sorted(paths, key=lambda item: _relative(*item).casefold()):
        relative = _relative(path, base)
        parts = relative.split("/")
        name = parts[-1]
        inside = next((item for item in excluded_roots if path.resolve().is_relative_to(item)), None)
        if inside is not None:
            skipped_under[f"{inside} (the FileDB archive itself)"] += 1
            continue
        if DERIVED_DIRECTORIES & set(parts[:-1]):
            skipped_under[f"{'/'.join(parts[:-1])} (derived output of the standalone tool)"] += 1
            continue
        if name.startswith("._"):
            discovery.excluded.append({"source_file": relative, "reason": "appledouble_sidecar"})
            continue
        if name.startswith("~$"):
            discovery.excluded.append({"source_file": relative, "reason": "excel_lock_file"})
            continue
        if name in JUNK_FILES:
            discovery.excluded.append({"source_file": relative, "reason": "filesystem_residue"})
            continue
        if archive and base is not None:
            if parts[0] == "superseded" or relative == ARCHIVE_MANIFEST:
                continue  # the archive's own history and index, not sources
            if parts[0] == "other":
                inner = "/".join(parts[1:])
                role = _token_role(inner) or (
                    "superseded_open_ended" if _describe(path) is not None
                    else "unrecognised_workbook" if path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}
                    else "other_file"
                )
                discovery.archived_only.append(ArchivedFile(path, role, relative))
                continue
            if parts[0] == "supplementary":
                first, last = _name_span(path)
                supplementary.append(SourceWorkbook(path, None, None, first, last, role=SUPPLEMENTARY,
                                                    archive_path=relative))
                continue
        if path.suffix.lower() not in {".xlsx", ".xlsm", ".xls"}:
            discovery.archived_only.append(ArchivedFile(path, "other_file", f"other/{relative}"))
            continue
        # The folder counts too: "Shot Log form/VEST shotlog_New_v3.xlsx" is a template.
        role = _token_role(relative)
        if role is not None:
            discovery.archived_only.append(ArchivedFile(path, role, f"other/{relative}"))
            continue
        monthly = _describe(path)
        if monthly is not None:
            by_month[(monthly.year, monthly.month)].append(
                replace(monthly, archive_path=f"{monthly.year:04d}/{name}")
            )
            continue
        if path.suffix.lower() == ".xls":
            discovery.archived_only.append(ArchivedFile(
                path, "unrecognised_workbook", f"other/{relative}",
                note="legacy .xls: kept, but openpyxl cannot read it"))
        elif has_shot_markers(path):
            first, last = _name_span(path)
            supplementary.append(SourceWorkbook(path, None, None, first, last, role=SUPPLEMENTARY,
                                                archive_path=f"supplementary/{name}"))
        else:
            discovery.archived_only.append(
                ArchivedFile(path, "unrecognised_workbook", f"other/{relative}",
                             note="no shot numbers in column A")
            )
    discovery.excluded.extend({"source_file": where, "reason": "skipped_directory", "files": count}
                              for where, count in sorted(skipped_under.items()))

    monthly_records: list[SourceWorkbook] = []
    for month in sorted(by_month):
        candidates = by_month[month]
        closed = [item for item in candidates if not item.open_ended]
        keep = closed or candidates
        for item in candidates:
            if item in keep:
                monthly_records.append(item)
            else:
                discovery.archived_only.append(ArchivedFile(
                    item.path, "superseded_open_ended", _other_path(item.path, root),
                    note="superseded by " + ", ".join(display_name(other.path) for other in closed),
                ))
    # An open-ended month is bounded by the next month's first shot.
    for index, item in enumerate(monthly_records):
        if item.open_ended:
            following = next((later.first_shot for later in monthly_records[index + 1:]
                              if later.first_shot is not None), None)
            monthly_records[index] = replace(item, span_limit=following)
    supplementary = [
        item if item.first_shot is not None else replace(item, content_span=_content_span(item.path))
        for item in supplementary
    ]
    discovery.included = _unique_paths(monthly_records + supplementary)
    discovery.archived_only = _unique_paths(discovery.archived_only)
    discovery.fallback = [
        _fallback(item) for item in discovery.archived_only
        if item.role in FALLBACK_ROLES and item.path.suffix.lower() in {".xlsx", ".xlsm"}
        and has_shot_markers(item.path)
    ]
    return discovery


def _other_path(path: Path, root: Path) -> str:
    try:
        return "other/" + _relative(path, root)
    except ValueError:
        return "other/" + display_name(path)


def _fallback(item: ArchivedFile) -> SourceWorkbook:
    described = _describe(item.path)
    first, last = _name_span(item.path)
    workbook = SourceWorkbook(
        item.path, described.year if described else None, described.month if described else None,
        first, last, role=item.role, archive_path=item.archive_path,
    )
    return workbook if first is not None else replace(workbook, content_span=_content_span(item.path))


def _unique_paths(items: list) -> list:
    """Give two different files that would share an archive path distinct ones.

    ``a/ShotLog_2023_05 #...xlsx`` and ``b/ShotLog_2023_05 #...xlsx`` flatten
    to one ``2023/`` name; archived under it they would replace each other on
    every run, and one month's shots would silently belong to the other file.
    The second keeps its bytes under a name carrying their hash.
    """
    seen: dict[str, Path] = {}
    result = []
    for item in items:
        target = item.archive_path
        if target in seen and seen[target].resolve() != item.path.resolve():
            stem, dot, suffix = target.rpartition(".")
            target = f"{stem}.sha-{sha256_file(item.path)[:8]}{dot}{suffix}"
            item = replace(item, archive_path=target)
        seen.setdefault(target, item.path)
        result.append(item)
    return result


def iter_sessions(
    sources: list[SourceWorkbook],
    registry: dict[str, Any] | None = None,
    overrides_path: Path | None = None,
    manifest: dict[str, Any] | None = None,
) -> Iterator[tuple[SourceWorkbook, dict[str, Any]]]:
    """Yield ``(workbook, session)`` for every sheet that logs a shot.

    A sheet no template matches is still yielded: its shots get records with
    their raw cells, marked ``unclassified``, because a shot with no record is
    a shot the archive has lost. Per-workbook and per-sheet failures are
    recorded in ``manifest`` (when given) and skipped: one corrupt month must
    not stop the archive.
    """
    registry = registry or packaged_registry()
    record = manifest if manifest is not None else _new_manifest(registry)
    for source in sources:
        source_hash = sha256_file(source.path)
        entry: dict[str, Any] = {"source_file": display_name(source.path), "source_sha256": source_hash,
                                 "role": source.role, "sheets": []}
        record["included_files"].append(entry)
        try:
            workbook = load_workbook(source.path, read_only=False, data_only=False)
        except Exception as error:  # noqa: BLE001 - recorded, the walk continues
            entry["error"] = repr(error)
            record["errors"].append({"source_file": display_name(source.path), "stage": "open_workbook", "error": repr(error)})
            continue
        try:
            for sheet_name in list(workbook.sheetnames):
                sheet: dict[str, Any] = {"sheet_name": sheet_name}
                entry["sheets"].append(sheet)
                try:
                    dataset = convert_sheet(source.path, sheet_name, registry, overrides_path,
                                            workbook=workbook, source_hash=source_hash,
                                            span=source.span())
                except Exception as error:  # noqa: BLE001
                    sheet.update({"status": "error", "error": repr(error)})
                    record["errors"].append({"source_file": display_name(source.path), "sheet_name": sheet_name,
                                             "stage": "convert_sheet", "error": repr(error)})
                    continue
                sheet["schema_version"] = dataset["schema_version"]
                shots = sum(len(group["shots"]) for group in dataset["run_groups"])
                if dataset["schema_version"] == "unclassified":
                    record["unclassified"].append({"source_file": display_name(source.path), "sheet_name": sheet_name,
                                                   "classification": dataset.get("classification", {}),
                                                   "shots": shots})
                    if not shots:
                        sheet["status"] = "unclassified"
                        continue
                errors = validate_dataset(dataset, registry)
                sheet.update({"status": "converted" if dataset["schema_version"] != "unclassified"
                              else "unclassified_raw", "session_id": dataset["experiment_session"]["id"],
                              "shots": shots, "validation_errors": errors})
                if errors:
                    record["errors"].append({"source_file": display_name(source.path), "sheet_name": sheet_name,
                                             "stage": "validation", "errors": errors})
                yield source, dataset
        finally:
            workbook.close()


def _new_manifest(registry: dict[str, Any]) -> dict[str, Any]:
    return {
        "batch_version": 3,
        "schema_versions": schema_versions(registry),
        "included_files": [],
        "excluded_files": [],
        "unclassified": [],
        "errors": [],
        "collisions": [],
    }


def summarise(manifest: dict[str, Any]) -> dict[str, int]:
    sheets = [sheet for item in manifest["included_files"] for sheet in item.get("sheets", [])]
    return {
        "xlsx_included": len(manifest["included_files"]),
        "xlsx_excluded": len(manifest["excluded_files"]),
        "sheets_converted": sum(1 for sheet in sheets if sheet.get("status") == "converted"),
        "sheets_unclassified": len(manifest["unclassified"]),
        "sheets_unclassified_with_shots": sum(1 for sheet in sheets if sheet.get("status") == "unclassified_raw"),
        "errors": len(manifest["errors"]),
        "collisions": len(manifest["collisions"]),
    }


def convert_directory(
    input_dir: str | Path,
    registry: dict[str, Any] | None = None,
    overrides_path: Path | None = None,
    extra: Iterable[str | Path] = (),
) -> tuple[list[tuple[SourceWorkbook, dict[str, Any]]], dict[str, Any]]:
    """Convert every chosen workbook; return the sessions and a batch manifest.

    Two sheets can name the same session (``250219`` in two workbooks). The
    second keeps its content under ``<session>__source-<sha8>`` and the pair is
    listed under ``collisions`` instead of one overwriting the other.
    """
    registry = registry or packaged_registry()
    discovery = discover_sources(input_dir, extra)
    manifest = _new_manifest(registry)
    manifest["excluded_files"] = discovery.excluded
    manifest["archived_only_files"] = [
        {"source_file": item.archive_path, "role": item.role, "note": item.note}
        for item in discovery.archived_only
    ]
    sessions: list[tuple[SourceWorkbook, dict[str, Any]]] = []
    claimed: dict[str, tuple[str, str]] = {}
    manifest["fallback_files"] = [item.archive_path for item in discovery.fallback]
    # Fallback workbooks come last, so a copy never claims a session id first.
    for source, dataset in iter_sessions(discovery.included + discovery.fallback, registry,
                                         overrides_path, manifest):
        session = dataset["experiment_session"]
        original = session["id"]
        owner = claimed.get(original)
        if owner is not None:
            provenance = dataset["provenance"]
            renamed = f"{original}__source-{provenance['source_sha256'][:8]}"
            if renamed in claimed:
                renamed = f"{renamed}-{len(manifest['collisions']) + 1}"
            manifest["collisions"].append({
                "session_id": original, "renamed_to": renamed,
                "source_file": display_name(source.path), "sheet_name": provenance["sheet_name"],
                "existing_source_file": owner[0], "existing_sheet_name": owner[1],
            })
            session["id"] = renamed
            for group in dataset["run_groups"]:
                group["id"] = renamed + group["id"][len(original):]
        claimed[session["id"]] = (display_name(source.path), dataset["provenance"]["sheet_name"])
        sessions.append((source, dataset))
    manifest["summary"] = summarise(manifest)
    return sessions, manifest


__all__ = [
    "ArchivedFile",
    "Discovery",
    "SourceWorkbook",
    "classify_source",
    "convert_directory",
    "discover_sources",
    "iter_sessions",
    "session_path",
    "summarise",
]

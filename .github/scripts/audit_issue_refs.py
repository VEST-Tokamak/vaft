"""Inventory the GitHub issue and pull-request references in VAFT's tracked text.

Code, docstrings and documentation should explain what the code does, why, and
under which assumptions, without sending the reader to a ticket (see
"References and provenance" in CONTRIBUTING.md). This script finds every
reference -- ``#123``, ``(#123)``, ``issue #123``, ``PR #123``, ``GH-123`` and
github.com issue/pull URLs -- in the files ``git ls-files`` reports, and says
which of them are recorded in the baseline.

Every detection is reported. A detection is ``excluded`` when it is not a
ticket reference at all (an HTML entity, a hex colour, a shot number, a
numbered probe) or sits in generated output; ``retained`` when the baseline
records it with a reason (compatibility, provenance, an open limitation,
release history); ``legacy`` when the baseline records it only as something
that predates the policy; and ``new`` when the baseline does not know it.

    python .github/scripts/audit_issue_refs.py                      # summary
    python .github/scripts/audit_issue_refs.py --format csv -o refs.csv
    python .github/scripts/audit_issue_refs.py --paths vaft/plot --format csv
    python .github/scripts/audit_issue_refs.py --new                # CI report
    python .github/scripts/audit_issue_refs.py --paths vaft/plot --write-baseline

The baseline (``.github/issue-refs-baseline.json``) counts references per
(path, number) rather than per line, so moving or re-wrapping a line does not
make a reference look new; a (path, number) pair counts as new only when the
file holds more of that number than the baseline records. ``--new`` exits 0
unless ``--strict`` is given: the CI job reports, it does not gate.

Standard library only; run it from anywhere inside the repository.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPOSITORY = "VEST-Tokamak/vaft"
BASELINE = Path(".github/issue-refs-baseline.json")

# VAFT's own tickets are numbered well below this; anything larger written as
# `#NNNNN` is a shot number (VEST shots run from ~30000 upward).
MAX_TICKET_NUMBER = 9999

BINARY_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".pdf", ".h5", ".hdf5", ".nc",
    ".npz", ".npy", ".pkl", ".pickle", ".gz", ".zip", ".xz", ".bz2", ".mp4", ".webm",
    ".woff", ".woff2", ".ttf", ".otf", ".eot", ".so", ".dylib", ".dll", ".exe", ".mat",
    ".fig", ".step", ".stp", ".stl", ".parquet", ".feather", ".db", ".sqlite",
}

# Generated output, recorded fixtures and release history: references there are
# reported but never edited by hand.
EXCLUDED_PATHS = (
    ("docs/assets/", "generated documentation asset"),
    ("test/data/", "recorded test fixture"),
    ("vaft/data/", "packaged data file", (".json", ".csv", ".yaml", ".yml", ".txt", ".dat")),
    (".github/issue-refs-baseline.json", "this audit's own baseline"),
    (".github/scripts/audit_issue_refs.py", "this audit's own patterns"),
    ("test/test_audit_issue_refs.py", "this audit's own test cases"),
    ("", "data file (values, not prose)", (".csv", ".tsv", ".json", ".jsonl", ".geojson")),
    ("package-lock.json", "lock file"),
    ("Gemfile.lock", "lock file"),
)
HISTORY_NAMES = ("CHANGELOG", "CHANGES", "RELEASE_NOTES", "HISTORY", "NEWS")

URL = re.compile(r"https?://github\.com/([\w.-]+/[\w.-]+)/(issues|pull)/(\d+)")
GH = re.compile(r"(?<![\w-])GH-(\d+)\b")
WORDED = re.compile(
    r"\b(PRs?|[Pp]ull [Rr]equests?|[Ii]ssues?|[Tt]ickets?)\s+#(\d+)\b"
)
# A bare `#123`: not part of a word, an HTML entity (`&#123;`), a URL fragment
# or a second `#`; ranges such as `#418-#421` are two references.
BARE = re.compile(r"(?<![\w&#/=?%.\\$])#(\d{1,5})(?![\w.]*\w)")
ADJOINING_WORDED = re.compile(r"(?:,\s*|\s+and\s+|\s*[-/]\s*)#(\d+)\b")

NUMBERED_ITEM = re.compile(
    r"(?i)\b(probes?|loops?|coils?|channels?|sensors?|items?|steps?|No\.|numbers?|"
    r"fl|bp|gauges?|cameras?|ports?|turns?|figures?|figs?\.?|tables?|eqs?\.?|"
    r"shots?|discharges?|pulses?|runs?|slices?|rank(?:ed)?|options?|"
    r"lanes?|row|column|col|cell|frames?|harmonics?|modes?|sectors?)\s*[-\u2013]?\s*$"
)
# `fill='#000'`, `color: #333;`, `"#123456"`: a colour value, not prose.
HEX_CONTEXT = re.compile(r"(?i)(fill|stroke|colou?r|background|border)[\w-]*\s*[:=]\s*[\"']?$")
HEX_END = re.compile(r"[\"';)]")
# A quoted digit-only colour without a keyword: `"#000000"`, `'#333'`. A quoted
# `"#358"` is a ticket a test matches on, so only six digits or a repeated digit.
QUOTED_HEX = re.compile(r"[\"']#(\d{6}|(\d)\2\2)[\"']")
# `loops #3, #4 and #6`, `#9-#11`: the later numbers share the first one's noun.
LIST_TAIL = re.compile(r"(?:#\d+\s*(?:[-\u2013,/]|and|or)\s*)+$")
OTHER_PROJECT = re.compile(r"(?i)\b(vfit|omas|imas|efit|gpec|chease|numpy|scipy|matplotlib|h5pyd|hsds)\s*$")
HEX_SUFFIXES = {".css", ".scss", ".sass", ".svg", ".less"}

CATEGORY_RULES = (
    ("compat", re.compile(
        r"(?i)deprecat|removed in|will be removed|migrat|renamed|backward|compatib|"
        r"since v?\d+\.\d+|until v?\d+\.\d+")),
    ("open-limitation", re.compile(
        r"(?i)\bopen\b|todo|fixme|not yet|tracked (in|by)|follow-?up|waits? (on|for)|"
        r"until #|pending|blocked|known (issue|limitation|bug)|workaround")),
    ("provenance", re.compile(
        r"(?i)validat|reproduc|provenance|regenerat|run record|baseline|benchmark|"
        r"qualified|measured|calibrat|study|studies")),
)


@dataclass
class Reference:
    path: str
    line: int
    number: int
    kind: str
    context: str
    cell: str = ""
    status: str = ""
    category: str = ""
    action: str = ""
    reason: str = ""
    title: str = ""
    url: str = field(default="")


def repository_root() -> Path:
    out = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], check=True, capture_output=True, text=True
    )
    return Path(out.stdout.strip())


def tracked_files(root: Path, pathspecs: list[str]) -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "-z", "--", *pathspecs],
        cwd=root, check=True, capture_output=True,
    )
    return sorted(p for p in out.stdout.decode("utf-8").split("\0") if p)


def head_commit(root: Path) -> str:
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=root, check=True, capture_output=True, text=True,
    ).stdout.strip()
    return sha + ("-dirty" if dirty else "")


def exclusion_for(path: str) -> str:
    for rule in EXCLUDED_PATHS:
        prefix, reason = rule[0], rule[1]
        suffixes = rule[2] if len(rule) > 2 else None
        if path == prefix or prefix == "" or (prefix.endswith("/") and path.startswith(prefix)):
            if suffixes is None or path.endswith(suffixes):
                return reason
    return ""


def is_history_file(path: str) -> bool:
    return Path(path).stem.upper().startswith(HISTORY_NAMES)


def text_lines(root: Path, path: str):
    """Yield (line number, cell label, text) for one tracked file, or nothing for binaries."""
    full = root / path
    if Path(path).suffix.lower() in BINARY_SUFFIXES or not full.is_file():
        return
    data = full.read_bytes()
    if b"\0" in data[:8192]:
        return
    text = data.decode("utf-8", errors="replace")
    if path.endswith(".ipynb"):
        try:
            notebook = json.loads(text)
        except json.JSONDecodeError:
            notebook = None
        if isinstance(notebook, dict) and isinstance(notebook.get("cells"), list):
            # Only what a person wrote: cell sources, not outputs.
            for index, cell in enumerate(notebook["cells"]):
                source = cell.get("source", "")
                source = "".join(source) if isinstance(source, list) else str(source)
                for lineno, line in enumerate(source.splitlines(), 1):
                    yield lineno, f"{cell.get('cell_type', 'cell')}[{index}]", line
            return
    for lineno, line in enumerate(text.splitlines(), 1):
        yield lineno, "", line


def false_positive(path: str, line: str, start: int, number: int) -> str:
    """Why a bare `#N` at ``start`` is not a ticket reference, or ''."""
    if number == 0:
        return "zero is not a ticket number"
    if number > MAX_TICKET_NUMBER:
        return "beyond the ticket range (shot number)"
    if number < 20:
        # Flux loops #1-#15, TeX macro arguments, list numbering. VAFT's first
        # tickets are cited worded ("issue #8") or by URL, which still count.
        return "small number (flux loop, TeX argument, list item)"
    before = line[:start]
    if OTHER_PROJECT.search(before[-20:]):
        return "another project's ticket"
    if NUMBERED_ITEM.search(LIST_TAIL.sub("", before)[-30:]):
        return "numbered item (probe, coil, shot, figure, ...)"
    if Path(path).suffix.lower() in HEX_SUFFIXES or (
        len(str(number)) in (3, 6) and HEX_CONTEXT.search(before)
        and HEX_END.match(line, start + 1 + len(str(number)))
    ) or QUOTED_HEX.match(line, max(start - 1, 0)):
        return "hex colour"
    stripped = before.rstrip()
    if stripped == "" and line[start + 1 + len(str(number)):].strip() == "":
        # A line that is only `#12` -- a Markdown/shell artefact, not prose.
        return "bare number on its own line"
    return ""


def scan_line(path: str, lineno: int, cell: str, line: str) -> list[Reference]:
    found: list[Reference] = []
    claimed: set[int] = set()  # character offsets of the `#` already reported

    def add(number: int, kind: str, offset: int, url: str = "") -> None:
        found.append(Reference(path, lineno, number, kind, line.strip()[:240], cell, url=url))
        claimed.add(offset)

    for m in URL.finditer(line):
        repo, what, number = m.group(1), m.group(2), int(m.group(3))
        ref = Reference(path, lineno, number, "url-pr" if what == "pull" else "url-issue",
                        line.strip()[:240], cell, url=m.group(0))
        if repo.lower() != REPOSITORY.lower():
            ref.status, ref.reason = "excluded", f"another repository ({repo})"
        found.append(ref)
        claimed.update(range(m.start(), m.end()))
    for m in GH.finditer(line):
        add(int(m.group(1)), "gh", m.start())
    for m in WORDED.finditer(line):
        word = m.group(1).lower()
        kind = "pr" if word.startswith(("pr", "pull")) else "issue"
        hash_at = m.start(2) - 1
        add(int(m.group(2)), kind, hash_at)
        # `issues #330, #529` / `PR #12 and #13`: the followers share the kind.
        tail_at = m.end()
        while (follow := ADJOINING_WORDED.match(line, tail_at)) is not None:
            add(int(follow.group(1)), kind, follow.start(1) - 1)
            tail_at = follow.end()
    for m in BARE.finditer(line):
        if m.start() in claimed:
            continue
        number = int(m.group(1))
        ref = Reference(path, lineno, number, "bare", line.strip()[:240], cell)
        why = false_positive(path, line, m.start(), number)
        if why:
            ref.status, ref.reason = "excluded", why
        found.append(ref)
    return found


def categorize(ref: Reference) -> None:
    if ref.status == "excluded":
        ref.category, ref.action = "false-positive", "none"
        return
    if is_history_file(ref.path):
        ref.category, ref.action = "history", "retain"
        return
    for category, pattern in CATEGORY_RULES:
        if pattern.search(ref.context):
            ref.category = category
            ref.action = "review (retain if the rationale needs the ticket)"
            return
    ref.category, ref.action = "development-history", "rewrite or remove"


def load_baseline(root: Path) -> dict:
    path = root / BASELINE
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("references", {})


def apply_baseline(refs: list[Reference], baseline: dict) -> None:
    """Mark each counted reference retained/legacy/new against the baseline."""
    seen: Counter = Counter()
    for ref in refs:
        exclusion = exclusion_for(ref.path)
        if ref.status == "excluded" and not exclusion:
            continue
        if exclusion:
            ref.status, ref.reason, ref.category, ref.action = (
                "excluded", exclusion, "generated-or-fixture", "none")
            continue
        key = str(ref.number)
        seen[(ref.path, key)] += 1
        entry = baseline.get(ref.path, {}).get(key)
        if isinstance(entry, int):
            entry = {"count": entry, "status": "legacy"}
        if entry is None or seen[(ref.path, key)] > entry.get("count", 0):
            ref.status = "new"
            continue
        ref.status = entry.get("status", "legacy")
        if entry.get("category"):
            ref.category = entry["category"]
        if entry.get("reason"):
            ref.reason = entry["reason"]
        if ref.status == "retained":
            ref.action = "retain"


def resolve_titles(refs: list[Reference], cache_path: Path) -> None:
    """Fill title/url from GitHub via `gh` (optional; cached between runs)."""
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    for number in sorted({r.number for r in refs if r.status != "excluded"}):
        key = str(number)
        if key not in cache:
            out = subprocess.run(
                ["gh", "api", f"repos/{REPOSITORY}/issues/{number}",
                 "--jq", "[.title, .state, (.pull_request != null), .html_url]"],
                capture_output=True, text=True,
            )
            cache[key] = json.loads(out.stdout) if out.returncode == 0 else None
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=1, sort_keys=True))
    for ref in refs:
        info = cache.get(str(ref.number))
        if ref.status != "excluded" and info:
            title, state, is_pr, url = info
            ref.title = f"[{'PR' if is_pr else 'issue'}, {state}] {title}"
            ref.url = ref.url or url


def scan(root: Path, pathspecs: list[str]) -> tuple[list[Reference], int, int]:
    refs: list[Reference] = []
    files = tracked_files(root, pathspecs)
    scanned = 0
    for path in files:
        any_line = False
        for lineno, cell, line in text_lines(root, path):
            any_line = True
            if "#" in line or "GH-" in line or "github.com" in line:
                refs.extend(scan_line(path, lineno, cell, line))
        scanned += any_line
    return refs, len(files), scanned


def write_baseline(root: Path, refs: list[Reference], pathspecs: list[str], status: str,
                   retain: list[str] = ()) -> int:
    """Replace the baseline entries of the scanned files with what they hold now.

    Entries for files outside the scan are kept untouched, so an area PR only
    rewrites its own part of the file. A reason already recorded for a (path,
    number) survives; new pairs get ``status``. Each ``retain`` item,
    ``PATH:NUMBER:CATEGORY:REASON``, marks one pair as kept on purpose.
    """
    retained = {}
    for item in retain:
        p, number, category, reason = item.split(":", 3)
        if not reason.strip():
            raise SystemExit(f"--retain {item!r}: a retained reference needs a reason")
        retained[(p, number)] = {"status": "retained", "category": category, "reason": reason}
    path = root / BASELINE
    current = load_baseline(root)
    scanned = set(tracked_files(root, pathspecs)) if pathspecs else None
    kept = {p: v for p, v in current.items()
            if scanned is not None and p not in scanned and (root / p).exists()}
    counts: Counter = Counter()
    categories: dict = {}
    for ref in refs:
        if ref.status == "excluded":
            continue
        counts[(ref.path, str(ref.number))] += 1
        categories.setdefault((ref.path, str(ref.number)), ref.category)
    for (p, number), count in sorted(counts.items()):
        old = current.get(p, {}).get(number, {})
        old = {"status": "legacy"} if isinstance(old, int) else old
        old = retained.pop((p, number), old)
        if old.get("status", status) == "legacy":
            entry = count  # a legacy pair is just its count
        else:
            entry = {"count": count, "status": "retained",
                     "category": old.get("category", categories[(p, number)]),
                     "reason": old.get("reason", "")}
        kept.setdefault(p, {})[number] = entry
    if retained:
        raise SystemExit(f"--retain names references the scan did not find: {sorted(retained)}")
    about = ("Issue/PR references known to .github/scripts/audit_issue_refs.py, per path and "
             "number. A bare count is 'legacy': it predates the convention in CONTRIBUTING.md "
             "and is still to be reviewed. An object is 'retained' on purpose, with its reason.")
    # One line per file keeps the diff of an area PR to the files it touched.
    rows = [
        f"  {json.dumps(p)}: "
        + json.dumps(dict(sorted(kept[p].items(), key=lambda kv: int(kv[0]))), ensure_ascii=False)
        for p in sorted(kept)
    ]
    text = ('{\n "schema": 1,\n "about": ' + json.dumps(about) + ',\n "references": {\n'
            + ",\n".join(rows) + "\n }\n}\n")
    path.write_text(text, encoding="utf-8")
    return sum(counts.values())


def summary(refs: list[Reference], total_files: int, scanned: int, commit: str) -> str:
    by_status = Counter(r.status for r in refs)
    by_category = Counter(r.category for r in refs if r.status != "excluded")
    by_area: Counter = Counter()
    for r in refs:
        if r.status != "excluded":
            parts = r.path.split("/")
            by_area[parts[0] if len(parts) < 3 or parts[0] != "vaft" else "/".join(parts[:2])] += 1
    lines = [
        f"Commit: {commit}",
        f"Files tracked: {total_files}, text files scanned: {scanned}",
        f"Detections: {len(refs)}  " + "  ".join(f"{k}={v}" for k, v in sorted(by_status.items())),
        "",
        "References (excluded removed) by category:",
        *[f"  {v:6d}  {k}" for k, v in by_category.most_common()],
        "",
        "References (excluded removed) by area:",
        *[f"  {v:6d}  {k}" for k, v in by_area.most_common()],
    ]
    return "\n".join(lines)


def to_csv(refs: list[Reference]) -> str:
    buffer = io.StringIO()
    names = list(asdict(refs[0]).keys()) if refs else list(Reference.__dataclass_fields__)
    writer = csv.DictWriter(buffer, fieldnames=names, lineterminator="\n")
    writer.writeheader()
    for ref in refs:
        writer.writerow(asdict(ref))
    return buffer.getvalue()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--paths", nargs="*", default=[], help="git pathspecs to scan (default: all)")
    parser.add_argument("--format", choices=("summary", "csv", "json"), default="summary")
    parser.add_argument("-o", "--output", help="write the report here instead of stdout")
    parser.add_argument("--new", action="store_true",
                        help="report only references the baseline does not know")
    parser.add_argument("--strict", action="store_true", help="with --new: exit 1 if any")
    parser.add_argument("--resolve", action="store_true",
                        help="look up each number's title and state with `gh` (cached)")
    parser.add_argument("--write-baseline", action="store_true",
                        help="record the scanned files' references in the baseline")
    parser.add_argument("--retain", action="append", default=[], metavar="PATH:NUMBER:CATEGORY:REASON",
                        help="with --write-baseline: keep this reference on purpose (repeatable)")
    parser.add_argument("--baseline-status", choices=("legacy", "retained"), default="legacy",
                        help="status given to pairs the baseline did not know (default legacy)")
    args = parser.parse_args(argv)

    root = repository_root()
    refs, total_files, scanned = scan(root, args.paths)
    for ref in refs:
        categorize(ref)
    apply_baseline(refs, load_baseline(root))
    commit = head_commit(root)

    if args.write_baseline:
        count = write_baseline(root, refs, args.paths, args.baseline_status, args.retain)
        print(f"Baseline: {count} references recorded for {args.paths or 'the whole tree'}.")
        return 0
    if args.resolve:
        cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "vaft-issue-refs.json"
        resolve_titles(refs, cache)

    report = [r for r in refs if r.status == "new"] if args.new else refs
    if args.format == "csv":
        text = to_csv(report)
    elif args.format == "json":
        text = json.dumps({"commit": commit, "files_tracked": total_files,
                           "files_scanned": scanned, "count": len(report),
                           "references": [asdict(r) for r in report]}, indent=1) + "\n"
    elif args.new:
        text = "\n".join(
            [f"{len(report)} issue/PR reference(s) not in {BASELINE} (commit {commit}):"]
            + [f"{r.path}:{r.line}{' ' + r.cell if r.cell else ''}: #{r.number} [{r.category}] {r.context}"
               for r in report]
            + ["", "Explain the behaviour in place instead, or record a deliberate reference "
                   "with --write-baseline and a reason (CONTRIBUTING.md, 'References and provenance')."]
        ) + "\n"
    else:
        text = summary(refs, total_files, scanned, commit) + "\n"

    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    if args.new and (step_summary := os.environ.get("GITHUB_STEP_SUMMARY")):
        with open(step_summary, "a", encoding="utf-8") as handle:
            handle.write(f"### Issue/PR references\n\n{len(report)} new reference(s) "
                         f"outside the baseline.\n\n")
            for r in report[:200]:
                handle.write(f"- `{r.path}:{r.line}` #{r.number} — {r.context[:120]}\n")
            if len(report) > 200:
                handle.write(f"- … {len(report) - 200} more in the job log (full list above).\n")
    return 1 if (args.new and args.strict and report) else 0


if __name__ == "__main__":
    raise SystemExit(main())

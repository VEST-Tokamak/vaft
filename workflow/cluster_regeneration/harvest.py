#!/usr/bin/env python3
"""Harvest a regen2147 lane: per-shot stage outcomes, failure classes, per-rule wall time, volume.

usage: harvest.py <lane dir> [<lane dir> ...] --out summary.json
Reads only the lane (real paths; products embed /srv/vest.filedb through srvns).
"""
import argparse
import collections
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

TS = re.compile(r"^\[(\w{3} \w{3} +\d+ \d\d:\d\d:\d\d \d{4})\]$")
START = re.compile(r"^(?:local)?(?:rule|checkpoint|localcheckpoint) (\w+):$")
JOBID = re.compile(r"^\s+jobid: (\d+)$")
WILD = re.compile(r"^\s+wildcards: (.*)$")
FIN = re.compile(r"^Finished jobid: (\d+) \(Rule: (\w+)\)")
ERR = re.compile(r"^Error in rule (\w+):")


def parse_log(path):
    """Finished jobs (rule, wildcards, wall) and rule errors of one Snakemake log.

    Job ids restart in every run appended to the same file, so a job is captured when it
    finishes rather than looked up by id at the end.
    """
    running, errors, finished = {}, [], []
    now = cur = None
    lines = path.read_text(errors="replace").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        m = TS.match(line)
        if m:
            now = datetime.strptime(m.group(1), "%a %b %d %H:%M:%S %Y")
        elif (m := START.match(line)):
            cur = {"rule": m.group(1), "start": now, "wild": ""}
        elif cur is not None and (m := JOBID.match(line)):
            running[int(m.group(1))] = cur
        elif cur is not None and (m := WILD.match(line)):
            cur["wild"] = m.group(1)
        elif (m := FIN.match(line)):
            j = running.pop(int(m.group(1)), None)
            if j and j["start"] and now:
                finished.append({**j, "wall": (now - j["start"]).total_seconds()})
        elif (m := ERR.match(line)):
            block = "\n".join(lines[i:i + 12])
            w = re.search(r"wildcards: (.*)", block)
            lg = re.search(r"log: (\S+)", block)
            errors.append({"rule": m.group(1), "wild": w.group(1) if w else "", "log": lg.group(1) if lg else None})
        i += 1
    return finished, errors


def shot_of(wild):
    m = re.search(r"shot=(\d+)", wild or "")
    return int(m.group(1)) if m else None


def classify(text):
    t = text[-6000:]
    for key, pat in [
        ("anti_alias_too_short", r"too few to anti-alias|too short to anti-alias"),
        ("no_plasma_current", r"NoPlasmaCurrentError"),
        ("degenerate_baseline", r"DegenerateBaselineWindow"),
        ("raw_signal_unavailable", r"RawSignalUnavailable"),
        ("unacquired_pf", r"UnacquiredPFCircuit"),
        ("eddy_no_output", r"EddyNoOutput"),
        ("efit_timeout", r"TimeoutExpired|timed out"),
        ("missing_input", r"MissingInput|FileNotFoundError"),
        ("memory", r"MemoryError|Killed"),
        ("value_error", r"ValueError"),
        ("key_error", r"KeyError"),
        ("runtime_error", r"RuntimeError"),
    ]:
        if re.search(pat, t, re.I):
            return key
    last = [row for row in t.splitlines() if row.strip()]
    return "other: " + (last[-1][:120] if last else "empty log")


def real(lane, p):
    return Path(str(p).replace("/srv/vest.filedb", str(lane / "filedb"), 1)) if p else None


def head(path, n=200):
    try:
        return path.read_text(errors="replace")[:n].strip()
    except OSError:
        return None


def du(path):
    try:
        return int(subprocess.run(["du", "-sb", str(path)], capture_output=True, text=True).stdout.split()[0])
    except Exception:
        return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lanes", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    shots, rule_wall, errs, vol = {}, collections.defaultdict(list), [], collections.Counter()
    shot_wall = collections.defaultdict(dict)
    for lane in a.lanes:
        F = lane / "filedb"
        # Wall time from every attempt (each finished job appears in the attempt that ran it);
        # errors from the latest attempt per chunk (<chunk>.snakemake.log) only.
        for log in sorted((lane / "logs").glob("*.attempt-*.log")) or sorted((lane / "logs").glob("*.snakemake.log")):
            finished, _ = parse_log(log)
            for j in finished:
                product = re.search(r"product=([\w-]+)", j["wild"] or "")
                rule = j["rule"] + (f":{product.group(1)}" if product else "")
                rule_wall[rule].append(j["wall"])
                sh = shot_of(j["wild"])
                if sh is not None:
                    shot_wall[sh][rule] = shot_wall[sh].get(rule, 0) + j["wall"]
        for log in sorted((lane / "logs").glob("*.snakemake.log")):
            _, errors = parse_log(log)
            for e in errors:
                lp = real(lane, e["log"])
                text = lp.read_text(errors="replace") if lp and lp.exists() else ""
                errs.append({**e, "shot": shot_of(e["wild"]) or (int(re.search(r"/(\d+)/log/", e["log"] or "").group(1)) if re.search(r"/(\d+)/log/", e["log"] or "") else None), "class": classify(text), "lane": str(lane)})
        # The lane-level checkpoint files hold only the last chunk's verdict; run_lane.sh keeps
        # every chunk's copy beside its shot list.
        pre = F / "pipeline/preflight/metadata"
        elig, impa, excl = set(), set(), {"excluded_shots": []}
        verdicts = [{n: pre / f"{n}.json" for n in ("eligible_shots", "excluded_shots", "impa_shots")}]
        for e in sorted((lane / "chunks").glob("shots.???.eligible_shots.json")):
            verdicts.append({n: e.with_name(e.name.replace("eligible_shots", n))
                             for n in ("eligible_shots", "excluded_shots", "impa_shots")})
        for v in verdicts:
            if v["eligible_shots"].exists():
                elig |= set(json.loads(v["eligible_shots"].read_text())["eligible_shots"])
            if v["excluded_shots"].exists():
                excl["excluded_shots"] += json.loads(v["excluded_shots"].read_text()).get("excluded_shots", [])
            if v["impa_shots"].exists():
                impa |= set(json.loads(v["impa_shots"].read_text()).get("impa_shots", []))
        for f in sorted((lane / "chunks").glob("shots.???")):
            for s in map(int, f.read_text().split()):
                d = {"eligible": s in elig, "impa_selected": s in impa}
                for st, p in [("diagnostics", F / f"omas/diagnostics/{s}/output/diagnostics.json.gz"),
                              ("eddy", F / f"omas/eddy/{s}/output/eddy.json.gz"),
                              ("constraints", F / f"omas/efit/magnetic/{s}/work/constraints.json"),
                              ("efit_ods", F / f"omas/efit/magnetic/{s}/output/efit.json.gz"),
                              ("impa", F / f"omas/impa/{s}/output/impa.json.gz")]:
                    d[st] = p.exists()
                m = F / f"omas/eddy/{s}/metadata/manifest.json"
                if m.exists():
                    try:
                        d["eddy_status"] = json.loads(m.read_text()).get("status")
                    except Exception:
                        pass
                d["efit_status"] = head(F / f"efit/magnetic/{s}/metadata/efit_status.txt")
                d["chease"] = (F / f"omas/chease/magnetic/{s}/output/chease.json.gz").exists()
                d["chease_status"] = head(F / f"chease/magnetic/{s}/metadata/chease_status.txt")
                d["mhd_linear"] = sorted(p.parts[-4] for p in F.glob(f"omas/mhd_linear/magnetic/chease/*/{s}/output/*.json.gz"))
                d["gpec_ideal"] = (F / f"omas/gpec_ideal/magnetic/chease/ideal-gpec/{s}/output/gpec_ideal.json.gz").exists()
                d["gpec_cells"] = {f"{p.parts[-5]}/{p.parts[-3]}": head(p, 160)
                                   for p in F.glob(f"gpec/magnetic/chease/*/{s}/n=*/metadata/status.txt")}
                kd, gd = F / f"efit/magnetic/{s}/kfile", F / f"efit/magnetic/{s}/gfile"
                d["kfiles"] = len(list(kd.glob("k[0-9]*"))) if kd.exists() else 0
                d["gfiles"] = len(list(gd.glob("g[0-9]*"))) if gd.exists() else 0
                b = {"raw": du(F / f"raw/{s}"),
                     "omas": sum(du(x) for pattern in (f"omas/*/{s}", f"omas/*/magnetic/{s}", f"omas/*/magnetic/chease/*/{s}")
                                 for x in F.glob(pattern)),
                     "efit": du(F / f"efit/magnetic/{s}"),
                     "chease": du(F / f"chease/magnetic/{s}"),
                     "gpec": sum(du(x) for x in F.glob(f"gpec/magnetic/chease/*/{s}"))}
                d["bytes"] = b
                d["wall_s"] = shot_wall.get(s, {})
                vol.update(b)
                shots[s] = d
        for entry in excl.get("excluded_shots", []) if isinstance(excl, dict) else []:
            if int(entry.get("shot", -1)) in shots:
                shots[int(entry["shot"])]["excluded_reason"] = (
                    f"{entry.get('reason')}: missing {entry.get('missing_field_codes')}")
    walls = {r: {"n": len(v), "sum_s": round(sum(v)), "median_s": sorted(v)[len(v) // 2], "max_s": max(v)}
             for r, v in sorted(rule_wall.items())}
    summary = {
        "n_shots": len(shots),
        "eligible": sum(d["eligible"] for d in shots.values()),
        "stage_counts": {k: sum(bool(d.get(k)) for d in shots.values())
                         for k in ["diagnostics", "eddy", "constraints", "efit_ods", "impa", "chease", "mhd_linear", "gpec_ideal"]},
        "mhd_linear_products": collections.Counter(x for d in shots.values() for x in d.get("mhd_linear", [])),
        "gpec_cells": collections.Counter(f"{cell.split('/')[0]} {(st or 'none').split(':')[0]}"
                                          for d in shots.values() for cell, st in d.get("gpec_cells", {}).items()),
        "efit_status": collections.Counter((d["efit_status"] or "none").split(";")[0].split(":")[0] for d in shots.values()),
        "eddy_status": collections.Counter(str(d.get("eddy_status")) for d in shots.values()),
        "errors_by_rule_class": collections.Counter(f"{e['rule']} | {e['class']}" for e in errs),
        "rule_wall": walls, "bytes_total": dict(vol),
    }
    a.out.write_text(json.dumps({"summary": summary, "errors": errs, "shots": {str(k): v for k, v in sorted(shots.items())}}, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()

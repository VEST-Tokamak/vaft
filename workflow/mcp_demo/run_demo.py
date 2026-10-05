"""Replay the VAFT MCP conference scenarios through a real stdio session (#188, #1532).

Starts ``python -m vaft.mcp`` as an MCP client would, calls the tools each
scenario needs, and writes every request and response to ``--out`` as JSON,
one file per scenario, plus ``summary.md`` with the headline numbers.  No
language model is involved: the transcripts show exactly what an agent
receives for the same questions.

    python workflow/mcp_demo/run_demo.py --atlas ~/runs/campaign/atlas --out demo-out

Scenarios:

1. ``equilibrium_39915``: equilibrium summary of shot 39915 (packaged sample, offline),
   q95 and beta_N at the slice nearest 0.320 s.
2. ``stability_n1``: Tier A states ideal-unstable at n = 1 with the full edge (stability atlas v2,
   physical layer ``ideal_unstable_full_edge``).
3. ``sat_spread``: surfaces whose total gyro-Bohm flux spreads most across TGLF SAT rules
   (transport sensitivity pairs).
4. ``zeff``: resistive Z_eff windows with status ok, with their conductivity model.

Needs ``vaft[mcp]``.  Scenarios 2-4 need ``--atlas`` (the campaign atlas
directory); without it only scenario 1 runs.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

SCENARIOS = {
    "equilibrium_39915": [
        ("inspect_dataset", {"shot": 39915}),
        ("list_equilibrium_times", {"shot": 39915}),
        ("get_equilibrium_summary", {"shot": 39915, "time": 0.320}),
    ],
    "stability_n1": [
        ("describe_atlas_table", {"name": "stability"}),
        ("query_atlas_table", {
            "name": "stability",
            "where": [{"column": "n_tor", "op": "==", "value": 1},
                      {"column": "ideal_unstable_full_edge", "op": "==", "value": True}],
            "count_by": ["efit_lineage", "efit_quality"],
        }),
        ("query_atlas_table", {
            "name": "stability",
            "where": [{"column": "n_tor", "op": "==", "value": 1}],
            "columns": ["dcon_full_status"],
            "count_by": ["dcon_full_status", "dcon_trunc_status"],
            "limit": 1,
        }),
    ],
    "sat_spread": [
        ("describe_atlas_table", {"name": "transport_sensitivity_pairs"}),
        ("query_atlas_table", {
            "name": "transport_sensitivity_pairs",
            "columns": ["efit_quality", "n_configs", "q_tot_gb_sat_spread_es", "q_tot_gb_sat_spread_em-bper"],
            "order_by": ["-q_tot_gb_sat_spread_em-bper"],
            "limit": 5,
        }),
    ],
    "zeff": [
        ("query_atlas_table", {
            "name": "zeff_windows",
            "where": [{"column": "status", "op": "==", "value": "ok"}],
            "order_by": ["shot", "t_start_s"],
        }),
    ],
}
ATLAS_SCENARIOS = {"stability_n1", "sat_spread", "zeff"}


async def _replay(env: dict[str, str], scenarios: list[str]) -> dict[str, list[dict]]:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    parameters = StdioServerParameters(command=sys.executable, args=["-m", "vaft.mcp"], env=env)
    transcripts: dict[str, list[dict]] = {}
    async with stdio_client(parameters) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            for name in scenarios:
                steps = []
                for tool, arguments in SCENARIOS[name]:
                    result = await session.call_tool(tool, arguments)
                    text = result.content[0].text if result.content else ""
                    try:
                        payload = json.loads(text)
                    except ValueError:
                        payload = text
                    steps.append({"tool": tool, "arguments": arguments, "is_error": bool(result.isError),
                                  "bytes": len(text), "result": payload})
                transcripts[name] = steps
    return transcripts


def _headline(name: str, steps: list[dict]) -> list[str]:
    last = steps[-1]["result"] if steps else {}
    if any(step["is_error"] for step in steps):
        return [f"error: {next(s['result'] for s in steps if s['is_error'])}"]
    if name == "equilibrium_39915":
        row = last["slices"][0]
        return [f"t = {last['matched']['time_s']} s (asked {last['matched']['requested_time_s']} s): "
                f"Ip = {row['ip_kA']:.1f} kA, q95 = {row['q_95']:.2f}, beta_N = {row['beta_normal']:.3f}, "
                f"li_3 = {row['li_3']:.2f}, kappa = {row['elongation']:.2f}"]
    if name == "stability_n1":
        unstable = steps[1]["result"]
        lines = [f"{unstable['total_matched']} state(s) ideal-unstable at n = 1 (full edge):"]
        lines += [f"- {r['shot']} t = {r['time_efit_s']} s {r['efit_lineage']} ({r['efit_quality']}): "
                  f"full {r.get('dcon_full_status')}, truncated {r.get('dcon_trunc_status')}" for r in unstable["rows"]]
        lines.append(f"n = 1 DCON full-edge statuses: {last['counts']['dcon_full_status']}")
        return lines
    if name == "sat_spread":
        return [f"- {r['shot']} t = {r['time_efit_s']} s {r['efit_lineage']} r/a = {r['r_over_a']}: "
                f"spread em-bper {r['q_tot_gb_sat_spread_em-bper']:.2f}, es {r['q_tot_gb_sat_spread_es']:.2f}"
                for r in last["rows"]]
    if name == "zeff":
        return [f"- {r['shot']} {r['t_start_s']}-{r['t_end_s']} s ({r['window_class']}): "
                f"Z_eff = {r['zeff']:.2f} +/- {r['zeff_uncertainty']:.2f} [{r['conductivity_model']}]"
                for r in last["rows"]]
    return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--atlas", type=Path, help="campaign atlas directory (VAFT_ATLAS_DIR)")
    parser.add_argument("--out", type=Path, required=True, help="directory for the JSON transcripts")
    parser.add_argument("--only", nargs="*", choices=sorted(SCENARIOS), help="run these scenarios only")
    args = parser.parse_args(argv)

    env = {k: v for k, v in os.environ.items() if not k.startswith("VAFT_")}
    env.setdefault("MPLBACKEND", "Agg")
    scenarios = list(args.only or SCENARIOS)
    if args.atlas:
        env["VAFT_ATLAS_DIR"] = str(args.atlas.expanduser().resolve())
    else:
        scenarios = [s for s in scenarios if s not in ATLAS_SCENARIOS]

    transcripts = asyncio.run(_replay(env, scenarios))
    args.out.mkdir(parents=True, exist_ok=True)
    summary = ["# VAFT MCP demo transcripts", ""]
    for name, steps in transcripts.items():
        (args.out / f"{name}.json").write_text(json.dumps(steps, indent=1, allow_nan=False) + "\n", encoding="utf-8")
        summary += [f"## {name}", "", *[f"`{s['tool']}({json.dumps(s['arguments'])})`: {s['bytes']} bytes"
                                         for s in steps], "", *_headline(name, steps), ""]
    (args.out / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    print("\n".join(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

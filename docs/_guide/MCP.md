---
title: MCP server for agents
author: VEST team
date: 2026-09-30 10:00
category: guide
layout: post
permalink: /reference/mcp/
guide:
  architecture: A thin, local, read-only Model Context Protocol adapter (vaft.mcp) over VAFT's existing discovery and extraction APIs (issue 1423).
  prerequisites: pip install 'vaft[mcp]' and an MCP client such as Claude Code or Codex.
  expected: An agent that can discover VAFT's formulas, processes, checks, plots and samples, read bounded numbers and equilibrium summaries from a shot or a local artifact, and query the campaign atlas tables with each lane's rules attached.
related:
  api: [formula, plot]
---

`vaft.mcp` lets an MCP client (Claude Code, Codex or any other) ask VAFT what it can do and read
small numerical views of the packaged reference data. You bring the agent; VAFT runs locally, in
your own environment, over the stdio transport.

The server is an adapter, not a second API. Each tool calls an existing VAFT function, such as
`vaft.help()`, the formula and process catalogs, the validation registry, `vaft.plot.available_plots`,
`vaft.plot.dd`, `vaft.plot.extract` or `vaft.data.sample_manifest`, and returns the answer as
bounded JSON.

## Install and register

```bash
pip install 'vaft[mcp]'
claude mcp add vaft -- python -m vaft.mcp     # Claude Code; `vaft mcp` is the same server
```

Any client that launches a stdio server can run `python -m vaft.mcp` the same way. Use the Python
interpreter of the environment where VAFT is installed. `import vaft` never needs the extra: only
`vaft.mcp.server.build_server()` imports the MCP SDK.

Do not start the server with the `vaft/` package directory as the working directory. Python puts the
working directory first on `sys.path`, so `vaft/mcp/` would shadow the SDK's own `mcp` package.
The repository root or any other directory is fine.

## Tools

| Tool | Answers from |
| --- | --- |
| `get_capabilities`, `get_capability(topic, item)` | `vaft.help()`: topics, defaults, sections, entry points |
| `search_formulas(text, category, limit)`, `describe_formula(name)` | `vaft.formula.catalog` (units, definitions, references; not the source code) |
| `search_processes(text, category, limit)`, `describe_process(name)` | `vaft.process.catalog` |
| `list_validation_checks(category)`, `describe_validation_check(key)` | `vaft.validation.registry` |
| `list_plots(query, domain, subject, view, status, limit)`, `describe_plot(name)` | `vaft.plot.available_plots` |
| `get_plot_requirements(name)` | `vaft.plot.dd(name)`: the Data Dictionary paths a plot reads |
| `extract_plot_data(name, shot, max_points, options)` | `vaft.plot.extract` on a packaged reference shot |
| `list_samples`, `describe_sample(shot)` | `vaft.data.available_samples` / `sample_manifest` |
| `list_boundaries(family)`, `describe_boundary(key)` | `vaft.formula.boundaries` (metadata only) |
| `inspect_dataset(shot \| artifact)` | which IDS a dataset holds, with time counts and ranges, plus provenance |
| `inspect_data_path(path, shot \| artifact, time, tolerance, max_points)` | one Data Dictionary path, read without creating it (`vaft.ods_access`) |
| `list_equilibrium_times(shot \| artifact)` | the equilibrium slice times |
| `get_equilibrium_summary(shot \| artifact, time, tolerance, limit)` | the `equilibrium_global` summary columns of `vaft.database.summary` (Ip, q95, β_N, li_3, W_mhd, shape, ...) |
| `list_atlas_tables`, `describe_atlas_table(name)` | the campaign atlas registry: each lane's table, schema, rules, caveats and README |
| `query_atlas_table(name, where, columns, order_by, count_by, limit)` | rows of one atlas table, filtered by value |

`extract_plot_data` returns the plot's view model, with labels and units as VAFT stores them. Each
array comes back as its shape, dtype, and finite minimum and maximum over all values, plus a
strided preview. The preview is `array[::stride]`, a transport bound, not a resampling.
`max_points` is a budget for the whole result, shared across its arrays, so a panel overview with
hundreds of arrays gets a few points per array. A result that would exceed about 50 kB keeps array
statistics only. `options` takes VAFT's extraction options only; the tool description lists the
allowed keys.

Every result carries `truncated`: the `count` of places something was shortened (a list cut to
`limit`, a long string, an array preview) and the first 20 of those `paths`. Shot 39915 ships in the
wheel; the other samples load from a Git checkout.

Tools run one at a time in a worker thread, so a slow extraction does not stall the protocol. The
server keeps the JSON-RPC stream on a private copy of standard output and points file descriptor 1
at stderr, so output printed by Python, C or Fortran code VAFT calls cannot corrupt the stream.

## Read-only by construction

Every tool is annotated `readOnlyHint`. None of them writes a file, publishes to the database,
reaches HSDS or the network, runs EFIT, CHEASE or any other solver or pipeline, or executes
caller-supplied code or shell commands. Results never carry credentials: help pages say whether
HSDS is configured, not with what, and the server also scrubs the values of `HS_PASSWORD` and
`HS_API_KEY` and replaces the home directory with `~` in every result and error message.
Tool inputs name scientific things: a plot, a formula or
a shot. They never name a data representation. During the DD migration (#1127, #1132, #1135),
the loading code can change behind the same tool schema.

## Datasets

A dataset is named by `shot` or by `artifact`, never by a storage format:

| Name | Read from | When |
| --- | --- | --- |
| `shot=39915` | the packaged sample (`vaft.omas.sample.sample_ods`) | always; offline |
| `artifact="run1/g039915.00319"` | a g-file, OMAS JSON, IMAS netCDF or IMAS HDF5 file or directory under `VAFT_ARTIFACT_DIR`, through `vaft.omas.load` | when `VAFT_ARTIFACT_DIR` is set |
| `shot=42929`, or any shot with `database_source="main"` | `vaft.database.load`, with your own HSDS configuration | only when the server runs with `VAFT_MCP_DATABASE=1` |

Every result carries the dataset's provenance: its kind, the shot or the artifact path relative to
its directory, the loader, the Data Dictionary version when the data records one, and the VAFT
version. Database failures report the exception class only, never the server's text.

Slices are matched by time, never by index. `get_equilibrium_summary(time=...)` and
`inspect_data_path("equilibrium.time_slice.*.global_quantities.q_95", time=...)` take the slice
nearest that time. They refuse when it is further away than `tolerance` (default: half the slice
spacing), and they report the time they matched.

## Atlas tables

The campaign atlas is the directory the lanes write their tables to (`~/runs/campaign/atlas` on
vestserver). Point `VAFT_ATLAS_DIR` at it. The server knows each table by name, together with
the lane that owns it, the schema that describes its columns, and the rules for reading it:

| Table | Lane | One row is |
| --- | --- | --- |
| `state`, `kinetic_profiles` | K (#1454) | an equilibrium state (shot, time_efit_s, efit_lineage), or one Thomson channel of it |
| `transport`, `transport_summary` | T (#1453) | a state and r/a (one TGLF configuration), or one state |
| `transport_sensitivity`, `transport_sensitivity_pairs` | T | a surface and a TGLF configuration, or the spread over configurations |
| `stability`, `stability_surfaces` | N (#1448) | a state and toroidal mode number, or one rational surface for one solver |
| `zeff_windows`, `zeff_slices` | Z (#1486) | a discharge window, or a state |
| `confinement` | D (#1490) | a magnetics state |
| `op_space_base` | V (#1456) | a state's operating-space coordinates |

`describe_atlas_table` returns each column's unit and definition from the lane's own schema, the
row key, the lane's rules and caveats, its README, and provenance: the file's sha256 and the build
commit. Read it before comparing rows.

`query_atlas_table` filters with `where` clauses (`{"column", "op", "value"}`; `==`, `!=`, `<`,
`<=`, `>`, `>=`, `in`, `not_in`, `isnull`, `notnull`), selects `columns`, sorts with `order_by`
(`"-name"` for descending) and counts rows per value with `count_by`. It does not join tables and
does not aggregate beyond counting. A lane's "never combine" rules are refusals:

- `stability` needs `n_tor ==`.
- `stability_surfaces` needs `n_tor ==` and `solver ==`.
- `transport_sensitivity` needs `tglf_config ==`.

Older spellings (`efit_label`, `magnetics-only`, `electron-kinetic`) are accepted and returned in
State key contract v1 spelling (`efit_quality`, `magnetics`, `electron_kinetic`). Tables are
read-only, limited to 50 MB, and every result stays under about 50 kB.

## Environment

| Variable | Effect |
| --- | --- |
| `VAFT_ATLAS_DIR` | the campaign atlas directory; the atlas tools refuse without it |
| `VAFT_ARTIFACT_DIR` | the directory `artifact=` paths are relative to; nothing outside it is read |
| `VAFT_MCP_DATABASE=1` | allow database shots (read-only, through your own HSDS configuration) |

```bash
claude mcp add vaft \
  -e VAFT_ATLAS_DIR=$HOME/runs/campaign/atlas \
  -e VAFT_ARTIFACT_DIR=$HOME/runs/artifacts \
  -- python -m vaft.mcp
```

A client configured by file (Claude Desktop, a project `.mcp.json`) takes the same command:

```json
{
  "mcpServers": {
    "vaft": {
      "command": "/path/to/env/bin/python",
      "args": ["-m", "vaft.mcp"],
      "env": {"VAFT_ATLAS_DIR": "/path/to/campaign/atlas"}
    }
  }
}
```

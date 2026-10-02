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
  expected: An agent that can discover VAFT's formulas, processes, checks, plots and samples, and read bounded numbers from the packaged reference shot.
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
| `get_atlas_summary(path, group_by, limit)` | a `.csv`/`.parquet` table under `VAFT_ATLAS_DIR` |

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

## Atlas tables

`get_atlas_summary` reads tables only from the directory in the `VAFT_ATLAS_DIR` environment
variable of the server process, and it refuses to run when that variable is unset. `path` must be
relative to that directory. Absolute, drive and UNC paths and `..` components are refused before
any file is touched. The path is then resolved, symlinks included, and refused if it lands outside
the directory. Every refusal gives the same message. Tables are limited to 50 MB and `group_by` to
three columns. The result gives the columns (first 200), the row count, the row counts per value
of each `group_by` column (default `efit_quality` and `efit_lineage`) and the first `limit` rows,
with cell text cut to 200 characters.

```bash
claude mcp add vaft -e VAFT_ATLAS_DIR=/path/to/atlas -- python -m vaft.mcp
```

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
array comes back as its shape, dtype, finite minimum and maximum over all values, and a strided
preview of at most `max_points` values. The preview is `array[::stride]`, a transport bound, not a
resampling. Lists are capped too, and every result names what was cut in `truncated`. Shot 39915
ships in the wheel; the other samples load from a Git checkout.

## Read-only by construction

Every tool is annotated `readOnlyHint`. None of them writes a file, publishes to the database,
reaches HSDS or the network, runs EFIT, CHEASE or any other solver or pipeline, or executes
caller-supplied code or shell commands. Tool inputs name scientific things: a plot, a formula or
a shot. They never name a data representation. During the DD migration (#1127, #1132, #1135),
the loading code can change behind the same tool schema.

## Atlas tables

`get_atlas_summary` reads tables only from the directory in the `VAFT_ATLAS_DIR` environment
variable of the server process, and it refuses to run when that variable is unset. Paths are
resolved, symlinks included, and any path outside that directory is refused. The result gives the
columns, the row count, the row counts per value of each `group_by` column (default `efit_quality`
and `efit_lineage`) and the first `limit` rows.

```bash
claude mcp add vaft -e VAFT_ATLAS_DIR=/path/to/atlas -- python -m vaft.mcp
```

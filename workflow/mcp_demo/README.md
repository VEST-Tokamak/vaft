# VAFT MCP conference demo (#188, log #1532)

These scenarios show an agent reading VAFT's equilibrium, stability, transport and Z_eff results
through the read-only MCP server (`python -m vaft.mcp`).

## Replay without a language model

```bash
pip install 'vaft[mcp]'
python workflow/mcp_demo/run_demo.py --atlas ~/runs/campaign/atlas --out demo-out
```

The script starts the stdio server the way an MCP client does. It calls the tools for each scenario
and writes `demo-out/<scenario>.json` (every request and the full response) and `demo-out/summary.md`.
Without `--atlas`, only the packaged-sample scenario runs.

## Live, with an agent

```bash
claude mcp add vaft -e VAFT_ATLAS_DIR=$HOME/runs/campaign/atlas -- python -m vaft.mcp
```

Then ask, for example:

1. "What are the equilibrium summary, q95 and β_N of shot 39915 at 320 ms?"
2. "Which Tier A states are ideal-unstable at n = 1? Respect the stability atlas rules."
3. "On which surfaces does the TGLF SAT rule change the total flux the most?"
4. "What resistive Z_eff values are identified, and with which conductivity model?"

The agent should call `describe_atlas_table` before querying. The stability table refuses a query
that does not pin `n_tor`.

## Reading the numbers

- Shot 39915 in scenario 1 is the packaged sample, which carries its own EFIT reconstruction. The
  Tier A atlas uses the `statistical_891` EFIT setting, so its 39915 states have different q95 and
  stored energy. Compare a shot within one source.
- Stability, transport and Z_eff values are the lane products as written (N #1448, T #1453,
  Z #1486). The MCP server adds no computation, joins or aggregation beyond row counts.

---
title: Browser GUI
author: VEST team
date: 2026-09-30 09:00
category: guide
layout: post
permalink: /workflows/gui/
guide:
  architecture: An optional Panel application over the same loading, discovery and plotting APIs a notebook uses.
  prerequisites: VAFT installed with the `gui` extra; a browser on the machine you sit at.
  expected: The plot browser at http://localhost:5006, locally or through a forwarded port.
  status: Experimental reference application (#1086); workspaces follow the GUI roadmap (#1359).
related:
  api: [omas, database, plot]
  data_sources: [sample-ods, hsds-public]
---

`vaft gui` serves a small browser application: pick a data source, pick one of the plots VAFT
can draw from it, and change that plot's controls. It is a presentation layer, not a separate
analysis tool. The plot list comes from `vaft.omas.available_plots`, the controls from the
plot's capability record, and every drawing from `vaft.omas.render_plot`, so what the GUI
shows is what the same calls give in a notebook.

The application runs where VAFT and the data are, and the browser only displays it. The same
command therefore works on a workstation, on a remote server over SSH, and on a cluster node,
with no X11 or remote desktop.

## Install

Panel is an optional dependency:

```bash
python -m pip install -e ".[gui]"
```

Without it, `import vaft` and every other workflow are unaffected; `vaft gui` stops with a
message naming the extra.

## Run locally

```bash
vaft gui                          # first packaged sample, opens a browser
vaft gui --sample 39915 41524     # two samples, compared on each plot
vaft gui --file equilibrium.json
vaft gui --shot 39915 41524       # database shots (needs HSDS read access)
vaft gui --plot equilibrium_field_psi --port 5010
```

The page shows the source picker and the plot selector in the sidebar, the plot's controls
beneath them, and the figure in the main area. Slices, channels, units and overlays appear
only for plots that offer them. A value a plot cannot draw leaves the previous figure on screen
and shows the reason above it.

- **Files.** Type server paths (one per line), pick them with **Browse server files** (the files
  of the machine the GUI runs on), or upload them from your own computer. Uploads are copied to a
  temporary folder on the server for the session (at most 512 MB per upload); select an IMAS
  entry's `master.h5` together with its per-IDS files to upload the entry.
- **Database shots.** Nothing is downloaded up front. Loading a shot lists the IDS it stores
  (`vaft.database.stored_ids`) and judges the plots from that list; each plot then fetches only the
  IDS it reads, the first time it is drawn, and keeps them in memory for every later plot and
  control change. The status line names the IDS in memory, and **Load chosen IDS** fetches IDS
  ahead of any plot. Reading needs HSDS read access to the shot's per-IDS domains.
- **Several shots.** Pick several samples, or type several database shots (`39915, 41524`); every
  plot then draws them together, labelled by shot.
- **Interactive or static.** Plots that have a Plotly rendering open interactive: box zoom, pan,
  scroll zoom, hover read-out, spike lines, and drawing tools to mark lines, regions and shapes.
  Zoom survives a change of slice or unit. **Static** switches to the Matplotlib image, and the
  choice is remembered for the next plot.
- **Figure.** Width and height in pixels, axis limits and log scales; empty fields stay automatic.
- **Export.** Downloads the current plot as PNG, SVG or PDF at the chosen DPI. The file is the
  Matplotlib rendering with the controls and figure settings on screen, whichever renderer is shown.

Loading a source computes its plot catalog, which takes a few seconds the first time.

## Run on a remote host over SSH

The server binds to `127.0.0.1` by default, so it is not reachable from other machines. Forward
the port instead:

```bash
ssh -L 5006:localhost:5006 user@server
vaft gui --no-show            # on the server
```

Then open <http://localhost:5006> on your own machine. `vaft gui` does not try to open a browser
when it runs inside an SSH session. Keep the local port equal to the server's: the page's
connection is only accepted from `localhost:<port>` of the server. To forward to another local
port, name it: `ssh -L 8080:localhost:5006 ...` with `vaft gui --allow-websocket-origin localhost:8080`.

Under SSH the page asks for a password, which `vaft gui` prints when it starts (any user name is
accepted). Set `VAFT_GUI_PASSWORD` to choose it yourself; see [Access](#access) for why.

## Run in VS Code Remote-SSH

Run `vaft gui` in the VS Code terminal of the remote window. VS Code detects the port and
forwards it; open it from the notification or the **Ports** panel. If VS Code forwards to a
different local port because 5006 is taken on your machine, the page stays blank: restart with
`--allow-websocket-origin localhost:<that port>`, or pick a free `--port` on both ends.

## Run on a cluster node

Start the GUI on a compute node (inside an interactive allocation), then tunnel to that node
through the login node. The tunnel must end on the compute node itself, because the server
listens only on that node's loopback:

```bash
# on the compute node
vaft gui --no-show --port 5006

# on your machine
ssh -J user@login-node -L 5006:localhost:5006 user@<compute-node>
```

If compute nodes do not accept SSH from the login node, ask the site how interactive jobs expose
ports; do not bind `--address 0.0.0.0` on a shared cluster.

## Access

Loopback keeps other *machines* out, not other *users*: on a login or compute node shared with
others, anyone logged in to that node can connect to `127.0.0.1:5006`. The application reads
files and database data with your permissions, so `vaft gui` protects it with a password whenever
it runs under SSH or binds another address (`--auth auto`, the default). Use `--auth password` to
require it locally too, and `--auth none` only on a machine nobody else uses.

Binding another address (`--address 0.0.0.0`) also prints a warning: the connection is not
encrypted. Multi-user deployment is a separate concern from this page, and belongs with the
database portal (#960).

## What comes next

The browser application is Track A of the GUI roadmap (#1359). An application shell and a
fuller plot explorer follow, then workspaces for routine diagnostics, equilibrium,
fluctuations and stability, operational space, start-up and pipeline monitoring.

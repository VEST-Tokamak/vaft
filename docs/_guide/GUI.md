---
title: Browser GUI
author: VEST team
date: 2026-09-30 09:00
category: guide
layout: post
permalink: /workflows/gui/
guide:
  architecture: A Panel application over the same loading, discovery and plotting APIs a notebook uses.
  prerequisites: VAFT installed; a browser on the machine you sit at.
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

Nothing beyond VAFT itself: Panel is one of its dependencies, so `vaft gui` works in any VAFT
environment. `import vaft` does not import Panel; only launching the GUI does. (Panel used to be
the optional `gui` extra; `pip install 'vaft[gui]'` still works and adds nothing.)

## Run locally

```bash
vaft gui                          # first packaged sample, opens a browser
vaft gui --sample 39915 41524     # two samples, compared on each plot
vaft gui --file equilibrium.json
vaft gui --shot 39915 41524       # database shots (needs HSDS read access)
vaft gui --plot equilibrium_field_psi --port 5010
vaft gui --workspace diagnostics  # start in another workspace (plots, diagnostics, equilibrium, database)
```

The page is an application shell. The sidebar starts with the **workspaces**:

- **Plots:** the plot explorer described below.
- **Diagnostics:** the same explorer narrowed to one diagnostic.
- **Equilibrium:** equilibrium plots by mode with one shared time slice, and validation verdicts.
- **Database:** the database sources, the connection, and opening database shots.

A strip above the main area shows the shared selection, which every workspace reads: what is open, the time on
screen (for a plot with a slice or time control), the database namespace and the state of the
database connection. Errors from any workspace appear under that strip.

In the **Plots** workspace, the sidebar holds the source picker, the plot selector and the
plot's controls, and the figure fills the main area. Slices, channels, units and overlays appear
only for plots that offer them. A value a plot cannot draw leaves the previous figure on screen
and shows the reason above it.

The plot selector is built from plot discovery (`available_plots`), not from a list kept in the
GUI:

- **Grouping and labels.** Plots are grouped by subject, with the subject's aliases, and labelled
  by view and quantity (`time / current` under `plasma_current [ip, ...]`). The canonical plot
  name stays the plot's identity and is shown under **About this plot**.
- **Search.** The box above the selector narrows the list. It uses the registry's own query, so
  `ip` finds the plasma current, plus a plain text match on names, labels and subjects. The plot
  on screen stays drawn while you search.
- **Available / All supported.** **Available** lists what the open source can draw. **All
  supported** adds every other plot VAFT has for this kind of data, marked *(unavailable)*.
  Choosing one draws nothing and shows discovery's reason, for example the missing IDS path.
  With several database shots, a plot is available only when every shot can draw it, and the
  reason names the shot that cannot.
- **About this plot.** The discovery record behind the plot: description, canonical name and
  function, IDS read, backends, controls and interaction modes.

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
- **Playback.** A plot with a slice or time control -- equilibrium slices, camera frames -- gets a
  player beneath it: play, pause, step, loop, and the frame interval. Each frame is the plot
  redrawn on the server with every other control as chosen, so a frame that cannot be drawn is
  reported and skipped. Writing the sequence to a video file is not part of the GUI yet (#1049/#1050).
- **Figure.** Width and height in pixels, axis limits and log scales; empty fields stay automatic.
- **Export.** Downloads the current plot as PNG, SVG or PDF at the chosen DPI. The file is the
  Matplotlib rendering with the controls and figure settings on screen, whichever renderer is shown.

Loading a source computes its plot catalog, which takes a few seconds the first time.

The **Diagnostics** workspace shows processed diagnostics by diagnostic, not by SQL field:

- **Choosing a diagnostic.** Plasma current, flux loops, B-pol probes, Thomson scattering and
  so on, grouped by category. The list is the diagnostic registry, the same source as the
  diagnostics table in these docs; a diagnostic with no plots is not listed.
- **Its plots.** They are the discovery records whose required data lies under the
  diagnostic's IDS path. The explorer below is the one from **Plots**, narrowed to them: the
  open shots compare on each plot, and controls, figure options and export work the same.
- **About this diagnostic.** The registry record: processed IDS path, family, availability,
  mapping status, measured and derived quantities, and the recorded source.
- **Not yet available.** Raw-versus-processed comparison and raw field inspection need an API
  that names each diagnostic's raw DAQ fields; they come when that API lands.

The **Equilibrium** workspace inspects reconstructed equilibria, after Tutorial 03:

- **One time slice.** Pick a slice on any plot that has one. Every other equilibrium plot you
  open shows the same slice, and the status strip names it.
- **Modes.**
  - **Inspect:** the 2-D state (flux map, boundary) and 1-D profiles.
  - **Constraints & fit:** constraints, their coverage and weights, residuals and convergence.
  - **Time evolution:** global quantities across the discharge.
  - **Quality:** fit-quality plots and table.

  The plots come from plot discovery. A plot no mode claims is shown under Inspect.
- **Validation verdicts.** In Quality, **Check this slice** (or **Check all slices**) runs
  `vaft.validation.equilibrium.validate_equilibrium` on each open shot. It shows the verdict of
  every check (verification, diagnostic fit, physical validity, independent validation) with its
  reason, exactly as the validation layer states it.
- **Several shots** compare on each plot as in **Plots**. Nothing in this workspace edits or
  reruns a reconstruction.

The **Database** workspace contains:

- **Namespaces.** A table of the namespaces a shot can be read from: what each holds, whether
  VAFT may write to it, and whether it covers every shot or only the shots its product was made
  for. It comes from `vaft.database.sources`.
- **Opening shots.** Pick a namespace, type shots and press **Open in Plots**. The plot explorer
  opens them and becomes the active workspace.
- **Credentials.** The HSDS configuration h5pyd will use: the file, the endpoint and the
  username. Environment variables override the file. Passwords and API keys show only as
  *configured* or *not set*. The GUI never shows, logs or stores a secret; change the
  configuration with `vaft hsds configure` in a terminal.
- **Test connection.** Asks the server whether it is ready, and puts the answer in the status
  strip. The page does not contact the server until you press it.

### Adding a workspace

Workspaces are registered rather than built in, and the later domain workspaces (#1359) plug
in the same way:

```python
from vaft.gui import register_workspace

class EquilibriumWorkspace:
    def __init__(self, shell):           # shell.selection, shell.report, shell.show
        self.shell = shell
    def sidebar(self): return [...]      # Panel objects
    def main(self): return [...]
    def activate(self): ...              # optional: called each time it is shown
    def close(self): ...                 # optional: called when the browser session ends

register_workspace("equilibrium", "Equilibrium", EquilibriumWorkspace, order=30)
```

A workspace is built the first time it is shown. It reads and changes the shared selection
through `shell.selection` (a `vaft.gui.SelectionState`), and draws through the public VAFT APIs
like every other workspace.

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
encrypted. To serve a team, put the GUI behind a proxy with HTTPS instead, as below.

## Host it for a team

`vaft gui --hosted` serves people who are not the server's user, behind a reverse proxy that
terminates HTTPS. What changes against a personal `vaft gui`:

- **Samples and database shots only.** The file source, the server file browser and uploads are
  left out of the page, and a file source is refused even if one is sent. A reader cannot reach
  the server's disk through the GUI.
  The Database workspace does not show the server's own HSDS configuration (file, endpoint,
  account name) either.
- **Every reader signs in.** With `--auth hsds`, readers sign in with their own HSDS account:
  the user name and password are checked against the HSDS the GUI reads from (`GET /about`,
  which answers 401 to a wrong account) and kept nowhere. Otherwise `--hosted` asks one shared
  password, which must be set in `VAFT_GUI_PASSWORD` (a random one would change unseen on every
  restart). `--auth none` is for a proxy that authenticates by itself.
- **The proxy's headers are trusted** (`X-Forwarded-For`, `X-Forwarded-Proto`), and `--prefix`
  serves the app under a path, so it can sit next to another service on the same host.

Whoever signs in, everyone reads the database with the credentials the service runs with:
`--auth hsds` decides who may enter, not what they may read. Give it a **read-only
HSDS account** through `HS_ENDPOINT`, `HS_USERNAME` and `HS_PASSWORD`, never an admin one.

The files in [`vaft/deploy/gui/`](https://github.com/VEST-Tokamak/vaft/tree/develop/vaft/deploy/gui)
are a working starting point for Ubuntu with nginx and systemd, serving
`https://<host>/gui/` next to HSDS on the same host. They ship with the package, so an
install from PyPI has them too: `importlib.resources.files("vaft.deploy.gui")` is the
installed directory.

| File | Where it goes |
| --- | --- |
| `vaft-gui.service` | `/etc/systemd/system/`; runs `vaft gui --hosted --prefix /gui` as an unprivileged `vaft-gui` user with systemd sandboxing and a memory cap |
| `vaft-gui.env.example` | `/etc/vaft-gui.env` (mode 0600); the page password and the HSDS read account |
| `nginx-vaft-gui.conf` | the nginx site; keeps HSDS on port 80 as before, serves `/gui/` over HTTPS only, and throttles the login form |
| `vaft-gui-proxy.conf` | `/etc/nginx/snippets/`; the websocket proxy settings the site includes |

The service's `--allow-websocket-origin` must name the host the browser opens (without a port
for the default 80/443). The page loads Panel's fonts and assets from public CDNs, so readers
need internet access; the server does not.

Hosted, one process holds every reader's session, and each session keeps the IDS it has read in
memory until its tab closes. `MemoryMax` in the unit bounds the whole service.

## What comes next

The browser application is Track A of the GUI roadmap (#1359). A fuller plot explorer follows,
then workspaces for routine diagnostics, equilibrium,
fluctuations and stability, operational space, start-up and pipeline monitoring.

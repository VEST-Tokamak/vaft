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
  status: Experimental reference application; the further domain workspaces are not built yet.
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
  reported and skipped. **Export** also writes the sequence as an MP4, WebM or GIF movie (MP4 and
  WebM need PyAV, `pip install vaft[video]`), at most 600 frames; a longer one takes every n-th state or
  is written in Python with `plot_*(..., animation=True).save()`. The frame times download alongside it.
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

Workspaces are registered rather than built in, and the later domain workspaces plug
in the same way:

```python
from vaft.gui import register_workspace

class MyWorkspace:
    def __init__(self, shell):           # shell.selection, shell.report, shell.show
        self.shell = shell
    def sidebar(self): return [...]      # Panel objects
    def main(self): return [...]
    def activate(self): ...              # optional: called each time it is shown
    def close(self): ...                 # optional: called when the browser session ends

register_workspace("my_workspace", "My workspace", MyWorkspace, order=90)
```

A workspace is built the first time it is shown. It reads and changes the shared selection
through `shell.selection` (a `vaft.gui.SelectionState`), and draws through the public VAFT APIs
like every other workspace.

## Phones and tablets

The GUI is one web page for every screen. There is no separate mobile app, and nothing else to
install.

- **Desktop:** the controls stay in a sidebar beside the figure.
- **Tablet** (narrower than 1100 px): the sidebar narrows to 300 px so the figure keeps room.
- **Phone** (narrower than 768 px, or under 500 px tall and 950 px wide: a phone held sideways, or a very small desktop window):
  - The controls become a drawer over the page, and the page opens with the drawer closed, so
    the figure and the status line are the first thing you see.
  - The ☰ button in the header opens the drawer to change the workspace, source, plot or
    figure options; press it again to see the figure.
  - Wide tables, such as the database namespaces, scroll sideways inside their own box instead
    of widening the page.

Every control is reachable at every width; nothing is hidden on small screens. The rules live
in `vaft.gui.layout`, and a new workspace gets them by using `vaft.gui.layout.page`, as the
shell does.

### Touch

On a phone or tablet the page renders at the device's own width (it declares a `viewport`), and
interactive plots start in a touch-friendly mode:

- **Drag on the plot** (under the renderer choice, for a single plot) decides what a one-finger drag over an
  interactive plot does: *Zoom box*, *Pan* or *Scroll the page*. Phones and tablets start on
  *Scroll the page*, so a plot as wide as the screen never traps the reader; desktops start on
  *Zoom box*. The choice is recognised from the browser's User-Agent and only sets the default.
  Switching keeps the current zoom. A composition uses the default; use its mode bar to zoom or
  pan.
  An iPad that asks for desktop sites starts on *Zoom box*; change it in the list.
- To zoom in on a touch screen, choose *Zoom box* and drag over the region you want, or use the
  zoom buttons in the plot's mode bar, which stays shown on a touch screen. To zoom back out, use the mode bar's *Autoscale* or
  *Reset axes*. A double tap does not reset the zoom, and Plotly has no two-finger pinch zoom on
  its 2-D plots.
- Legends move underneath the plot on a touch screen, so they do not take the plot's width.
  *Figure options* can still hide the legend.
- A static (Matplotlib) image is scaled to the screen width, so its text gets small on a phone.
  Pinch-zoom the page or turn the phone sideways, or switch to the interactive renderer when
  the plot has one.

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

### Sessions

Every browser tab is its own session on the server.

- **Isolation.** A session has its own shell, workspaces, loaded data, selection, time slice and
  uploads. What one reader opens or changes is never visible to another. Sessions share
  code, the list of workspaces and the server process's settings. (**Test connection** does not
  change the process-wide logging level; it only quietens the `h5pyd` and `urllib3` loggers, and
  only when they were not configured.)
- **Clean-up.** When a tab is closed, the server discards its session once Bokeh's
  unused-session lifetime has passed: 15 s, checked every 17 s, so within about half a minute.
  Discarding a session closes its shell, which releases its figures, loaded data and uploaded
  files.
- **Not limited yet.**
  - A tab left open keeps its session, and its data, until it is closed.
  - There is no idle timeout, per-session memory cap or concurrency limit.
  - These limits will be set from measurements on the hosted test instance (Open: #1862, #1863), not
    guessed in advance.
- **Shared data account.** With `--auth hsds`, sign-in only decides who may open the page. Every
  session reads data with the service's own HSDS credentials, so the GUI does not apply each
  user's HSDS permissions.

### Validate a test instance before team access

A hosted GUI is first validated as an isolated **test instance** next to any production one. It
gets its own systemd unit, user, venv, environment file, port (5016) and URL prefix
(`/gui-test/`). Nothing in the production unit, site or HSDS accounts changes, and every step on
the host needs the maintainer's approval of the concrete configuration first.

1. **Service account.** Create a dedicated HSDS login (for example `gui-test-reader`) through
   the server's HSDS account procedure.
   - Adding a login means restarting the HSDS stack, which is a production change.
   - Grant it **read only** (`+r`) on the approved dataset domains, preferably through a group.
     Existing domains do not inherit a later folder ACL, so check every in-scope domain.
   - Never reuse the production GUI's or an administrator's credentials.
2. **Install the service.** Either run `sudo bash test-instance.sh install <commit>` from the
   installed `vaft/deploy/gui/` directory (add `--dry-run` to only print the commands; it never
   starts the service while the environment file still holds the placeholder), or follow the
   comments in `vaft-gui-test.service`:
   - create the user and the venv, installing a fixed commit;
   - put the account in `/etc/vaft-gui-test.env` (mode 0600);
   - install and start the unit.

   It listens on 127.0.0.1:5016 only. Local accounts on the host can still reach that port
   without passing through Nginx's allow list or rate limit, as with production's 5006. Sign-in
   still applies there.
3. **First check, through an SSH tunnel** (no Nginx change):
   `ssh -L 5016:127.0.0.1:5016 <host>`, then
   `VAFT_SMOKE_USERNAME=... VAFT_SMOKE_PASSWORD=... python -m vaft.deploy.gui.smoke http://localhost:5016/gui-test/`.
   The checks:
   - the page needs sign-in;
   - a wrong password is refused;
   - the account signs in;
   - the websocket upgrades.

   The password comes from the environment and is never printed.
4. **Restricted route.**
   - Install `nginx-vaft-gui-test-http.conf` as `/etc/nginx/conf.d/vaft-gui-test.conf`. It gives
     the test instance its own login rate limit, so test sign-ins never use up production's.
   - Include `nginx-vaft-gui-test-locations.conf` in the 443 server of the hosted GUI's live
     site, next to its `/gui/` locations.
   - Fill in the `allow` lines with the approved internal ranges; `deny all` stays last.
     `sudo bash test-instance.sh nginx <cidr>...` does these three steps. It keeps a copy of the
     site, runs `nginx -t`, and restores the copy if the test fails.
   - Run `nginx -t`, reload, and repeat the smoke check against `https://<host>/gui-test/` **from
     an allowed address**. From anywhere else the check reports "blocked with 403", which is the
     restriction working, not a sign-in result.
   - `--insecure` turns certificate checks off entirely. Use it only for the self-signed or
     IP-only certificate of the test host.
5. **Record the evidence:**
   - the tested commit, install method and redacted host/proxy settings;
   - each smoke result and a failed-sign-in result;
   - the hosted restrictions seen in the page (no server file browser, no uploads, no file
     sources);
   - the remaining risks.
6. **Rollback** (`sudo bash test-instance.sh remove` does the first two items).
   - `systemctl disable --now vaft-gui-test`, then remove the unit, venv, environment file and
     user.
   - Remove the `include` line and the `conf.d` file, run `nginx -t` and reload.
   - Remove the HSDS login through the account procedure: drop its ACL rows first, then the
     login, then restart.

What sign-in does **not** do: `--auth hsds` decides who may open the page. It is not per-user
data authorization, because every session reads with the service account. Give that account
only what every signed-in reader may see. Also note that this HSDS deployment allows
unauthenticated reads, so its default ACL is what anonymous clients of HSDS itself get.

## What comes next

The browser application is Track A of the GUI roadmap (Open: #1359). A fuller plot explorer follows,
then workspaces for routine diagnostics, equilibrium,
fluctuations and stability, operational space, start-up and pipeline monitoring.

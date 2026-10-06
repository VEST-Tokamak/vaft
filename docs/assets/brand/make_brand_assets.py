#!/usr/bin/env python
"""Generate the VAFT logo assets from a real Solov'ev double-null equilibrium (#1763).

The geometry is computed, not drawn: a Cerfon-Freidberg double-null equilibrium
is solved with ``vaft.process.equilibrium`` (kappa 1.75, delta 0.45), the
separatrix is the psi_X contour (so the X-points are true crossings), and three
closed flux surfaces, the magnetic axis and the dashed open surfaces outside the
separatrix (psi_N 1.25, 1.5) come from the same psi.  Graphic
normalization is limited to a 90 degree rotation, a mild vertical stretch and
the length of the short X-point branches (box fit).  The SOL and divertor legs are cut.

Usage::

    PYTHONPATH=<repo> python docs/assets/brand/make_brand_assets.py   # regenerates every asset here

Rasterization uses ``rsvg-convert`` (librsvg) and the ``.ico`` uses Pillow.
These assets are deliberately not package data (the wheel is size-capped).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from contourpy import contour_generator
from scipy.constants import mu_0 as MU0

from vaft.process.equilibrium import (
    evaluate_solovev,
    solovev_shape_constraints,
    solve_solovev_constraints,
)

HERE = Path(__file__).resolve().parent

R0, A_MINOR, KAPPA, DELTA = 1.0, 0.55, 1.75, 0.45
BRANCH_LENGTH = 0.13 * 1.5        # natural local branch x lengthening factor [R0]
SURFACE_FRACTIONS = (0.85, 0.62, 0.40)  # psi/psi_axis of the three closed surfaces
HALO_FRACTIONS = (0.25, 0.5)  # psi_N = 1.25, 1.5 (psi_N = 1 is the separatrix)
Y_STRETCH = 1.50                 # mild affine: letter-A proportions after rotation



# --------------------------------------------------------------------------- physics
def solve_equilibrium():
    constraints = solovev_shape_constraints(
        major_radius=R0, minor_radius=A_MINOR, elongation=KAPPA, triangularity=DELTA,
        topology="double_null",
    )
    psi0, a_cf = 0.01, 0.9
    model = solve_solovev_constraints(
        constraints, pprime=-psi0 * (1 - a_cf) / (MU0 * R0**4), ffprime=-psi0 * a_cf / R0**2,
        rref=R0, f_boundary=0.1 * R0, basis="cerfon_freidberg_even",
    )
    x_point = (R0 - 1.1 * DELTA * A_MINOR, 1.1 * KAPPA * A_MINOR)
    return model, x_point


def geometry():
    """Return separatrix pieces, closed surfaces and the axis in (R, Z) metres."""
    model, x_point = solve_equilibrium()
    r = np.linspace(0.25, 1.8, 1241)
    z = np.linspace(-1.35, 1.35, 2161)
    rm, zm = np.meshgrid(r, z, indexing="ij")
    psi = evaluate_solovev(model, rm, zm)["psi"]
    # contourpy takes z[row, col] with x along columns -> pass the transposed grid.
    gen = lambda level: [np.asarray(p) for p in contour_generator(r, z, psi.T, name="serial").lines(level)]

    core = (np.abs(rm - R0) < 0.4) & (np.abs(zm) < 0.5)
    iaxis = np.unravel_index(np.argmin(np.where(core, psi, np.inf)), psi.shape)
    axis = (float(rm[iaxis]), float(zm[iaxis]))
    psi_axis = float(psi[iaxis])

    # Separatrix: keep the confined arcs between the nulls and a short branch beyond each.
    xr, xz = x_point
    pieces: list[np.ndarray] = []
    for line in gen(0.0):
        d = np.hypot(line[:, 0] - xr, np.abs(line[:, 1]) - xz)
        inside = np.abs(line[:, 1]) <= xz
        keep = inside | (d <= BRANCH_LENGTH)
        keep &= (line[:, 0] > 0.3) & (np.abs(line[:, 1]) < xz + BRANCH_LENGTH)
        idx = np.flatnonzero(keep)
        for chunk in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
            if len(chunk) > 3:
                pieces.append(line[chunk])

    surfaces = []
    for f in SURFACE_FRACTIONS:
        loops = [p for p in gen(f * psi_axis) if np.allclose(p[0], p[-1])]
        surfaces.append(max(loops, key=len))
    # Open flux surfaces just outside the separatrix (psi > psi_X), kept on the outboard side
    # between the X-points; drawn dashed so they read as flux surfaces, not a drawn outline.
    halo = []
    for f in HALO_FRACTIONS:
        pts = []
        for line in gen(-f * psi_axis):
            keep = (line[:, 0] > 0.3) & (np.abs(line[:, 1]) < xz + 0.10 * R0)
            idx = np.flatnonzero(keep)
            pts += [line[c] for c in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1) if len(c) > 3]
        halo.append(pts)
    return {"separatrix": pieces, "surfaces": surfaces, "axis": axis, "x_point": x_point, "halo": halo}


def to_graphic(points, scale=1.0):
    """Rotate 90 deg (outboard apex up) and apply the mild vertical stretch.

    (R, Z) -> x = Z, y = -R * Y_STRETCH so SVG's y-down puts the outboard side on top.
    """
    pts = np.atleast_2d(points)
    return np.column_stack([pts[:, 1], -pts[:, 0] * Y_STRETCH]) * scale


# --------------------------------------------------------------------------- drawing
_GEO = None
NAVY2, BLUE2, CYAN2 = "#0b1f3b", "#007bff", "#00d4ff"
BG_DARK = "#0b1a2e"
GAP = 10.0
BOX_W = 94.0       # V, F, T and the LCFS body of the A share one box ...
BOX_H = 90.0       # ... of this height; the dashed outer flux fills the rest of the 100-unit line
STEM = 20.0
OVER = 6.0         # room beside the A for the X-point branch tips
SEP_W, SURF_W = 6.0, 3.2
DOT_R = 0.54 * SEP_W


def geometry_cached():
    global _GEO
    if _GEO is None:
        _GEO = geometry()
    return _GEO


def _thin(points, spacing=0.3):
    """Drop points closer than *spacing* units to the last kept one (the contours are ~2000 points)."""
    pts = np.asarray(points)
    keep = [0]
    for i in range(1, len(pts) - 1):
        if np.hypot(*(pts[i] - pts[keep[-1]])) >= spacing:
            keep.append(i)
    return pts[keep + [len(pts) - 1]]


def _path(points, close=False, spacing=0.3):
    points = _thin(points, spacing)
    d = "M" + " L".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return d + (" Z" if close else "")


def _letter_paths(w: float, h: float):
    """Bold geometric V, F, T (flat terminals, sharp joins) filling the w x h LCFS box."""
    t, bar = STEM, 0.18 * h
    return {
        "V": f"M0,0 H{t + 2} L{w / 2},{0.72 * h} L{w - t - 2},0 H{w} L{w / 2 + t / 2 + 1},{h} H{w / 2 - t / 2 - 1} Z",
        "F": f"M0,0 H{w} V{bar} H{t} V{0.40 * h} H{w - 12} V{0.40 * h + bar} H{t} V{h} H0 Z",
        "T": f"M0,0 H{w} V{bar} H{w / 2 + t / 2} V{h} H{w / 2 - t / 2} V{bar} H0 Z",
    }


def _palette(variant: str):
    """(ink, gradient?, background) for color | mono | dark."""
    return {"color": (NAVY2, True, None), "mono": ("#14212e", False, None), "dark": ("#ffffff", True, BG_DARK)}[variant]


def mark_elements(variant: str, simplified: bool = False) -> list[str]:
    """The A glyph in a BOX_W x BOX_H box whose bottom sits on y=100.

    The body (separatrix + closed surfaces) fills the box X-point to X-point; the dashed
    open flux surfaces (psi_N 1.25, 1.5) sit outside it.  ``simplified`` keeps one closed
    surface, drops the halo and thickens the strokes for 16-48 px use.
    """
    geo = geometry_cached()
    ink, grad, _ = _palette(variant)
    sep = [to_graphic(p) for p in geo["separatrix"]]
    y0 = np.vstack(sep)[:, 1].min()
    y1 = np.vstack(sep)[:, 1].max()
    xb = abs(to_graphic(np.array([geo["x_point"]]))[0, 0])
    w_sep, w_surf = (SEP_W * 1.7, SURF_W * 2.2) if simplified else (SEP_W, SURF_W)
    half = w_sep / 2
    sx, sy = (BOX_W - 2 * half) / (2 * xb), (BOX_H - 2 * half) / (y1 - y0)
    tf = lambda pts: (np.atleast_2d(pts) - [-xb, y0]) * [sx, sy] + [half, 100 - BOX_H + half]
    stroke = "url(#g)" if grad else ink
    out = []
    spacing = 1.0 if simplified else 0.3
    for idx in ((1,) if simplified else (0, 1, 2)):
        out.append(f'<path d="{_path(tf(to_graphic(geo["surfaces"][idx])), True, spacing)}" stroke="{stroke}" stroke-width="{w_surf}"/>')
    for piece in sep:
        out.append(f'<path d="{_path(tf(piece), False, spacing)}" stroke="{stroke}" stroke-width="{w_sep}"/>')
    if not simplified:
        for i, group in enumerate(geo["halo"]):
            for piece in group:
                out.append(f'<path d="{_path(tf(to_graphic(piece)))}" stroke="{stroke}" stroke-width="{SURF_W * 0.5:.2f}" '
                           f'stroke-linecap="butt" stroke-dasharray="2.4 3.2" opacity="{0.9 - 0.2 * i:.2f}"/>')
    a = tf(to_graphic(geo["axis"]))[0]
    dot = ink if not grad else BLUE2
    out.append(f'<circle cx="{a[0]:.2f}" cy="{a[1]:.2f}" r="{(DOT_R * 1.5 if simplified else DOT_R):.2f}" fill="{dot}" stroke="none"/>')
    return out


def _svg(width_units, height_units, x0, y0, body, variant, px_height, label, bg_radius=0.0) -> str:
    _, grad, bg = _palette(variant)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" height="{px_height}" '
           f'width="{px_height * width_units / height_units:.0f}" '
           f'viewBox="{x0:.2f} {y0:.2f} {width_units:.2f} {height_units:.2f}" role="img" aria-label="{label}">']
    if grad:
        out.append(f'<defs><linearGradient id="g" gradientUnits="userSpaceOnUse" x1="0" y1="100" x2="0" y2="0">'
                   f'<stop offset="0" stop-color="{BLUE2}"/><stop offset="1" stop-color="{CYAN2}"/></linearGradient></defs>')
    if bg:
        out.append(f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{width_units:.2f}" height="{height_units:.2f}" '
                   f'rx="{bg_radius:.2f}" fill="{bg}"/>')
    out += body
    out.append("</svg>")
    return "\n".join(out)


def _wordmark_body(variant: str) -> tuple[list[str], float]:
    ink = _palette(variant)[0]
    paths = _letter_paths(BOX_W, BOX_H)
    xs = [i * (BOX_W + GAP) + (OVER if i == 1 else 2 * OVER if i > 1 else 0) for i in range(4)]
    body = ['<g fill="none" stroke-linecap="round" stroke-linejoin="round">']
    for ch, dx in zip("VAFT", xs):
        if ch == "A":
            body.append(f'<g transform="translate({dx:.2f},0)">' + "".join(mark_elements(variant)) + "</g>")
        else:
            body.append(f'<path transform="translate({dx:.2f},{100 - BOX_H})" d="{paths[ch]}" fill="{ink}" stroke="none"/>')
    body.append("</g>")
    return body, 4 * BOX_W + 3 * GAP + 2 * OVER


def wordmark_svg(variant: str = "color", px_height: int = 120) -> str:
    body, total = _wordmark_body(variant)
    pad = 22
    return _svg(total + 2 * pad, 100 + 2 * pad, -pad, -pad, body, variant, px_height, "VAFT")


def mark_svg(variant: str = "color", simplified: bool = False, px_height: int = 256) -> str:
    """The A alone on a square canvas (icon / favicon); the dark variant is a rounded tile."""
    # The small glyph has no dashed halo, so give it more of the canvas.  At
    # favicon size the old 140-unit canvas left only ~10 pixels for the mark.
    side = 112.0 if simplified else 140.0
    x0, y0 = BOX_W / 2 - side / 2, 52 - side / 2
    body = ['<g fill="none" stroke-linecap="round" stroke-linejoin="round">'] + mark_elements(variant, simplified) + ["</g>"]
    return _svg(side, side, x0, y0, body, variant, px_height, "VAFT mark", bg_radius=side * 0.22)


def social_svg() -> str:
    """1280x640 social preview: the dark wordmark over the project name and the pipeline."""
    body, total = _wordmark_body("dark")
    pad, width = 22, 960.0
    scale = width / (total + 2 * pad)
    font = 'font-family="Helvetica Neue, Arial, sans-serif" text-anchor="middle"'
    return "\n".join([
        '<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="640" viewBox="0 0 1280 640" role="img" '
        'aria-label="VAFT social preview">',
        f'<defs><linearGradient id="g" gradientUnits="userSpaceOnUse" x1="0" y1="100" x2="0" y2="0">'
        f'<stop offset="0" stop-color="{BLUE2}"/><stop offset="1" stop-color="{CYAN2}"/></linearGradient></defs>',
        f'<rect width="1280" height="640" fill="{BG_DARK}"/>',
        f'<g transform="translate({(1280 - width) / 2:.2f},{150 - 22 * scale:.2f}) scale({scale:.4f}) translate({pad},{pad})">',
        *body, "</g>",
        f'<text x="640" y="450" {font} font-size="36" letter-spacing="1.5" fill="#cfe6ff">'
        "Versatile Analysis Framework for Tokamak</text>",
        f'<text x="640" y="508" {font} font-size="24" fill="#7fa6cc">'
        "Integrating fusion knowledge for shared discovery.</text>",
        "</svg>",
    ])


# --------------------------------------------------------------------------- outputs
def rsvg(svg: Path, png: Path, width: int, height: int | None = None):
    cmd = ["rsvg-convert", "-w", str(width)] + (["-h", str(height)] if height else []) + ["-o", str(png), str(svg)]
    subprocess.run(cmd, check=True)


def build(outdir: Path):
    from PIL import Image

    outdir.mkdir(exist_ok=True)
    files = {"vaft-social-preview.svg": social_svg()}
    for variant, suffix in (("color", ""), ("mono", "-mono"), ("dark", "-dark")):
        files[f"vaft-wordmark{suffix}.svg"] = wordmark_svg(variant)
        files[f"vaft-mark{suffix}.svg"] = mark_svg(variant)
        files[f"vaft-mark-small{suffix}.svg"] = mark_svg(variant, simplified=True)
    for name, text in files.items():
        (outdir / name).write_text(text + "\n")
    for size in (512, 256, 128):
        source = "vaft-mark-small.svg" if size == 128 else "vaft-mark.svg"
        rsvg(outdir / source, outdir / f"vaft-mark-{size}.png", size, size)
    rsvg(outdir / "vaft-wordmark.svg", outdir / "vaft-wordmark-1024.png", 1024)
    rsvg(outdir / "vaft-wordmark-dark.svg", outdir / "vaft-wordmark-dark-1024.png", 1024)
    rsvg(outdir / "vaft-mark-small-dark.svg", outdir / "apple-touch-icon-152.png", 152, 152)
    rsvg(outdir / "vaft-social-preview.svg", outdir / "vaft-social-preview.png", 1280, 640)
    for size in (32, 16):
        rsvg(outdir / "vaft-mark-small-dark.svg", outdir / f"favicon-{size}.png", size, size)
    big = outdir / "_favicon-256.png"
    rsvg(outdir / "vaft-mark-small-dark.svg", big, 256, 256)
    Image.open(big).save(outdir / "favicon.ico", sizes=[(32, 32), (16, 16)])
    big.unlink()
    write_gui_module(outdir.parents[2] / "vaft" / "gui" / "_brand.py")


def write_gui_module(path: Path):
    """Embed the small mark and favicon for an offline ``vaft gui`` (not package data)."""
    import base64

    svg = mark_svg("color", simplified=True, px_height=40).replace("\n", "")
    uri = "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()
    # Panel infers the favicon MIME type from the URL suffix.  A fragment gives
    # the embedded icon a .png suffix without changing the data URL payload.
    favicon = (
        "data:image/png;base64,"
        + base64.b64encode((HERE / "favicon-32.png").read_bytes()).decode()
        + "#favicon.png"
    )
    path.write_text(
        '"""The VAFT mark for the ``vaft gui`` header, generated by ``docs/assets/brand/make_brand_assets.py``."""\n\n'
        "from __future__ import annotations\n\n"
        f'LOGO_DATA_URI = (\n    "{uri[:70]}"\n'
        + "".join(f'    "{uri[i:i + 98]}"\n' for i in range(70, len(uri), 98))
        + ")\n\n"
        '#: Panel infers the favicon type from the URL suffix.\n'
        'FAVICON_URL = (\n'
        + "".join(f'    "{favicon[i:i + 98]}"\n' for i in range(0, len(favicon), 98))
        + ")\n"
    )


if __name__ == "__main__":
    build(HERE)

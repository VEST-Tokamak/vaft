"""The orthographic 3-D camera every vaft.diagram 3-D view shares.

One camera keeps the 3-D figures of different families (islands, particle
drifts) looking like one set: same azimuth, elevation and hidden-line style.
"""

from __future__ import annotations

import math
from typing import List

import numpy as np

from ._scene import Polyline

#: centimetres per unit length in 3-D views
THREE_D_SCALE = 1.0
#: orthographic camera of the 3-D views [rad]
CAMERA_AZIMUTH = math.radians(-62.0)
CAMERA_ELEVATION = math.radians(26.0)


def camera():
    """``(view, u, v)``: the unit vector towards the viewer and the screen axes."""
    az, el = CAMERA_AZIMUTH, CAMERA_ELEVATION
    view = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    u = np.array([-math.sin(az), math.cos(az), 0.0])
    v = np.cross(view, u)
    return view, u, v


def project(points: np.ndarray, scale: float = THREE_D_SCALE) -> np.ndarray:
    """Screen coordinates [cm] of 3-D points, shape ``(..., 2)``."""
    _, u, v = camera()
    points = np.asarray(points, dtype=float)
    return scale * np.stack([points @ u, points @ v], axis=-1)


def split(xy: np.ndarray, visible: np.ndarray, style: str, hidden_style: str, role: str) -> List:
    """Break a polyline into runs drawn in the visible or hidden style."""
    items: List = []
    start = 0
    for i in range(1, len(xy) + 1):
        if i == len(xy) or visible[i] != visible[start]:
            end = min(i + 1, len(xy))
            if end - start >= 2:
                vis = bool(visible[start])
                items.append(Polyline.of(xy[start:end], style if vis else hidden_style,
                                         role=role if vis else f"{role}_hidden"))
            start = i
    return items

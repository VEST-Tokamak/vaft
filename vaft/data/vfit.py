"""Read VFIT FEM/GSE MATLAB results and convert them to OMAS.

VFIT stores its internal poloidal flux in Wb/rad.  This module converts it to
the Wb convention used by GEQDSK and the OMAS equilibrium IDS.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np
from scipy.io import loadmat


VFITKind = Literal["fem", "gse"]
_TWO_PI = 2.0 * np.pi


def _native(value: Any) -> Any:
    """Turn scipy's MATLAB objects into plain dictionaries and ndarrays."""
    if hasattr(value, "_fieldnames"):
        return {name: _native(getattr(value, name)) for name in value._fieldnames}
    if isinstance(value, np.ndarray) and value.dtype == object:
        return [_native(item) for item in value.flat]
    return value


def _as_array(value: Any, name: str) -> np.ndarray:
    if value is None:
        raise ValueError(f"VFIT result is missing required field {name}")
    result = np.asarray(value, dtype=float)
    if result.size == 0:
        raise ValueError(f"VFIT result field {name} is empty")
    return result


def _scalar(value: Any, name: str) -> float:
    return float(_as_array(value, name).reshape(-1)[0])


def _field(mapping: Mapping[str, Any], name: str, *, required: bool = True) -> Any:
    if name in mapping:
        return mapping[name]
    if required:
        raise ValueError(f"VFIT result is missing required field {name}")
    return None


def _time_value(
    value: Any, index: int, count: int, axis: int | None = None
) -> np.ndarray:
    """Select one VFIT time slice while preserving profile/grid axes."""
    arr = np.asarray(value)
    if arr.ndim == 0 or count == 1:
        return arr
    if axis is not None:
        normalized_axis = axis if axis >= 0 else arr.ndim + axis
        if 0 <= normalized_axis < arr.ndim and arr.shape[normalized_axis] == count:
            return np.take(arr, index, axis=normalized_axis)
    if arr.shape[-1] == count:
        return arr[..., index]
    if arr.shape[0] == count:
        return arr[index, ...]
    # A scalar field can be shared by all slices.  Arrays with no identifiable
    # time axis are intentionally shared rather than silently reshaped.
    return arr


def _slice_field(
    mapping: Mapping[str, Any],
    name: str,
    index: int,
    count: int,
    *,
    required: bool = True,
    axis: int | None = None,
) -> np.ndarray | None:
    value = _field(mapping, name, required=required)
    if value is None:
        return None
    return _time_value(value, index, count, axis=axis)


def _profile(value: Any, name: str) -> np.ndarray:
    return _as_array(value, name).reshape(-1)


def _derivative(values: np.ndarray, psi: np.ndarray) -> np.ndarray:
    if values.size < 2 or psi.size < 2:
        return np.zeros_like(values)
    return np.gradient(values, psi)


def _find_slice_count(data: Mapping[str, Any], kind: VFITKind) -> int:
    time_name = "FitTime" if kind == "fem" else "ProfileTime"
    values = (
        np.asarray(_field(data, time_name, required=False))
        if _field(data, time_name, required=False) is not None
        else np.array([])
    )
    if values.size > 1:
        return int(values.size)
    contour_name = "Contour" if kind == "fem" else "ProfFitContour"
    contour = _field(data, contour_name)
    psi = _as_array(_field(contour, "Psi"), f"{contour_name}.Psi")
    return int(psi.shape[-1]) if psi.ndim >= 3 else 1


def _times_seconds(data: Mapping[str, Any], kind: VFITKind, count: int) -> np.ndarray:
    time_name = "FitTime" if kind == "fem" else "ProfileTime"
    raw = _field(data, time_name, required=False)
    if raw is None:
        return np.zeros(count, dtype=float)
    times = np.asarray(raw, dtype=float).reshape(-1)
    if times.size == 1:
        return np.full(count, times[0] / 1000.0, dtype=float)
    if times.size != count:
        raise ValueError(
            f"{time_name} has {times.size} values but VFIT fields contain {count} time slices"
        )
    return times / 1000.0


def _kind_from_data(data: Mapping[str, Any]) -> VFITKind:
    if "ProfFitContour" in data and "ProfFitProfile" in data:
        return "gse"
    if "Contour" in data and "Profile" in data:
        return "fem"
    raise ValueError(
        "Unsupported VFIT MAT result: expected FEM fields Contour/Profile or "
        "GSE fields ProfFitContour/ProfFitProfile"
    )


def _validate(data: Mapping[str, Any], kind: VFITKind) -> None:
    common = ("Grid", "VESTGeometry", "shotNumber")
    for name in common:
        _field(data, name)
    _field(_field(data, "Grid"), "ContourR")
    _field(_field(data, "Grid"), "ContourZ")
    if kind == "fem":
        for name in ("Contour", "ConstShape", "ConstMHD", "FluxSurface", "Profile"):
            _field(data, name)
        _field(_field(data, "Contour"), "Psi")
    else:
        for name in (
            "ProfFitContour",
            "ProfFitShape",
            "ProfFitConstMHD",
            "ProfFitFluxSurf",
            "ProfFitProfile",
        ):
            _field(data, name)
        _field(_field(data, "ProfFitContour"), "Psi")


@dataclass
class VFITResult:
    """A normalized VFIT FEM or Grad--Shafranov-equilibrium MAT result."""

    kind: VFITKind
    source: Path
    shot: int
    times: np.ndarray
    raw: Mapping[str, Any]

    @property
    def time_count(self) -> int:
        return int(self.times.size)

    def to_omas(
        self, ods: Any = None, time_index: int = 0, include_pf_passive: bool = True
    ) -> Any:
        """Convert all result times to OMAS equilibrium slices.

        ``time_index`` is the first destination index.  With an existing ODS,
        the default zero appends after existing equilibrium slices.
        """
        from omas import ODS

        if ods is None:
            ods = ODS()
        if time_index == 0 and "equilibrium.time_slice" in ods:
            try:
                time_index = len(ods["equilibrium.time_slice"])
            except TypeError:
                pass

        ods["dataset_description.data_entry.pulse"] = self.shot
        ods["equilibrium.ids_properties.homogeneous_time"] = 1
        ods[
            "equilibrium.ids_properties.comment"
        ] = f"VFIT {self.kind.upper()} MAT import; internal psi converted from Wb/rad to Wb (COCOS 11)"
        ods["equilibrium.code.name"] = "VFIT"
        ods["equilibrium.code.version"] = "MAT import"

        for source_index, time in enumerate(self.times):
            destination = time_index + source_index
            if self.kind == "gse":
                self._gse_slice(ods, destination, source_index, float(time))
            else:
                self._fem_slice(ods, destination, source_index, float(time))
            try:
                ods.set_time_array("equilibrium.time", destination, float(time))
            except Exception:
                ods[f"equilibrium.time.{destination}"] = float(time)

        if include_pf_passive:
            self._pf_passive(ods)
        return ods

    def to_imas(
        self,
        target: str | Path,
        *,
        occurrence: dict | None = None,
        imas_version: str | None = None,
        new: bool = False,
    ) -> Any:
        """Convert through OMAS and save using VAFT's IMAS writer."""
        from vaft.imas.omas_imas import save_omas_imas

        uri = str(target)
        if not uri.startswith("imas:"):
            uri = "imas:hdf5?path=" + str(Path(target).expanduser())
        return save_omas_imas(
            self.to_omas(),
            occurrence=occurrence or {},
            imas_version=imas_version,
            new=new,
            uri=uri,
        )

    def _validate_time_index(self, time_index: int) -> None:
        if not 0 <= time_index < self.time_count:
            raise IndexError(
                f"time_index={time_index} is outside the VFIT result range 0..{self.time_count - 1}"
            )

    def _contour_data(
        self, time_index: int, field: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return a 2D VFIT field in matplotlib's ``(Z, R)`` orientation."""
        self._validate_time_index(time_index)
        grid = _field(self.raw, "Grid")
        r = _profile(_field(grid, "ContourR"), "Grid.ContourR")
        z = _profile(_field(grid, "ContourZ"), "Grid.ContourZ")
        contour_name = "Contour" if self.kind == "fem" else "ProfFitContour"
        contour = _field(self.raw, contour_name)
        values = _as_array(
            _slice_field(contour, field, time_index, self.time_count, axis=-1),
            f"{contour_name}.{field}",
        )
        if values.shape == (r.size, z.size):
            values = values.T
        if values.shape != (z.size, r.size):
            raise ValueError(
                f"VFIT {field} has shape {values.shape}; expected (Z, R) or (R, Z) "
                f"for grid {(z.size, r.size)}"
            )
        return r, z, values

    def _surface_data(self, time_index: int) -> tuple[np.ndarray, np.ndarray]:
        flux_name = "FluxSurface" if self.kind == "fem" else "ProfFitFluxSurf"
        flux_surface = _field(self.raw, flux_name)
        r = _as_array(
            _slice_field(flux_surface, "R", time_index, self.time_count, axis=-1),
            f"{flux_name}.R",
        )
        z = _as_array(
            _slice_field(flux_surface, "Z", time_index, self.time_count, axis=-1),
            f"{flux_name}.Z",
        )
        if r.ndim != 2 or z.ndim != 2:
            raise ValueError("VFIT flux surfaces must have (surface, point) dimensions")
        return r, z

    def plot_equilibrium(
        self,
        time_index: int = 0,
        quantity: Literal["psi", "j_tor", "pressure", "b_pol"] = "psi",
        *,
        ax: Any = None,
        levels: int = 40,
        show_flux_surfaces: bool = True,
        flux_surface_stride: int = 5,
        colorbar: bool = True,
    ) -> Any:
        """Plot a VFIT equilibrium field with boundary and flux-surface overlays.

        ``psi`` is converted from VFIT's Wb/rad to Wb.  The other quantities
        are plotted in their native VFIT units.
        """
        import matplotlib.pyplot as plt

        fields = {
            "psi": ("Psi", "Poloidal flux [Wb]"),
            "j_tor": (
                "Jelement" if self.kind == "fem" else "Jtor",
                "Toroidal current density",
            ),
            "pressure": ("P", "Pressure"),
            "b_pol": ("Bpn", "Poloidal magnetic field [T]"),
        }
        if quantity == "pressure" and self.kind != "gse":
            raise ValueError("Pressure is not available in FEM VFIT results")
        field, label = fields[quantity]
        r, z, values = self._contour_data(time_index, field)
        if quantity == "psi":
            values = values * _TWO_PI

        if ax is None:
            _, ax = plt.subplots()
        contour = ax.contourf(r, z, values, levels=levels)
        if colorbar:
            ax.figure.colorbar(contour, ax=ax, label=label)
        surface_r, surface_z = self._surface_data(time_index)
        if show_flux_surfaces:
            stride = max(1, int(flux_surface_stride))
            for index in range(0, surface_r.shape[0], stride):
                ax.plot(
                    surface_r[index],
                    surface_z[index],
                    color="white",
                    linewidth=0.35,
                    alpha=0.7,
                )
        ax.plot(
            surface_r[-1],
            surface_z[-1],
            color="tab:red",
            linewidth=1.5,
            label="plasma boundary",
        )
        ax.plot(
            surface_r[-1, 0],
            surface_z[-1, 0],
            color="tab:red",
            marker="o",
            markersize=2,
        )
        ax.set_aspect("equal")
        ax.set_xlabel("R [m]")
        ax.set_ylabel("Z [m]")
        ax.set_title(
            f"VFIT {self.kind.upper()} {quantity}: shot {self.shot}, t={self.times[time_index]:.4f} s"
        )
        return ax

    def plot_profiles(
        self, time_index: int = 0, *, axes: Any = None
    ) -> tuple[Any, np.ndarray]:
        """Plot available 1D VFIT profiles (q/J; additionally P/F for GSE)."""
        import matplotlib.pyplot as plt

        self._validate_time_index(time_index)
        profile_name = "Profile" if self.kind == "fem" else "ProfFitProfile"
        profile = _field(self.raw, profile_name)
        psin = _profile(
            _slice_field(profile, "PsiN", time_index, self.time_count, axis=0),
            f"{profile_name}.PsiN",
        )
        fields: list[tuple[str, str]] = [("q", "q"), ("J", "J")]
        if self.kind == "gse":
            fields.extend((("P", "Pressure [Pa]"), ("F", "F [T m]")))

        if axes is None:
            rows = int(np.ceil(len(fields) / 2.0))
            figure, axes = plt.subplots(
                rows, 2, squeeze=False, figsize=(10, 3.5 * rows)
            )
        else:
            axes = np.asarray(axes, dtype=object)
            figure = axes.reshape(-1)[0].figure
        flat_axes = axes.reshape(-1)
        if flat_axes.size < len(fields):
            raise ValueError(
                f"At least {len(fields)} axes are required for {self.kind.upper()} profiles"
            )
        for ax, (field, label) in zip(flat_axes, fields):
            values = _profile(
                _slice_field(profile, field, time_index, self.time_count, axis=0),
                f"{profile_name}.{field}",
            )
            ax.plot(psin, values, marker="o", markersize=3)
            ax.set_xlabel("psi_N")
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.3)
        for ax in flat_axes[len(fields) :]:
            ax.set_visible(False)
        figure.suptitle(
            f"VFIT {self.kind.upper()} profiles: shot {self.shot}, t={self.times[time_index]:.4f} s"
        )
        figure.tight_layout()
        return figure, axes

    def plot_wall_currents(
        self, time_index: int = 0, *, ax: Any = None, colorbar: bool = True
    ) -> Any:
        """Plot discretized VEST passive-wall current at one equilibrium time."""
        import matplotlib.pyplot as plt
        from matplotlib.collections import PatchCollection
        from matplotlib.patches import Rectangle

        self._validate_time_index(time_index)
        geometry = _field(self.raw, "VESTGeometry")
        wall = _as_array(_field(geometry, "Wall"), "VESTGeometry.Wall")
        eddy = _field(self.raw, "Eddy")
        currents = _as_array(_field(eddy, "WallCurrent"), "Eddy.WallCurrent")
        if currents.ndim == 1:
            values = currents
        elif currents.ndim == 2 and currents.shape[0] == wall.shape[0]:
            values = currents[:, min(time_index, currents.shape[1] - 1)]
        elif currents.ndim == 2 and currents.shape[1] == wall.shape[0]:
            values = currents[min(time_index, currents.shape[0] - 1), :]
        else:
            raise ValueError(
                "Eddy.WallCurrent does not match VESTGeometry.Wall elements"
            )
        if values.size != wall.shape[0]:
            raise ValueError(
                "Eddy.WallCurrent does not have one value per wall element"
            )

        if ax is None:
            _, ax = plt.subplots()
        patches = [
            Rectangle((r - width / 2.0, z - height / 2.0), width, height)
            for r, z, width, height in wall[:, :4]
        ]
        collection = PatchCollection(patches, cmap="coolwarm", edgecolor="none")
        collection.set_array(np.asarray(values, dtype=float))
        ax.add_collection(collection)
        ax.autoscale_view()
        if colorbar:
            ax.figure.colorbar(collection, ax=ax, label="Passive-wall current [A]")
        ax.set_aspect("equal")
        ax.set_xlabel("R [m]")
        ax.set_ylabel("Z [m]")
        ax.set_title(
            f"VFIT passive-wall current: shot {self.shot}, t={self.times[time_index]:.4f} s"
        )
        return ax

    def plot_magnetics_residuals(
        self, time_index: int = 0, *, ax: Any = None, colorbar: bool = True
    ) -> Any:
        """Plot VFIT magnetic-constraint residuals at their R-Z probe locations."""
        import matplotlib.pyplot as plt

        self._validate_time_index(time_index)
        magnetics = _field(self.raw, "Magnetics")
        r = _profile(_field(magnetics, "R"), "Magnetics.R")
        z = _profile(_field(magnetics, "Z"), "Magnetics.Z")
        residual = _profile(
            _slice_field(
                magnetics,
                "Residual",
                time_index,
                self.time_count,
                required=False,
                axis=0,
            )
            if _field(magnetics, "Residual", required=False) is not None
            else _slice_field(magnetics, "Error", time_index, self.time_count, axis=0),
            "Magnetics.Residual",
        )
        if not (r.size == z.size == residual.size):
            raise ValueError(
                "Magnetics R, Z, and residual arrays must have equal lengths"
            )
        if ax is None:
            _, ax = plt.subplots()
        scatter = ax.scatter(
            r, z, c=residual, cmap="coolwarm", edgecolors="black", linewidths=0.25
        )
        if colorbar:
            ax.figure.colorbar(scatter, ax=ax, label="Magnetic residual")
        ax.set_aspect("equal")
        ax.set_xlabel("R [m]")
        ax.set_ylabel("Z [m]")
        ax.set_title(
            f"VFIT magnetics residual: shot {self.shot}, t={self.times[time_index]:.4f} s"
        )
        return ax

    def _common_slice(
        self,
        ods: Any,
        destination: int,
        time: float,
        contour: Mapping[str, Any],
        shape: Mapping[str, Any],
        mhd: Mapping[str, Any],
        flux_surface: Mapping[str, Any],
        source_index: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        grid = _field(self.raw, "Grid")
        r = _profile(_field(grid, "ContourR"), "Grid.ContourR")
        z = _profile(_field(grid, "ContourZ"), "Grid.ContourZ")
        psi = _as_array(
            _slice_field(contour, "Psi", source_index, self.time_count, axis=-1),
            "Contour.Psi",
        )
        if psi.shape == (z.size, r.size):
            psi = psi.T
        if psi.shape != (r.size, z.size):
            raise ValueError(
                f"VFIT psi has shape {psi.shape}; expected (Z, R) or (R, Z) for grid {(z.size, r.size)}"
            )
        psi *= _TWO_PI

        eqt = ods[f"equilibrium.time_slice.{destination}"]
        eqt["time"] = time
        axis_r = _scalar(
            _slice_field(shape, "Rmag", source_index, self.time_count, axis=0),
            "Shape.Rmag",
        )
        axis_z = _scalar(
            _slice_field(shape, "Zmag", source_index, self.time_count, axis=0),
            "Shape.Zmag",
        )
        psi_axis = (
            _scalar(
                _slice_field(shape, "PsiA", source_index, self.time_count, axis=0),
                "Shape.PsiA",
            )
            * _TWO_PI
        )
        psi_boundary = (
            _scalar(
                _slice_field(shape, "PsiB", source_index, self.time_count, axis=0),
                "Shape.PsiB",
            )
            * _TWO_PI
        )
        ip = _scalar(
            _slice_field(mhd, "Ip", source_index, self.time_count, axis=0), "MHD.Ip"
        )

        eqt["global_quantities.magnetic_axis.r"] = axis_r
        eqt["global_quantities.magnetic_axis.z"] = axis_z
        eqt["global_quantities.psi_axis"] = psi_axis
        eqt["global_quantities.psi_boundary"] = psi_boundary
        eqt["global_quantities.ip"] = ip
        prof2d = eqt["profiles_2d.0"]
        prof2d["grid_type.index"] = 1
        prof2d["grid.dim1"] = r
        prof2d["grid.dim2"] = z
        prof2d["psi"] = psi

        boundary_r = _as_array(
            _slice_field(flux_surface, "R", source_index, self.time_count, axis=-1),
            "FluxSurface.R",
        )
        boundary_z = _as_array(
            _slice_field(flux_surface, "Z", source_index, self.time_count, axis=-1),
            "FluxSurface.Z",
        )
        if boundary_r.ndim < 2 or boundary_z.ndim < 2:
            raise ValueError("VFIT flux surfaces must have (surface, point) dimensions")
        eqt["boundary.outline.r"] = boundary_r[-1, :].reshape(-1)
        eqt["boundary.outline.z"] = boundary_z[-1, :].reshape(-1)
        return eqt, r, z, psi_axis, psi_boundary

    def _write_profiles(
        self,
        eqt: Any,
        psin: np.ndarray,
        psi_axis: float,
        psi_boundary: float,
        q: np.ndarray,
        **profiles: np.ndarray,
    ) -> None:
        if psin.size != q.size:
            raise ValueError("VFIT PsiN and q profiles must have the same length")
        psi = psi_axis + psin * (psi_boundary - psi_axis)
        eqt["profiles_1d.psi"] = psi
        eqt["profiles_1d.rho_tor_norm"] = np.sqrt(np.clip(psin, 0.0, 1.0))
        eqt["profiles_1d.q"] = q
        for name, values in profiles.items():
            if values.size == psin.size:
                eqt[f"profiles_1d.{name}"] = values
        if q.size:
            eqt["global_quantities.q_axis"] = float(q[0])
            qmin = int(np.argmin(np.abs(q)))
            eqt["global_quantities.q_min.value"] = float(q[qmin])
            eqt["global_quantities.q_min.rho_tor_norm"] = float(
                eqt["profiles_1d.rho_tor_norm"][qmin]
            )

    def _fem_slice(
        self, ods: Any, destination: int, source_index: int, time: float
    ) -> None:
        contour = _field(self.raw, "Contour")
        shape = _field(self.raw, "ConstShape")
        mhd = _field(self.raw, "ConstMHD")
        flux_surface = _field(self.raw, "FluxSurface")
        profile = _field(self.raw, "Profile")
        eqt, _r, _z, psi_axis, psi_boundary = self._common_slice(
            ods, destination, time, contour, shape, mhd, flux_surface, source_index
        )
        psin = _profile(
            _slice_field(profile, "PsiN", source_index, self.time_count, axis=0),
            "Profile.PsiN",
        )
        q = _profile(
            _slice_field(profile, "q", source_index, self.time_count, axis=0),
            "Profile.q",
        )
        current = _profile(
            _slice_field(profile, "J", source_index, self.time_count, axis=0),
            "Profile.J",
        )
        self._write_profiles(eqt, psin, psi_axis, psi_boundary, q, j_tor=current)
        for source_name, target_name in (("Lint", "li_3"),):
            value = _slice_field(
                mhd, source_name, source_index, self.time_count, required=False, axis=0
            )
            if value is not None:
                eqt[f"global_quantities.{target_name}"] = _scalar(
                    value, f"ConstMHD.{source_name}"
                )

    def _gse_slice(
        self, ods: Any, destination: int, source_index: int, time: float
    ) -> None:
        contour = _field(self.raw, "ProfFitContour")
        shape = _field(self.raw, "ProfFitShape")
        mhd = _field(self.raw, "ProfFitConstMHD")
        flux_surface = _field(self.raw, "ProfFitFluxSurf")
        profile = _field(self.raw, "ProfFitProfile")
        eqt, _r, _z, psi_axis, psi_boundary = self._common_slice(
            ods, destination, time, contour, shape, mhd, flux_surface, source_index
        )
        btor = _slice_field(
            self.raw, "ConstBtor", source_index, self.time_count, required=False, axis=0
        )
        if btor is not None:
            # VFIT's GEQDSK exporter uses the same fixed vacuum-field
            # reference radius.
            ods["equilibrium.vacuum_toroidal_field.r0"] = 0.4
            try:
                ods.set_time_array(
                    "equilibrium.vacuum_toroidal_field.b0",
                    destination,
                    _scalar(btor, "ConstBtor"),
                )
            except Exception:
                ods[f"equilibrium.vacuum_toroidal_field.b0.{destination}"] = _scalar(
                    btor, "ConstBtor"
                )
        psin = _profile(
            _slice_field(profile, "PsiN", source_index, self.time_count, axis=0),
            "ProfFitProfile.PsiN",
        )
        f = _profile(
            _slice_field(profile, "F", source_index, self.time_count, axis=0),
            "ProfFitProfile.F",
        )
        pressure = _profile(
            _slice_field(profile, "P", source_index, self.time_count, axis=0),
            "ProfFitProfile.P",
        )
        q = _profile(
            _slice_field(profile, "q", source_index, self.time_count, axis=0),
            "ProfFitProfile.q",
        )
        current = _profile(
            _slice_field(profile, "J", source_index, self.time_count, axis=0),
            "ProfFitProfile.J",
        )
        psi = psi_axis + psin * (psi_boundary - psi_axis)
        self._write_profiles(
            eqt,
            psin,
            psi_axis,
            psi_boundary,
            q,
            f=f,
            pressure=pressure,
            j_tor=current,
            f_df_dpsi=f * _derivative(f, psi),
            dpressure_dpsi=_derivative(pressure, psi),
        )
        global_names = {
            "BetaP": "beta_pol",
            "BetaT": "beta_tor",
            "BetaN": "beta_normal",
            "Lint": "li_3",
            "Wmag": "energy_mhd",
            "q0": "q_axis",
        }
        for source_name, target_name in global_names.items():
            value = _slice_field(
                mhd, source_name, source_index, self.time_count, required=False, axis=0
            )
            if value is not None:
                eqt[f"global_quantities.{target_name}"] = _scalar(
                    value, f"ProfFitConstMHD.{source_name}"
                )

    def _pf_passive(self, ods: Any) -> None:
        geometry = _field(self.raw, "VESTGeometry", required=False)
        if not isinstance(geometry, Mapping) or "Wall" not in geometry:
            return
        wall = _as_array(geometry["Wall"], "VESTGeometry.Wall")
        if wall.ndim != 2 or wall.shape[1] < 4:
            raise ValueError("VESTGeometry.Wall must be an (element, >=4) array")
        currents = None
        eddy = _field(self.raw, "Eddy", required=False)
        if isinstance(eddy, Mapping):
            raw_current = _field(eddy, "WallCurrent", required=False)
            if raw_current is not None:
                currents = np.asarray(raw_current, dtype=float)

        ods["pf_passive.ids_properties.homogeneous_time"] = 1
        ods["pf_passive.time"] = self.times
        for index, row in enumerate(wall):
            r, z, width, height = map(float, row[:4])
            base = f"pf_passive.loop.{index}"
            ods[f"{base}.name"] = f"VFIT wall element {index + 1}"
            ods[f"{base}.element.0.identifier"] = f"W{index + 1}"
            ods[f"{base}.element.0.turns_with_sign"] = 1.0
            ods[f"{base}.element.0.area"] = width * height
            ods[f"{base}.element.0.geometry.geometry_type"] = 2
            ods[f"{base}.element.0.geometry.rectangle.r"] = r
            ods[f"{base}.element.0.geometry.rectangle.z"] = z
            ods[f"{base}.element.0.geometry.rectangle.width"] = width
            ods[f"{base}.element.0.geometry.rectangle.height"] = height
            if currents is not None:
                if currents.ndim == 1 and currents.size == wall.shape[0]:
                    values = np.full(self.time_count, currents[index])
                elif currents.ndim == 2 and currents.shape[0] == wall.shape[0]:
                    values = currents[index, : self.time_count]
                elif currents.ndim == 2 and currents.shape[1] == wall.shape[0]:
                    values = currents[: self.time_count, index]
                else:
                    continue
                # ``pf_passive`` stores homogeneous loop currents directly,
                # unlike the signal structures used by ``pf_active``.
                ods[f"{base}.current"] = np.asarray(values, dtype=float)


def read_vfit(
    path: str | Path, kind: Literal["auto", "fem", "gse"] = "auto"
) -> VFITResult:
    """Read a VFIT v5/v7 MAT result.

    MATLAB v7.3 files are HDF5 containers and deliberately unsupported here;
    export the VFIT result with MATLAB's default ``save`` format instead.
    """
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"VFIT MAT result does not exist: {source}")
    try:
        loaded = loadmat(source, squeeze_me=True, struct_as_record=False)
    except (NotImplementedError, ValueError) as error:
        # scipy raises NotImplementedError for MATLAB-created v7.3 files and
        # ValueError for a minimal/third-party HDF5 v7.3 container.
        try:
            import h5py

            is_hdf5 = h5py.is_hdf5(source)
        except ImportError:
            is_hdf5 = isinstance(error, NotImplementedError)
        if not is_hdf5:
            raise
        raise ValueError(
            f"MATLAB v7.3/HDF5 MAT files are not supported: {source}. "
            "Re-save it with MATLAB's default v5/v7 MAT format."
        ) from error
    data = {
        name: _native(value)
        for name, value in loaded.items()
        if not name.startswith("__")
    }
    detected = _kind_from_data(data)
    if kind != "auto" and kind != detected:
        raise ValueError(
            f"Requested kind={kind!r}, but {source.name} contains a {detected.upper()} VFIT result"
        )
    _validate(data, detected)
    count = _find_slice_count(data, detected)
    shot = int(round(_scalar(_field(data, "shotNumber"), "shotNumber")))
    return VFITResult(
        detected, source, shot, _times_seconds(data, detected, count), data
    )


__all__ = ["VFITKind", "VFITResult", "read_vfit"]

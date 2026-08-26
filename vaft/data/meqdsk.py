"""Standalone parser and OMAS mapping for current EFIT NetCDF m-files.

The supported names follow ``EFIT/efit/write_m.F90``.  In particular, the
m-file contains the post-input-processing weights ``fwtpasma`` and ``fwtdia``;
the k-file names ``fwtcur`` and ``fwtdlc`` are intentionally not aliases here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.io import netcdf_file

EFIT_MAPPING_SOURCE_REVISION = "4d10ed592f8c9d295d393d0cf331f2d8f6be3034"

# k-file input -> current EFIT internal/m-file output.  These are processing
# stages, not interchangeable aliases.
EFIT_K_TO_M_TRANSFORMS = {
    "coils": ("silopt", "sigsil", "fwtsi", "csilop", "saisil"),
    "expmp2": ("expmpi", "sigmpi", "fwtmp2", "cmpr2", "saimpi"),
    "brsp": ("fccurt", "sigfcc", "fwtfc", "ccbrsp", "chifcc"),
    "plasma": ("plasma", "sigpasma", "fwtpasma", "cpasma", "chipasma"),
    "dflux": ("diamag", "sigdia", "fwtdia", "cdflux", "chidflux"),
}


@dataclass(frozen=True)
class MVariable:
    data: np.ndarray
    dimensions: tuple[str, ...] = ()
    attributes: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class MEQDSK:
    """Parsed EFIT m-file with native variables and NetCDF metadata."""

    variables: Mapping[str, MVariable]
    dimensions: Mapping[str, int | None]
    attributes: Mapping[str, Any]
    source: Path | None = None

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key.lower() in self.variables

    def __getitem__(self, key: str) -> MVariable:
        return self.variables[key.lower()]

    def _time_count(self) -> int:
        for name, size in self.dimensions.items():
            if "time" in name.lower() and size is not None:
                return int(size)
        return 1

    def time_seconds(self, time_index_efit: int = 0) -> float | None:
        """Return embedded EFIT time in seconds (``write_m.F90`` stores msec)."""
        value = self._at("time", time_index_efit)
        if value is None:
            return None
        return float(np.asarray(value).reshape(-1)[0]) / 1000.0

    def _at(self, name: str, time_index_efit: int) -> Any | None:
        variable = self.variables.get(name)
        if variable is None:
            return None
        data = np.asarray(variable.data)
        time_axis = next(
            (axis for axis, dim in enumerate(variable.dimensions) if "time" in dim.lower()),
            None,
        )
        if time_axis is not None and data.ndim:
            if time_index_efit >= data.shape[time_axis]:
                raise IndexError(f"m-file time index {time_index_efit} out of range for {name}")
            data = np.take(data, time_index_efit, axis=time_axis)
        if data.ndim == 0:
            return data.item()
        return data

    @staticmethod
    def _write_family(
        ods: Any,
        root: str,
        values: Mapping[str, Any | None],
        *,
        start_index: int = 0,
    ) -> None:
        present = {name: value for name, value in values.items() if value is not None}
        if not present:
            return
        arrays = {name: np.asarray(value) for name, value in present.items()}
        count = max((array.size for array in arrays.values()), default=0)
        for index in range(count):
            for field, array in arrays.items():
                flat = array.reshape(-1)
                if index < flat.size:
                    ods[f"{root}.{start_index + index}.{field}"] = flat[index].item()
            ods[f"{root}.{start_index + index}.exact"] = 0

    def to_omas(
        self,
        ods: Any = None,
        *,
        time_index: int = 0,
        time_index_efit: int = 0,
        preserve_raw: bool = True,
    ) -> Any:
        """Map current EFIT m-file output to schema-native equilibrium fields."""
        from omas import ODS

        if ods is None:
            ods = ODS()
        constraints = f"equilibrium.time_slice.{time_index}.constraints"
        families = {
            "bpol_probe": {
                "measured": self._at("expmpi", time_index_efit),
                "reconstructed": self._at("cmpr2", time_index_efit),
                "measured_error_upper": self._at("sigmpi", time_index_efit),
                "weight": self._at("fwtmp2", time_index_efit),
                "chi_squared": self._at("saimpi", time_index_efit),
            },
            "flux_loop": {
                "measured": self._at("silopt", time_index_efit),
                "reconstructed": self._at("csilop", time_index_efit),
                "measured_error_upper": self._at("sigsil", time_index_efit),
                "weight": self._at("fwtsi", time_index_efit),
                "chi_squared": self._at("saisil", time_index_efit),
            },
            "pressure": {
                "measured": self._at("pressr", time_index_efit),
                "reconstructed": self._at("cpress", time_index_efit),
                "measured_error_upper": self._at("sigpre", time_index_efit),
                "weight": self._at("fwtpre", time_index_efit),
                "chi_squared": self._at("saipre", time_index_efit),
            },
            "pressure_rotational": {
                "measured": self._at("presw", time_index_efit),
                "reconstructed": self._at("cpresw", time_index_efit),
                "measured_error_upper": self._at("sigprw", time_index_efit),
                "weight": self._at("fwtprw", time_index_efit),
                "chi_squared": self._at("saiprw", time_index_efit),
            },
            "j_tor": {
                "measured": self._at("vzeroj", time_index_efit),
                "reconstructed": self._at("cjtr", time_index_efit),
                "measured_error_upper": self._at("sigjtr", time_index_efit),
                "weight": self._at("fwtjtr", time_index_efit),
                "chi_squared": self._at("chijtr", time_index_efit),
            },
        }
        for family, values in families.items():
            self._write_family(ods, f"{constraints}.{family}", values)

        mse_values = {
            "measured": self._at("tangam", time_index_efit),
            "reconstructed": self._at("cmgam", time_index_efit),
            "measured_error_upper": self._at("siggam", time_index_efit),
            "weight": self._at("fwtgam", time_index_efit),
            "chi_squared": self._at("chigam", time_index_efit),
        }
        # EFIT stores tan(gamma); OMAS stores the polarisation angle.
        for key in ("measured", "reconstructed", "measured_error_upper"):
            if mse_values[key] is not None:
                mse_values[key] = np.arctan(mse_values[key])
        self._write_family(ods, f"{constraints}.mse_polarisation_angle", mse_values)

        pf_groups = (
            ("eccurt", "cecurr", "sigecc", "fwtec", "chiecc"),
            ("fccurt", "ccbrsp", "sigfcc", "fwtfc", "chifcc"),
            ("accurt", "caccurt", None, None, None),
        )
        pf_offset = 0
        for measured, reconstructed, error, weight, chi in pf_groups:
            values = {
                "measured": self._at(measured, time_index_efit),
                "reconstructed": self._at(reconstructed, time_index_efit),
                "measured_error_upper": self._at(error, time_index_efit) if error else None,
                "weight": self._at(weight, time_index_efit) if weight else None,
                "chi_squared": self._at(chi, time_index_efit) if chi else None,
            }
            # Current write_m builds can emit ``caccurt`` without matching
            # measured A-coil currents.  Keep that reconstruction-only vector
            # in raw meqdsk parameters instead of inventing an OMAS constraint.
            if values["measured"] is None or values["reconstructed"] is None:
                continue
            count = max((np.asarray(v).size for v in values.values() if v is not None), default=0)
            if count:
                self._write_family(
                    ods,
                    f"{constraints}.pf_current",
                    values,
                    start_index=pf_offset,
                )
                pf_offset += count

        scalars = {
            "ip": {
                "measured": "plasma", "reconstructed": "cpasma",
                "measured_error_upper": "sigpasma", "weight": "fwtpasma",
                "chi_squared": "chipasma",
            },
            "diamagnetic_flux": {
                "measured": "diamag", "reconstructed": "cdflux",
                "measured_error_upper": "sigdia", "weight": "fwtdia",
                "chi_squared": "chidflux",
            },
        }
        for family, mapping in scalars.items():
            wrote = False
            for field, variable in mapping.items():
                value = self._at(variable, time_index_efit)
                if value is not None:
                    ods[f"{constraints}.{family}.{field}"] = float(np.asarray(value))
                    wrote = True
            if wrote:
                ods[f"{constraints}.{family}.exact"] = 0

        cerror = self._at("cerror", time_index_efit)
        if cerror is not None:
            errors = np.asarray(cerror, dtype=float).reshape(-1)
            ods[f"equilibrium.time_slice.{time_index}.convergence.iterations_n"] = len(errors)
            if errors.size:
                root = f"equilibrium.time_slice.{time_index}.convergence"
                ods[f"{root}.grad_shafranov_deviation_expression.index"] = 3
                ods[f"{root}.grad_shafranov_deviation_expression.name"] = "max_absolute_psi_residual"
                ods[f"{root}.grad_shafranov_deviation_value"] = float(errors[-1])
        for name in ("chitot", "chifin", "cchisq"):
            value = self._at(name, time_index_efit)
            if value is not None:
                array = np.asarray(value, dtype=float).reshape(-1)
                if array.size:
                    ods[f"{constraints}.chi_squared_reduced"] = float(array[-1])
                    ods[f"{constraints}.freedom_degrees_n"] = 1
                    break

        coil_variables = (
            ("curc79", "C79"), ("curc139", "C139"), ("curc199", "C199"),
            ("curiu30", "IU30"), ("curil30", "IL30"),
            ("curiu90", "IU90"), ("curil90", "IL90"),
            ("curiu150", "IU150"), ("curil150", "IL150"),
        )
        try:
            time_value = float(ods[f"equilibrium.time_slice.{time_index}.time"])
        except Exception:
            time_value = self.time_seconds(time_index_efit) or 0.0
        coil_index = 0
        for variable, identifier in coil_variables:
            value = self._at(variable, time_index_efit)
            if value is None:
                continue
            root = f"coils_non_axisymmetric.coil.{coil_index}"
            ods[f"{root}.identifier"] = identifier
            ods[f"{root}.current.time"] = np.asarray([time_value])
            ods[f"{root}.current.data"] = np.asarray([float(np.asarray(value))])
            coil_index += 1

        if preserve_raw:
            raw = f"equilibrium.code.parameters.time_slice.{time_index}.meqdsk"
            ods[f"{raw}.mapping_source_revision"] = EFIT_MAPPING_SOURCE_REVISION
            for name, size in self.dimensions.items():
                ods[f"{raw}.dimensions.{name}"] = -1 if size is None else int(size)
            for name, variable in self.variables.items():
                value = self._at(name, time_index_efit)
                if value is not None:
                    ods[f"{raw}.variables.{name}.data"] = value
                ods[f"{raw}.variables.{name}.dimensions"] = list(variable.dimensions)
                for attr, attr_value in (variable.attributes or {}).items():
                    ods[f"{raw}.variables.{name}.attributes.{attr}"] = attr_value
            for attr, value in self.attributes.items():
                ods[f"{raw}.attributes.{attr}"] = value
        return ods


def _decode(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").rstrip("\x00")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray) and value.dtype.kind in {"S", "U"}:
        return np.asarray([_decode(item) for item in value.reshape(-1)]).reshape(value.shape)
    return value


def read_meqdsk(path: str | Path) -> MEQDSK:
    """Read a NetCDF m-file emitted by the current EFIT ``write_m.F90``."""
    source = Path(path).expanduser()
    with netcdf_file(str(source), "r", mmap=False) as dataset:
        dimensions = {str(name): size for name, size in dataset.dimensions.items()}
        attributes = {str(name): _decode(value) for name, value in dataset._attributes.items()}
        variables = {}
        for name, variable in dataset.variables.items():
            attrs = {str(key): _decode(value) for key, value in variable._attributes.items()}
            variables[str(name).lower()] = MVariable(
                data=np.array(variable[:], copy=True),
                dimensions=tuple(str(dim) for dim in variable.dimensions),
                attributes=attrs,
            )
    return MEQDSK(variables, dimensions, attributes, source=source)


__all__ = [
    "EFIT_K_TO_M_TRANSFORMS",
    "EFIT_MAPPING_SOURCE_REVISION",
    "MEQDSK",
    "MVariable",
    "read_meqdsk",
]

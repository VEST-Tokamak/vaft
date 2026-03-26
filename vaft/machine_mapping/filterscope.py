"""Backward-compatible filterscope alias for canonical spectrometer_uv mapping."""

from __future__ import annotations

from .spectrometer_uv import spectrometer_uv as filterscope, vfit_filterscope

__all__ = ["filterscope", "vfit_filterscope"]

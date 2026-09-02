"""DataFrame-based parameter history plotting."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import pandas as pd


def plot_parameter_history(
    df: pd.DataFrame,
    *,
    y: str | Sequence[str],
    x: str = "shot",
    ax=None,
    show: bool = False,
):
    """Plot one or more numeric summary columns against another table column."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    columns = (y,) if isinstance(y, str) else tuple(y)
    if not columns or any(not isinstance(column, str) or not column for column in columns):
        raise ValueError("y must be a column name or a non-empty sequence of column names")
    requested = (x, *columns)
    missing = [column for column in requested if column not in df]
    if missing:
        raise ValueError(f"DataFrame is missing plot columns: {missing}")
    non_numeric = [
        column for column in requested
        if not df.empty and not pd.api.types.is_numeric_dtype(df[column])
    ]
    if non_numeric:
        raise TypeError(f"history plot columns must be numeric: {non_numeric}")

    if ax is None:
        figure, axes = plt.subplots()
    else:
        axes = ax
        figure = axes.figure
    for column in columns:
        axes.plot(df[x], df[column], marker="o", label=column)
    axes.set_xlabel(x)
    axes.set_ylabel(columns[0] if len(columns) == 1 else "value")
    if len(columns) > 1:
        axes.legend()
    axes.grid(True, alpha=0.3)
    if show:
        plt.show()
    return figure, axes


__all__ = ["plot_parameter_history"]

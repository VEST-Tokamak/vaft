"""DataFrame-based parameter history plotting."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedFormatter, FixedLocator


def _parse_dates(values: pd.Series) -> list[pd.Timestamp | None]:
    """Parse each timestamp independently, including mixed ISO 8601 offsets."""
    parsed = []
    for value in values:
        if pd.isna(value) or isinstance(value, (bool, int, float, np.number)):
            parsed.append(None)
            continue
        try:
            stamp = pd.Timestamp(value)
        except (TypeError, ValueError, OverflowError):
            stamp = pd.NaT
        parsed.append(None if pd.isna(stamp) else stamp)
    return parsed


def _date_label(stamp: pd.Timestamp, span_days: float) -> str:
    if span_days >= 730:
        return stamp.strftime("%Y")
    if span_days >= 90:
        return stamp.strftime("%b %Y")
    if span_days >= 2:
        return stamp.strftime("%b %d")
    return stamp.strftime("%b %d %H:%M")


def _observed_ticks(
    positions: Sequence[float], labels: Sequence[object], *, coordinate: str
) -> tuple[list[float], list[object]]:
    """Select sparse observed positions; refuse contradictory labels."""
    observed: dict[float, object] = {}
    for position, label in zip(positions, labels):
        if not np.isfinite(position) or label is None:
            continue
        position = float(position)
        if position in observed and observed[position] != label:
            raise ValueError(
                f"the same {coordinate} position has different secondary-axis values"
            )
        observed[position] = label
    if not observed:
        return [], []
    locations = sorted(observed)
    if len(locations) > 6:
        indices = np.linspace(0, len(locations) - 1, 6, dtype=int)
        locations = [locations[index] for index in indices]
    return locations, [observed[location] for location in locations]


def plot_parameter_history(
    df: pd.DataFrame,
    *,
    y: str | Sequence[str],
    x: str = "shot",
    secondary_x: str | None = None,
    date_column: str = "pulse_time_begin",
    ax=None,
    show: bool = False,
):
    """Plot summary rows against shot, acquisition time, or a numeric column.

    A secondary axis labels observed positions only; it does not interpolate
    between shot number and acquisition time. Rows are never aggregated.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    columns = (y,) if isinstance(y, str) else tuple(y)
    if not columns or any(not isinstance(column, str) or not column for column in columns):
        raise ValueError("y must be a column name or a non-empty sequence of column names")
    if not isinstance(x, str) or not x:
        raise ValueError("x must be a column name or 'date'")
    if secondary_x not in (None, "shot", "date"):
        raise ValueError("secondary_x must be None, 'shot', or 'date'")
    if secondary_x == x:
        raise ValueError("secondary_x must differ from x")
    if not isinstance(date_column, str) or not date_column:
        raise ValueError("date_column must be a non-empty column name")

    requested = (date_column if x == "date" else x, *columns)
    if secondary_x == "date":
        requested += (date_column,)
    elif secondary_x == "shot":
        requested += ("shot",)
    missing = [column for column in dict.fromkeys(requested) if column not in df]
    if missing:
        raise ValueError(f"DataFrame is missing plot columns: {missing}")
    numeric = (*columns, *((x,) if x != "date" else ()))
    non_numeric = [
        column for column in numeric
        if not df.empty and not pd.api.types.is_numeric_dtype(df[column])
    ]
    if non_numeric:
        raise TypeError(f"history plot columns must be numeric: {non_numeric}")

    dates = None
    if x == "date" or secondary_x == "date":
        dates = _parse_dates(df[date_column])
        if not any(date is not None for date in dates):
            raise ValueError(f"no usable acquisition dates in {date_column!r}")

    if x == "date":
        positions = np.array([
            mdates.date2num(date.to_pydatetime()) if date is not None else np.nan
            for date in dates
        ])
        plotted = np.isfinite(positions)
    else:
        positions = np.asarray(df[x], dtype=float)
        plotted = np.ones(len(df), dtype=bool)

    if ax is None:
        figure, axes = plt.subplots()
    else:
        axes = ax
        figure = axes.figure
    for column in columns:
        axes.plot(positions[plotted], df[column].to_numpy()[plotted], marker="o", label=column)
    if x == "date":
        locator = mdates.AutoDateLocator()
        axes.xaxis.set_major_locator(locator)
        axes.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    axes.set_xlabel(date_column if x == "date" else x)
    axes.set_ylabel(columns[0] if len(columns) == 1 else "value")
    if len(columns) > 1:
        axes.legend()
    axes.grid(True, alpha=0.3)

    if secondary_x is not None:
        labels = dates if secondary_x == "date" else df["shot"].tolist()
        tick_positions, tick_values = _observed_ticks(
            positions, labels, coordinate="date" if x == "date" else x
        )
        if not tick_positions:
            raise ValueError(f"no usable secondary-axis values for {secondary_x!r}")
        if secondary_x == "date":
            valid_dates = [date for date in dates if date is not None]
            date_positions = [mdates.date2num(date.to_pydatetime()) for date in valid_dates]
            span_days = max(date_positions) - min(date_positions)
            tick_labels = [_date_label(value, span_days) for value in tick_values]
        else:
            tick_labels = [str(value) for value in tick_values]
        secondary = axes.secondary_xaxis("top", functions=(lambda value: value, lambda value: value))
        secondary.xaxis.set_major_locator(FixedLocator(tick_positions))
        secondary.xaxis.set_major_formatter(FixedFormatter(tick_labels))
        secondary.set_xlabel(date_column if secondary_x == "date" else "shot")

    if show:
        plt.show()
    return figure, axes


__all__ = ["plot_parameter_history"]

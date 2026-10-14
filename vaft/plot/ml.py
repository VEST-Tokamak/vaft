"""ML experiment adapters to immutable canonical plot view models."""
__all__ = ['learning_curves', 'group_metrics', 'resource_comparison']

import numpy as np

from .models import LineSeries, Series


def learning_curves(history, *, title="Training history", keys=None):
    """Adapt loss components or learning rates; units follow the recorded loss."""
    keys = tuple(history) if keys is None else tuple(keys)
    return LineSeries(tuple(Series(np.arange(len(history[k])), np.asarray(history[k]), label=k) for k in keys),
                      x_label="Epoch", y_label="Recorded metric", title=title)


def group_metrics(evaluation, metric):
    """Independent-group metrics, using group labels as trace identities."""
    values = [(group, scores[metric]) for group, scores in evaluation.per_group.items() if metric in scores]
    return LineSeries(tuple(Series(np.array([i]), np.array([value]), label=group) for i, (group, value) in enumerate(values)),
                      x_label="Independent group index", y_label=metric, title="Group evaluation")


def resource_comparison(records, metric="fit_seconds"):
    """Compare explicitly recorded compute metrics across local experiment runs."""
    values = [(str(r.get("run_id", i)), r.get("training", {}).get("resources", {}).get(metric)) for i, r in enumerate(records)]
    return LineSeries(tuple(Series(np.array([i]), np.array([value]), label=run) for i, (run, value) in enumerate(values) if value is not None),
                      x_label="Run index", y_label=metric, title="Experiment resources")

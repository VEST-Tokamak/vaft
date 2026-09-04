"""Log-linear regression of confinement time against engineering parameters.

The workflow behind the confinement-scaling study: take a table of shots
with their engineering parameters and energy confinement time, take
natural logarithms, fit ``ln tau_E = c0 + sum_i c_i ln p_i`` by ordinary
least squares, and read the scaling exponents off the coefficients.
Everything operates on ``pandas`` DataFrames whose columns are named by
the caller; nothing here knows which columns exist except through the
``eng_params`` list it is handed.

Two functions are not statistics and are here for historical reasons:
:func:`generate_core_profiles_history_dataframe` builds the input table by
running a workflow script, and :func:`confinement_time_histogram` draws
one.  Both are documented as they are; their home is decided by #263.

Notation
--------
tau_E     : energy confinement time            [s]
p_i       : an engineering parameter           [any]
c_i       : the scaling exponent of ``p_i``    [-]
alpha     : significance level                 [-]

Provenance
----------
.. [NB] ``notebooks/confinement_time_scaling.ipynb``, the study these
   functions were extracted from.
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from typing import Dict, List, Tuple, Optional, Union
import logging
import math


__all__ = [
    "RegressionResults",
    "analyze_significance",
    "compute_metrics",
    "confinement_time_histogram",
    "filter_dataframe",
    "generate_core_profiles_history_dataframe",
    "get_correlation_matrix",
    "get_individual_correlations",
    "get_residuals",
    "load_data_from_excel",
    "log_transform",
    "perform_ols_regression",
]

logger = logging.getLogger(__name__)


class RegressionResults:
    """Container for regression analysis results."""
    
    def __init__(self, model, log_df, eng_params, target_param):
        self.model = model
        self.log_df = log_df
        self.eng_params = eng_params
        self.target_param = target_param
        self.coefficients = model.params
        self.pvalues = model.pvalues
        self.rsquared = model.rsquared
        self.rsquared_adj = model.rsquared_adj
        self.residuals = model.resid
        self.fitted_values = model.fittedvalues
        
    def get_summary(self) -> pd.DataFrame:
        """Get summary of regression coefficients and p-values."""
        idx = self.coefficients.index
        summary = pd.DataFrame(index=idx)
        summary['Coefficient'] = self.coefficients
        summary['P-value'] = self.pvalues.reindex(idx)
        summary['Significant'] = self.pvalues.reindex(idx) < 0.05
        return summary
    
    def get_exponents(self) -> Dict[str, float]:
        """Get scaling law exponents (excluding constant term)."""
        exponents = {}
        for param in self.eng_params:
            param_name = f'ln_{param}'
            if param_name in self.coefficients:
                exponents[param] = float(self.coefficients[param_name])
        return exponents


def load_data_from_excel(filepath: str) -> pd.DataFrame:
    """Load a confinement-scaling table from an Excel file.

    Parameters
    ----------
    filepath : str
        Path to the ``.xlsx`` file; one row per shot, columns as the study
        named them [-].

    Returns
    -------
    pd.DataFrame
        The table as read, untouched [-].

    Raises
    ------
    Exception
        Whatever ``pandas.read_excel`` raises, after logging it.

    Applicability
    -------------
    Machine-independent.  The tables under
    ``workflow/automatic_pipeline_3_data_summary/`` are VEST, but nothing here
    depends on that.

    Provenance
    ----------
    .. [1] The study notebook [NB]_.
    """
    try:
        df = pd.read_excel(filepath)
        logger.info(f"Loaded {len(df)} rows from {filepath}")
        return df
    except Exception as e:
        logger.error(f"Error loading Excel file: {e}")
        raise

def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Drop shots with unrealistic loss power and rectify the toroidal field sign.

    Parameters
    ----------
    df : pd.DataFrame
        The scaling table; ``Ploss_MW`` and ``Bt_T`` are acted on when present
        [-].

    Returns
    -------
    pd.DataFrame
        The filtered table [-].

    Processing steps
    ----------------
    1. Keep rows with ``Ploss_MW <= 3``.
    2. Replace ``Bt_T`` by its absolute value.

    Defaults
    --------
    The 3 MW ceiling is a hard-coded empirical cut from the study; its
    derivation is not recorded.  It is not a parameter.

    Convention
    ----------
    ``Bt_T`` is made positive regardless of the field direction actually run,
    so the fit sees magnitude only.  Any sign convention on the field is
    discarded here.

    Applicability
    -------------
    VEST-specific.  The ceiling and the column names are the VEST study's.

    Limitations
    -----------
    Assigns into the frame it was given (``df["Bt_T"] = ...``), so the
    caller's table may be modified and pandas may warn about a copy.
    """
    if 'Ploss_MW' in df.columns:
        df = df[df['Ploss_MW'] <= 3]

    # Ensure Bt_T is always positive
    if 'Bt_T' in df.columns:
        df['Bt_T'] = df['Bt_T'].abs()
    return df

def generate_core_profiles_history_dataframe(max_shots: Optional[int] = None, 
                                            Z_eff: float = 2.0) -> pd.DataFrame:
    """Build the scaling table by running the core-profiles history workflow.

    Parameters
    ----------
    max_shots : int or None, optional
        Stop after this many shots; ``None`` for all [-].
    Z_eff : float, optional
        Effective charge assumed for every shot when computing the ion
        contribution [-].

    Returns
    -------
    pd.DataFrame
        One row per shot with the engineering parameters and ``tauE_s`` [-].

    Raises
    ------
    ImportError
        The workflow script cannot be found or loaded.
    ValueError
        The script returned nothing.

    Processing steps
    ----------------
    1. Locate ``workflow/automatic_pipeline_3_data_summary/gen_core_profiles_history.py``
       relative to this file and load it by path.
    2. Call its ``generate_core_profiles_history_excel(max_shots, Z_eff)``.

    Defaults
    --------
    ``Z_eff = 2.0`` is an assumed value, used because no shot-resolved
    measurement exists; its origin is not recorded.

    Applicability
    -------------
    VEST-specific.  The script reads the VEST database.

    Limitations
    -----------
    Loads a workflow script by file path and inserts its directory into
    ``sys.path``, which only works from a source checkout and is a
    layer-boundary violation: a process function reaching up into a workflow.
    Tracked in #263.

    Provenance
    ----------
    .. [1] ``workflow/automatic_pipeline_3_data_summary/gen_core_profiles_history.py``.
    """
    import sys
    import os
    from pathlib import Path
    
    # Import gen_core_profiles_history module
    # Try to find the module in the workflow directory
    vaft_root = Path(__file__).parent.parent.parent
    workflow_path = vaft_root / 'workflow' / 'automatic_pipeline_3_data_summary'
    
    if str(workflow_path) not in sys.path:
        sys.path.insert(0, str(workflow_path))
    
    try:
        # Dynamic import to avoid linter warnings
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "gen_core_profiles_history",
            workflow_path / "gen_core_profiles_history.py"
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load gen_core_profiles_history from {workflow_path}")
        gen_core_profiles_history = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gen_core_profiles_history)
        
        df = gen_core_profiles_history.generate_core_profiles_history_excel(
            max_shots=max_shots, Z_eff=Z_eff
        )
        if df is None:
            raise ValueError("generate_core_profiles_history_excel returned None")
        logger.info(f"Generated DataFrame with {len(df)} rows")
        return df
    except ImportError as e:
        logger.error(f"Could not import gen_core_profiles_history: {e}")
        raise
    except Exception as e:
        logger.error(f"Error generating core profiles history: {e}")
        raise


def log_transform(df: pd.DataFrame, 
                  eng_params: List[str], 
                  target_param: str = 'tauE_s') -> pd.DataFrame:
    """Take natural logarithms of the engineering parameters and the target.

    Parameters
    ----------
    df : pd.DataFrame
        The scaling table [-].
    eng_params : list of str
        Column names of the engineering parameters [-].
    target_param : str, optional
        Column name of the confinement time [-].

    Returns
    -------
    pd.DataFrame
        Columns ``ln_<name>`` for each parameter and the target, same index as
        ``df``; a missing column becomes all-NaN with a warning [-].

    Convention
    ----------
    Natural logarithm, so the fitted coefficients are the exponents of a
    power law directly.  A value that is not strictly positive has no
    logarithm and becomes NaN, to be dropped by :func:`perform_ols_regression`.

    Applicability
    -------------
    Machine-independent.
    """
    log_df = pd.DataFrame(index=df.index)
    
    # Log transform engineering parameters
    for param in eng_params:
        if param in df.columns:
            # Handle NaN and non-positive values
            valid_mask = (df[param] > 0) & ~np.isnan(df[param])
            log_df[f'ln_{param}'] = np.where(valid_mask, np.log(df[param]), np.nan)
        else:
            logger.warning(f"Parameter {param} not found in DataFrame")
            log_df[f'ln_{param}'] = np.nan
    
    # Log transform target parameter
    if target_param in df.columns:
        valid_mask = (df[target_param] > 0) & ~np.isnan(df[target_param])
        log_df[f'ln_{target_param}'] = np.where(valid_mask, np.log(df[target_param]), np.nan)
    else:
        logger.warning(f"Target parameter {target_param} not found in DataFrame")
        log_df[f'ln_{target_param}'] = np.nan
    
    return log_df


def perform_ols_regression(df: pd.DataFrame,
                           eng_params: List[str],
                           target_param: str = 'tauE_s',
                           dropna: bool = True) -> RegressionResults:
    """Fit the log-linear scaling by ordinary least squares.

    Parameters
    ----------
    df : pd.DataFrame
        The scaling table, in original units [-].
    eng_params : list of str
        Column names of the engineering parameters to regress on [-].
    target_param : str, optional
        Column name of the confinement time [-].
    dropna : bool, optional
        Drop rows with any NaN among the regressors or the target [-].

    Returns
    -------
    RegressionResults
        The fitted ``statsmodels`` model with the log table, the parameter
        list and the target name; ``get_exponents()`` reads the scaling
        exponents off it [-].

    Raises
    ------
    ValueError
        No rows remain after dropping NaN.

    Processing steps
    ----------------
    1. :func:`log_transform` the table.
    2. Select the ``ln_`` regressors and the ``ln_`` target.
    3. Drop rows with NaN, if ``dropna``.
    4. Add an intercept column and fit ``statsmodels.api.OLS``.

    Defaults
    --------
    ``dropna = True`` is a validated-workflow default; ``statsmodels`` would
    otherwise raise on NaN.

    Convention
    ----------
    The fit is in log space, so the coefficients are exponents and the
    residuals are relative errors.  Metrics in original units come from
    :func:`compute_metrics`, which exponentiates the fitted values.

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [1] The study notebook [NB]_.
    """
    # statsmodels is imported here, not at module scope: it is the heaviest
    # dependency in vaft.process and only this one regression needs it (#249).
    import statsmodels.api as sm

    # Log transform
    log_df = log_transform(df, eng_params, target_param)
    
    # Prepare independent variables
    X_cols = [f'ln_{p}' for p in eng_params]
    X = log_df[X_cols].copy()
    y = log_df[f'ln_{target_param}'].copy()
    
    # Drop rows with NaN values if requested
    if dropna:
        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_mask]
        y = y[valid_mask]
        logger.info(f"Using {len(X)} valid data points after dropping NaN values")
    
    if len(X) == 0:
        raise ValueError("No valid data points after preprocessing")
    
    # Add constant term for intercept
    X_with_const = sm.add_constant(X)
    
    # Perform OLS regression
    model = sm.OLS(y, X_with_const).fit()
    
    logger.info(f"OLS Regression completed:")
    logger.info(f"  R-squared: {model.rsquared:.4f}")
    logger.info(f"  Adjusted R-squared: {model.rsquared_adj:.4f}")
    logger.info(f"  Number of observations: {len(X)}")
    
    return RegressionResults(model, log_df, eng_params, target_param)


def analyze_significance(results: RegressionResults, 
                         alpha: float = 0.05) -> Dict[str, bool]:
    """Flag which scaling exponents are statistically significant.

    Parameters
    ----------
    results : RegressionResults
        A fitted regression [-].
    alpha : float, optional
        Significance level; a coefficient with ``p < alpha`` is significant
        [-].

    Returns
    -------
    dict of str to bool
        Parameter name to whether its exponent is significant; a parameter
        absent from the fit is ``False`` [-].

    Defaults
    --------
    ``alpha = 0.05`` is the conventional literature value, no more.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    A p-value from a small sample with correlated regressors overstates
    significance; see :func:`get_correlation_matrix` before trusting it.
    """
    significance = {}
    for param in results.eng_params:
        param_name = f'ln_{param}'
        if param_name in results.pvalues:
            pval = results.pvalues[param_name]
            significance[param] = pval < alpha
        else:
            significance[param] = False
    
    return significance


def compute_metrics(results: RegressionResults, 
                   df: pd.DataFrame,
                   target_param: str = 'tauE_s') -> Dict[str, float]:
    """Goodness-of-fit metrics of the regression in original units.

    Parameters
    ----------
    results : RegressionResults
        A fitted regression [-].
    df : pd.DataFrame
        The original table the regression was fitted from [-].
    target_param : str, optional
        Column name of the confinement time [-].

    Returns
    -------
    dict of str to float
        ``R2``, ``RMSE``, ``MAE``, ``Mean_Relative_Error_%`` and
        ``Median_Relative_Error_%`` [-].

    Processing steps
    ----------------
    1. Exponentiate the fitted log values to get predicted confinement times.
    2. Select the actual values at the rows the fit used.
    3. Compute the metrics with ``sklearn.metrics`` and NumPy.

    Convention
    ----------
    Metrics are in the *original* units, so ``R2`` here is not the ``rsquared``
    of the log fit on the results object; the two differ and both are
    reported deliberately.  ``RMSE`` and ``MAE`` carry the target's unit,
    seconds for ``tauE_s``, and are returned as plain floats.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Imports ``scikit-learn`` on call for two one-line metrics.  When the
    fitted values carry no index the actual values are truncated to the
    shorter length, which silently misaligns rows if any were dropped.
    """
    from sklearn.metrics import r2_score, mean_squared_error
    
    # Get predicted values in original scale
    y_pred_log = results.fitted_values
    y_pred = np.exp(y_pred_log)
    
    # Get actual values using the same indices as fitted_values
    # fitted_values has the index of rows that were actually used in regression
    if hasattr(y_pred_log, 'index'):
        # fitted_values is a pandas Series with indices
        valid_indices = y_pred_log.index
        y_actual = df.loc[valid_indices, target_param].values
    else:
        # fitted_values is a numpy array (fallback)
        log_df = results.log_df
        valid_mask = ~log_df[f'ln_{target_param}'].isna()
        y_actual = df.loc[log_df.index[valid_mask], target_param].values
        
        # Filter to matching length if needed
        if len(y_pred) != len(y_actual):
            min_len = min(len(y_pred), len(y_actual))
            y_pred = y_pred[:min_len]
            y_actual = y_actual[:min_len]
    
    # Compute metrics
    r2 = r2_score(y_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    mae = np.mean(np.abs(y_actual - y_pred))
    
    # Relative errors
    relative_error = np.abs((y_actual - y_pred) / y_actual) * 100
    mean_relative_error = np.mean(relative_error)
    median_relative_error = np.median(relative_error)
    
    metrics = {
        'R2': r2,
        'RMSE': rmse,
        'MAE': mae,
        'Mean_Relative_Error_%': mean_relative_error,
        'Median_Relative_Error_%': median_relative_error
    }
    
    return metrics


def get_residuals(results: RegressionResults) -> pd.Series:
    """Residuals of the log-space fit.

    Parameters
    ----------
    results : RegressionResults
        A fitted regression [-].

    Returns
    -------
    pd.Series
        ``ln(actual) - ln(fitted)`` per row used in the fit [-].

    Applicability
    -------------
    Machine-independent.
    """
    return results.residuals


def get_correlation_matrix(log_df: pd.DataFrame, 
                          eng_params: List[str],
                          target_param: str = 'tauE_s') -> pd.DataFrame:
    """Pearson correlation matrix of the log-transformed regressors and target.

    Parameters
    ----------
    log_df : pd.DataFrame
        Output of :func:`log_transform` [-].
    eng_params : list of str
        Column names of the engineering parameters [-].
    target_param : str, optional
        Column name of the confinement time [-].

    Returns
    -------
    pd.DataFrame
        The correlation matrix over the available ``ln_`` columns; empty when
        fewer than two exist [-].

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Pairwise-complete, as ``DataFrame.corr`` is; rows are not dropped
    consistently across pairs.
    """
    cols = [f'ln_{p}' for p in eng_params] + [f'ln_{target_param}']
    available_cols = [c for c in cols if c in log_df.columns]
    
    if len(available_cols) < 2:
        logger.warning("Not enough columns for correlation matrix")
        return pd.DataFrame()
    
    corr_df = log_df[available_cols].corr()
    return corr_df


def get_individual_correlations(log_df: pd.DataFrame,
                               eng_params: List[str],
                               target_param: str = 'tauE_s') -> Dict[str, float]:
    """Pearson correlation of each log regressor with the log target.

    Parameters
    ----------
    log_df : pd.DataFrame
        Output of :func:`log_transform` [-].
    eng_params : list of str
        Column names of the engineering parameters [-].
    target_param : str, optional
        Column name of the confinement time [-].

    Returns
    -------
    dict of str to float
        Parameter name to its correlation with the target; NaN when the
        column is missing or fewer than two complete rows exist [-].

    Applicability
    -------------
    Machine-independent.
    """
    correlations = {}
    target_col = f'ln_{target_param}'
    
    if target_col not in log_df.columns:
        logger.warning(f"Target column {target_col} not found")
        return correlations
    
    for param in eng_params:
        param_col = f'ln_{param}'
        if param_col in log_df.columns:
            # Remove NaN values for correlation calculation
            valid_mask = ~(log_df[param_col].isna() | log_df[target_col].isna())
            if valid_mask.sum() > 1:
                corr, _ = pearsonr(log_df[param_col][valid_mask], 
                                  log_df[target_col][valid_mask])
                correlations[param] = corr
            else:
                correlations[param] = np.nan
        else:
            correlations[param] = np.nan
    
    return correlations



# Setup logger to prevent NameError
logger = logging.getLogger(__name__)

def confinement_time_histogram(df: pd.DataFrame,
                               eng_params: Optional[List[str]] = None,
                               m: Optional[int] = None,
                               n: Optional[int] = None,
                               figsize: Optional[Tuple[float, float]] = None,
                               bins: Union[int, str] = 30,
                               alpha: float = 0.7,
                               edgecolor: str = 'black',
                               **kwargs):
    """Plot histograms of the scaling parameters in an ``m x n`` grid.

    Parameters
    ----------
    df : pd.DataFrame
        The scaling table [-].
    eng_params : list of str or None, optional
        Columns to plot; ``None`` for the renderer's default set [-].
    m : int or None, optional
        Grid rows [-].
    n : int or None, optional
        Grid columns [-].
    figsize : tuple of (float, float) or None, optional
        Figure size in inches [-].
    bins : int or str, optional
        Histogram bins, as Matplotlib takes them [-].
    alpha : float, optional
        Bar opacity [-].
    edgecolor : str, optional
        Bar edge colour [-].

    Returns
    -------
    matplotlib.figure.Figure
        The figure, as :func:`vaft.plot.history.confinement_time_histogram`
        returns it [-].

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Rendering lives in :func:`vaft.plot.history.confinement_time_histogram`
    (issue #63); this is a compatibility shim that keeps the old call site
    and pulls Matplotlib in on call.  Its removal is tracked in #263.
    """
    from vaft.plot.history import confinement_time_histogram as _render

    return _render(df, eng_params=eng_params, m=m, n=n, figsize=figsize,
                   bins=bins, alpha=alpha, edgecolor=edgecolor, **kwargs)


"""
Statistical Analysis Module for Confinement Time Scaling

This module provides functions for performing statistical analysis on confinement time
scaling data, including log transformation, OLS regression, and significance testing.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import pearsonr
from typing import Dict, List, Tuple, Optional, Union
import logging
import matplotlib.pyplot as plt
import math

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
        summary = pd.DataFrame({
            'Coefficient': self.coefficients,
            'P-value': self.pvalues,
            'Significant': self.pvalues < 0.05
        })
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
    """Load confinement time scaling data from Excel file.
    
    Args:
        filepath: Path to Excel file
        
    Returns:
        DataFrame with confinement time scaling parameters
    """
    try:
        df = pd.read_excel(filepath)
        logger.info(f"Loaded {len(df)} rows from {filepath}")
        return df
    except Exception as e:
        logger.error(f"Error loading Excel file: {e}")
        raise

def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Filter dataframe to exclude shots with unrealistic values or correct sign errors"""
    if 'Ploss_MW' in df.columns:
        df = df[df['Ploss_MW'] <= 3]

    # Ensure Bt_T is always positive
    if 'Bt_T' in df.columns:
        df['Bt_T'] = df['Bt_T'].abs()
    return df

def generate_core_profiles_history_dataframe(max_shots: Optional[int] = None, 
                                            Z_eff: float = 2.0) -> pd.DataFrame:
    """Generate core profiles history DataFrame using gen_core_profiles_history.
    
    This function wraps the gen_core_profiles_history module to generate
    a DataFrame with confinement time scaling parameters.
    
    Args:
        max_shots: Maximum number of shots to process (None for all)
        Z_eff: Effective charge number (default: 2.0)
        
    Returns:
        DataFrame with confinement time scaling parameters
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
    """Perform log transformation on engineering parameters and target parameter.
    
    Args:
        df: DataFrame with original parameters
        eng_params: List of engineering parameter column names
        target_param: Target parameter column name (default: 'tauE_s')
        
    Returns:
        DataFrame with log-transformed parameters
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
    """Perform OLS regression on log-transformed data.
    
    Args:
        df: DataFrame with original parameters
        eng_params: List of engineering parameter column names
        target_param: Target parameter column name (default: 'tauE_s')
        dropna: Whether to drop rows with NaN values (default: True)
        
    Returns:
        RegressionResults object with regression analysis results
    """
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
    """Analyze statistical significance of regression coefficients.
    
    Args:
        results: RegressionResults object
        alpha: Significance level (default: 0.05)
        
    Returns:
        Dictionary mapping parameter names to significance status
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
    """Compute performance metrics for the regression model.
    
    Args:
        results: RegressionResults object
        df: Original DataFrame
        target_param: Target parameter column name
        
    Returns:
        Dictionary with performance metrics
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
    """Get residuals from regression model.
    
    Args:
        results: RegressionResults object
        
    Returns:
        Series with residuals
    """
    return results.residuals


def get_correlation_matrix(log_df: pd.DataFrame, 
                          eng_params: List[str],
                          target_param: str = 'tauE_s') -> pd.DataFrame:
    """Compute correlation matrix for log-transformed parameters.
    
    Args:
        log_df: DataFrame with log-transformed parameters
        eng_params: List of engineering parameter column names
        target_param: Target parameter column name
        
    Returns:
        Correlation matrix as DataFrame
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
    """Get correlation coefficients between each parameter and target.
    
    Args:
        log_df: DataFrame with log-transformed parameters
        eng_params: List of engineering parameter column names
        target_param: Target parameter column name
        
    Returns:
        Dictionary mapping parameter names to correlation coefficients
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
                               **kwargs) -> plt.Figure:
    """
    Plot histograms for confinement_time parameter sets in an m x n grid layout.
    Improved version with error handling for grid sizing and data anomalies.
    """
    
    # 1. Parameter Selection
    if eng_params is None:
        exclude_cols = {'shot', 'time'}
        # Only select numeric columns to avoid plotting errors
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        available_params = [col for col in numeric_cols if col not in exclude_cols]
    else:
        # Filter provided list to columns actually present in the DataFrame
        available_params = [p for p in eng_params if p in df.columns]
    
    num_params = len(available_params)
    
    if num_params == 0:
        logger.warning("No valid numeric parameters found to plot.")
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.text(0.5, 0.5, 'No valid parameters to plot', 
                ha='center', va='center', transform=ax.transAxes)
        return fig
    
    # 2. Dynamic Grid Calculation with Error Prevention
    # Ensure m * n is always >= num_params to avoid IndexError
    if m is None and n is None:
        n = math.ceil(math.sqrt(num_params))
        m = math.ceil(num_params / n)
    elif m is None:
        m = math.ceil(num_params / n)
    elif n is None:
        n = math.ceil(num_params / m)
    
    # Safeguard: Force grid expansion if user-provided dimensions are too small
    if m * n < num_params:
        logger.warning(f"Provided grid {m}x{n} is smaller than {num_params} params. Adjusting rows.")
        m = math.ceil(num_params / n)
    
    # 3. Figure Initialization
    if figsize is None:
        width = max(4, n * 3)
        height = max(3, m * 2.5)
        figsize = (width, height)
    
    # Use squeeze=False so 'axes' is always a 2D array
    fig, axes = plt.subplots(m, n, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()
    
    # 4. Keyword Argument Handling
    # Merges default values with user-provided kwargs to avoid "multiple values for argument" errors
    plot_settings = {
        'bins': bins,
        'alpha': alpha,
        'edgecolor': edgecolor
    }
    plot_settings.update(kwargs)
    
    # 5. Plotting Loop
    for idx, param in enumerate(available_params):
        ax = axes_flat[idx]
        
        # Data Cleaning: Remove NaNs and handle infinite values
        # Infinite values can break histogram binning logic
        data = df[param].replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(data) == 0:
            ax.set_title(f"{param} (No Data)", fontsize=10)
            ax.axis('off')
            continue
        
        try:
            ax.hist(data, **plot_settings)
            ax.set_title(param, fontsize=10, fontweight='bold')
            ax.set_xlabel('Value', fontsize=9)
            ax.set_ylabel('Frequency', fontsize=9)
            ax.grid(True, alpha=0.3)
        except Exception as e:
            logger.error(f"Error plotting parameter '{param}': {e}")
            ax.set_title(f"Error: {param}", color='red')

    # 6. Post-processing: Hide unused subplots
    for idx in range(num_params, len(axes_flat)):
        axes_flat[idx].axis('off')
    
    plt.tight_layout()
    
    return fig
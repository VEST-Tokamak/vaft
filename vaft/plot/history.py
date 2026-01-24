"""
Plotting functions for confinement time scaling analysis history.

This module provides visualization functions for statistical analysis results
of confinement time scaling laws.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import ScalarFormatter
from typing import Optional, List, Dict
import logging

logger = logging.getLogger(__name__)


def plot_scaling_fit(results, df, target_param='tauE_s', figsize=(12, 5)):
    """Plot overall fitting results and residual distribution.
    
    Args:
        results: RegressionResults object from statistical_analysis
        df: Original DataFrame with parameters
        target_param: Target parameter name (default: 'tauE_s')
        figsize: Figure size tuple (default: (12, 5))
    """
    # Get predicted and actual values
    y_pred_log = results.fitted_values
    tauE_pred = np.exp(y_pred_log)
    
    # Get actual values (filter to valid indices)
    log_df = results.log_df
    valid_mask = ~log_df[f'ln_{target_param}'].isna()
    tauE_exp = df[target_param][valid_mask].values
    
    # Ensure same length
    min_len = min(len(tauE_pred), len(tauE_exp))
    tauE_pred = tauE_pred[:min_len]
    tauE_exp = tauE_exp[:min_len]
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Experimental vs Predicted
    ax = axes[0]
    sns.regplot(x=tauE_exp, y=tauE_pred, ax=ax, scatter_kws={'alpha': 0.5})
    # Add x=y line
    min_val = min(tauE_exp.min(), tauE_pred.min())
    max_val = max(tauE_exp.max(), tauE_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', 
            label='y=x', linewidth=2)
    ax.set_xlabel(f'Experimental $\\tau_E$ [s]', fontsize=12)
    ax.set_ylabel(f'Predicted $\\tau_E$ [s]', fontsize=12)
    ax.set_title('Global Scaling Law Fit', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Residual distribution
    ax = axes[1]
    residuals = tauE_exp - tauE_pred
    sns.histplot(residuals, kde=True, ax=ax)
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero residual')
    ax.set_xlabel('Residual [s]', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Residual Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_individual_parameter_effects(log_df, eng_params, target_param='tauE_s', 
                                     figsize=(8, None)):
    """Plot individual parameter effects on confinement time (log-log scale).
    
    Args:
        log_df: DataFrame with log-transformed parameters
        eng_params: List of engineering parameter names
        target_param: Target parameter name (default: 'tauE_s')
        figsize: Figure size tuple. If height is None, calculated automatically.
    """
    from scipy.stats import pearsonr
    
    m = len(eng_params)
    if figsize[1] is None:
        figsize = (figsize[0], 4 * m)
    
    fig, axes = plt.subplots(nrows=m, ncols=1, figsize=figsize)
    fig.suptitle('Individual Parameter Variation vs Confinement Time (Log-Log Scale)', 
                 fontsize=16, y=1.02)
    
    target_col = f'ln_{target_param}'
    
    for i, param in enumerate(eng_params):
        param_col = f'ln_{param}'
        
        if param_col not in log_df.columns or target_col not in log_df.columns:
            axes[i].text(0.5, 0.5, f'Data not available for {param}', 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(f'{param} (Data unavailable)', fontsize=12)
            continue
        
        # Remove NaN values
        valid_mask = ~(log_df[param_col].isna() | log_df[target_col].isna())
        x_data = log_df[param_col][valid_mask]
        y_data = log_df[target_col][valid_mask]
        
        if len(x_data) == 0:
            axes[i].text(0.5, 0.5, f'No valid data for {param}', 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(f'{param} (No data)', fontsize=12)
            continue
        
        # Calculate correlation
        corr, _ = pearsonr(x_data, y_data)
        
        # Plot
        sns.regplot(x=x_data, y=y_data, ax=axes[i], 
                   scatter_kws={'alpha': 0.4, 'color': 'gray'}, 
                   line_kws={'color': 'blue', 'linewidth': 2})
        axes[i].set_title(f'Effect of {param} (Corr: {corr:.3f})', fontsize=12)
        axes[i].set_xlabel(f'ln({param})', fontsize=11)
        axes[i].set_ylabel('ln($\\tau_E$)', fontsize=11)
        axes[i].grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(log_df, eng_params, target_param='tauE_s', 
                             figsize=(10, 8)):
    """Plot correlation heatmap for all log-transformed parameters.
    
    Args:
        log_df: DataFrame with log-transformed parameters
        eng_params: List of engineering parameter names
        target_param: Target parameter name (default: 'tauE_s')
        figsize: Figure size tuple (default: (10, 8))
    """
    from vaft.process.statistical_analysis import get_correlation_matrix
    
    corr_df = get_correlation_matrix(log_df, eng_params, target_param)
    
    if corr_df.empty:
        logger.warning("Correlation matrix is empty")
        return None
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr_df, annot=True, cmap='coolwarm', fmt=".2f", 
                center=0, vmin=-1, vmax=1, ax=ax, square=True)
    ax.set_title('Correlation Matrix of Log-Transformed Parameters', 
                fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    return fig


def plot_regression_summary(results, figsize=(10, 6)):
    """Plot regression coefficients (exponents) and p-values as a bar chart.
    
    Args:
        results: RegressionResults object
        figsize: Figure size tuple (default: (10, 6))
    """
    summary = results.get_summary()
    
    # Filter to engineering parameters (exclude constant)
    eng_params = results.eng_params
    param_names = [f'ln_{p}' for p in eng_params]
    summary_filtered = summary.loc[summary.index.isin(param_names)]
    
    # Extract parameter names (remove 'ln_' prefix)
    summary_filtered.index = [idx.replace('ln_', '') for idx in summary_filtered.index]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Coefficients (exponents)
    colors = ['green' if sig else 'red' for sig in summary_filtered['Significant']]
    ax1.barh(summary_filtered.index, summary_filtered['Coefficient'], color=colors)
    ax1.axvline(0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_xlabel('Coefficient (Exponent)', fontsize=12)
    ax1.set_title('Regression Coefficients (Scaling Exponents)', 
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Plot 2: P-values
    colors = ['green' if sig else 'red' for sig in summary_filtered['Significant']]
    ax2.barh(summary_filtered.index, summary_filtered['P-value'], color=colors)
    ax2.axvline(0.05, color='red', linestyle='--', linewidth=2, label='α = 0.05')
    ax2.set_xlabel('P-value', fontsize=12)
    ax2.set_title('Statistical Significance (P-values)', 
                  fontsize=14, fontweight='bold')
    ax2.set_xlim(0, max(0.1, summary_filtered['P-value'].max() * 1.1))
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    return fig


def confinement_time_exp_vs_scaling(ods_or_odc, scaling_raw='IPB', time_slice=None, 
                                    figsize=(10, 6)):
    """Plot experimental vs scaling law confinement time.
    
    Args:
        ods_or_odc: ODS or ODC object
        scaling_raw: Scaling law name (default: 'IPB')
        time_slice: Time slice index (default: None for all)
        figsize: Figure size tuple (default: (10, 6))
    """
    # This function can be implemented later if needed
    # For now, it's a placeholder
    pass



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


def _get_param_latex_label(param_name):
    """Convert parameter name to LaTeX format with units.
    
    Args:
        param_name: Parameter name (e.g., 'Ip_MA', 'Bt_T', 'Ploss_MW')
        
    Returns:
        Tuple of (latex_symbol, unit) for the parameter
    """
    param_mapping = {
        'Ip_MA': ('$I_p$', '[MA]'),
        'I_p': ('$I_p$', '[A]'),
        'Bt_T': ('$B_t$', '[T]'),
        'B_t': ('$B_t$', '[T]'),
        'Ploss_MW': ('$P_{loss}$', '[MW]'),
        'P_loss': ('$P_{loss}$', '[W]'),
        'ne_19m3': ('$n_e$', '[$10^{19}$ m$^{-3}$]'),
        'n_e': ('$n_e$', '[m$^{-3}$]'),
        'R_m': ('$R$', '[m]'),
        'R': ('$R$', '[m]'),
        'epsilon': ('$\\epsilon$', ''),
        'kappa': ('$\\kappa$', ''),
        'M_amu': ('$M$', '[amu]'),
        'M': ('$M$', '[amu]'),
    }
    
    if param_name in param_mapping:
        return param_mapping[param_name]
    else:
        # Default: use parameter name as is
        return (param_name, '')


def plot_individual_parameter_effects(df, eng_params, target_param='tauE_s', 
                                     figsize=(8, 5), ncols=None, results=None):
    """Plot individual parameter effects on confinement time (linear scale).
    
    Args:
        df: DataFrame with original (non-log-transformed) parameters
        eng_params: List of engineering parameter names
        target_param: Target parameter name (default: 'tauE_s')
        figsize: Figure size tuple. If height is None, calculated automatically.
        ncols: Number of columns in subplot grid. If None, calculated automatically.
        results: RegressionResults object from statistical_analysis (optional).
                If provided, first subplot will show experimental vs predicted tauE.
    """
    from scipy.stats import pearsonr
    import math
    
    # Fixed layout: 2 rows x 4 columns = 8 subplots total
    nrows = 2
    ncols = 4
        
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    fig.suptitle('Individual Parameter Variation vs Confinement Time', 
                 fontsize=16, y=1.02)
    
    # Flatten axes array for easier indexing
    axes = axes.flatten()
    
    target_col = target_param
    
    # First subplot: tauE_exp vs tauE_predicted (from results) or tauE_exp vs tauE_fitted
    ax = axes[0]
    if results is not None:
        # Use results to create the same plot as plot_scaling_fit's first figure
        try:
            # Get predicted and actual values (same logic as plot_scaling_fit)
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
            
            if len(tauE_exp) > 0:
                # Plot with same style as other subplots
                sns.regplot(x=tauE_exp, y=tauE_pred, ax=ax, 
                           scatter_kws={'alpha': 0.4, 'color': 'gray'}, 
                           line_kws={'color': 'blue', 'linewidth': 2})
                # Add x=y line (similar style to other plots)
                min_val = min(tauE_exp.min(), tauE_pred.min())
                max_val = max(tauE_exp.max(), tauE_pred.max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', 
                        linewidth=2, alpha=0.7)
                ax.set_xlabel(f'Experimental $\\tau_E$ [s]', fontsize=11)
                ax.set_ylabel(f'Predicted $\\tau_E$ [s]', fontsize=11)
                ax.grid(True, linestyle=':', alpha=0.6)
                
                # Add R-squared value as legend
                r_squared = results.rsquared
                from matplotlib.patches import Rectangle
                empty_patch = Rectangle((0, 0), 0, 0, fill=False, edgecolor='none', visible=False)
                legend_text = f'R²: {r_squared:.3f}'
                ax.legend([empty_patch], [legend_text], 
                         loc='upper left', frameon=True, fontsize=9, 
                         handlelength=0, framealpha=0.7)
            else:
                ax.text(0.5, 0.5, 'No valid data for tauE comparison', 
                        ha='center', va='center', transform=ax.transAxes)
        except Exception as e:
            logger.warning(f"Error plotting from results: {e}")
            ax.text(0.5, 0.5, f'Error plotting from results: {e}', 
                    ha='center', va='center', transform=ax.transAxes)
    elif 'tauE_exp' in df.columns and 'tauE_fitted' in df.columns:
        # Fallback: use tauE_exp and tauE_fitted from DataFrame if results not provided
        # Remove NaN values
        valid_mask = ~(df['tauE_exp'].isna() | df['tauE_fitted'].isna())
        x_data = df['tauE_exp'][valid_mask]
        y_data = df['tauE_fitted'][valid_mask]
        
        if len(x_data) > 0:
            # Calculate correlation
            corr, _ = pearsonr(x_data, y_data)
            
            # Plot
            sns.regplot(x=x_data, y=y_data, ax=ax, 
                       scatter_kws={'alpha': 0.4, 'color': 'gray'}, 
                       line_kws={'color': 'blue', 'linewidth': 2})
            
            # Add y=x line for reference
            min_val = min(x_data.min(), y_data.min())
            max_val = max(x_data.max(), y_data.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', 
                    linewidth=1.5, alpha=0.7, label='y=x')
            
            ax.set_xlabel('$\\tau_E$ exp [s]', fontsize=11)
            ax.set_ylabel('$\\tau_E$ fitted [s]', fontsize=11)
            ax.grid(True, linestyle=':', alpha=0.6)
            
            # Add correlation as legend
            from matplotlib.patches import Rectangle
            empty_patch = Rectangle((0, 0), 0, 0, fill=False, edgecolor='none', visible=False)
            legend_text = f'Coeff\n{corr:.3f}'
            ax.legend([empty_patch], [legend_text], 
                     loc='upper left', frameon=True, fontsize=9, 
                     handlelength=0, framealpha=0.7)
        else:
            ax.text(0.5, 0.5, 'No valid data for tauE comparison', 
                    ha='center', va='center', transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, 'Results or tauE columns not available', 
                ha='center', va='center', transform=ax.transAxes)
    
    # Remaining subplots: individual parameter effects
    m = len(eng_params)
    max_params = nrows * ncols - 1  # -1 for the tauE comparison plot
    
    for i, param in enumerate(eng_params[:max_params]):
        param_col = param
        ax = axes[i + 1]  # +1 to skip the first subplot
        
        if param_col not in df.columns or target_col not in df.columns:
            ax.text(0.5, 0.5, f'Data not available for {param}', 
                        ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{param} (Data unavailable)', fontsize=12)
            continue
        
        # Remove NaN values
        valid_mask = ~(df[param_col].isna() | df[target_col].isna())
        x_data = df[param_col][valid_mask]
        y_data = df[target_col][valid_mask]
        
        if len(x_data) == 0:
            ax.text(0.5, 0.5, f'No valid data for {param}', 
                        ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{param} (No data)', fontsize=12)
            continue
        
        # Calculate correlation
        corr, _ = pearsonr(x_data, y_data)
        
        # Get P-value and significance from results if available
        p_value = None
        is_significant = None
        if results is not None:
            param_name_log = f'ln_{param}'
            if param_name_log in results.pvalues:
                p_value = float(results.pvalues[param_name_log])
                is_significant = p_value < 0.05
        
        # Get LaTeX label and unit
        latex_symbol, unit = _get_param_latex_label(param)
        
        # Plot
        sns.regplot(x=x_data, y=y_data, ax=ax, 
                   scatter_kws={'alpha': 0.4, 'color': 'gray'}, 
                   line_kws={'color': 'blue', 'linewidth': 2})
        
        # Build legend text in the requested format
        if p_value is not None:
            p_value_rounded = round(p_value, 3)
            sig_text = 'Significant' if is_significant else 'Not Significant'
            legend_text = f'Correlation: {corr:.3f}\nP-val: {p_value_rounded:.3f}\n{sig_text}'
        else:
            legend_text = f'Correlation: {corr:.3f}'
        
        # Add legend with custom text using empty handle (semi-transparent)
        from matplotlib.patches import Rectangle
        empty_patch = Rectangle((0, 0), 0, 0, fill=False, edgecolor='none', visible=False)
        ax.legend([empty_patch], [legend_text], 
                 loc='upper left', frameon=True, fontsize=9, 
                 handlelength=0, framealpha=0.7)
        
        # Build xlabel with LaTeX symbol and unit
        if unit:
            ax.set_xlabel(f'{latex_symbol} {unit}', fontsize=11)
        else:
            ax.set_xlabel(f'{latex_symbol}', fontsize=11)
        ax.set_ylabel('$\\tau_E$ [s]', fontsize=11)
        ax.grid(True, linestyle=':', alpha=0.6)
    
    # Hide unused subplots
    total_used = min(m, max_params) + 1  # +1 for tauE comparison
    for i in range(total_used, len(axes)):
        axes[i].set_visible(False)
    
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


def plot_confinement_time_exp_vs_scaling(df, scaling_raw=None, figsize=(10, 6)):
    """Plot experimental vs scaling law confinement time.
    
    Args:
        df: DataFrame with engineering parameters and experimental confinement time.
            Expected columns: I_p (or Ip_MA), B_t (or Bt_T), P_loss (or Ploss_MW),
            n_e (or ne_19m3), R (or R_m), epsilon, kappa, M (or M_amu, optional, default=2.0),
            tauE_s (experimental confinement time)
        scaling_raw: Scaling law name(s) - can be a string, list of strings, or None.
                     If None, uses all available scaling laws: ['IPB89', 'H98y2', 'NSTX'].
                     Options: 'IPB89', 'H98y2', 'NSTX'
        figsize: Figure size tuple (default: (10, 6))
    
    Returns:
        matplotlib figure object
    """
    from vaft.formula.equilibrium import confinement_time_from_engineering_parameters
    from scipy.stats import pearsonr
    
    # Map scaling law names (handle variations like 'IPB' -> 'IPB89')
    scaling_map = {
        'IPB': 'IPB89',
        'IPB89': 'IPB89',
        'H98y2': 'H98y2',
        'H98': 'H98y2',
        'NSTX': 'NSTX'
    }
    
    # Handle scaling_raw: convert to list if needed, default to all scaling laws
    if scaling_raw is None:
        scaling_raw_list = ['IPB89', 'H98y2', 'NSTX']
    elif isinstance(scaling_raw, str):
        scaling_raw_list = [scaling_raw]
    else:
        scaling_raw_list = scaling_raw
    
    # Normalize scaling law names
    scaling_list = [scaling_map.get(s, s) for s in scaling_raw_list]
    
    # Map column names to handle variations
    def get_column(df, possible_names, default=None):
        """Get column from DataFrame, trying multiple possible names."""
        for name in possible_names:
            if name in df.columns:
                return df[name]
        if default is not None:
            return default
        raise ValueError(f"Could not find column. Tried: {possible_names}")
    
    # Extract parameters with flexible column name handling
    try:
        # I_p: try I_p (A) or Ip_MA (MA) - if Ip_MA, convert to A
        if 'I_p' in df.columns:
            I_p = df['I_p'].values
        elif 'Ip_MA' in df.columns:
            I_p = df['Ip_MA'].values * 1e6  # Convert MA to A
        else:
            raise ValueError("Could not find I_p or Ip_MA column")
        
        # B_t: try B_t or Bt_T
        B_t = get_column(df, ['B_t', 'Bt_T']).values
        
        # P_loss: try P_loss (W) or Ploss_MW (MW) - if Ploss_MW, convert to W
        if 'P_loss' in df.columns:
            P_loss = df['P_loss'].values
        elif 'Ploss_MW' in df.columns:
            P_loss = df['Ploss_MW'].values * 1e6  # Convert MW to W
        else:
            raise ValueError("Could not find P_loss or Ploss_MW column")
        
        # n_e: try n_e (m^-3) or ne_19m3 (10^19 m^-3) - if ne_19m3, convert to m^-3
        if 'n_e' in df.columns:
            n_e = df['n_e'].values
        elif 'ne_19m3' in df.columns:
            n_e = df['ne_19m3'].values * 1e19  # Convert 10^19 m^-3 to m^-3
        else:
            raise ValueError("Could not find n_e or ne_19m3 column")
        
        # R: try R or R_m
        R = get_column(df, ['R', 'R_m']).values
        
        # epsilon
        epsilon = get_column(df, ['epsilon']).values
        
        # kappa
        kappa = get_column(df, ['kappa']).values
        
        # M: try M, M_amu, or default to 2.0 (deuterium)
        if 'M' in df.columns:
            M = df['M'].values
        elif 'M_amu' in df.columns:
            M = df['M_amu'].values
        else:
            M = np.full(len(df), 2.0)  # Default to deuterium
            logger.info("Using default M=2.0 (deuterium) for scaling law calculation")
        
        # tauE_s (experimental)
        tauE_exp = get_column(df, ['tauE_s']).values
        
    except (KeyError, ValueError) as e:
        logger.error(f"Error extracting parameters from DataFrame: {e}")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f'Error: {e}', ha='center', va='center', 
                transform=ax.transAxes, fontsize=12)
        ax.set_title('Confinement Time: Experimental vs Scaling', fontsize=14)
        return fig
    
    # Calculate scaling law confinement time for each scaling law
    scaling_data = {}  # {scaling_name: (tauE_scaling_array, valid_indices, correlation)}
    
    for scaling in scaling_list:
        tauE_scaling = []
        valid_indices = []
        
        for i in range(len(df)):
            try:
                tauE_scaling_val = confinement_time_from_engineering_parameters(
                    I_p=I_p[i], B_t=B_t[i], P_loss=P_loss[i], n_e=n_e[i],
                    M=M[i] if isinstance(M, np.ndarray) else M,
                    R=R[i], epsilon=epsilon[i], kappa=kappa[i],
                    scaling=scaling
                )
                tauE_scaling.append(tauE_scaling_val)
                valid_indices.append(i)
            except (ValueError, ZeroDivisionError) as e:
                logger.debug(f"Row {i}, scaling {scaling}: Could not calculate scaling law confinement time: {e}")
                continue
        
        if len(tauE_scaling) > 0:
            tauE_scaling_arr = np.array(tauE_scaling)
            tauE_exp_valid = tauE_exp[valid_indices]
            
            # Calculate correlation and R²
            valid_mask = ~(np.isnan(tauE_exp_valid) | np.isnan(tauE_scaling_arr))
            if np.sum(valid_mask) > 1:
                corr, _ = pearsonr(tauE_exp_valid[valid_mask], tauE_scaling_arr[valid_mask])
                r_squared = corr ** 2
            else:
                corr = np.nan
                r_squared = np.nan
            
            scaling_data[scaling] = (tauE_scaling_arr, tauE_exp_valid, r_squared)
    
    if len(scaling_data) == 0:
        logger.error("No valid scaling law confinement times calculated")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No valid data available', ha='center', va='center', 
                transform=ax.transAxes, fontsize=14)
        ax.set_title('Confinement Time: Experimental vs Scaling', fontsize=14)
        return fig
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Define colors and markers for different scaling laws
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    markers = ['o', 's', '^', 'v', 'D', 'p']
    
    # Track overall min/max for y=x line
    all_tauE_exp = []
    all_tauE_scaling = []
    
    # Plot each scaling law
    for idx, scaling in enumerate(scaling_list):
        if scaling not in scaling_data:
            continue
        
        tauE_scaling_arr, tauE_exp_valid, r_squared = scaling_data[scaling]
        
        # Store for overall range calculation
        all_tauE_exp.extend(tauE_exp_valid)
        all_tauE_scaling.extend(tauE_scaling_arr)
        
        # Create label with scaling name and R²
        if not np.isnan(r_squared):
            label = f'{scaling} (R²={r_squared:.3f})'
        else:
            label = f'{scaling} (R²=N/A)'
        
        # Scatter plot with different color and marker for each scaling law
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        ax.scatter(tauE_exp_valid, tauE_scaling_arr, alpha=0.6, s=50, 
                  edgecolors='black', linewidth=0.5, color=color, marker=marker,
                  label=label)
    
    # Add y=x line for reference (using overall range)
    if len(all_tauE_exp) > 0 and len(all_tauE_scaling) > 0:
        min_val = min(min(all_tauE_exp), min(all_tauE_scaling))
        max_val = max(max(all_tauE_exp), max(all_tauE_scaling))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', 
                linewidth=2, label='y=x', alpha=0.7)
    
    ax.set_xlabel('Experimental $\\tau_E$ [s]', fontsize=12)
    ax.set_ylabel('Scaling Law $\\tau_E$ [s]', fontsize=12)
    
    # Create title with scaling law names
    scaling_names_str = ', '.join(scaling_list)
    ax.set_title(f'Confinement Time: Experimental vs Scaling Laws ({scaling_names_str})', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    return fig


def plot_bremsstrahlung_power_scaling_vs_fundamental_method(ods_or_odc, Z_eff=2.0, figsize=(5, 6)):
    """Plot bremsstrahlung power scaling vs fundamental method.
    
    Compares two methods of calculating bremsstrahlung power:
    - Pressure-based method: P_B_pressure = ∫_V S_B(pressure) dV
    - Electron density-based method: P_B_electron = ∫_V S_B(n_e, T_e) dV
    
    Args:
        ods_or_odc: ODS or ODC object
        Z_eff: Effective charge (default: 2.0)
        figsize: Figure size tuple (default: (5, 6))
    
    Returns:
        matplotlib figure object
    """
    from vaft.omas import odc_or_ods_check
    from vaft.omas.formula_wrapper import compute_bremsstrahlung_power
    
    odc = odc_or_ods_check(ods_or_odc)
    
    # Collect data for all shots/time slices
    P_B_pressure_list = []
    P_B_electron_list = []
    shot_labels = []
    time_slices = []
    
    for key in odc.keys():
        ods = odc[key]
        
        # Get number of core profile time slices
        if 'core_profiles.profiles_1d' not in ods:
            logger.warning(f"Skipping {key}: core_profiles.profiles_1d not found")
            continue
        
        n_slices = len(ods['core_profiles.profiles_1d'])
        
        # Extract shot label
        if hasattr(odc, 'get') and hasattr(odc.get(key, {}), 'get'):
            shot_label = odc.get(key, {}).get('shot', {}).get('data', [key])[0] if isinstance(odc.get(key, {}).get('shot', {}), dict) else str(key)
        else:
            shot_label = str(key)
        
        # Process each time slice
        for time_slice in range(n_slices):
            try:
                P_B_pressure, P_B_electron = compute_bremsstrahlung_power(
                    ods, time_slice=time_slice, Z_eff=Z_eff
                )
                P_B_pressure_list.append(P_B_pressure)
                P_B_electron_list.append(P_B_electron)
                shot_labels.append(shot_label)
                time_slices.append(time_slice)
            except Exception as e:
                logger.warning(f"Error computing bremsstrahlung power for {key}, time_slice={time_slice}: {e}")
                continue
    
    if len(P_B_pressure_list) == 0:
        logger.error("No valid bremsstrahlung power data found")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', 
                transform=ax.transAxes, fontsize=14)
        ax.set_title('Bremsstrahlung Power Comparison', fontsize=14)
        return fig
    
    P_B_pressure_arr = np.asarray(P_B_pressure_list)
    P_B_electron_arr = np.asarray(P_B_electron_list)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Scatter plot
    ax.scatter(P_B_electron_arr, P_B_pressure_arr, alpha=0.6, s=50, 
              c=range(len(P_B_electron_arr)), cmap='viridis', edgecolors='black', linewidth=0.5)
    
    # Add y=x line for reference
    min_val = min(P_B_electron_arr.min(), P_B_pressure_arr.min())
    max_val = max(P_B_electron_arr.max(), P_B_pressure_arr.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', 
            linewidth=2, label='y=x (perfect agreement)', alpha=0.7)
    
    # Calculate correlation
    from scipy.stats import pearsonr
    valid_mask = ~(np.isnan(P_B_electron_arr) | np.isnan(P_B_pressure_arr))
    if np.sum(valid_mask) > 1:
        corr, _ = pearsonr(P_B_electron_arr[valid_mask], P_B_pressure_arr[valid_mask])
        ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('P_B (Electron Method) [W]', fontsize=12)
    ax.set_ylabel('P_B (Pressure Method) [W]', fontsize=12)
    ax.set_title('Bremsstrahlung Power: Pressure vs Electron Method', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    return fig


def plot_ohmic_power_flux_vs_dissipation_method(ods_or_odc, figsize=(5, 6)):
    """Plot ohmic power: flux-based vs dissipation-based method comparison.
    
    Compares two methods of calculating ohmic heating power:
    - Flux-based method: P_ohm_flux = I_p * V_res
    - Dissipation-based method: P_ohm_diss = ∫_V η J_φ² dV
    
    Args:
        ods_or_odc: ODS or ODC object
        figsize: Figure size tuple (default: (5, 6))
    
    Returns:
        matplotlib figure object
    """
    from vaft.omas import odc_or_ods_check
    from vaft.omas.formula_wrapper import compute_power_balance
    
    odc = odc_or_ods_check(ods_or_odc)
    
    # Collect data for all shots/time slices
    P_ohm_flux_list = []
    P_ohm_diss_list = []
    shot_labels = []
    time_slices = []
    
    for key in odc.keys():
        ods = odc[key]
        
        # Check if equilibrium data exists
        if 'equilibrium.time_slice' not in ods or len(ods['equilibrium.time_slice']) == 0:
            logger.warning(f"Skipping {key}: equilibrium.time_slice not found")
            continue
        
        # Extract shot label
        if hasattr(odc, 'get') and hasattr(odc.get(key, {}), 'get'):
            shot_label = odc.get(key, {}).get('shot', {}).get('data', [key])[0] if isinstance(odc.get(key, {}).get('shot', {}), dict) else str(key)
        else:
            shot_label = str(key)
        
        # Process all time slices
        n_slices = len(ods['equilibrium.time_slice'])
        for time_slice in range(n_slices):
            try:
                power_balance = compute_power_balance(ods)
                
                P_ohm_flux = power_balance['P_ohm_flux']
                P_ohm_diss = power_balance['P_ohm_diss']
                
                # Handle scalar or array values
                if np.isscalar(P_ohm_flux):
                    P_ohm_flux_list.append(float(P_ohm_flux))
                    P_ohm_diss_list.append(float(P_ohm_diss))
                    shot_labels.append(shot_label)
                    time_slices.append(time_slice)
                else:
                    # If arrays, append all values
                    for i in range(len(P_ohm_flux)):
                        P_ohm_flux_list.append(float(P_ohm_flux[i]))
                        P_ohm_diss_list.append(float(P_ohm_diss[i]))
                        shot_labels.append(shot_label)
                        time_slices.append(time_slice)
                        
            except Exception as e:
                logger.warning(f"Error computing power balance for {key}, time_slice={time_slice}: {e}")
                continue
    
    if len(P_ohm_flux_list) == 0:
        logger.error("No valid ohmic power data found")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', 
                transform=ax.transAxes, fontsize=14)
        ax.set_title('Ohmic Power Comparison', fontsize=14)
        return fig
    
    P_ohm_flux_arr = np.asarray(P_ohm_flux_list)
    P_ohm_diss_arr = np.asarray(P_ohm_diss_list)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Scatter plot
    ax.scatter(P_ohm_flux_arr, P_ohm_diss_arr, alpha=0.6, s=50, 
              c=range(len(P_ohm_flux_arr)), cmap='viridis', edgecolors='black', linewidth=0.5)
    
    # Add y=x line for reference
    min_val = min(P_ohm_flux_arr.min(), P_ohm_diss_arr.min())
    max_val = max(P_ohm_flux_arr.max(), P_ohm_diss_arr.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', 
            linewidth=2, label='y=x (perfect agreement)', alpha=0.7)
    
    # Calculate correlation
    from scipy.stats import pearsonr
    valid_mask = ~(np.isnan(P_ohm_flux_arr) | np.isnan(P_ohm_diss_arr))
    if np.sum(valid_mask) > 1:
        corr, _ = pearsonr(P_ohm_flux_arr[valid_mask], P_ohm_diss_arr[valid_mask])
        ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('P_ohm (Flux Method) [W]', fontsize=12)
    ax.set_ylabel('P_ohm (Dissipation Method) [W]', fontsize=12)
    ax.set_title('Ohmic Power: Flux vs Dissipation Method', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    return fig
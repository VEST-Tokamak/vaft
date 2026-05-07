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

# Human-readable y-axis labels for confinement scaling regression bars (mathtext).
_ENG_PARAM_PLOT_LABELS: Dict[str, str] = {
    'Ip_MA': r'$I_{\mathrm{p}}$ [MA]',
    'Bt_T': r'$B_{\mathrm{t}}$ [T]',
    'Ploss_MW': r'$P_{\mathrm{loss}}$ [MW]',
    'ne_19m3': r'$n_{\mathrm{e}}$ [$10^{19}$ m$^{-3}$]',
    'R_m': r'$R$ [m]',
    'epsilon': r'$\epsilon$',
    'kappa': r'$\kappa$',
    'T_eV': r'$T_{\mathrm{e}}$ [eV]',
    'a_m': r'$a$ [m]',
}


def _eng_param_y_label(column_name: str) -> str:
    """Label for regression summary bar plots; falls back to the column name."""
    return _ENG_PARAM_PLOT_LABELS.get(column_name, column_name)


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
                                     figsize=None, ncols=None, results=None):
    """Plot individual parameter effects on confinement time (linear scale).
    
    Args:
        df: DataFrame with original (non-log-transformed) parameters
        eng_params: List of engineering parameter names
        target_param: Target parameter name (default: 'tauE_s')
        figsize: Figure size tuple. If None, calculated automatically.
        ncols: Number of columns in subplot grid. If None, calculated automatically.
        results: RegressionResults object from statistical_analysis (optional).
                If provided, first subplot will show experimental vs predicted tauE.
    """
    from scipy.stats import pearsonr
    import math
    
    has_results_plot = results is not None
    has_tau_cols_plot = 'tauE_exp' in df.columns and 'tauE_fitted' in df.columns
    include_tau_comparison = has_results_plot or has_tau_cols_plot

    total_plots = len(eng_params) + (1 if include_tau_comparison else 0)
    if total_plots == 0:
        fig, ax = plt.subplots(figsize=figsize or (8, 4))
        ax.text(0.5, 0.5, 'No engineering parameters to plot',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    if ncols is None:
        ncols = min(4, max(1, total_plots))
    ncols = max(1, int(ncols))
    nrows = int(math.ceil(total_plots / ncols))

    if figsize is None:
        figsize = (4.3 * ncols, 3.6 * nrows)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    fig.suptitle('Individual Parameter Variation vs Confinement Time', 
                 fontsize=16, y=1.02)
    
    # Flatten axes array for easier indexing
    axes = np.atleast_1d(axes).flatten()
    
    target_col = target_param
    
    plot_idx = 0

    # Optional first subplot: tauE_exp vs tauE_predicted (from results)
    # or tauE_exp vs tauE_fitted (from DataFrame).
    if include_tau_comparison:
        ax = axes[plot_idx]
        plot_idx += 1
    if has_results_plot:
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
    elif has_tau_cols_plot:
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
    # Remaining subplots: individual parameter effects
    m = len(eng_params)
    max_params = len(axes) - plot_idx
    
    for i, param in enumerate(eng_params[:max_params]):
        param_col = param
        ax = axes[plot_idx + i]
        
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
    total_used = min(m, max_params) + plot_idx
    for i in range(total_used, len(axes)):
        axes[i].set_visible(False)
    
    fig.tight_layout(rect=[0, 0, 1, 0.97])
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
    eng_params = results.eng_params
    coefs = []
    pvals = []
    y_labels = []
    for p in eng_params:
        key = f'ln_{p}'
        coefs.append(float(results.coefficients[key]))
        pvals.append(float(results.pvalues[key]))
        y_labels.append(_eng_param_y_label(p))
    significant = [pv < 0.05 for pv in pvals]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Coefficients (exponents)
    colors = ['green' if sig else 'red' for sig in significant]
    ax1.barh(y_labels, coefs, color=colors)
    ax1.axvline(0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_xlabel('Coefficient (Exponent)', fontsize=12)
    ax1.set_title('Regression Coefficients (Scaling Exponents)', 
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Plot 2: P-values
    colors = ['green' if sig else 'red' for sig in significant]
    ax2.barh(y_labels, pvals, color=colors)
    ax2.axvline(0.05, color='red', linestyle='--', linewidth=2, label='α = 0.05')
    ax2.set_xlabel('P-value', fontsize=12)
    ax2.set_title('Statistical Significance (P-values)', 
                  fontsize=14, fontweight='bold')
    ax2.set_xlim(0, max(0.1, max(pvals) * 1.1))
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='x')
    # First entry in eng_params at the top (matches list order)
    ax1.invert_yaxis()
    ax2.invert_yaxis()
    
    plt.tight_layout()
    return fig


def plot_H_factor_vs_greenwald_fraction(
    df: pd.DataFrame,
    tauE_exp_col: str = 'tauE_s',
    tauE_scaling_col: Optional[str] = None,
    figsize=(8, 6),
):
    """Plot confinement enhancement factor versus Greenwald fraction.

    This function computes:
    - ``H = tau_E,exp / tau_E,scaling`` using
      ``confinement_factor_ITER89P``
    - ``f_G = n_e / n_G`` using ``greenwald_density`` and ``greenwald_fraction``

    Accepted column aliases:
    - Plasma current: ``Ip_MA`` (MA) or ``I_p`` (A)
    - Electron density: ``ne_19m3`` (1e19 m^-3) or ``n_e`` (m^-3)
    - Minor radius: ``a`` (m), or computed from ``R_m * epsilon``
    - Scaling confinement time:
      ``tauE_H98y2`` / ``tauE_IPB89`` / ``tauE_ITER89P`` (priority order)

    Args:
        df: DataFrame containing confinement and geometry parameters.
        tauE_exp_col: Experimental confinement time column.
        tauE_scaling_col: Scaling confinement time column. If None, inferred.
        figsize: Figure size tuple.

    Returns:
        matplotlib figure object
    """
    from vaft.formula.equilibrium import confinement_factor_ITER89P
    from vaft.formula.stability import greenwald_density, greenwald_fraction

    def _pick_column(column_candidates):
        for col in column_candidates:
            if col in df.columns:
                return col
        return None

    if tauE_exp_col not in df.columns:
        raise ValueError(f"Missing required column: {tauE_exp_col}")

    if tauE_scaling_col is None:
        tauE_scaling_col = _pick_column(['tauE_H98y2', 'tauE_IPB89', 'tauE_ITER89P'])
    if tauE_scaling_col is None or tauE_scaling_col not in df.columns:
        raise ValueError(
            "Could not find scaling tau_E column. Tried: "
            "['tauE_H98y2', 'tauE_IPB89', 'tauE_ITER89P']"
        )

    ip_col = _pick_column(['Ip_MA', 'I_p'])
    if ip_col is None:
        raise ValueError("Could not find plasma current column. Tried: ['Ip_MA', 'I_p']")

    ne_col = _pick_column(['ne_19m3', 'n_e'])
    if ne_col is None:
        raise ValueError("Could not find density column. Tried: ['ne_19m3', 'n_e']")

    if 'a' in df.columns:
        a_m = pd.to_numeric(df['a'], errors='coerce').to_numpy(dtype=float)
    elif 'R_m' in df.columns and 'epsilon' in df.columns:
        a_m = (
            pd.to_numeric(df['R_m'], errors='coerce').to_numpy(dtype=float)
            * pd.to_numeric(df['epsilon'], errors='coerce').to_numpy(dtype=float)
        )
    else:
        raise ValueError("Need minor radius column 'a' or both 'R_m' and 'epsilon'")

    tauE_exp = pd.to_numeric(df[tauE_exp_col], errors='coerce').to_numpy(dtype=float)
    tauE_scaling = pd.to_numeric(df[tauE_scaling_col], errors='coerce').to_numpy(dtype=float)

    if ip_col == 'Ip_MA':
        I_p_MA = pd.to_numeric(df[ip_col], errors='coerce').to_numpy(dtype=float)
    else:
        I_p_MA = pd.to_numeric(df[ip_col], errors='coerce').to_numpy(dtype=float) / 1e6

    if ne_col == 'ne_19m3':
        n_e_19 = pd.to_numeric(df[ne_col], errors='coerce').to_numpy(dtype=float)
    else:
        n_e_19 = pd.to_numeric(df[ne_col], errors='coerce').to_numpy(dtype=float) / 1e19

    valid_mask = (
        np.isfinite(tauE_exp)
        & np.isfinite(tauE_scaling)
        & np.isfinite(I_p_MA)
        & np.isfinite(a_m)
        & np.isfinite(n_e_19)
        & (tauE_exp > 0)
        & (tauE_scaling > 0)
        & (I_p_MA > 0)
        & (a_m > 0)
        & (n_e_19 > 0)
    )

    if np.sum(valid_mask) < 2:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5,
            0.5,
            'Not enough valid data for H-factor vs Greenwald fraction',
            ha='center',
            va='center',
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return fig

    tauE_exp_valid = tauE_exp[valid_mask]
    tauE_scaling_valid = tauE_scaling[valid_mask]
    I_p_MA_valid = I_p_MA[valid_mask]
    a_m_valid = a_m[valid_mask]
    n_e_19_valid = n_e_19[valid_mask]

    n_G_19 = greenwald_density(I_p=I_p_MA_valid, a=a_m_valid)
    f_G = greenwald_fraction(n_e=n_e_19_valid, n_G=n_G_19)
    H_factor = confinement_factor_ITER89P(
        tau_E_exp=tauE_exp_valid,
        tau_E_ITER89P=tauE_scaling_valid,
    )

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(
        f_G,
        H_factor,
        alpha=0.65,
        s=45,
        edgecolors='black',
        linewidth=0.5,
        color='#1f77b4',
        label=f'N={len(f_G)}',
    )

    if len(f_G) >= 3:
        coeff = np.polyfit(f_G, H_factor, 1)
        x_fit = np.linspace(np.nanmin(f_G), np.nanmax(f_G), 100)
        y_fit = np.polyval(coeff, x_fit)
        ax.plot(x_fit, y_fit, 'r-', linewidth=2, label=f'Linear fit: y={coeff[0]:.3f}x+{coeff[1]:.3f}')

    ax.axhline(1.0, color='gray', linestyle='--', linewidth=1.5, label='H=1')
    ax.set_xlabel('Greenwald fraction $f_G = n_e / n_G$', fontsize=12)
    ax.set_ylabel('Confinement factor $H$', fontsize=12)
    ax.set_title('H-factor vs Greenwald Fraction', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)

    plt.tight_layout()
    return fig


def _resolve_scaling_series(
    df: pd.DataFrame,
    scaling_cols: Optional[Dict[str, str]] = None,
) -> Dict[str, pd.Series]:
    """Resolve available scaling-law tau_E columns from a DataFrame."""
    if scaling_cols is None:
        scaling_candidates = {
            'ITER89P': ['tauE_ITER89P', 'tauE_IPB89'],
            'H98y2': ['tauE_H98y2', 'tauE_H98'],
            # Keep NSTX2006H/L distinct so plotting/metrics don't collapse them into one entry.
            # Retain legacy 'NSTX' as a fallback only when a more specific column isn't present.
            'NSTX2006H': ['tauE_NSTX2006H', 'tauE_NSTX2007'],
            'NSTX2006L': ['tauE_NSTX2006L'],
            'NSTX': ['tauE_NSTX'],
            'Kurskiev2022': ['tauE_Kurskiev2022'],
        }
    else:
        scaling_candidates = {name: [col] for name, col in scaling_cols.items()}

    resolved = {}
    for scaling_name, candidates in scaling_candidates.items():
        for col in candidates:
            if col in df.columns:
                resolved[scaling_name] = pd.to_numeric(df[col], errors='coerce')
                break
    return resolved


def _resolve_parameter_series(df: pd.DataFrame, aliases: List[str]) -> Optional[pd.Series]:
    """Return the first matching parameter column converted to numeric."""
    for col in aliases:
        if col in df.columns:
            return pd.to_numeric(df[col], errors='coerce')
    return None


def compute_confinement_scaling_metrics(
    df: pd.DataFrame,
    tauE_exp_col: str = 'tauE_s',
    scaling_cols: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Compute agreement metrics between tau_E(exp) and tau_E(scaling).

    Returns per scaling law:
    - Pearson correlation
    - RMSE, MAE
    - Mean relative error
    - Mean log error and std(log H)
    - Log-space fit parameters: y = a + b x
      where y=log(tau_E,exp), x=log(tau_E,scaling)
    """
    if tauE_exp_col not in df.columns:
        raise ValueError(f"Missing required column: {tauE_exp_col}")

    tau_exp = pd.to_numeric(df[tauE_exp_col], errors='coerce')
    scaling_series = _resolve_scaling_series(df, scaling_cols=scaling_cols)
    if not scaling_series:
        raise ValueError("No scaling-law tau_E columns found in DataFrame.")

    rows = []
    for scaling_name, tau_scal in scaling_series.items():
        mask = tau_exp.notna() & tau_scal.notna() & (tau_exp > 0) & (tau_scal > 0)
        if mask.sum() < 2:
            rows.append(
                {
                    'Scaling': scaling_name,
                    'N': int(mask.sum()),
                    'Pearson_r': np.nan,
                    'RMSE': np.nan,
                    'MAE': np.nan,
                    'MeanRelativeError': np.nan,
                    'MeanLogError': np.nan,
                    'StdLogH': np.nan,
                    'LogFit_a': np.nan,
                    'LogFit_b': np.nan,
                }
            )
            continue

        y_exp = tau_exp[mask].to_numpy(dtype=float)
        x_scal = tau_scal[mask].to_numpy(dtype=float)
        diff = y_exp - x_scal
        rel_err = diff / x_scal
        log_h = np.log(y_exp / x_scal)

        pearson_r = np.corrcoef(x_scal, y_exp)[0, 1] if len(x_scal) > 1 else np.nan
        rmse = float(np.sqrt(np.mean(diff**2)))
        mae = float(np.mean(np.abs(diff)))
        mean_rel_error = float(np.mean(rel_err))
        mean_log_error = float(np.mean(log_h))
        std_log_h = float(np.std(log_h, ddof=1)) if len(log_h) > 1 else np.nan

        x_log = np.log(x_scal)
        y_log = np.log(y_exp)
        fit_b, fit_a = np.polyfit(x_log, y_log, 1)

        rows.append(
            {
                'Scaling': scaling_name,
                'N': int(mask.sum()),
                'Pearson_r': float(pearson_r),
                'RMSE': rmse,
                'MAE': mae,
                'MeanRelativeError': mean_rel_error,
                'MeanLogError': mean_log_error,
                'StdLogH': std_log_h,
                'LogFit_a': float(fit_a),
                'LogFit_b': float(fit_b),
            }
        )

    metrics_df = pd.DataFrame(rows).set_index('Scaling').sort_index()
    return metrics_df


def plot_tauE_exp_vs_scaling_loglog(
    df: pd.DataFrame,
    tauE_exp_col: str = 'tauE_s',
    scaling_cols: Optional[Dict[str, str]] = None,
    figsize=(12, 8),
):
    """Plot tau_E(exp) vs tau_E(scaling) in log-log space per scaling law."""
    metrics_df = compute_confinement_scaling_metrics(
        df,
        tauE_exp_col=tauE_exp_col,
        scaling_cols=scaling_cols,
    )
    tau_exp = pd.to_numeric(df[tauE_exp_col], errors='coerce')
    scaling_series = _resolve_scaling_series(df, scaling_cols=scaling_cols)

    n_scalings = len(scaling_series)
    ncols = min(2, max(1, n_scalings))
    nrows = int(np.ceil(n_scalings / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).flatten()

    for ax, (scaling_name, tau_scal) in zip(axes, scaling_series.items()):
        mask = tau_exp.notna() & tau_scal.notna() & (tau_exp > 0) & (tau_scal > 0)
        if mask.sum() < 2:
            ax.text(0.5, 0.5, f'Not enough data: {scaling_name}', ha='center', va='center', transform=ax.transAxes)
            ax.set_axis_off()
            continue

        x = tau_scal[mask].to_numpy(dtype=float)
        y = tau_exp[mask].to_numpy(dtype=float)
        ax.scatter(x, y, s=36, alpha=0.6, edgecolors='black', linewidth=0.4)

        low = min(np.min(x), np.min(y))
        high = max(np.max(x), np.max(y))
        ax.plot([low, high], [low, high], 'k--', linewidth=1.5, label='y=x')

        x_log = np.log(x)
        y_log = np.log(y)
        fit_b, fit_a = np.polyfit(x_log, y_log, 1)
        x_line = np.linspace(np.min(x_log), np.max(x_log), 100)
        y_line = fit_a + fit_b * x_line
        ax.plot(np.exp(x_line), np.exp(y_line), 'r-', linewidth=2, label=f'log-fit: a={fit_a:.2f}, b={fit_b:.2f}')

        stats = metrics_df.loc[scaling_name]
        info = (
            f"r={stats['Pearson_r']:.3f}\n"
            f"RMSE={stats['RMSE']:.3e}\n"
            f"MAE={stats['MAE']:.3e}\n"
            f"MRE={stats['MeanRelativeError']:.3f}"
        )
        ax.text(
            0.03,
            0.97,
            info,
            transform=ax.transAxes,
            fontsize=9,
            va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.75),
        )

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$\tau_{E,\mathrm{scaling}}$ [s]')
        ax.set_ylabel(r'$\tau_{E,\mathrm{exp}}$ [s]')
        ax.set_title(scaling_name)
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(loc='lower right', fontsize=8)

    for ax in axes[n_scalings:]:
        ax.set_visible(False)

    fig.suptitle(r'$\tau_{E,\mathrm{exp}}$ vs $\tau_{E,\mathrm{scaling}}$ (log-log)', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def plot_H_factor_distribution(
    df: pd.DataFrame,
    tauE_exp_col: str = 'tauE_s',
    scaling_cols: Optional[Dict[str, str]] = None,
    figsize=(12, 5),
    bins: int = 20,
):
    """Plot H-factor distributions for available scaling laws."""
    if tauE_exp_col not in df.columns:
        raise ValueError(f"Missing required column: {tauE_exp_col}")

    tau_exp = pd.to_numeric(df[tauE_exp_col], errors='coerce')
    scaling_series = _resolve_scaling_series(df, scaling_cols=scaling_cols)
    if not scaling_series:
        raise ValueError("No scaling-law tau_E columns found in DataFrame.")

    h_data = {}
    for scaling_name, tau_scal in scaling_series.items():
        mask = tau_exp.notna() & tau_scal.notna() & (tau_exp > 0) & (tau_scal > 0)
        if mask.sum() >= 2:
            h_data[scaling_name] = (tau_exp[mask] / tau_scal[mask]).to_numpy(dtype=float)

    if not h_data:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No valid H-factor data', ha='center', va='center', transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    fig, (ax_hist, ax_box) = plt.subplots(1, 2, figsize=figsize)

    for scaling_name, h_val in h_data.items():
        ax_hist.hist(h_val, bins=bins, alpha=0.45, label=f'{scaling_name} (mean={np.mean(h_val):.2f})')
    ax_hist.axvline(1.0, color='k', linestyle='--', linewidth=1.5)
    ax_hist.set_xlabel(r'$H=\tau_{E,\mathrm{exp}}/\tau_{E,\mathrm{scaling}}$')
    ax_hist.set_ylabel('Count')
    ax_hist.set_title('H-factor histogram')
    ax_hist.grid(True, alpha=0.3)
    ax_hist.legend(fontsize=9)

    names = list(h_data.keys())
    vals = [h_data[n] for n in names]
    ax_box.boxplot(vals, labels=names, showmeans=True)
    ax_box.axhline(1.0, color='k', linestyle='--', linewidth=1.5)
    ax_box.set_ylabel(r'$H=\tau_{E,\mathrm{exp}}/\tau_{E,\mathrm{scaling}}$')
    ax_box.set_title('H-factor boxplot')
    ax_box.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig


def plot_H_factor_vs_parameters(
    df: pd.DataFrame,
    tauE_exp_col: str = 'tauE_s',
    scaling_cols: Optional[Dict[str, str]] = None,
    parameter_aliases: Optional[Dict[str, List[str]]] = None,
    figsize=(14, 8),
):
    """Plot H-factor against key engineering parameters."""
    if tauE_exp_col not in df.columns:
        raise ValueError(f"Missing required column: {tauE_exp_col}")

    if parameter_aliases is None:
        parameter_aliases = {
            'I_p': ['Ip_MA', 'I_p'],
            'B_t': ['Bt_T', 'B_t'],
            'n_e': ['ne_19m3', 'n_e'],
            'P_loss': ['Ploss_MW', 'P_loss'],
        }

    tau_exp = pd.to_numeric(df[tauE_exp_col], errors='coerce')
    scaling_series = _resolve_scaling_series(df, scaling_cols=scaling_cols)
    if not scaling_series:
        raise ValueError("No scaling-law tau_E columns found in DataFrame.")

    n_params = len(parameter_aliases)
    ncols = min(2, max(1, n_params))
    nrows = int(np.ceil(n_params / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).flatten()

    for ax, (param_label, aliases) in zip(axes, parameter_aliases.items()):
        x_param = _resolve_parameter_series(df, aliases)
        if x_param is None:
            ax.text(0.5, 0.5, f'Missing parameter: {param_label}', ha='center', va='center', transform=ax.transAxes)
            ax.set_axis_off()
            continue

        for scaling_name, tau_scal in scaling_series.items():
            mask = (
                tau_exp.notna()
                & tau_scal.notna()
                & x_param.notna()
                & (tau_exp > 0)
                & (tau_scal > 0)
            )
            if mask.sum() < 2:
                continue

            h_val = (tau_exp[mask] / tau_scal[mask]).to_numpy(dtype=float)
            x_val = x_param[mask].to_numpy(dtype=float)
            ax.scatter(x_val, h_val, s=28, alpha=0.5, label=scaling_name)

        ax.axhline(1.0, color='k', linestyle='--', linewidth=1.2)
        ax.set_xlabel(param_label)
        ax.set_ylabel(r'$H=\tau_{E,\mathrm{exp}}/\tau_{E,\mathrm{scaling}}$')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='best')

    for ax in axes[n_params:]:
        ax.set_visible(False)

    fig.suptitle('H-factor vs engineering parameters', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def plot_scaling_metrics_bars(
    metrics_df: pd.DataFrame,
    metric_names: Optional[List[str]] = None,
    figsize=(12, 6),
    include_r: bool = True,
    r_col: str = "Pearson_r",
    tick_rotation: int = 20,
    ncols: int = 2,
):
    """Plot bar charts for selected scaling-law metrics."""
    if metric_names is None:
        metric_names = ['RMSE', 'MeanLogError', 'StdLogH']

    valid_metrics = [m for m in metric_names if m in metrics_df.columns]
    if not valid_metrics:
        raise ValueError(f"No valid metrics found. Available: {list(metrics_df.columns)}")

    scaling_names = [str(x) for x in metrics_df.index.tolist()]
    x = np.arange(len(scaling_names))

    tick_labels = scaling_names

    # Add an extra bar panel for correlation (r) rather than embedding it in tick labels.
    if include_r and (r_col in metrics_df.columns) and (r_col not in valid_metrics):
        valid_metrics = list(valid_metrics) + [r_col]

    n_metrics = len(valid_metrics)
    ncols = int(max(1, min(ncols, n_metrics)))
    nrows = int(np.ceil(n_metrics / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).flatten()

    for ax, metric_name in zip(axes, valid_metrics):
        vals = metrics_df[metric_name]
        ax.bar(x, vals.to_numpy(), color='tab:blue', alpha=0.75)
        ax.set_title(metric_name)
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels, rotation=tick_rotation, ha="right")
        ax.grid(True, alpha=0.3, axis='y')
        if metric_name == r_col:
            ax.axhline(0.0, color='k', linestyle='--', linewidth=1.2, alpha=0.7)
            ax.set_ylim(-1.0, 1.0)

    for ax in axes[n_metrics:]:
        ax.set_visible(False)

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
            n_e (or ne_19m3 / ne_line_19m3 / ne_vol_19m3), R (or R_m), epsilon, kappa,
            M (or M_amu, optional, default=2.0),
            tauE_s (experimental confinement time)
        scaling_raw: Scaling law name(s) - can be a string, list of strings, or None.
                     If None, uses all available scaling laws: ['ITER89P', 'H98y2', 'NSTX2006H', 'NSTX2006L'].
                     Options: 'ITER89P', 'H98y2', 'NSTX2006H', 'NSTX2006L', 'Kurskiev2022'
        figsize: Figure size tuple (default: (10, 6))
    
    Returns:
        matplotlib figure object
    """
    from vaft.formula.equilibrium import confinement_time_from_engineering_parameters
    from scipy.stats import pearsonr
    
    # Map scaling law names (keep accepted short aliases only where intended).
    scaling_map = {
        'IPB': 'ITER89P',
        'IPB89': 'ITER89P',
        'ITER89P': 'ITER89P',
        'H98y2': 'H98y2',
        'H98': 'H98y2',
        'NSTX2006H': 'NSTX2006H',
        'NSTX2006L': 'NSTX2006L',
        'Kurskiev2022': 'Kurskiev2022',
    }
    
    # Handle scaling_raw: convert to list if needed, default to all scaling laws
    if scaling_raw is None:
        scaling_raw_list = ['ITER89P', 'H98y2', 'NSTX2006H', 'NSTX2006L']
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
        
        # n_e:
        # - many scalings use line-averaged density
        # - Kurskiev2022 is defined in this codebase as volume-averaged density
        #
        # We therefore resolve both line- and volume-averaged density if available.
        n_e_line = None
        n_e_vol = None
        if 'n_e' in df.columns:
            # Ambiguous, but historically treated as line-averaged in this plotting helper.
            n_e_line = df['n_e'].values
        if 'ne_line_19m3' in df.columns:
            n_e_line = df['ne_line_19m3'].values * 1e19  # 10^19 m^-3 -> m^-3
        elif 'ne_19m3' in df.columns and n_e_line is None:
            # legacy compatibility: ne_19m3 was stored as line-averaged in the pipeline
            n_e_line = df['ne_19m3'].values * 1e19  # 10^19 m^-3 -> m^-3
        if 'ne_vol_19m3' in df.columns:
            n_e_vol = df['ne_vol_19m3'].values * 1e19  # 10^19 m^-3 -> m^-3
        if n_e_line is None and n_e_vol is None:
            raise ValueError("Could not find any density column among: n_e, ne_19m3, ne_line_19m3, ne_vol_19m3")
        
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
                # Kurskiev2022 expects volume-averaged density (see vaft.formula.constants).
                if scaling == 'Kurskiev2022':
                    if n_e_vol is not None and np.isfinite(n_e_vol[i]) and n_e_vol[i] > 0:
                        n_e_val = n_e_vol[i]
                        input_density_definition = 'volume_avg'
                        line_to_volume_factor = None
                    elif n_e_line is not None and np.isfinite(n_e_line[i]) and n_e_line[i] > 0 and n_e_vol is not None and np.isfinite(n_e_vol[i]) and n_e_vol[i] > 0:
                        # Fallback: convert line-avg to volume-avg if both are present.
                        n_e_val = n_e_line[i]
                        input_density_definition = 'line_avg'
                        line_to_volume_factor = float(n_e_vol[i] / n_e_line[i])
                    else:
                        raise ValueError("Missing valid volume-averaged density for Kurskiev2022")
                else:
                    # Default behaviour: use line-averaged density if available, else volume-averaged.
                    if n_e_line is not None and np.isfinite(n_e_line[i]) and n_e_line[i] > 0:
                        n_e_val = n_e_line[i]
                        input_density_definition = 'line_avg'
                        line_to_volume_factor = None
                    elif n_e_vol is not None and np.isfinite(n_e_vol[i]) and n_e_vol[i] > 0:
                        n_e_val = n_e_vol[i]
                        input_density_definition = 'volume_avg'
                        line_to_volume_factor = None
                    else:
                        raise ValueError("Missing valid density")

                tauE_scaling_val = confinement_time_from_engineering_parameters(
                    I_p=I_p[i], B_t=B_t[i], P_loss=P_loss[i], n_e=n_e_val,
                    M=M[i] if isinstance(M, np.ndarray) else M,
                    R=R[i], epsilon=epsilon[i], kappa=kappa[i],
                    scaling=scaling,
                    input_density_definition=input_density_definition,
                    line_to_volume_factor=line_to_volume_factor,
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
        
        try:
            # This plot compares ohmic-method consistency only; radiation is not required.
            power_balance = compute_power_balance(ods, include_line_radiation=False)
            P_ohm_flux = np.asarray(power_balance['P_ohm_flux'], dtype=float)
            P_ohm_diss = np.asarray(power_balance['P_ohm_diss'], dtype=float)

            if P_ohm_flux.ndim == 0:
                P_ohm_flux = np.asarray([float(P_ohm_flux)], dtype=float)
            if P_ohm_diss.ndim == 0:
                P_ohm_diss = np.asarray([float(P_ohm_diss)], dtype=float)

            n_pts = min(len(P_ohm_flux), len(P_ohm_diss))
            if n_pts == 0:
                continue

            valid = np.isfinite(P_ohm_flux[:n_pts]) & np.isfinite(P_ohm_diss[:n_pts])
            for i in np.where(valid)[0]:
                P_ohm_flux_list.append(float(P_ohm_flux[i]))
                P_ohm_diss_list.append(float(P_ohm_diss[i]))
                shot_labels.append(shot_label)
                time_slices.append(int(i))

        except Exception as e:
            logger.warning(f"Error computing power balance for {key}: {e}")
            continue
    
    if len(P_ohm_flux_list) == 0:
        logger.warning("No valid ohmic power data found")
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

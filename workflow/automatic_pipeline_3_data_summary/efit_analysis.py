#!/usr/bin/env python3
"""
EFIT Data Analysis Script

This script analyzes EFIT data from efit_history.xlsx file, creating histograms
and calculating basic statistics for all numeric columns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style for better visualization
plt.style.use('seaborn')
sns.set_palette('husl')

def load_efit_data(file_path='efit_history.xlsx'):
    """Load EFIT data from Excel file.
    
    Args:
        file_path (str): Path to the Excel file
        
    Returns:
        pd.DataFrame: Loaded EFIT data
    """
    try:
        df = pd.read_excel(file_path)
        print(f"Successfully loaded data from {file_path}")
        print(f"\nDataFrame Info:")
        print(f"Shape: {df.shape}")
        print(f"\nColumns:")
        for col in df.columns:
            print(f"- {col}")
        return df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def plot_efit_histograms(df):
    """Plot histograms for all numeric columns except 'shot' and 'time'.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing EFIT data
    """
    if df is None:
        print("No data available for plotting")
        return
    
    # Get numeric columns excluding 'shot' and 'time'
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    plot_cols = [col for col in numeric_cols if col not in ['shot', 'time']]
    
    if not plot_cols:
        print("No numeric columns found for plotting")
        return
    
    # Calculate number of rows and columns for subplot layout
    n_cols = 3
    n_rows = (len(plot_cols) + n_cols - 1) // n_cols
    
    # Create figure and subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten()
    
    # Plot histograms
    for idx, col in enumerate(plot_cols):
        sns.histplot(data=df, x=col, ax=axes[idx], kde=True)
        axes[idx].set_title(f'{col} Distribution')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Count')
    
    # Hide empty subplots
    for idx in range(len(plot_cols), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def calculate_efit_statistics(df):
    """Calculate and display statistics for all numeric columns except 'shot' and 'time'.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing EFIT data
    """
    if df is None:
        print("No data available for statistics calculation")
        return
    
    # Get numeric columns excluding 'shot' and 'time'
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    stat_cols = [col for col in numeric_cols if col not in ['shot', 'time']]
    
    if not stat_cols:
        print("No numeric columns found for statistics calculation")
        return
    
    # Calculate statistics
    stats = df[stat_cols].agg(['mean', 'std', 'min', 'max']).round(4)
    
    # Display statistics in a formatted table
    print("\nEFIT Data Statistics:")
    print("-" * 80)
    print(f"{'Column':<30} {'Mean':>12} {'Std Dev':>12} {'Min':>12} {'Max':>12}")
    print("-" * 80)
    
    for col in stat_cols:
        mean = stats.loc['mean', col]
        std = stats.loc['std', col]
        min_val = stats.loc['min', col]
        max_val = stats.loc['max', col]
        print(f"{col:<30} {mean:>12.4f} {std:>12.4f} {min_val:>12.4f} {max_val:>12.4f}")
    
    print("-" * 80)

if __name__ == "__main__":
    # Load the data
    efit_df = load_efit_data()
    
    if efit_df is not None:
        # Plot histograms
        plot_efit_histograms(efit_df)
        
        # Calculate and display statistics
        calculate_efit_statistics(efit_df) 
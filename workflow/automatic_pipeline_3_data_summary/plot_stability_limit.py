import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import itertools

def normalized_plasma_current(Ip, R, a, Bt):
    """
    Calculate normalized plasma current
    
    Parameters:
    Ip : float or np.array
        Plasma current (A)
    R : float or np.array
        Major radius (m)
    a : float or np.array
        Minor radius (m)
    Bt : float or np.array
        Toroidal field (T)
    
    Returns:
    Ip_norm : float or np.array
        Normalized plasma current
    """
    Ip = Ip / 1e6  # Convert A to MA
    Ip_norm = Ip / (a * Bt)
    return Ip_norm

def kink_safety_factor(R, a, kappa, Ip, Bt, type_='conventional'):
    """
    Calculate kink safety factor and related parameters
    """
    mu0 = 4 * np.pi * 1e-7
    epsilon = a / R

    if type_ == 'conventional':
        q_kink = 2 * np.pi * a**2 * kappa * Bt / (mu0 * Ip * R)
        g_factor = 1 / kappa * (1 + 4 / np.pi**2 * (kappa**2 - 1))
        q_kink *= g_factor
        beta_max = np.pi**2 / 16 * kappa * epsilon / q_kink**2
        beta_crit = 0.14 * epsilon * kappa / q_kink
    else:
        raise ValueError("Only 'conventional' type is supported")

    q_min = 1 + kappa / 2
    ip_max = q_kink * Ip * 2 / (1 + kappa)
    ip_max /= 1e6  # Convert MA to A

    return q_kink, q_min, beta_max, beta_crit, ip_max

def plot_stability_limits(input_file="chease_dcon_history.xlsx"):
    """
    Plot stability limits with stability classification coloring
    """
    # Read the joined data
    print(f"Reading data from {input_file}...")
    df = pd.read_excel(input_file)
    
    # Calculate normalized plasma current
    df['Ip_norm'] = normalized_plasma_current(
        df['ip [kA]'] * 1000,  # Convert kA to A
        df['major_radius [m]'],
        df['minor_radius [m]'],
        abs(df['b_field_tor_axis [T]'])
    )
    
    # Calculate kink safety factor
    df['q_kink'], df['q_min'], df['beta_max'], df['beta_crit'], df['ip_max'] = kink_safety_factor(
        df['major_radius [m]'],
        df['minor_radius [m]'],
        df['elongation'],
        df['ip [kA]'] * 1000,  # Convert kA to A
        abs(df['b_field_tor_axis [T]'])
    )
    
    # Create stability classification (3 classes)
    df['stability_class'] = 'stable'
    df.loc[(df['ideal_stability'] == 'Stable') & (df['resistive_stability'] == 'Unstable'), 'stability_class'] = 'resistive_unstable'
    df.loc[df['ideal_stability'] == 'Unstable', 'stability_class'] = 'ideal_unstable'

    colors = {
        'stable': 'green',
        'resistive_unstable': 'orange',
        'ideal_unstable': 'red'
    }
    class_titles = {
        'stable': 'Stable',
        'resistive_unstable': 'Resistive Unstable',
        'ideal_unstable': 'Ideal Unstable'
    }
    
    # Create figure
    fig, ax = plt.subplots(2, 2, figsize=(15, 15))
    fontsize = 26
    plt.rcParams.update({'font.size': fontsize})
    plt.rcParams.update({'axes.titlesize': fontsize})
    plt.rcParams.update({'axes.labelsize': fontsize})
    plt.rcParams.update({'xtick.labelsize': fontsize*0.8})
    plt.rcParams.update({'ytick.labelsize': fontsize*0.8})
    
    scatter_size = 60
    
    # Balloning-Kink limit
    for stability_class in colors.keys():
        mask = df['stability_class'] == stability_class
        ax[0, 0].scatter(
            df.loc[mask, 'Ip_norm'],
            df.loc[mask, 'q_kink'],
            s=scatter_size,
            color=colors[stability_class],
            label=stability_class.replace('_', ' ').title()
        )
    ax[0, 0].axhline(y=df['q_min'].max(), color='k', linestyle='--', linewidth=2)
    ax[0, 0].set_xlabel(r'Normalized Plasma Current $I_N$ [MA/m.T]')
    ax[0, 0].set_ylabel(r'Kink Safety Factor $q^*$')
    ax[0, 0].set_title('Balloning-Kink limit')
    ax[0, 0].legend()
    
    # Low-beta kink current limit
    for stability_class in colors.keys():
        mask = df['stability_class'] == stability_class
        ax[0, 1].scatter(
            df.loc[mask, 'ip [kA]'],
            df.loc[mask, 'ip_max']*1000,
            s=scatter_size,
            color=colors[stability_class],
            label=stability_class.replace('_', ' ').title()
        )
    ax[0, 1].plot(ax[0, 1].get_xlim(), ax[0, 1].get_ylim(), ls="--", c=".3", linewidth=2)
    ax[0, 1].set_xlabel(r'Plasma Current $I_p$ [kA]')
    ax[0, 1].set_ylabel(r'Kink limit on maximum $I_p$ [kA]')
    ax[0, 1].set_title('Low-beta kink current limit')
    ax[0, 1].legend()
    
    # Sawtooth limit
    for stability_class in colors.keys():
        mask = df['stability_class'] == stability_class
        ax[1, 0].scatter(
            df.loc[mask, 'ip [kA]'],
            df.loc[mask, 'q_axis'],
            s=scatter_size,
            color=colors[stability_class],
            label=stability_class.replace('_', ' ').title()
        )
    ax[1, 0].axhline(y=1, color='k', linestyle='--', linewidth=2)
    ax[1, 0].set_xlabel(r'Plasma Current $I_p$ [kA]')
    ax[1, 0].set_ylabel(r'Axis safety factor $q_0$')
    ax[1, 0].set_title('Sawtooth limit')
    ax[1, 0].set_ylim([0, 8])
    ax[1, 0].legend()
    
    # [1, 1] Empirical Kink & double tearing limit
    def empirical_li_qa():
        qa = np.array([2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10])
        li = np.array([0.95, 0.68, 0.93, 0.61, 0.86, 0.5, 0.71, 0.435, 0.7, 0.35, 0.67, 0.3, 0.67, 0.3, 0.67, 0.3, 0.67, 0.3])
        return qa, li

    qa_line, li_line = empirical_li_qa()
    for stability_class in colors.keys():
        mask = df['stability_class'] == stability_class
        ax[1, 1].scatter(
            df.loc[mask, 'q95'],
            df.loc[mask, 'li_3'],
            s=scatter_size,
            color=colors[stability_class],
            label=stability_class.replace('_', ' ').title()
        )
    ax[1, 1].plot(qa_line, li_line, 'k--', linewidth=2)
    ax[1, 1].set_xlabel(r'Edge safety factor $q_{95}$')
    ax[1, 1].set_ylabel(r'Normalized plasma inductance $li$')
    ax[1, 1].set_title('Empirical Kink & double tearing limit')
    ax[1, 1].set_xlim([0, 10])
    ax[1, 1].set_ylim([0, 1])
    ax[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('stability_limit.png', dpi=300, bbox_inches='tight')
    print("Plot saved as stability_limit.png")

def export_statistics(input_file="chease_dcon_history.xlsx", output_file="chease_dcon_history_stats.xlsx"):
    """
    Extract descriptive statistics for all numeric columns and save to Excel.
    """
    df = pd.read_excel(input_file)
    stats = df.describe(percentiles=[.25, .5, .75]).T  # Transpose for better readability
    stats.to_excel(output_file)
    print(f"Descriptive statistics saved to {output_file}")

def plot_histograms(input_file="chease_dcon_history.xlsx", output_file="histograms.png"):
    """
    Plot 4x4 histograms for selected columns, with mean and std in the legend.
    """
    columns = [
        'ip [kA]', 'major_radius [m]', 'minor_radius [m]', 'aspect_ratio',
        'elongation', 'triangularity', 'q_axis', 'q_min',
        'q95', 'magnetic_axis_r [m]', 'beta_poloidal',
        'b_field_tor_axis [T]', 'plasma_volume [m^3]', 'beta_toroidal', 'beta_normal', 'li_3'
    ]
    # Remove duplicates if any
    columns = list(dict.fromkeys(columns))
    df = pd.read_excel(input_file)
    fig, axes = plt.subplots(4, 4, figsize=(24, 20))
    axes = axes.flatten()
    for i, col in enumerate(columns):
        if col not in df.columns:
            axes[i].set_visible(False)
            continue
        data = df[col].dropna()
        mean = data.mean()
        std = data.std()
        axes[i].hist(data, bins=30, color='skyblue', edgecolor='k', alpha=0.8)
        axes[i].set_title(col, fontsize=16)
        axes[i].legend([f"Mean: {mean:.3g}\nStd: {std:.3g}"], fontsize=12, loc='upper right')
        axes[i].tick_params(axis='both', which='major', labelsize=12)
    # Hide any unused subplots
    for j in range(len(columns), 16):
        axes[j].set_visible(False)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.suptitle('Histograms of Key Parameters', fontsize=28, y=1.02)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Histogram plot saved as {output_file}")

def plot_all_pairwise_scatters(input_file="chease_dcon_history.xlsx", output_dir="corr"):
    """
    For all pairs of selected columns, plot scatter plots and save as corr/var1_var2.png
    """
    columns = [
        'ip [kA]', 'major_radius [m]', 'minor_radius [m]', 'aspect_ratio',
        'elongation', 'triangularity', 'q_axis', 'q_min',
        'q95', 'magnetic_axis_r [m]', 'beta_poloidal',
        'b_field_tor_axis [T]', 'plasma_volume [m^3]', 'beta_toroidal', 'beta_normal', 'li_3'
    ]
    # Remove duplicates if any
    columns = list(dict.fromkeys(columns))
    df = pd.read_excel(input_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for var1, var2 in itertools.combinations(columns, 2):
        if var1 not in df.columns or var2 not in df.columns:
            continue
        x = df[var1].dropna()
        y = df[var2].dropna()
        # Only keep rows where both are not NaN
        valid = df[[var1, var2]].dropna()
        if valid.empty:
            continue
        plt.figure(figsize=(7, 6))
        plt.scatter(valid[var1], valid[var2], alpha=0.6, s=20, color='royalblue', edgecolor='k')
        plt.xlabel(var1)
        plt.ylabel(var2)
        plt.title(f"{var1} vs {var2}")
        plt.tight_layout()
        fname = f"{output_dir}/{var1.replace(' ', '_').replace('[','').replace(']','')}_{var2.replace(' ', '_').replace('[','').replace(']','')}.png"
        plt.savefig(fname, dpi=200)
        plt.close()
    print(f"All pairwise scatter plots saved in '{output_dir}' folder.")

if __name__ == '__main__':
    # setting the path
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    plot_stability_limits()
    export_statistics()
    plot_histograms()
    plot_all_pairwise_scatters()

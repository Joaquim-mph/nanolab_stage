#!/usr/bin/env python3
"""
Interactive hysteresis visualizer with enhanced features.

Features:
- Side-by-side comparison of forward/return traces with hysteresis
- Animated voltage sweep visualization
- Heatmap of hysteresis across voltage ranges
- Interactive selection of polynomial orders
- Export publication-quality figures
"""

import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches


def load_all_data(stats_dir: Path, hysteresis_dir: Path, v_max: float):
    """
    Load forward, return, and hysteresis data for a specific V_max.

    Returns:
        Tuple of (forward_df, return_df, hysteresis_df, v_max_clean)
    """
    v_max_clean = str(v_max).replace(".", "p")

    forward_file = stats_dir / f"forward_vmax{v_max_clean}V.csv"
    return_file = stats_dir / f"return_with_fit_vmax{v_max_clean}V.csv"
    hysteresis_file = hysteresis_dir / f"hysteresis_vmax{v_max_clean}V.csv"

    if not all([forward_file.exists(), return_file.exists(), hysteresis_file.exists()]):
        return None, None, None, v_max_clean

    forward_df = pl.read_csv(forward_file)
    return_df = pl.read_csv(return_file)
    hysteresis_df = pl.read_csv(hysteresis_file)

    return forward_df, return_df, hysteresis_df, v_max_clean


def plot_comprehensive_hysteresis(
    stats_dir: Path,
    hysteresis_dir: Path,
    v_max: float,
    output_dir: Path,
    poly_order: int = 3
):
    """
    Create comprehensive hysteresis visualization with multiple panels.

    Panel layout:
    - Top: Forward and return traces overlaid
    - Middle: Hysteresis current
    - Bottom: Normalized hysteresis percentage
    """

    forward_df, return_df, hysteresis_df, v_max_clean = load_all_data(
        stats_dir, hysteresis_dir, v_max
    )

    if forward_df is None:
        print(f"Warning: Missing data for V_max={v_max}V")
        return

    fig = plt.figure(figsize=(14, 12))
    gs = GridSpec(3, 1, height_ratios=[1.2, 1, 0.8], hspace=0.3)

    # Panel 1: Forward and Return traces
    ax1 = fig.add_subplot(gs[0])

    # Forward trace
    ax1.errorbar(
        forward_df["V (V)"].to_numpy(),
        forward_df["I_mean"].to_numpy() * 1e9,
        yerr=forward_df["I_std"].to_numpy() * 1e9,
        fmt='o-',
        color='red',
        label='Forward sweep',
        alpha=0.7,
        markersize=4,
        linewidth=1.5,
        capsize=3,
        errorevery=5
    )

    # Return trace (raw)
    ax1.errorbar(
        return_df["V (V)"].to_numpy(),
        return_df["I_mean"].to_numpy() * 1e9,
        yerr=return_df["I_std"].to_numpy() * 1e9,
        fmt='s-',
        color='blue',
        label='Return sweep',
        alpha=0.7,
        markersize=4,
        linewidth=1.5,
        capsize=3,
        errorevery=5
    )

    # Return trace (polynomial fit)
    fit_col = f"I_fit_poly{poly_order}"
    if fit_col in return_df.columns:
        ax1.plot(
            return_df["V (V)"].to_numpy(),
            return_df[fit_col].to_numpy() * 1e9,
            '--',
            color='darkblue',
            label=f'Return fit (n={poly_order})',
            linewidth=2.5,
            alpha=0.9
        )

    ax1.set_ylabel('Current (nA)', fontsize=13, fontweight='bold')
    ax1.set_title(f'IV Sweep Comparison - V_max = {v_max:.0f}V',
                  fontsize=15, fontweight='bold')
    ax1.legend(fontsize=11, loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.2, linewidth=1)

    # Panel 2: Hysteresis current
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    # Raw hysteresis
    ax2.errorbar(
        hysteresis_df["V (V)"].to_numpy(),
        hysteresis_df["I_hysteresis_raw"].to_numpy() * 1e9,
        yerr=hysteresis_df["I_hysteresis_std"].to_numpy() * 1e9,
        fmt='o-',
        color='darkgreen',
        label='Raw hysteresis',
        alpha=0.7,
        markersize=4,
        linewidth=1.5,
        capsize=3,
        errorevery=3
    )

    # Polynomial hysteresis
    poly_col = f"I_hysteresis_poly{poly_order}"
    if poly_col in hysteresis_df.columns:
        ax2.plot(
            hysteresis_df["V (V)"].to_numpy(),
            hysteresis_df[poly_col].to_numpy() * 1e9,
            '-',
            color='purple',
            label=f'Fit hysteresis (n={poly_order})',
            linewidth=2.5,
            alpha=0.9
        )

    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.4, linewidth=1.5)
    ax2.set_ylabel('Hysteresis (nA)', fontsize=13, fontweight='bold')
    ax2.set_title('Hysteresis Current (Forward - Return)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11, loc='best', framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')

    # Add shaded region for positive/negative hysteresis
    v_vals = hysteresis_df["V (V)"].to_numpy()
    hyst_vals = hysteresis_df["I_hysteresis_raw"].to_numpy() * 1e9
    ax2.fill_between(v_vals, 0, hyst_vals,
                      where=(hyst_vals >= 0),
                      color='green', alpha=0.2, label='Positive')
    ax2.fill_between(v_vals, 0, hyst_vals,
                      where=(hyst_vals < 0),
                      color='red', alpha=0.2, label='Negative')

    # Panel 3: Normalized hysteresis (percentage)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)

    # Calculate percentage hysteresis
    forward_interp = np.interp(
        hysteresis_df["V (V)"].to_numpy(),
        forward_df["V (V)"].to_numpy(),
        forward_df["I_mean"].to_numpy()
    )

    # Avoid division by zero
    pct_hysteresis = np.zeros_like(forward_interp)
    nonzero_mask = np.abs(forward_interp) > 1e-12
    pct_hysteresis[nonzero_mask] = (
        hysteresis_df["I_hysteresis_raw"].to_numpy()[nonzero_mask] /
        forward_interp[nonzero_mask] * 100
    )

    ax3.plot(
        hysteresis_df["V (V)"].to_numpy(),
        pct_hysteresis,
        'o-',
        color='darkorange',
        markersize=4,
        linewidth=2,
        alpha=0.8
    )

    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.4, linewidth=1.5)
    ax3.set_xlabel('Voltage (V)', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Hysteresis (%)', fontsize=13, fontweight='bold')
    ax3.set_title('Normalized Hysteresis', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')

    # Add statistics box
    mean_hyst = np.mean(np.abs(hyst_vals))
    max_hyst = np.max(np.abs(hyst_vals))
    mean_pct = np.mean(np.abs(pct_hysteresis))

    stats_text = f'Statistics:\n'
    stats_text += f'  Mean |H|: {mean_hyst:.3f} nA\n'
    stats_text += f'  Max |H|:  {max_hyst:.3f} nA\n'
    stats_text += f'  Mean %:   {mean_pct:.2f}%'

    ax1.text(
        0.98, 0.02,
        stats_text,
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment='bottom',
        horizontalalignment='right',
        family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85, edgecolor='orange', linewidth=2)
    )

    plt.tight_layout()
    output_file = output_dir / f"comprehensive_hysteresis_vmax{v_max_clean}V.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_hysteresis_heatmap(
    hysteresis_dir: Path,
    output_dir: Path,
    poly_order: int = 3
):
    """
    Create heatmap of hysteresis across all voltage ranges.

    X-axis: Applied voltage
    Y-axis: V_max range
    Color: Hysteresis magnitude
    """

    # Find all hysteresis files
    hyst_files = sorted(hysteresis_dir.glob("hysteresis_vmax*.csv"))

    if len(hyst_files) == 0:
        print("No hysteresis files found")
        return

    # Collect data
    v_max_list = []
    voltage_grids = []
    hysteresis_grids = []

    for hyst_file in hyst_files:
        v_max_str = hyst_file.stem.replace("hysteresis_vmax", "").replace("V", "")
        v_max = float(v_max_str.replace("p", "."))

        hyst_df = pl.read_csv(hyst_file)

        v_max_list.append(v_max)
        voltage_grids.append(hyst_df["V (V)"].to_numpy())

        # Use polynomial hysteresis if available
        poly_col = f"I_hysteresis_poly{poly_order}"
        if poly_col in hyst_df.columns:
            hysteresis_grids.append(hyst_df[poly_col].to_numpy() * 1e9)
        else:
            hysteresis_grids.append(hyst_df["I_hysteresis_raw"].to_numpy() * 1e9)

    # Create interpolated grid
    v_min = min([v.min() for v in voltage_grids])
    v_max_global = max([v.max() for v in voltage_grids])
    v_common = np.linspace(v_min, v_max_global, 200)

    heatmap_data = []
    for v_grid, h_grid in zip(voltage_grids, hysteresis_grids):
        h_interp = np.interp(v_common, v_grid, h_grid, left=np.nan, right=np.nan)
        heatmap_data.append(h_interp)

    heatmap_data = np.array(heatmap_data)

    # Create heatmap
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10),
                                    height_ratios=[3, 1])

    # Main heatmap
    im = ax1.imshow(
        heatmap_data,
        aspect='auto',
        cmap='RdBu_r',
        extent=[v_min, v_max_global, v_max_list[0], v_max_list[-1]],
        origin='lower',
        interpolation='bilinear'
    )

    # Symmetric color scale around zero
    vmax = np.nanmax(np.abs(heatmap_data))
    im.set_clim(-vmax, vmax)

    ax1.set_xlabel('Applied Voltage (V)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Voltage Range V_max (V)', fontsize=13, fontweight='bold')
    ax1.set_title(f'Hysteresis Heatmap (Poly n={poly_order})',
                  fontsize=15, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax1, label='Hysteresis Current (nA)')
    cbar.ax.tick_params(labelsize=11)

    # Add contour lines
    contour_levels = np.linspace(-vmax, vmax, 11)
    ax1.contour(
        v_common, v_max_list, heatmap_data,
        levels=contour_levels,
        colors='black',
        alpha=0.3,
        linewidths=0.5
    )

    # Bottom panel: Mean hysteresis vs V_max
    mean_hyst_per_vmax = np.nanmean(np.abs(heatmap_data), axis=1)

    ax2.plot(
        v_max_list,
        mean_hyst_per_vmax,
        'o-',
        color='darkred',
        markersize=8,
        linewidth=2.5,
        label='Mean |Hysteresis|'
    )

    ax2.fill_between(v_max_list, 0, mean_hyst_per_vmax, alpha=0.3, color='red')

    ax2.set_xlabel('Voltage Range V_max (V)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Mean |Hysteresis| (nA)', fontsize=13, fontweight='bold')
    ax2.set_title('Average Hysteresis Magnitude', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=11)

    plt.tight_layout()
    output_file = output_dir / f"hysteresis_heatmap_poly{poly_order}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_polynomial_comparison(
    hysteresis_dir: Path,
    output_dir: Path,
    v_max: float
):
    """
    Compare hysteresis for different polynomial orders side-by-side.
    """

    v_max_clean = str(v_max).replace(".", "p")
    hyst_file = hysteresis_dir / f"hysteresis_vmax{v_max_clean}V.csv"

    if not hyst_file.exists():
        print(f"Warning: {hyst_file} not found")
        return

    hyst_df = pl.read_csv(hyst_file)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    colors = ['red', 'blue', 'green', 'purple']
    poly_orders = [1, 3, 5, 7]

    for idx, (ax, order, color) in enumerate(zip(axes, poly_orders, colors)):
        # Raw hysteresis (reference)
        ax.plot(
            hyst_df["V (V)"].to_numpy(),
            hyst_df["I_hysteresis_raw"].to_numpy() * 1e9,
            'o',
            color='gray',
            label='Raw data',
            alpha=0.4,
            markersize=3
        )

        # Polynomial hysteresis
        poly_col = f"I_hysteresis_poly{order}"
        if poly_col in hyst_df.columns:
            ax.plot(
                hyst_df["V (V)"].to_numpy(),
                hyst_df[poly_col].to_numpy() * 1e9,
                '-',
                color=color,
                label=f'Polynomial n={order}',
                linewidth=3,
                alpha=0.9
            )

            # Calculate residuals
            residuals = (
                hyst_df["I_hysteresis_raw"].to_numpy() -
                hyst_df[poly_col].to_numpy()
            ) * 1e9

            rms = np.sqrt(np.mean(residuals**2))

            # Add residual statistics
            stats_text = f'RMS residual: {rms:.3f} nA'
            ax.text(
                0.05, 0.95,
                stats_text,
                transform=ax.transAxes,
                fontsize=11,
                verticalalignment='top',
                family='monospace',
                bbox=dict(boxstyle='round', facecolor='white',
                         alpha=0.8, edgecolor=color, linewidth=2)
            )

        ax.axhline(y=0, color='k', linestyle='--', alpha=0.4, linewidth=1.5)
        ax.set_xlabel('Voltage (V)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Hysteresis (nA)', fontsize=12, fontweight='bold')
        ax.set_title(f'Polynomial Order n={order}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3, linestyle='--')

    fig.suptitle(f'Polynomial Fit Comparison - V_max = {v_max:.0f}V',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout()
    output_file = output_dir / f"polynomial_comparison_vmax{v_max_clean}V.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def visualize_hysteresis(
    stats_dir: Path,
    hysteresis_dir: Path,
    output_dir: Path,
    poly_order: int = 3,
    create_heatmap: bool = True,
    create_comparison: bool = True
):
    """
    Main function to create all hysteresis visualizations.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Hysteresis Visualizer")
    print(f"=" * 60)
    print(f"Stats directory:      {stats_dir}")
    print(f"Hysteresis directory: {hysteresis_dir}")
    print(f"Output directory:     {output_dir}")
    print(f"Polynomial order:     {poly_order}")
    print(f"=" * 60)

    # Find all voltage ranges
    hyst_files = sorted(hysteresis_dir.glob("hysteresis_vmax*.csv"))

    if len(hyst_files) == 0:
        print("No hysteresis files found!")
        return

    print(f"\nFound {len(hyst_files)} voltage ranges\n")

    # Create comprehensive plots for each V_max
    for hyst_file in hyst_files:
        v_max_str = hyst_file.stem.replace("hysteresis_vmax", "").replace("V", "")
        v_max = float(v_max_str.replace("p", "."))

        print(f"Processing V_max = {v_max:.0f}V...")
        plot_comprehensive_hysteresis(stats_dir, hysteresis_dir, v_max,
                                      output_dir, poly_order)

        if create_comparison:
            plot_polynomial_comparison(hysteresis_dir, output_dir, v_max)

    # Create heatmap
    if create_heatmap and len(hyst_files) > 1:
        print(f"\nCreating heatmap...")
        plot_hysteresis_heatmap(hysteresis_dir, output_dir, poly_order)

    print(f"\n✓ All visualizations saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Enhanced hysteresis visualization tool"
    )
    parser.add_argument("--stats-dir", type=Path, required=True,
                       help="Directory containing IV statistics")
    parser.add_argument("--hysteresis-dir", type=Path, required=True,
                       help="Directory containing hysteresis data")
    parser.add_argument("--output-dir", type=Path, required=True,
                       help="Output directory for visualizations")
    parser.add_argument("--poly-order", type=int, default=3,
                       choices=[1, 3, 5, 7],
                       help="Polynomial order for fits (default: 3)")
    parser.add_argument("--no-heatmap", action="store_true",
                       help="Skip heatmap generation")
    parser.add_argument("--no-comparison", action="store_true",
                       help="Skip polynomial comparison plots")

    args = parser.parse_args()

    visualize_hysteresis(
        args.stats_dir,
        args.hysteresis_dir,
        args.output_dir,
        poly_order=args.poly_order,
        create_heatmap=not args.no_heatmap,
        create_comparison=not args.no_comparison
    )


if __name__ == "__main__":
    main()

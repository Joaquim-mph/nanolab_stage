#!/usr/bin/env python3
"""
Comprehensive hysteresis exploration tool.

Creates detailed comparative visualizations across all V_max ranges:
- Overlay of all hysteresis traces
- Statistical comparison panels
- Voltage-dependent trends
- Distribution analysis
- Interactive gridded layouts
"""

import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
from matplotlib.gridspec import GridSpec
import matplotlib.cm as cm


def load_all_hysteresis_data(hysteresis_dir: Path):
    """
    Load all hysteresis data files.

    Returns:
        List of tuples: (v_max, hysteresis_df)
    """
    hyst_files = sorted(hysteresis_dir.glob("hysteresis_vmax*.csv"))

    if len(hyst_files) == 0:
        print("No hysteresis files found")
        return []

    data_list = []
    for hyst_file in hyst_files:
        v_max_str = hyst_file.stem.replace("hysteresis_vmax", "").replace("V", "")
        v_max = float(v_max_str.replace("p", "."))

        hyst_df = pl.read_csv(hyst_file)
        data_list.append((v_max, hyst_df))

    return sorted(data_list, key=lambda x: x[0])


def plot_all_overlaid(data_list, output_dir: Path, poly_order: int = 3):
    """
    Overlay all hysteresis traces on a single plot.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

    n_ranges = len(data_list)
    colors = cm.viridis(np.linspace(0, 1, n_ranges))

    # Panel 1: Raw hysteresis
    for (v_max, hyst_df), color in zip(data_list, colors):
        ax1.plot(
            hyst_df["V (V)"].to_numpy(),
            hyst_df["I_hysteresis_raw"].to_numpy() * 1e9,
            '-',
            color=color,
            label=f'V_max={v_max:.0f}V',
            linewidth=2,
            alpha=0.8
        )

    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.4, linewidth=1.5)
    ax1.set_xlabel('Applied Voltage (V)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Hysteresis Current (nA)', fontsize=14, fontweight='bold')
    ax1.set_title('Raw Hysteresis - All Voltage Ranges Overlaid',
                  fontsize=16, fontweight='bold')
    ax1.legend(fontsize=10, loc='best', ncol=2, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')

    # Panel 2: Polynomial fits
    poly_col = f"I_hysteresis_poly{poly_order}"

    for (v_max, hyst_df), color in zip(data_list, colors):
        if poly_col in hyst_df.columns:
            ax2.plot(
                hyst_df["V (V)"].to_numpy(),
                hyst_df[poly_col].to_numpy() * 1e9,
                '-',
                color=color,
                label=f'V_max={v_max:.0f}V',
                linewidth=2.5,
                alpha=0.9
            )

    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.4, linewidth=1.5)
    ax2.set_xlabel('Applied Voltage (V)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Hysteresis Current (nA)', fontsize=14, fontweight='bold')
    ax2.set_title(f'Polynomial Hysteresis (n={poly_order}) - All Voltage Ranges Overlaid',
                  fontsize=16, fontweight='bold')
    ax2.legend(fontsize=10, loc='best', ncol=2, framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    output_file = output_dir / f"overlay_all_ranges_poly{poly_order}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_gridded_comparison(data_list, output_dir: Path, poly_order: int = 3):
    """
    Create a grid of all hysteresis traces for easy comparison.
    """
    n_ranges = len(data_list)
    n_cols = 4
    n_rows = (n_ranges + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))

    if n_ranges == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    poly_col = f"I_hysteresis_poly{poly_order}"

    for idx, (v_max, hyst_df) in enumerate(data_list):
        ax = axes[idx]

        # Raw data
        ax.plot(
            hyst_df["V (V)"].to_numpy(),
            hyst_df["I_hysteresis_raw"].to_numpy() * 1e9,
            'o',
            color='gray',
            markersize=2,
            alpha=0.5,
            label='Raw'
        )

        # Polynomial fit
        if poly_col in hyst_df.columns:
            ax.plot(
                hyst_df["V (V)"].to_numpy(),
                hyst_df[poly_col].to_numpy() * 1e9,
                '-',
                color='red',
                linewidth=2.5,
                alpha=0.9,
                label=f'Poly n={poly_order}'
            )

        # Calculate statistics
        hyst_mean = hyst_df["I_hysteresis_raw"].mean() * 1e9
        hyst_std = hyst_df["I_hysteresis_raw"].std() * 1e9
        hyst_max = hyst_df["I_hysteresis_raw"].abs().max() * 1e9

        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=1)
        ax.set_xlabel('V (V)', fontsize=10)
        ax.set_ylabel('Hysteresis (nA)', fontsize=10)
        ax.set_title(f'V_max = {v_max:.0f}V\nμ={hyst_mean:.2f}, σ={hyst_std:.2f}, max={hyst_max:.2f} nA',
                    fontsize=10, fontweight='bold')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(len(data_list), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Hysteresis Comparison - All Voltage Ranges',
                 fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()

    output_file = output_dir / f"grid_comparison_poly{poly_order}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_statistics_summary(data_list, summary_df, output_dir: Path):
    """
    Statistical analysis across V_max ranges.
    """
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 2, hspace=0.3, wspace=0.3)

    v_max_vals = np.array([v for v, _ in data_list])

    # Collect statistics
    mean_raw = []
    std_raw = []
    max_raw = []
    min_raw = []
    rms_raw = []

    for v_max, hyst_df in data_list:
        hyst_vals = hyst_df["I_hysteresis_raw"].to_numpy()
        mean_raw.append(np.mean(hyst_vals) * 1e9)
        std_raw.append(np.std(hyst_vals) * 1e9)
        max_raw.append(np.max(hyst_vals) * 1e9)
        min_raw.append(np.min(hyst_vals) * 1e9)
        rms_raw.append(np.sqrt(np.mean(hyst_vals**2)) * 1e9)

    mean_raw = np.array(mean_raw)
    std_raw = np.array(std_raw)
    max_raw = np.array(max_raw)
    min_raw = np.array(min_raw)
    rms_raw = np.array(rms_raw)

    # Plot 1: Mean hysteresis
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.errorbar(v_max_vals, mean_raw, yerr=std_raw,
                 fmt='o-', markersize=10, linewidth=2.5, capsize=5,
                 color='darkblue', label='Mean ± Std')
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.4)
    ax1.fill_between(v_max_vals, mean_raw - std_raw, mean_raw + std_raw,
                      alpha=0.3, color='blue')
    ax1.set_xlabel('V_max (V)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Mean Hysteresis (nA)', fontsize=12, fontweight='bold')
    ax1.set_title('Mean Hysteresis vs Voltage Range', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    # Plot 2: Max/Min range
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(v_max_vals, max_raw, 'o-', markersize=8, linewidth=2,
             color='red', label='Maximum')
    ax2.plot(v_max_vals, min_raw, 's-', markersize=8, linewidth=2,
             color='blue', label='Minimum')
    ax2.fill_between(v_max_vals, min_raw, max_raw, alpha=0.2, color='purple')
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.4)
    ax2.set_xlabel('V_max (V)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Hysteresis (nA)', fontsize=12, fontweight='bold')
    ax2.set_title('Hysteresis Range (Min/Max)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    # Plot 3: Standard deviation
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.bar(v_max_vals, std_raw, width=0.6, color='orange', alpha=0.7, edgecolor='black')
    ax3.set_xlabel('V_max (V)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Std Deviation (nA)', fontsize=12, fontweight='bold')
    ax3.set_title('Hysteresis Variability', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: RMS hysteresis
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(v_max_vals, rms_raw, 'o-', markersize=10, linewidth=2.5,
             color='purple', label='RMS')
    ax4.fill_between(v_max_vals, 0, rms_raw, alpha=0.3, color='purple')
    ax4.set_xlabel('V_max (V)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('RMS Hysteresis (nA)', fontsize=12, fontweight='bold')
    ax4.set_title('RMS Hysteresis Magnitude', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=10)

    # Plot 5: Absolute hysteresis
    ax5 = fig.add_subplot(gs[2, 0])
    abs_mean = np.abs(mean_raw)
    ax5.plot(v_max_vals, abs_mean, 'o-', markersize=10, linewidth=2.5,
             color='darkred')
    ax5.fill_between(v_max_vals, 0, abs_mean, alpha=0.3, color='red')
    ax5.set_xlabel('V_max (V)', fontsize=12, fontweight='bold')
    ax5.set_ylabel('|Mean Hysteresis| (nA)', fontsize=12, fontweight='bold')
    ax5.set_title('Absolute Mean Hysteresis', fontsize=13, fontweight='bold')
    ax5.grid(True, alpha=0.3)

    # Plot 6: Polynomial comparison (if available)
    ax6 = fig.add_subplot(gs[2, 1])

    poly_orders = [1, 3, 5, 7]
    poly_colors = ['red', 'blue', 'green', 'purple']

    for order, color in zip(poly_orders, poly_colors):
        col = f"hyst_mean_poly{order}"
        if col in summary_df.columns:
            vals = (summary_df[col].abs() * 1e9).to_numpy()
            ax6.plot(v_max_vals, vals, 'o-', markersize=6, linewidth=2,
                    color=color, label=f'Poly n={order}', alpha=0.8)

    ax6.plot(v_max_vals, abs_mean, 's--', markersize=8, linewidth=2,
            color='black', label='Raw', alpha=0.6)
    ax6.set_xlabel('V_max (V)', fontsize=12, fontweight='bold')
    ax6.set_ylabel('|Mean Hysteresis| (nA)', fontsize=12, fontweight='bold')
    ax6.set_title('Polynomial Fit Comparison', fontsize=13, fontweight='bold')
    ax6.legend(fontsize=9, loc='best')
    ax6.grid(True, alpha=0.3)

    plt.suptitle('Statistical Analysis - Hysteresis vs Voltage Range',
                 fontsize=18, fontweight='bold', y=0.998)

    output_file = output_dir / "statistics_summary.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_normalized_comparison(data_list, stats_dir: Path, output_dir: Path):
    """
    Compare normalized hysteresis (percentage of forward current).
    """
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))

    n_ranges = len(data_list)
    colors = cm.plasma(np.linspace(0, 1, n_ranges))

    # Collect normalized data
    all_pct_data = []

    for (v_max, hyst_df), color in zip(data_list, colors):
        v_max_clean = str(v_max).replace(".", "p")
        forward_file = stats_dir / f"forward_vmax{v_max_clean}V.csv"

        if not forward_file.exists():
            continue

        forward_df = pl.read_csv(forward_file)

        # Interpolate forward current to hysteresis voltage points
        forward_interp = np.interp(
            hyst_df["V (V)"].to_numpy(),
            forward_df["V (V)"].to_numpy(),
            forward_df["I_mean"].to_numpy()
        )

        # Calculate percentage
        pct_hysteresis = np.zeros_like(forward_interp)
        nonzero_mask = np.abs(forward_interp) > 1e-12
        pct_hysteresis[nonzero_mask] = (
            hyst_df["I_hysteresis_raw"].to_numpy()[nonzero_mask] /
            forward_interp[nonzero_mask] * 100
        )

        all_pct_data.append((v_max, hyst_df["V (V)"].to_numpy(), pct_hysteresis))

        # Plot
        axes[0].plot(
            hyst_df["V (V)"].to_numpy(),
            pct_hysteresis,
            '-',
            color=color,
            label=f'V_max={v_max:.0f}V',
            linewidth=2,
            alpha=0.8
        )

    axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.4, linewidth=1.5)
    axes[0].set_xlabel('Applied Voltage (V)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Hysteresis (%)', fontsize=14, fontweight='bold')
    axes[0].set_title('Normalized Hysteresis - All Voltage Ranges',
                      fontsize=16, fontweight='bold')
    axes[0].legend(fontsize=10, loc='best', ncol=2, framealpha=0.9)
    axes[0].grid(True, alpha=0.3, linestyle='--')

    # Bottom panel: Mean percentage vs V_max
    v_max_vals = []
    mean_pct_vals = []

    for v_max, v_vals, pct_vals in all_pct_data:
        v_max_vals.append(v_max)
        mean_pct_vals.append(np.mean(np.abs(pct_vals)))

    axes[1].bar(v_max_vals, mean_pct_vals, width=0.6,
                color='coral', alpha=0.7, edgecolor='black', linewidth=1.5)
    axes[1].set_xlabel('V_max (V)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Mean |Hysteresis| (%)', fontsize=14, fontweight='bold')
    axes[1].set_title('Average Normalized Hysteresis vs Voltage Range',
                      fontsize=16, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_file = output_dir / "normalized_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_distribution_analysis(data_list, output_dir: Path):
    """
    Histogram and distribution analysis of hysteresis values.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Collect all data
    all_hyst_raw = []
    for v_max, hyst_df in data_list:
        all_hyst_raw.extend(hyst_df["I_hysteresis_raw"].to_numpy() * 1e9)

    # Plot 1: Overall histogram
    axes[0, 0].hist(all_hyst_raw, bins=50, color='steelblue',
                    alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    axes[0, 0].set_xlabel('Hysteresis (nA)', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Count', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Overall Hysteresis Distribution', fontsize=13, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    # Plot 2: Stacked histograms by V_max
    n_ranges = len(data_list)
    colors = cm.tab10(np.linspace(0, 1, n_ranges))

    for (v_max, hyst_df), color in zip(data_list, colors):
        hyst_vals = hyst_df["I_hysteresis_raw"].to_numpy() * 1e9
        axes[0, 1].hist(hyst_vals, bins=30, alpha=0.5, label=f'V_max={v_max:.0f}V',
                       color=color, edgecolor='black', linewidth=0.5)

    axes[0, 1].axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    axes[0, 1].set_xlabel('Hysteresis (nA)', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Count', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Hysteresis Distribution by V_max', fontsize=13, fontweight='bold')
    axes[0, 1].legend(fontsize=8, loc='best')
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # Plot 3: Box plot comparison
    box_data = []
    box_labels = []
    for v_max, hyst_df in data_list:
        box_data.append(hyst_df["I_hysteresis_raw"].to_numpy() * 1e9)
        box_labels.append(f'{v_max:.0f}V')

    bp = axes[1, 0].boxplot(box_data, labels=box_labels, patch_artist=True,
                            showmeans=True, meanline=True)

    # Color boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    axes[1, 0].axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.5)
    axes[1, 0].set_xlabel('V_max Range', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Hysteresis (nA)', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Box Plot Comparison', fontsize=13, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Plot 4: Cumulative distribution
    for (v_max, hyst_df), color in zip(data_list, colors):
        hyst_vals = np.sort(hyst_df["I_hysteresis_raw"].to_numpy() * 1e9)
        cumulative = np.arange(1, len(hyst_vals) + 1) / len(hyst_vals) * 100
        axes[1, 1].plot(hyst_vals, cumulative, '-', color=color,
                       linewidth=2, alpha=0.8, label=f'V_max={v_max:.0f}V')

    axes[1, 1].axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    axes[1, 1].set_xlabel('Hysteresis (nA)', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Cumulative Probability (%)', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Cumulative Distribution Function', fontsize=13, fontweight='bold')
    axes[1, 1].legend(fontsize=8, loc='best')
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('Distribution Analysis - Hysteresis Characteristics',
                 fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()

    output_file = output_dir / "distribution_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def explore_hysteresis(
    stats_dir: Path,
    hysteresis_dir: Path,
    output_dir: Path,
    poly_order: int = 3
):
    """
    Main exploration function - creates all comparative visualizations.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nHysteresis Data Explorer")
    print(f"=" * 70)
    print(f"Stats directory:      {stats_dir}")
    print(f"Hysteresis directory: {hysteresis_dir}")
    print(f"Output directory:     {output_dir}")
    print(f"Polynomial order:     {poly_order}")
    print(f"=" * 70)

    # Load all data
    data_list = load_all_hysteresis_data(hysteresis_dir)

    if len(data_list) == 0:
        print("\nError: No hysteresis data found!")
        return

    print(f"\nFound {len(data_list)} voltage ranges:")
    for v_max, hyst_df in data_list:
        print(f"  - V_max = {v_max:.1f}V ({len(hyst_df)} points)")

    # Load summary
    summary_file = hysteresis_dir / "hysteresis_summary.csv"
    summary_df = None
    if summary_file.exists():
        summary_df = pl.read_csv(summary_file)

    # Create visualizations
    print(f"\nGenerating comparative visualizations...")

    print(f"\n[1/5] Creating overlay plots...")
    plot_all_overlaid(data_list, output_dir, poly_order)

    print(f"[2/5] Creating gridded comparison...")
    plot_gridded_comparison(data_list, output_dir, poly_order)

    print(f"[3/5] Creating statistical summary...")
    if summary_df is not None:
        plot_statistics_summary(data_list, summary_df, output_dir)

    print(f"[4/5] Creating normalized comparison...")
    plot_normalized_comparison(data_list, stats_dir, output_dir)

    print(f"[5/5] Creating distribution analysis...")
    plot_distribution_analysis(data_list, output_dir)

    print(f"\n{'='*70}")
    print(f"✓ Exploration complete! All plots saved to: {output_dir}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive hysteresis exploration and comparison tool"
    )
    parser.add_argument("--stats-dir", type=Path, required=True,
                       help="Directory containing IV statistics")
    parser.add_argument("--hysteresis-dir", type=Path, required=True,
                       help="Directory containing hysteresis data")
    parser.add_argument("--output-dir", type=Path, required=True,
                       help="Output directory for exploration plots")
    parser.add_argument("--poly-order", type=int, default=3,
                       choices=[1, 3, 5, 7],
                       help="Polynomial order for fits (default: 3)")

    args = parser.parse_args()

    explore_hysteresis(
        args.stats_dir,
        args.hysteresis_dir,
        args.output_dir,
        poly_order=args.poly_order
    )


if __name__ == "__main__":
    main()

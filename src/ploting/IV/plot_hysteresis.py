#!/usr/bin/env python3
"""
Plot hysteresis current (forward - return difference).

Creates visualization of hysteresis for both raw data and polynomial fits.
"""

import polars as pl
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def plot_hysteresis(hysteresis_dir: Path, output_dir: Path):
    """
    Create plots from hysteresis data.

    Args:
        hysteresis_dir: Directory containing hysteresis data
        output_dir: Directory to save plots
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load summary
    summary_file = hysteresis_dir / "hysteresis_summary.csv"
    if not summary_file.exists():
        print(f"Error: {summary_file} not found")
        return

    summary = pl.read_csv(summary_file)
    print(f"Found {len(summary)} voltage ranges")
    print(summary)

    # Find all hysteresis files
    hyst_files = sorted(hysteresis_dir.glob("hysteresis_vmax*.csv"))
    n_ranges = len(hyst_files)

    if n_ranges == 0:
        print("No hysteresis files found")
        return

    # Create combined plot with all V_max ranges
    n_cols = min(4, n_ranges)
    n_rows = (n_ranges + n_cols - 1) // n_cols

    # Plot 1: Raw hysteresis
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))

    if n_ranges == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_ranges > 1 else [axes]

    for i, hyst_file in enumerate(hyst_files):
        # Extract V_max from filename
        v_max_str = hyst_file.stem.replace("hysteresis_vmax", "").replace("V", "")
        v_max = float(v_max_str.replace("p", "."))

        # Load data
        hyst_df = pl.read_csv(hyst_file)

        ax = axes[i]

        # Plot raw hysteresis
        ax.errorbar(
            hyst_df["V (V)"].to_numpy(),
            hyst_df["I_hysteresis_raw"].to_numpy() * 1e9,  # Convert to nA
            yerr=hyst_df["I_hysteresis_std"].to_numpy() * 1e9,
            fmt='o-',
            color='darkblue',
            label='Raw hysteresis',
            alpha=0.7,
            markersize=3,
            linewidth=1,
            errorevery=5,
            capsize=2
        )

        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=1)
        ax.set_xlabel('Voltage (V)', fontsize=10)
        ax.set_ylabel('Hysteresis Current (nA)', fontsize=10)
        ax.set_title(f'V_max = {v_max:.0f}V', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)

        # Get summary stats for this V_max
        sum_row = summary.filter(pl.col("v_max") == v_max)
        if len(sum_row) > 0:
            mean_val = sum_row["hyst_mean_raw"][0] * 1e9
            std_val = sum_row["hyst_std_raw"][0] * 1e9
            max_val = sum_row["hyst_max_raw"][0] * 1e9
            min_val = sum_row["hyst_min_raw"][0] * 1e9

            stats_text = f'Mean: {mean_val:.2f} nA\nStd: {std_val:.2f} nA'
            ax.text(
                0.05, 0.95,
                stats_text,
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.6)
            )

    # Hide unused subplots
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plot_file = output_dir / "hysteresis_raw_all.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {plot_file}")
    plt.close()

    # Plot 2: Polynomial fits comparison
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))

    if n_ranges == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_ranges > 1 else [axes]

    colors = ['red', 'blue', 'green', 'purple']
    linestyles = ['-', '--', '-.', ':']

    for i, hyst_file in enumerate(hyst_files):
        v_max_str = hyst_file.stem.replace("hysteresis_vmax", "").replace("V", "")
        v_max = float(v_max_str.replace("p", "."))

        hyst_df = pl.read_csv(hyst_file)
        ax = axes[i]

        # Plot polynomial hysteresis
        for idx, order in enumerate([1, 3, 5, 7]):
            col = f"I_hysteresis_poly{order}"
            if col in hyst_df.columns:
                ax.plot(
                    hyst_df["V (V)"].to_numpy(),
                    hyst_df[col].to_numpy() * 1e9,
                    color=colors[idx],
                    linestyle=linestyles[idx],
                    label=f'Poly n={order}',
                    linewidth=2,
                    alpha=0.8
                )

        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=1)
        ax.set_xlabel('Voltage (V)', fontsize=10)
        ax.set_ylabel('Hysteresis Current (nA)', fontsize=10)
        ax.set_title(f'V_max = {v_max:.0f}V (Polynomial Fits)', fontsize=11, fontweight='bold')
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plot_file = output_dir / "hysteresis_polynomial_all.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {plot_file}")
    plt.close()

    # Create individual detailed plots
    for hyst_file in hyst_files:
        v_max_str = hyst_file.stem.replace("hysteresis_vmax", "").replace("V", "")
        v_max = float(v_max_str.replace("p", "."))

        hyst_df = pl.read_csv(hyst_file)

        # Detailed plot with both raw and fits
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Top: Raw hysteresis with error bars
        ax1.errorbar(
            hyst_df["V (V)"].to_numpy(),
            hyst_df["I_hysteresis_raw"].to_numpy() * 1e9,
            yerr=hyst_df["I_hysteresis_std"].to_numpy() * 1e9,
            fmt='o-',
            color='darkblue',
            label='Raw hysteresis',
            alpha=0.6,
            markersize=4,
            linewidth=1.5,
            capsize=3,
            errorevery=3
        )
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=1.5)
        ax1.set_xlabel('Voltage (V)', fontsize=14)
        ax1.set_ylabel('Hysteresis Current (nA)', fontsize=14)
        ax1.set_title(f'Raw Hysteresis - V_max = {v_max:.0f}V', fontsize=15, fontweight='bold')
        ax1.legend(fontsize=12, loc='best')
        ax1.grid(True, alpha=0.3)

        # Stats box for raw
        sum_row = summary.filter(pl.col("v_max") == v_max)
        if len(sum_row) > 0:
            stats_text = f'Raw Hysteresis:\n'
            stats_text += f'  Mean: {sum_row["hyst_mean_raw"][0]*1e9:.3f} nA\n'
            stats_text += f'  Std:  {sum_row["hyst_std_raw"][0]*1e9:.3f} nA\n'
            stats_text += f'  Max:  {sum_row["hyst_max_raw"][0]*1e9:.3f} nA\n'
            stats_text += f'  Min:  {sum_row["hyst_min_raw"][0]*1e9:.3f} nA'

            ax1.text(
                0.05, 0.95,
                stats_text,
                transform=ax1.transAxes,
                fontsize=10,
                verticalalignment='top',
                family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
            )

        # Bottom: Polynomial fits
        for idx, order in enumerate([1, 3, 5, 7]):
            col = f"I_hysteresis_poly{order}"
            if col in hyst_df.columns:
                ax2.plot(
                    hyst_df["V (V)"].to_numpy(),
                    hyst_df[col].to_numpy() * 1e9,
                    color=colors[idx],
                    linestyle=linestyles[idx],
                    label=f'Polynomial n={order}',
                    linewidth=2.5,
                    alpha=0.9
                )

        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=1.5)
        ax2.set_xlabel('Voltage (V)', fontsize=14)
        ax2.set_ylabel('Hysteresis Current (nA)', fontsize=14)
        ax2.set_title(f'Polynomial Fit Hysteresis - V_max = {v_max:.0f}V', fontsize=15, fontweight='bold')
        ax2.legend(fontsize=12, loc='best')
        ax2.grid(True, alpha=0.3)

        # Stats box for polynomials
        if len(sum_row) > 0:
            poly_stats = 'Polynomial Hysteresis (Mean):\n'
            for order in [1, 3, 5, 7]:
                col = f"hyst_mean_poly{order}"
                if col in sum_row.columns and sum_row[col][0] is not None:
                    poly_stats += f'  n={order}: {sum_row[col][0]*1e9:.3f} nA\n'

            ax2.text(
                0.05, 0.95,
                poly_stats,
                transform=ax2.transAxes,
                fontsize=10,
                verticalalignment='top',
                family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
            )

        plt.tight_layout()
        plot_file = output_dir / f"hysteresis_vmax{v_max_str}V.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {plot_file}")
        plt.close()

    # Summary plot: Hysteresis vs V_max
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot mean hysteresis for different polynomial orders
    v_max_vals = summary["v_max"].to_numpy()

    # Raw hysteresis
    ax.errorbar(
        v_max_vals,
        (summary["hyst_mean_raw"].abs() * 1e9).to_numpy(),
        yerr=(summary["hyst_std_raw"] * 1e9).to_numpy(),
        fmt='o-',
        markersize=8,
        linewidth=2,
        capsize=4,
        label='Raw',
        color='darkblue'
    )

    # Polynomial hysteresis
    poly_colors = ['red', 'blue', 'green', 'purple']
    for idx, order in enumerate([1, 3, 5, 7]):
        col_mean = f"hyst_mean_poly{order}"
        col_std = f"hyst_std_poly{order}"
        if col_mean in summary.columns:
            ax.plot(
                v_max_vals,
                (summary[col_mean].abs() * 1e9).to_numpy(),
                marker='s',
                markersize=6,
                linewidth=2,
                label=f'Poly n={order}',
                color=poly_colors[idx],
                alpha=0.8
            )

    ax.set_xlabel('Maximum Voltage V_max (V)', fontsize=14)
    ax.set_ylabel('|Hysteresis Current| (nA)', fontsize=14)
    ax.set_title('Hysteresis vs Voltage Range', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_file = output_dir / "hysteresis_vs_vmax.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {plot_file}")
    plt.close()

    print(f"\n✓ All hysteresis plots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Plot hysteresis data")
    parser.add_argument("--hysteresis-dir", type=Path, required=True,
                       help="Directory containing hysteresis data")
    parser.add_argument("--output-dir", type=Path, required=True,
                       help="Output directory for plots")

    args = parser.parse_args()

    plot_hysteresis(args.hysteresis_dir, args.output_dir)


if __name__ == "__main__":
    main()

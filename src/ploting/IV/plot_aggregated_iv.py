#!/usr/bin/env python3
"""
Plot aggregated IV statistics with mean traces and error bars.
"""

import polars as pl
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def plot_iv_stats(stats_dir: Path, output_dir: Path):
    """
    Create plots from aggregated IV statistics.

    Args:
        stats_dir: Directory containing aggregated IV statistics
        output_dir: Directory to save plots
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load fit summary
    fit_summary = pl.read_csv(stats_dir / "fit_summary.csv")
    print(f"Found {len(fit_summary)} voltage ranges")
    print(fit_summary)

    # Find all forward/backward files
    forward_files = sorted(stats_dir.glob("forward_vmax*.csv"))
    backward_files = sorted(stats_dir.glob("backward_with_fit_vmax*.csv"))

    n_ranges = len(forward_files)
    if n_ranges == 0:
        print("No data files found")
        return

    # Create combined plot with all V_max ranges
    # Dynamic layout: 2 rows, ceiling(n/2) columns
    n_cols = min(4, n_ranges)  # Max 4 columns
    n_rows = (n_ranges + n_cols - 1) // n_cols  # Ceiling division

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))

    # Flatten axes array for easier iteration
    if n_ranges == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_ranges > 1 else [axes]

    for i, (fwd_file, bwd_file) in enumerate(zip(forward_files, backward_files)):
        # Extract V_max from filename
        v_max_str = fwd_file.stem.replace("forward_vmax", "").replace("V", "")
        v_max = float(v_max_str.replace("p", "."))

        # Load data
        fwd_df = pl.read_csv(fwd_file)
        bwd_df = pl.read_csv(bwd_file)

        # Get fit parameters
        fit_row = fit_summary.filter(pl.col("v_max") == v_max)
        slope = fit_row["slope"][0]
        intercept = fit_row["intercept"][0]
        r_squared = fit_row["r_squared"][0]
        n_runs = fit_row["n_runs"][0]
        resistance = fit_row["resistance_ohm"][0]

        # Plot forward trace
        ax = axes[i]
        # Detect voltage column name
        v_col = "V (V)" if "V (V)" in fwd_df.columns else "Vg (V)"

        ax.errorbar(
            fwd_df[v_col].to_numpy(),
            fwd_df["I_mean"].to_numpy() * 1e6,  # Convert to µA
            yerr=fwd_df["I_std"].to_numpy() * 1e6,
            fmt='o-',
            label='Forward (mean ± std)',
            alpha=0.6,
            markersize=3,
            linewidth=1
        )

        # Plot backward trace
        ax.errorbar(
            bwd_df[v_col].to_numpy(),
            bwd_df["I_mean"].to_numpy() * 1e6,  # Convert to µA
            yerr=bwd_df["I_std"].to_numpy() * 1e6,
            fmt='s-',
            label='Backward (mean ± std)',
            alpha=0.6,
            markersize=3,
            linewidth=1
        )

        # Plot fit line
        if "I_fit" in bwd_df.columns:
            ax.plot(
                bwd_df[v_col].to_numpy(),
                bwd_df["I_fit"].to_numpy() * 1e6,
                'r--',
                label=f'Linear fit (R²={r_squared:.4f})',
                linewidth=2
            )

        ax.set_xlabel('Voltage (V)', fontsize=12)
        ax.set_ylabel('Current I (µA)', fontsize=12)
        ax.set_title(f'V_max = {v_max:.0f}V (n={n_runs} runs)', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Add resistance annotation
        ax.text(
            0.05, 0.95,
            f'R = {resistance/1e6:.2f} MΩ\nSlope = {slope:.2e} A/V',
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

    # Hide unused subplots
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plot_file = output_dir / "iv_aggregated_all.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved combined plot: {plot_file}")
    plt.close()

    # Create individual plots for each V_max
    for fwd_file, bwd_file in zip(forward_files, backward_files):
        v_max_str = fwd_file.stem.replace("forward_vmax", "").replace("V", "")
        v_max = float(v_max_str.replace("p", "."))

        fwd_df = pl.read_csv(fwd_file)
        bwd_df = pl.read_csv(bwd_file)

        fit_row = fit_summary.filter(pl.col("v_max") == v_max)
        slope = fit_row["slope"][0]
        intercept = fit_row["intercept"][0]
        r_squared = fit_row["r_squared"][0]
        n_runs = fit_row["n_runs"][0]
        resistance = fit_row["resistance_ohm"][0]

        fig, ax = plt.subplots(figsize=(10, 6))

        # Detect voltage column
        v_col = "V (V)" if "V (V)" in fwd_df.columns else "Vg (V)"

        # Forward
        ax.errorbar(
            fwd_df[v_col].to_numpy(),
            fwd_df["I_mean"].to_numpy() * 1e6,
            yerr=fwd_df["I_std"].to_numpy() * 1e6,
            fmt='o-',
            label='Forward (mean ± std)',
            alpha=0.7,
            markersize=4,
            linewidth=1.5,
            capsize=2
        )

        # Backward
        ax.errorbar(
            bwd_df[v_col].to_numpy(),
            bwd_df["I_mean"].to_numpy() * 1e6,
            yerr=bwd_df["I_std"].to_numpy() * 1e6,
            fmt='s-',
            label='Backward (mean ± std)',
            alpha=0.7,
            markersize=4,
            linewidth=1.5,
            capsize=2
        )

        # Fit
        if "I_fit" in bwd_df.columns:
            ax.plot(
                bwd_df[v_col].to_numpy(),
                bwd_df["I_fit"].to_numpy() * 1e6,
                'r--',
                label=f'Linear fit',
                linewidth=2.5
            )

        ax.set_xlabel('Voltage (V)', fontsize=14)
        ax.set_ylabel('Current I (µA)', fontsize=14)
        ax.set_title(
            f'Aggregated IV Characteristics (V_max = {v_max:.0f}V, n={n_runs} runs)',
            fontsize=15,
            fontweight='bold'
        )
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)

        # Stats box
        stats_text = f'Fit statistics:\n'
        stats_text += f'  R² = {r_squared:.6f}\n'
        stats_text += f'  Slope = {slope:.3e} A/V\n'
        stats_text += f'  Intercept = {intercept:.3e} A\n'
        stats_text += f'  Resistance = {abs(resistance)/1e6:.2f} MΩ'

        ax.text(
            0.05, 0.95,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7)
        )

        plt.tight_layout()
        plot_file = output_dir / f"iv_aggregated_vmax{v_max_str}V.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {plot_file}")
        plt.close()

    # Create resistance vs V_max plot
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.errorbar(
        fit_summary["v_max"].to_numpy(),
        (fit_summary["resistance_ohm"].abs() / 1e6).to_numpy(),
        fmt='o-',
        markersize=10,
        linewidth=2,
        capsize=5
    )

    ax.set_xlabel('Maximum Gate Voltage V_max (V)', fontsize=14)
    ax.set_ylabel('Resistance |R| (MΩ)', fontsize=14)
    ax.set_title('Resistance vs Voltage Range', fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Annotate n_runs
    for i, row in enumerate(fit_summary.iter_rows(named=True)):
        ax.annotate(
            f'n={row["n_runs"]}',
            (row["v_max"], abs(row["resistance_ohm"])/1e6),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center',
            fontsize=9
        )

    plt.tight_layout()
    plot_file = output_dir / "resistance_vs_vmax.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {plot_file}")
    plt.close()

    print(f"\n✓ All plots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Plot aggregated IV statistics")
    parser.add_argument("--stats-dir", type=Path, required=True,
                       help="Directory containing aggregated IV stats")
    parser.add_argument("--output-dir", type=Path, required=True,
                       help="Output directory for plots")

    args = parser.parse_args()

    plot_iv_stats(args.stats_dir, args.output_dir)


if __name__ == "__main__":
    main()

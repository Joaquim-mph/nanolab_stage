#!/usr/bin/env python3
"""
Create a single comprehensive figure showing all V_max ranges with all polynomial orders.

Layout: 8 subplots (one per V_max range)
Each subplot shows:
- Raw hysteresis data (scatter points)
- Polynomial fits for orders 1, 3, 5, 7 (different colors/styles)
"""

import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse


def plot_all_ranges_all_polynomials(hysteresis_dir: Path, output_dir: Path):
    """
    Create single figure with 8 subplots comparing all polynomial orders.
    """

    # Load all hysteresis files
    hyst_files = sorted(hysteresis_dir.glob("hysteresis_vmax*.csv"))

    if len(hyst_files) == 0:
        print("No hysteresis files found")
        return

    n_ranges = len(hyst_files)

    # Create figure with 2 rows, 4 columns
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    axes = axes.flatten()

    # Define colors and styles for polynomial orders
    poly_styles = {
        1: {'color': 'red', 'linestyle': '-', 'linewidth': 2, 'alpha': 0.8},
        3: {'color': 'blue', 'linestyle': '-', 'linewidth': 2, 'alpha': 0.8},
        5: {'color': 'green', 'linestyle': '-', 'linewidth': 2, 'alpha': 0.8},
        7: {'color': 'purple', 'linestyle': '-', 'linewidth': 2, 'alpha': 0.8}
    }

    for idx, hyst_file in enumerate(hyst_files):
        # Extract V_max from filename
        v_max_str = hyst_file.stem.replace("hysteresis_vmax", "").replace("V", "")
        v_max = float(v_max_str.replace("p", "."))

        # Load data
        hyst_df = pl.read_csv(hyst_file)

        ax = axes[idx]

        # Get voltage and raw hysteresis data
        V_vals = hyst_df["V (V)"].to_numpy()
        I_raw = hyst_df["I_hysteresis_raw"].to_numpy() * 1e9  # Convert to nA
        I_std = hyst_df["I_hysteresis_std"].to_numpy() * 1e9

        # Plot raw data with error bars
        ax.errorbar(
            V_vals,
            I_raw,
            yerr=I_std,
            fmt='o',
            color='gray',
            markersize=3,
            alpha=0.4,
            label='Raw data',
            capsize=2,
            errorevery=5,
            linewidth=1,
            elinewidth=0.8
        )

        # Plot polynomial fits
        for order in [1, 3, 5, 7]:
            poly_col = f"I_hysteresis_poly{order}"

            if poly_col in hyst_df.columns:
                I_poly = hyst_df[poly_col].to_numpy() * 1e9

                ax.plot(
                    V_vals,
                    I_poly,
                    label=f'n={order}',
                    **poly_styles[order]
                )

        # Add zero reference line
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3, linewidth=1)

        # Calculate statistics for raw data
        mean_raw = np.mean(I_raw)
        std_raw = np.std(I_raw)
        max_abs_raw = np.max(np.abs(I_raw))

        # Labels and formatting
        ax.set_xlabel('Voltage (V)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Hysteresis (nA)', fontsize=11, fontweight='bold')
        ax.set_title(
            f'V_max = {v_max:.0f}V\n' +
            f'μ={mean_raw:.2f} nA, σ={std_raw:.2f} nA, max|H|={max_abs_raw:.2f} nA',
            fontsize=11,
            fontweight='bold'
        )

        # Add legend
        if idx == 0:
            ax.legend(fontsize=9, loc='best', framealpha=0.9, ncol=1)

        # Grid
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

        # Add subplot label
        ax.text(
            0.02, 0.98,
            f'({chr(97+idx)})',  # (a), (b), (c), etc.
            transform=ax.transAxes,
            fontsize=13,
            fontweight='bold',
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                     edgecolor='black', alpha=0.8)
        )

    # Hide unused subplots if less than 8
    for idx in range(n_ranges, 8):
        axes[idx].set_visible(False)

    # Main title
    fig.suptitle(
        'Polynomial Order Comparison - All Voltage Ranges\n' +
        'Raw Hysteresis vs Polynomial Fits (n=1, 3, 5, 7)',
        fontsize=16,
        fontweight='bold',
        y=0.995
    )

    plt.tight_layout(rect=[0, 0, 1, 0.985])

    output_file = output_dir / "all_ranges_all_polynomials.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_file}")
    print(f"Figure shows {n_ranges} voltage ranges with polynomial orders 1, 3, 5, 7")

    plt.close()


def plot_all_ranges_all_polynomials_compact(hysteresis_dir: Path, output_dir: Path):
    """
    Create compact version with smaller subplots for publication.
    """

    # Load all hysteresis files
    hyst_files = sorted(hysteresis_dir.glob("hysteresis_vmax*.csv"))

    if len(hyst_files) == 0:
        print("No hysteresis files found")
        return

    n_ranges = len(hyst_files)

    # Create figure with 2 rows, 4 columns (more compact)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    # Define colors and styles for polynomial orders
    poly_styles = {
        1: {'color': 'red', 'linestyle': '-', 'linewidth': 1.5, 'alpha': 0.9},
        3: {'color': 'blue', 'linestyle': '-', 'linewidth': 2.5, 'alpha': 0.95},
        5: {'color': 'green', 'linestyle': '-', 'linewidth': 1.5, 'alpha': 0.9},
        7: {'color': 'purple', 'linestyle': '-', 'linewidth': 1.5, 'alpha': 0.9}
    }

    for idx, hyst_file in enumerate(hyst_files):
        # Extract V_max from filename
        v_max_str = hyst_file.stem.replace("hysteresis_vmax", "").replace("V", "")
        v_max = float(v_max_str.replace("p", "."))

        # Load data
        hyst_df = pl.read_csv(hyst_file)

        ax = axes[idx]

        # Get voltage and raw hysteresis data
        V_vals = hyst_df["V (V)"].to_numpy()
        I_raw = hyst_df["I_hysteresis_raw"].to_numpy() * 1e9  # Convert to nA

        # Plot raw data (no error bars for compact view)
        ax.scatter(
            V_vals,
            I_raw,
            s=10,
            color='lightgray',
            alpha=0.6,
            label='Raw',
            zorder=1
        )

        # Plot polynomial fits
        for order in [1, 3, 5, 7]:
            poly_col = f"I_hysteresis_poly{order}"

            if poly_col in hyst_df.columns:
                I_poly = hyst_df[poly_col].to_numpy() * 1e9

                ax.plot(
                    V_vals,
                    I_poly,
                    label=f'n={order}',
                    zorder=2 if order != 3 else 3,  # Highlight n=3
                    **poly_styles[order]
                )

        # Add zero reference line
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.4, linewidth=1)

        # Labels and formatting
        ax.set_xlabel('V (V)', fontsize=10, fontweight='bold')
        ax.set_ylabel('H (nA)', fontsize=10, fontweight='bold')
        ax.set_title(f'V_max = {v_max:.0f}V', fontsize=11, fontweight='bold')

        # Add legend only to first subplot
        if idx == 0:
            ax.legend(fontsize=8, loc='best', framealpha=0.95, ncol=1)

        # Grid
        ax.grid(True, alpha=0.25, linestyle=':', linewidth=0.5)

    # Hide unused subplots if less than 8
    for idx in range(n_ranges, 8):
        axes[idx].set_visible(False)

    # Main title
    fig.suptitle(
        'Polynomial Fit Comparison Across Voltage Ranges',
        fontsize=15,
        fontweight='bold',
        y=0.995
    )

    plt.tight_layout(rect=[0, 0, 1, 0.985])

    output_file = output_dir / "all_ranges_all_polynomials_compact.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    print(f"Compact figure shows {n_ranges} voltage ranges")

    plt.close()


def plot_residuals_comparison(hysteresis_dir: Path, output_dir: Path):
    """
    Create figure showing residuals (raw - polynomial fit) for all orders.
    """

    # Load all hysteresis files
    hyst_files = sorted(hysteresis_dir.glob("hysteresis_vmax*.csv"))

    if len(hyst_files) == 0:
        print("No hysteresis files found")
        return

    n_ranges = len(hyst_files)

    # Create figure
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    axes = axes.flatten()

    # Define colors for polynomial orders
    poly_colors = {1: 'red', 3: 'blue', 5: 'green', 7: 'purple'}

    for idx, hyst_file in enumerate(hyst_files):
        # Extract V_max from filename
        v_max_str = hyst_file.stem.replace("hysteresis_vmax", "").replace("V", "")
        v_max = float(v_max_str.replace("p", "."))

        # Load data
        hyst_df = pl.read_csv(hyst_file)

        ax = axes[idx]

        # Get voltage and raw hysteresis data
        V_vals = hyst_df["V (V)"].to_numpy()
        I_raw = hyst_df["I_hysteresis_raw"].to_numpy() * 1e9

        # Calculate and plot residuals for each polynomial order
        rms_values = {}

        for order in [1, 3, 5, 7]:
            poly_col = f"I_hysteresis_poly{order}"

            if poly_col in hyst_df.columns:
                I_poly = hyst_df[poly_col].to_numpy() * 1e9
                residuals = I_raw - I_poly

                # Calculate RMS
                rms = np.sqrt(np.mean(residuals**2))
                rms_values[order] = rms

                ax.plot(
                    V_vals,
                    residuals,
                    'o-',
                    color=poly_colors[order],
                    markersize=2,
                    linewidth=1.5,
                    alpha=0.7,
                    label=f'n={order} (RMS={rms:.2f})'
                )

        # Add zero reference line
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.4, linewidth=1.5)

        # Labels and formatting
        ax.set_xlabel('Voltage (V)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Residual (nA)', fontsize=11, fontweight='bold')
        ax.set_title(f'V_max = {v_max:.0f}V - Fit Residuals',
                    fontsize=11, fontweight='bold')

        # Add legend
        ax.legend(fontsize=8, loc='best', framealpha=0.9, ncol=2)

        # Grid
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

    # Hide unused subplots if less than 8
    for idx in range(n_ranges, 8):
        axes[idx].set_visible(False)

    # Main title
    fig.suptitle(
        'Polynomial Fit Residuals - Raw Hysteresis minus Polynomial Fit\n' +
        'Lower RMS indicates better fit quality',
        fontsize=16,
        fontweight='bold',
        y=0.995
    )

    plt.tight_layout(rect=[0, 0, 1, 0.985])

    output_file = output_dir / "residuals_all_ranges_all_polynomials.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    print(f"Residuals figure shows fit quality for all polynomial orders")

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Compare all polynomial orders across all V_max ranges in single figure"
    )
    parser.add_argument("--hysteresis-dir", type=Path, required=True,
                       help="Directory containing hysteresis data")
    parser.add_argument("--output-dir", type=Path, required=True,
                       help="Output directory for plots")
    parser.add_argument("--compact", action="store_true",
                       help="Create compact version without error bars")
    parser.add_argument("--residuals", action="store_true",
                       help="Also create residuals comparison plot")

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nPolynomial Order Comparison Tool")
    print(f"=" * 70)
    print(f"Hysteresis directory: {args.hysteresis_dir}")
    print(f"Output directory:     {args.output_dir}")
    print(f"=" * 70)

    # Create main figure
    print(f"\nCreating comprehensive comparison figure...")
    plot_all_ranges_all_polynomials(args.hysteresis_dir, args.output_dir)

    # Create compact version if requested
    if args.compact:
        print(f"\nCreating compact version...")
        plot_all_ranges_all_polynomials_compact(args.hysteresis_dir, args.output_dir)

    # Create residuals plot if requested
    if args.residuals:
        print(f"\nCreating residuals comparison...")
        plot_residuals_comparison(args.hysteresis_dir, args.output_dir)

    print(f"\n{'='*70}")
    print(f"✓ Complete! All plots saved to: {args.output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

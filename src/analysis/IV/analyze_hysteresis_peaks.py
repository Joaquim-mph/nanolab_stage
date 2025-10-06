#!/usr/bin/env python3
"""
Analyze hysteresis peaks - find voltage locations of maximum hysteresis.

For each V_max range and each polynomial order, identifies:
- Voltage where maximum positive hysteresis occurs
- Voltage where maximum negative hysteresis occurs
- Voltage where maximum absolute hysteresis occurs

Outputs:
- CSV files with peak locations
- Visualization with vertical lines marking peak positions
"""

import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse


def find_hysteresis_peaks(hysteresis_dir: Path, output_dir: Path):
    """
    Find voltage locations of maximum hysteresis for all ranges and polynomial orders.
    """

    # Load all hysteresis files
    hyst_files = sorted(hysteresis_dir.glob("hysteresis_vmax*.csv"))

    if len(hyst_files) == 0:
        print("No hysteresis files found")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect peak data
    all_peaks = []

    for hyst_file in hyst_files:
        # Extract V_max from filename
        v_max_str = hyst_file.stem.replace("hysteresis_vmax", "").replace("V", "")
        v_max = float(v_max_str.replace("p", "."))

        # Load data
        hyst_df = pl.read_csv(hyst_file)

        V_vals = hyst_df["V (V)"].to_numpy()

        # Analyze raw hysteresis
        I_raw = hyst_df["I_hysteresis_raw"].to_numpy()

        idx_max_pos = np.argmax(I_raw)
        idx_max_neg = np.argmin(I_raw)
        idx_max_abs = np.argmax(np.abs(I_raw))

        peak_data = {
            'v_max': v_max,
            'method': 'raw',
            'V_max_positive': V_vals[idx_max_pos],
            'I_max_positive': I_raw[idx_max_pos],
            'V_max_negative': V_vals[idx_max_neg],
            'I_max_negative': I_raw[idx_max_neg],
            'V_max_absolute': V_vals[idx_max_abs],
            'I_max_absolute': I_raw[idx_max_abs],
        }
        all_peaks.append(peak_data)

        # Analyze polynomial fits
        for order in [1, 3, 5, 7]:
            poly_col = f"I_hysteresis_poly{order}"

            if poly_col in hyst_df.columns:
                I_poly = hyst_df[poly_col].to_numpy()

                idx_max_pos = np.argmax(I_poly)
                idx_max_neg = np.argmin(I_poly)
                idx_max_abs = np.argmax(np.abs(I_poly))

                peak_data = {
                    'v_max': v_max,
                    'method': f'poly{order}',
                    'V_max_positive': V_vals[idx_max_pos],
                    'I_max_positive': I_poly[idx_max_pos],
                    'V_max_negative': V_vals[idx_max_neg],
                    'I_max_negative': I_poly[idx_max_neg],
                    'V_max_absolute': V_vals[idx_max_abs],
                    'I_max_absolute': I_poly[idx_max_abs],
                }
                all_peaks.append(peak_data)

    # Create dataframe and save
    peaks_df = pl.DataFrame(all_peaks)

    # Sort by v_max and method
    peaks_df = peaks_df.sort(['v_max', 'method'])

    # Save to CSV
    output_file = output_dir / "hysteresis_peaks.csv"
    peaks_df.write_csv(output_file)
    print(f"\nSaved peak analysis: {output_file}")

    # Print summary
    print(f"\nHysteresis Peak Locations:")
    print(peaks_df)

    return peaks_df


def plot_hysteresis_with_peaks(
    hysteresis_dir: Path,
    peaks_df: pl.DataFrame,
    output_dir: Path
):
    """
    Create plots showing hysteresis with vertical lines at peak locations.
    """

    # Load all hysteresis files
    hyst_files = sorted(hysteresis_dir.glob("hysteresis_vmax*.csv"))

    if len(hyst_files) == 0:
        print("No hysteresis files found")
        return

    n_ranges = len(hyst_files)

    # Create figure
    fig, axes = plt.subplots(2, 4, figsize=(28, 14))
    axes = axes.flatten()

    # Define colors for polynomial orders
    poly_styles = {
        1: {'color': 'red', 'linestyle': '-', 'linewidth': 2, 'alpha': 0.8},
        3: {'color': 'blue', 'linestyle': '-', 'linewidth': 2, 'alpha': 0.8},
        5: {'color': 'green', 'linestyle': '-', 'linewidth': 2, 'alpha': 0.8},
        7: {'color': 'purple', 'linestyle': '-', 'linewidth': 2, 'alpha': 0.8}
    }

    # Peak marker styles
    peak_marker_styles = {
        1: {'marker': '^', 'markersize': 10, 'markeredgewidth': 2},
        3: {'marker': 's', 'markersize': 10, 'markeredgewidth': 2},
        5: {'marker': 'D', 'markersize': 10, 'markeredgewidth': 2},
        7: {'marker': 'v', 'markersize': 10, 'markeredgewidth': 2}
    }

    for idx, hyst_file in enumerate(hyst_files):
        # Extract V_max from filename
        v_max_str = hyst_file.stem.replace("hysteresis_vmax", "").replace("V", "")
        v_max = float(v_max_str.replace("p", "."))

        # Load data
        hyst_df = pl.read_csv(hyst_file)

        ax = axes[idx]

        V_vals = hyst_df["V (V)"].to_numpy()
        I_raw = hyst_df["I_hysteresis_raw"].to_numpy() * 1e9

        # Plot raw data
        ax.errorbar(
            V_vals,
            I_raw,
            yerr=hyst_df["I_hysteresis_std"].to_numpy() * 1e9,
            fmt='o',
            color='gray',
            markersize=3,
            alpha=0.4,
            label='Raw',
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

        # Get peaks for this V_max
        peaks_for_vmax = peaks_df.filter(pl.col('v_max') == v_max)

        # Plot peak markers for each polynomial order
        for order in [1, 3, 5, 7]:
            method = f'poly{order}'
            peak_row = peaks_for_vmax.filter(pl.col('method') == method)

            if len(peak_row) > 0:
                # Mark maximum absolute hysteresis
                V_peak = peak_row['V_max_absolute'][0]
                I_peak = peak_row['I_max_absolute'][0] * 1e9

                ax.plot(
                    V_peak,
                    I_peak,
                    color=poly_styles[order]['color'],
                    fillstyle='none',
                    markeredgecolor=poly_styles[order]['color'],
                    **peak_marker_styles[order],
                    label=f'Peak n={order} @ {V_peak:.2f}V',
                    zorder=10
                )

                # Add vertical line at peak location
                ax.axvline(
                    x=V_peak,
                    color=poly_styles[order]['color'],
                    linestyle=':',
                    alpha=0.4,
                    linewidth=1.5
                )

        # Add zero reference line
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3, linewidth=1)

        # Labels and formatting
        ax.set_xlabel('Voltage (V)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Hysteresis (nA)', fontsize=12, fontweight='bold')
        ax.set_title(f'V_max = {v_max:.0f}V - Peak Locations',
                    fontsize=12, fontweight='bold')

        # Legend
        if idx == 0:
            ax.legend(fontsize=8, loc='best', framealpha=0.9, ncol=2)

        # Grid
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

    # Hide unused subplots
    for idx in range(n_ranges, 8):
        axes[idx].set_visible(False)

    # Main title
    fig.suptitle(
        'Hysteresis with Peak Markers\n' +
        'Markers show voltage location of maximum absolute hysteresis for each polynomial order',
        fontsize=16,
        fontweight='bold',
        y=0.995
    )

    plt.tight_layout(rect=[0, 0, 1, 0.985])

    output_file = output_dir / "hysteresis_with_peaks.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")

    plt.close()


def plot_peak_voltage_trends(peaks_df: pl.DataFrame, output_dir: Path):
    """
    Plot how peak voltage locations change with V_max for each polynomial order.
    """

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Define colors
    poly_colors = {
        'raw': 'gray',
        'poly1': 'red',
        'poly3': 'blue',
        'poly5': 'green',
        'poly7': 'purple'
    }

    # Plot 1: Maximum absolute hysteresis voltage location
    ax1 = axes[0, 0]
    for method in ['raw', 'poly1', 'poly3', 'poly5', 'poly7']:
        method_data = peaks_df.filter(pl.col('method') == method)
        if len(method_data) > 0:
            ax1.plot(
                method_data['v_max'].to_numpy(),
                method_data['V_max_absolute'].to_numpy(),
                'o-',
                color=poly_colors[method],
                markersize=8,
                linewidth=2,
                label=method,
                alpha=0.8
            )

    ax1.set_xlabel('V_max Range (V)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Voltage at Max |Hysteresis| (V)', fontsize=12, fontweight='bold')
    ax1.set_title('Peak Location vs Voltage Range', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Maximum absolute hysteresis magnitude
    ax2 = axes[0, 1]
    for method in ['raw', 'poly1', 'poly3', 'poly5', 'poly7']:
        method_data = peaks_df.filter(pl.col('method') == method)
        if len(method_data) > 0:
            ax2.plot(
                method_data['v_max'].to_numpy(),
                np.abs(method_data['I_max_absolute'].to_numpy()) * 1e9,
                'o-',
                color=poly_colors[method],
                markersize=8,
                linewidth=2,
                label=method,
                alpha=0.8
            )

    ax2.set_xlabel('V_max Range (V)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Max |Hysteresis| (nA)', fontsize=12, fontweight='bold')
    ax2.set_title('Peak Magnitude vs Voltage Range', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Positive peak locations
    ax3 = axes[1, 0]
    for method in ['raw', 'poly1', 'poly3', 'poly5', 'poly7']:
        method_data = peaks_df.filter(pl.col('method') == method)
        if len(method_data) > 0:
            ax3.plot(
                method_data['v_max'].to_numpy(),
                method_data['V_max_positive'].to_numpy(),
                'o-',
                color=poly_colors[method],
                markersize=8,
                linewidth=2,
                label=method,
                alpha=0.8
            )

    ax3.set_xlabel('V_max Range (V)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Voltage at Max Positive H (V)', fontsize=12, fontweight='bold')
    ax3.set_title('Positive Peak Location', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10, loc='best')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Negative peak locations
    ax4 = axes[1, 1]
    for method in ['raw', 'poly1', 'poly3', 'poly5', 'poly7']:
        method_data = peaks_df.filter(pl.col('method') == method)
        if len(method_data) > 0:
            ax4.plot(
                method_data['v_max'].to_numpy(),
                method_data['V_max_negative'].to_numpy(),
                'o-',
                color=poly_colors[method],
                markersize=8,
                linewidth=2,
                label=method,
                alpha=0.8
            )

    ax4.set_xlabel('V_max Range (V)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Voltage at Max Negative H (V)', fontsize=12, fontweight='bold')
    ax4.set_title('Negative Peak Location', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=10, loc='best')
    ax4.grid(True, alpha=0.3)

    plt.suptitle('Hysteresis Peak Analysis - Trends Across Voltage Ranges',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.985])

    output_file = output_dir / "peak_voltage_trends.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")

    plt.close()


def create_peak_summary_table(peaks_df: pl.DataFrame, output_dir: Path):
    """
    Create a formatted summary table showing peak voltages for each method.
    """

    # Pivot table: V_max as rows, methods as columns
    summary_data = []

    v_max_values = sorted(peaks_df['v_max'].unique().to_list())

    for v_max in v_max_values:
        row_data = {'V_max': v_max}

        for method in ['raw', 'poly1', 'poly3', 'poly5', 'poly7']:
            peak_row = peaks_df.filter(
                (pl.col('v_max') == v_max) & (pl.col('method') == method)
            )

            if len(peak_row) > 0:
                v_peak = peak_row['V_max_absolute'][0]
                i_peak = peak_row['I_max_absolute'][0] * 1e9
                row_data[f'{method}_V'] = v_peak
                row_data[f'{method}_I'] = i_peak

        summary_data.append(row_data)

    summary_df = pl.DataFrame(summary_data)

    output_file = output_dir / "peak_summary_table.csv"
    summary_df.write_csv(output_file)
    print(f"Saved: {output_file}")

    print("\nPeak Summary Table (Voltage at Max |Hysteresis|):")
    print(summary_df)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze hysteresis peak locations for all polynomial orders"
    )
    parser.add_argument("--hysteresis-dir", type=Path, required=True,
                       help="Directory containing hysteresis data")
    parser.add_argument("--output-dir", type=Path, required=True,
                       help="Output directory for analysis results")

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nHysteresis Peak Analysis")
    print(f"=" * 70)
    print(f"Hysteresis directory: {args.hysteresis_dir}")
    print(f"Output directory:     {args.output_dir}")
    print(f"=" * 70)

    # Find peaks
    peaks_df = find_hysteresis_peaks(args.hysteresis_dir, args.output_dir)

    # Create visualizations
    print(f"\nCreating visualizations...")
    plot_hysteresis_with_peaks(args.hysteresis_dir, peaks_df, args.output_dir)
    plot_peak_voltage_trends(peaks_df, args.output_dir)
    create_peak_summary_table(peaks_df, args.output_dir)

    print(f"\n{'='*70}")
    print(f"✓ Peak analysis complete!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

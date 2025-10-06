#!/usr/bin/env python3
"""
Compare IV analysis results across multiple days.

Creates comprehensive comparison visualizations and statistics tables.
"""

import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse


def load_day_data(date: str, base_dir: Path = Path("data/04_analysis")):
    """Load all analysis data for a given date."""
    stats_dir = base_dir / "iv_stats" / date
    hyst_dir = base_dir / "hysteresis" / date
    peaks_dir = base_dir / "hysteresis_peaks" / date

    # Load fit summary
    fit_summary = pl.read_csv(stats_dir / "fit_summary.csv")

    # Load hysteresis summary
    hyst_summary = pl.read_csv(hyst_dir / "hysteresis_summary.csv")

    # Load peak summary
    peak_summary = pl.read_csv(peaks_dir / "peak_summary_table.csv")

    # Load detailed hysteresis for each V_max
    hyst_files = sorted(hyst_dir.glob("hysteresis_vmax*.csv"))
    hyst_detailed = {}
    for f in hyst_files:
        v_max = f.stem.replace("hysteresis_vmax", "").replace("p", ".")
        hyst_detailed[float(v_max.rstrip("V"))] = pl.read_csv(f)

    return {
        "fit_summary": fit_summary,
        "hyst_summary": hyst_summary,
        "peak_summary": peak_summary,
        "hyst_detailed": hyst_detailed,
        "date": date
    }


def create_comparison_plots(days_data: list, output_dir: Path):
    """Create comprehensive comparison plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    dates = [d["date"] for d in days_data]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green

    # 1. Resistance comparison across voltage ranges
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, day in enumerate(days_data):
        fit_sum = day["fit_summary"]
        v_max = fit_sum["v_max"].to_numpy()
        resistance = fit_sum["resistance_ohm"].to_numpy() / 1e6  # Convert to MΩ

        axes[0].plot(v_max, resistance, 'o-', label=day["date"],
                    color=colors[idx], linewidth=2, markersize=8)
        axes[0].set_xlabel("V_max (V)", fontsize=12)
        axes[0].set_ylabel("Resistance (MΩ)", fontsize=12)
        axes[0].set_title("Resistance vs V_max", fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

    # R² comparison
    for idx, day in enumerate(days_data):
        fit_sum = day["fit_summary"]
        v_max = fit_sum["v_max"].to_numpy()
        r_squared = fit_sum["r_squared"].to_numpy()

        axes[1].plot(v_max, r_squared, 'o-', label=day["date"],
                    color=colors[idx], linewidth=2, markersize=8)
        axes[1].set_xlabel("V_max (V)", fontsize=12)
        axes[1].set_ylabel("R²", fontsize=12)
        axes[1].set_title("Linear Fit Quality vs V_max", fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0.9, 1.0])

    plt.tight_layout()
    plt.savefig(output_dir / "resistance_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_dir / 'resistance_comparison.png'}")

    # 2. Hysteresis comparison for each V_max (6 subplots)
    # Find common V_max values across all days
    common_v_max = set(days_data[0]["hyst_detailed"].keys())
    for day in days_data[1:]:
        common_v_max = common_v_max.intersection(day["hyst_detailed"].keys())
    v_max_values = sorted(common_v_max)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for v_idx, v_max in enumerate(v_max_values):
        ax = axes[v_idx]

        for day_idx, day in enumerate(days_data):
            if v_max in day["hyst_detailed"]:
                hyst_df = day["hyst_detailed"][v_max]
                V = hyst_df["V (V)"].to_numpy()
                I_hyst_poly3 = hyst_df["I_hysteresis_poly3"].to_numpy() * 1e9  # Convert to nA

                ax.plot(V, I_hyst_poly3, label=day["date"],
                       color=colors[day_idx], linewidth=2, alpha=0.8)

        ax.set_xlabel("V (V)", fontsize=11)
        ax.set_ylabel("Hysteresis Current (nA)", fontsize=11)
        ax.set_title(f"V_max = {v_max}V", fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)

    plt.suptitle("Hysteresis Comparison Across Days (Polynomial n=3)",
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / "hysteresis_comparison_all_vmax.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_dir / 'hysteresis_comparison_all_vmax.png'}")

    # 3. Hysteresis magnitude comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    poly_orders = [1, 3, 5, 7]

    for poly_idx, poly_order in enumerate(poly_orders):
        ax = axes[poly_idx]

        for day_idx, day in enumerate(days_data):
            hyst_sum = day["hyst_summary"]
            v_max = hyst_sum["v_max"].to_numpy()

            # Get mean and std for this polynomial order
            mean_col = f"hyst_mean_poly{poly_order}"
            std_col = f"hyst_std_poly{poly_order}"

            mean_hyst = hyst_sum[mean_col].to_numpy() * 1e9  # nA
            std_hyst = hyst_sum[std_col].to_numpy() * 1e9  # nA

            ax.errorbar(v_max, mean_hyst, yerr=std_hyst,
                       fmt='o-', label=day["date"], color=colors[day_idx],
                       linewidth=2, markersize=8, capsize=5, alpha=0.8)

        ax.set_xlabel("V_max (V)", fontsize=11)
        ax.set_ylabel("Mean Hysteresis (nA)", fontsize=11)
        ax.set_title(f"Polynomial Order {poly_order}", fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)

    plt.suptitle("Hysteresis Magnitude Comparison",
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / "hysteresis_magnitude_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_dir / 'hysteresis_magnitude_comparison.png'}")

    # 4. Peak voltage comparison
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    for day_idx, day in enumerate(days_data):
        peak_sum = day["peak_summary"]
        v_max = peak_sum["V_max"].to_numpy()

        # Use poly3 peaks
        peak_V = peak_sum["poly3_V"].to_numpy()
        peak_I = peak_sum["poly3_I"].to_numpy()

        ax.scatter(v_max, peak_V, s=100, label=f"{day['date']} (Peak V)",
                  color=colors[day_idx], marker='o', alpha=0.7)

    ax.set_xlabel("V_max (V)", fontsize=12)
    ax.set_ylabel("Voltage at Peak Hysteresis (V)", fontsize=12)
    ax.set_title("Peak Hysteresis Location Comparison (Poly n=3)",
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / "peak_voltage_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_dir / 'peak_voltage_comparison.png'}")


def create_comparison_tables(days_data: list, output_dir: Path):
    """Create comparison tables."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Resistance summary table
    resistance_rows = []
    for day in days_data:
        fit_sum = day["fit_summary"]
        for row in fit_sum.iter_rows(named=True):
            resistance_rows.append({
                "Date": day["date"],
                "V_max": row["v_max"],
                "Resistance_MΩ": row["resistance_ohm"] / 1e6,
                "R²": row["r_squared"],
                "n_runs": row["n_runs"]
            })

    resistance_df = pl.DataFrame(resistance_rows)
    resistance_df.write_csv(output_dir / "resistance_comparison.csv")
    print(f"\nSaved: {output_dir / 'resistance_comparison.csv'}")

    # Print summary
    print("\n" + "="*70)
    print("RESISTANCE COMPARISON SUMMARY")
    print("="*70)
    print(resistance_df)

    # 2. Hysteresis summary table (poly3)
    hyst_rows = []
    for day in days_data:
        hyst_sum = day["hyst_summary"]
        for row in hyst_sum.iter_rows(named=True):
            hyst_rows.append({
                "Date": day["date"],
                "V_max": row["v_max"],
                "Mean_Hyst_poly3_nA": row["hyst_mean_poly3"] * 1e9,
                "Std_Hyst_poly3_nA": row["hyst_std_poly3"] * 1e9,
                "Max_Hyst_poly3_nA": row["hyst_max_poly3"] * 1e9,
                "Min_Hyst_poly3_nA": row["hyst_min_poly3"] * 1e9,
                "n_points": row["n_points"]
            })

    hyst_df = pl.DataFrame(hyst_rows)
    hyst_df.write_csv(output_dir / "hysteresis_comparison.csv")
    print(f"\nSaved: {output_dir / 'hysteresis_comparison.csv'}")

    print("\n" + "="*70)
    print("HYSTERESIS COMPARISON SUMMARY (Polynomial n=3)")
    print("="*70)
    print(hyst_df)

    # 3. Peak locations comparison
    peak_rows = []
    for day in days_data:
        peak_sum = day["peak_summary"]
        for row in peak_sum.iter_rows(named=True):
            peak_rows.append({
                "Date": day["date"],
                "V_max": row["V_max"],
                "Peak_V_poly3": row["poly3_V"],
                "Peak_I_poly3_nA": row["poly3_I"],
                "Peak_V_poly7": row["poly7_V"],
                "Peak_I_poly7_nA": row["poly7_I"],
            })

    peak_df = pl.DataFrame(peak_rows)
    peak_df.write_csv(output_dir / "peak_comparison.csv")
    print(f"\nSaved: {output_dir / 'peak_comparison.csv'}")

    print("\n" + "="*70)
    print("PEAK LOCATION COMPARISON")
    print("="*70)
    print(peak_df)


def main():
    parser = argparse.ArgumentParser(
        description="Compare IV analysis results across multiple days",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--dates", type=str, nargs='+', required=True,
                       help="Dates to compare (YYYY-MM-DD format)")
    parser.add_argument("--output-dir", type=Path,
                       default=Path("plots/comparison"),
                       help="Output directory for comparison plots")
    parser.add_argument("--base-dir", type=Path,
                       default=Path("data/04_analysis"),
                       help="Base directory containing analysis results")

    args = parser.parse_args()

    print("="*70)
    print("IV ANALYSIS - DAY COMPARISON")
    print("="*70)
    print(f"Dates: {', '.join(args.dates)}")
    print(f"Output: {args.output_dir}")
    print("="*70)

    # Load data for all dates
    days_data = []
    for date in args.dates:
        print(f"\nLoading data for {date}...")
        data = load_day_data(date, args.base_dir)
        days_data.append(data)
        print(f"  ✓ Loaded {len(data['fit_summary'])} voltage ranges")

    # Create comparison plots
    print("\n" + "="*70)
    print("CREATING COMPARISON PLOTS")
    print("="*70)
    create_comparison_plots(days_data, args.output_dir)

    # Create comparison tables
    print("\n" + "="*70)
    print("CREATING COMPARISON TABLES")
    print("="*70)
    create_comparison_tables(days_data, args.output_dir)

    print("\n" + "="*70)
    print("✓ COMPARISON COMPLETE!")
    print("="*70)
    print(f"\nAll outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

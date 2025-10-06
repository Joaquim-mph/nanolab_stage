#!/usr/bin/env python3
"""
Complete IV preprocessing pipeline runner.

Runs the full IV analysis workflow:
1. Aggregate statistics from repeated IV experiments
2. Generate plots with polynomial fits

Usage:
    python run_iv_pipeline.py --date 2025-09-11 --procedure IV
    python run_iv_pipeline.py --date 2025-09-11 --procedure IV --chip-number 71
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        print(f"\n❌ Error: {description} failed with code {result.returncode}")
        sys.exit(result.returncode)

    print(f"\n✓ {description} completed successfully")


def main():
    parser = argparse.ArgumentParser(
        description="Run complete IV preprocessing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all IV data for a specific date
  python run_iv_pipeline.py --date 2025-09-11 --procedure IV

  # Process specific chip and voltage range
  python run_iv_pipeline.py --date 2025-09-11 --procedure IV --chip-number 71 --v-max 8

  # Custom output directories
  python run_iv_pipeline.py --date 2025-09-11 --procedure IV --output-suffix custom_run
        """
    )

    # Input parameters
    parser.add_argument("--date", required=True,
                       help="Date in YYYY-MM-DD format")
    parser.add_argument("--procedure", type=str, default="IV",
                       help="Procedure name (IV, IVg, etc.) [default: IV]")
    parser.add_argument("--chip-number",
                       help="Filter by chip number (optional)")
    parser.add_argument("--v-max", type=float,
                       help="Filter by specific V_max value (optional)")

    # Directory parameters
    parser.add_argument("--stage-root", type=Path,
                       default=Path("data/02_stage/raw_measurements"),
                       help="Root of staged data")
    parser.add_argument("--output-suffix", type=str, default=None,
                       help="Suffix for output directories (default: uses date)")

    # Processing options
    parser.add_argument("--skip-stats", action="store_true",
                       help="Skip statistics aggregation (only plot existing data)")
    parser.add_argument("--skip-plots", action="store_true",
                       help="Skip plotting (only run statistics)")
    parser.add_argument("--compute-hysteresis", action="store_true",
                       help="Compute hysteresis current (forward - return)")

    args = parser.parse_args()

    # Determine output directory suffix
    output_suffix = args.output_suffix or args.date.replace("-", "_")

    # Set up paths
    stats_dir = Path(f"data/04_analysis/iv_stats/{output_suffix}")
    plots_dir = Path(f"plots/iv_stats/{output_suffix}")

    # Find script paths (relative to this file)
    script_dir = Path(__file__).parent.parent.parent  # Go up to src/
    aggregate_script = script_dir / "analysis" / "aggregate_iv_stats.py"
    plot_script = script_dir / "analysis" / "plot_aggregated_iv_odd.py"

    # Build command for statistics aggregation
    if not args.skip_stats:
        stats_cmd = [
            "python3",
            str(aggregate_script),
            "--date", args.date,
            "--procedure", args.procedure,
            "--stage-root", str(args.stage_root),
            "--output-dir", str(stats_dir)
        ]

        if args.chip_number:
            stats_cmd.extend(["--chip-number", args.chip_number])
        if args.v_max:
            stats_cmd.extend(["--v-max", str(args.v_max)])

        run_command(stats_cmd, "Statistics Aggregation")

    # Build command for plotting
    if not args.skip_plots:
        plot_cmd = [
            "python3",
            str(plot_script),
            "--stats-dir", str(stats_dir),
            "--output-dir", str(plots_dir)
        ]

        run_command(plot_cmd, "Plot Generation")

    # Build command for hysteresis computation
    if args.compute_hysteresis:
        hysteresis_script = script_dir / "analysis" / "compute_hysteresis.py"
        hysteresis_dir = Path(f"data/04_analysis/hysteresis/{output_suffix}")
        hysteresis_plots_dir = Path(f"plots/hysteresis/{output_suffix}")

        hysteresis_cmd = [
            "python3",
            str(hysteresis_script),
            "--stats-dir", str(stats_dir),
            "--output-dir", str(hysteresis_dir)
        ]

        run_command(hysteresis_cmd, "Hysteresis Computation")

        # Plot hysteresis
        hysteresis_plot_script = script_dir / "analysis" / "plot_hysteresis.py"
        hysteresis_plot_cmd = [
            "python3",
            str(hysteresis_plot_script),
            "--hysteresis-dir", str(hysteresis_dir),
            "--output-dir", str(hysteresis_plots_dir)
        ]

        run_command(hysteresis_plot_cmd, "Hysteresis Plotting")

    # Summary
    print(f"\n{'='*60}")
    print("✓ IV Pipeline Complete!")
    print(f"{'='*60}")
    print(f"\nResults:")
    print(f"  Statistics:     {stats_dir}")
    print(f"  IV Plots:       {plots_dir}")
    if args.compute_hysteresis:
        print(f"  Hysteresis:     {hysteresis_dir}")
        print(f"  Hyst Plots:     {hysteresis_plots_dir}")
    print(f"\nKey files:")
    print(f"  - {stats_dir}/fit_summary.csv")
    print(f"  - {stats_dir}/polynomial_fits_summary.csv")
    print(f"  - {plots_dir}/iv_aggregated_all_return_fit.png")
    print(f"  - {plots_dir}/resistance_vs_vmax_return_fit.png")
    if args.compute_hysteresis:
        print(f"  - {hysteresis_dir}/hysteresis_summary.csv")
        print(f"  - {hysteresis_plots_dir}/hysteresis_vs_vmax.png")
    print()


if __name__ == "__main__":
    main()

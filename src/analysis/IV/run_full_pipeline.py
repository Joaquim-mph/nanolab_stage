#!/usr/bin/env python3
"""
IV Full Pipeline Runner - Complete analysis and visualization workflow.

Executes:
1. Analysis: Statistics → Hysteresis → Peaks
2. Plotting: All visualization types

One-command solution for complete IV analysis workflow.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*70}")
    print(f"▶ {description}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\n❌ Error: {description} failed with exit code {result.returncode}")
        return False

    print(f"\n✓ {description} completed successfully")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run complete IV analysis and plotting pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This is the master runner that executes both analysis and plotting.

Examples:
  # Full pipeline with defaults
  python src/analysis/IV/run_full_pipeline.py --date 2025-09-11

  # Specific chip with custom output
  python src/analysis/IV/run_full_pipeline.py \\
    --date 2025-09-11 \\
    --chip-number 71 \\
    --output-suffix my_analysis

  # Analysis only (no plots)
  python src/analysis/IV/run_full_pipeline.py \\
    --date 2025-09-11 \\
    --analysis-only

  # Plotting only (skip analysis)
  python src/analysis/IV/run_full_pipeline.py \\
    --date 2025-09-11 \\
    --plotting-only

Pipeline Steps:
  1. Analysis:
     - Aggregate IV statistics
     - Compute hysteresis
     - Analyze peak locations

  2. Plotting:
     - IV traces (forward/return)
     - Hysteresis comparison (⭐ main figure)
     - Statistical exploration
     - Detailed per-range plots
     - Basic hysteresis plots
        """
    )

    # Input parameters
    parser.add_argument("--date", type=str, required=True,
                       help="Date to process (YYYY-MM-DD)")
    parser.add_argument("--procedure", type=str, default="IV",
                       help="Procedure name (default: IV)")
    parser.add_argument("--chip-number", type=str,
                       help="Specific chip number to analyze")
    parser.add_argument("--stage-root", type=Path,
                       default=Path("data/02_stage/raw_measurements"),
                       help="Root directory of staged data")

    # Output parameters
    parser.add_argument("--output-suffix", type=str,
                       help="Suffix for output directories (default: date)")

    # Processing options
    parser.add_argument("--poly-orders", type=int, nargs='+',
                       default=[1, 3, 5, 7],
                       help="Polynomial orders for fitting (default: 1 3 5 7)")
    parser.add_argument("--poly-order", type=int, default=3,
                       choices=[1, 3, 5, 7],
                       help="Polynomial order for exploration plots (default: 3)")

    # Workflow control
    parser.add_argument("--analysis-only", action="store_true",
                       help="Run analysis only (skip plotting)")
    parser.add_argument("--plotting-only", action="store_true",
                       help="Run plotting only (skip analysis)")
    parser.add_argument("--skip-peaks", action="store_true",
                       help="Skip peak analysis")

    # Plot options
    parser.add_argument("--plots", type=str, nargs='+',
                       choices=['traces', 'comparison', 'exploration', 'detailed', 'basic', 'all'],
                       default=['all'],
                       help="Which plot types to generate (default: all)")
    parser.add_argument("--compact", action="store_true",
                       help="Generate compact versions")
    parser.add_argument("--residuals", action="store_true",
                       help="Generate residual plots")

    args = parser.parse_args()

    # Validate
    if args.analysis_only and args.plotting_only:
        print("Error: Cannot specify both --analysis-only and --plotting-only")
        sys.exit(1)

    # Determine output directories
    output_suffix = args.output_suffix or args.date
    stats_dir = Path(f"data/04_analysis/iv_stats/{output_suffix}")
    hysteresis_dir = Path(f"data/04_analysis/hysteresis/{output_suffix}")
    peaks_dir = Path(f"data/04_analysis/hysteresis_peaks/{output_suffix}")
    plots_dir = Path(f"plots/{output_suffix}")

    # Get script directory
    script_dir = Path(__file__).parent
    plotting_dir = Path(__file__).parent.parent.parent / "ploting" / "IV"

    print(f"\n{'='*70}")
    print(f"IV FULL PIPELINE")
    print(f"{'='*70}")
    print(f"Date:          {args.date}")
    print(f"Procedure:     {args.procedure}")
    print(f"Chip number:   {args.chip_number or 'All'}")
    print(f"Stage root:    {args.stage_root}")
    print(f"")
    print(f"Output directories:")
    print(f"  Statistics:  {stats_dir}")
    print(f"  Hysteresis:  {hysteresis_dir}")
    print(f"  Peaks:       {peaks_dir}")
    print(f"  Plots:       {plots_dir}")
    print(f"{'='*70}")

    # Phase 1: Analysis
    if not args.plotting_only:
        print(f"\n{'#'*70}")
        print(f"# PHASE 1: ANALYSIS")
        print(f"{'#'*70}")

        cmd_analysis = [
            sys.executable,
            str(script_dir / "run_analysis.py"),
            "--date", args.date,
            "--procedure", args.procedure,
            "--stage-root", str(args.stage_root),
            "--stats-dir", str(stats_dir),
            "--hysteresis-dir", str(hysteresis_dir),
            "--peaks-dir", str(peaks_dir),
            "--poly-orders", *[str(o) for o in args.poly_orders]
        ]

        if args.chip_number:
            cmd_analysis.extend(["--chip-number", args.chip_number])

        if args.skip_peaks:
            cmd_analysis.append("--skip-peaks")

        if not run_command(cmd_analysis, "PHASE 1: Analysis Pipeline"):
            sys.exit(1)

    # Phase 2: Plotting
    if not args.analysis_only:
        print(f"\n{'#'*70}")
        print(f"# PHASE 2: PLOTTING")
        print(f"{'#'*70}")

        cmd_plotting = [
            sys.executable,
            str(plotting_dir / "run_plotting.py"),
            "--stats-dir", str(stats_dir),
            "--hysteresis-dir", str(hysteresis_dir),
            "--output-dir", str(plots_dir),
            "--poly-order", str(args.poly_order),
            "--plots", *args.plots
        ]

        if not args.skip_peaks and peaks_dir.exists():
            cmd_plotting.extend(["--peaks-dir", str(peaks_dir)])

        if args.compact:
            cmd_plotting.append("--compact")
        if args.residuals:
            cmd_plotting.append("--residuals")

        if not run_command(cmd_plotting, "PHASE 2: Plotting Pipeline"):
            sys.exit(1)

    # Final summary
    print(f"\n{'='*70}")
    print(f"✓ PIPELINE COMPLETE!")
    print(f"{'='*70}")
    print(f"\nGenerated outputs:")
    if not args.plotting_only:
        print(f"  📊 Analysis data:")
        print(f"     - Statistics:  {stats_dir}")
        print(f"     - Hysteresis:  {hysteresis_dir}")
        if not args.skip_peaks:
            print(f"     - Peaks:       {peaks_dir}")
    if not args.analysis_only:
        print(f"  📈 Plots:")
        print(f"     - All figures: {plots_dir}")
        if 'comparison' in args.plots or 'all' in args.plots:
            print(f"     - ⭐ Main:     {plots_dir}/hysteresis_comparison/all_ranges_all_polynomials.png")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
IV Analysis Runner - Executes complete IV analysis pipeline.

Runs all analysis steps in sequence:
1. Aggregate IV statistics (forward/return separation, polynomial fits)
2. Compute hysteresis (forward - return)
3. Analyze peak locations

Output: Analysis-ready datasets for visualization
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

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        print(f"\n❌ Error: {description} failed with exit code {result.returncode}")
        return False

    print(f"\n✓ {description} completed successfully")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run complete IV analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all IV data for a date
  python src/analysis/IV/run_analysis.py --date 2025-09-11

  # Analyze specific chip
  python src/analysis/IV/run_analysis.py --date 2025-09-11 --chip-number 71

  # Custom output location
  python src/analysis/IV/run_analysis.py --date 2025-09-11 --output-suffix my_test

  # Skip hysteresis and peak analysis
  python src/analysis/IV/run_analysis.py --date 2025-09-11 --stats-only

Output Structure:
  data/04_analysis/iv_stats/{date}/
    ├── fit_summary.csv
    ├── polynomial_fits_summary.csv
    ├── forward_vmax*.csv
    ├── return_vmax*.csv
    └── return_with_fit_vmax*.csv

  data/04_analysis/hysteresis/{date}/
    ├── hysteresis_summary.csv
    └── hysteresis_vmax*.csv

  data/04_analysis/hysteresis_peaks/{date}/
    ├── hysteresis_peaks.csv
    ├── peak_summary_table.csv
    └── *.png (visualizations)
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
    parser.add_argument("--stats-dir", type=Path,
                       help="Custom output directory for statistics (overrides default)")
    parser.add_argument("--hysteresis-dir", type=Path,
                       help="Custom output directory for hysteresis (overrides default)")
    parser.add_argument("--peaks-dir", type=Path,
                       help="Custom output directory for peaks (overrides default)")

    # Processing options
    parser.add_argument("--poly-orders", type=int, nargs='+',
                       default=[1, 3, 5, 7],
                       help="Polynomial orders for fitting (default: 1 3 5 7)")
    parser.add_argument("--stats-only", action="store_true",
                       help="Only run statistics aggregation (skip hysteresis and peaks)")
    parser.add_argument("--skip-peaks", action="store_true",
                       help="Skip peak analysis (run stats + hysteresis only)")

    args = parser.parse_args()

    # Determine output directories
    output_suffix = args.output_suffix or args.date

    stats_dir = args.stats_dir or Path(f"data/04_analysis/iv_stats/{output_suffix}")
    hysteresis_dir = args.hysteresis_dir or Path(f"data/04_analysis/hysteresis/{output_suffix}")
    peaks_dir = args.peaks_dir or Path(f"data/04_analysis/hysteresis_peaks/{output_suffix}")

    # Get script directory
    script_dir = Path(__file__).parent

    print(f"\n{'='*70}")
    print(f"IV ANALYSIS PIPELINE")
    print(f"{'='*70}")
    print(f"Date:          {args.date}")
    print(f"Procedure:     {args.procedure}")
    print(f"Chip number:   {args.chip_number or 'All'}")
    print(f"Stage root:    {args.stage_root}")
    print(f"Stats output:  {stats_dir}")
    if not args.stats_only:
        print(f"Hysteresis:    {hysteresis_dir}")
        if not args.skip_peaks:
            print(f"Peaks:         {peaks_dir}")
    print(f"Poly orders:   {args.poly_orders}")
    print(f"{'='*70}")

    # Step 1: Aggregate IV statistics
    cmd_stats = [
        sys.executable,
        str(script_dir / "aggregate_iv_stats.py"),
        "--stage-root", str(args.stage_root),
        "--date", args.date,
        "--procedure", args.procedure,
        "--output-dir", str(stats_dir)
    ]

    if args.chip_number:
        cmd_stats.extend(["--chip-number", args.chip_number])

    if not run_command(cmd_stats, "Step 1/3: Aggregate IV Statistics"):
        sys.exit(1)

    if args.stats_only:
        print(f"\n{'='*70}")
        print(f"✓ Analysis complete! (stats only)")
        print(f"{'='*70}\n")
        return

    # Step 2: Compute hysteresis
    cmd_hysteresis = [
        sys.executable,
        str(script_dir / "compute_hysteresis.py"),
        "--stats-dir", str(stats_dir),
        "--output-dir", str(hysteresis_dir)
    ]

    if not run_command(cmd_hysteresis, "Step 2/3: Compute Hysteresis"):
        sys.exit(1)

    if args.skip_peaks:
        print(f"\n{'='*70}")
        print(f"✓ Analysis complete! (stats + hysteresis)")
        print(f"{'='*70}\n")
        return

    # Step 3: Analyze peaks
    cmd_peaks = [
        sys.executable,
        str(script_dir / "analyze_hysteresis_peaks.py"),
        "--hysteresis-dir", str(hysteresis_dir),
        "--output-dir", str(peaks_dir)
    ]

    if not run_command(cmd_peaks, "Step 3/3: Analyze Hysteresis Peaks"):
        sys.exit(1)

    # Summary
    print(f"\n{'='*70}")
    print(f"✓ COMPLETE! All analysis steps finished successfully")
    print(f"{'='*70}")
    print(f"\nOutput locations:")
    print(f"  Statistics:  {stats_dir}")
    print(f"  Hysteresis:  {hysteresis_dir}")
    print(f"  Peaks:       {peaks_dir}")
    print(f"\nNext step: Generate plots with:")
    print(f"  python src/ploting/IV/run_plotting.py \\")
    print(f"    --stats-dir {stats_dir} \\")
    print(f"    --hysteresis-dir {hysteresis_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

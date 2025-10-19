#!/usr/bin/env python3
"""
Unified pipeline runner using Pydantic configuration.

This is the main entry point for running the complete nanolab data pipeline
with type-safe, validated parameters.

Supports:
- Staging: Raw CSV → Parquet
- Analysis: IV aggregation, hysteresis computation, peak detection
- Plotting: Publication-quality figures

Usage:
    # From JSON config (recommended)
    python run_pipeline.py --config config/my_pipeline.json

    # Quick analysis + plotting (assumes data already staged)
    python run_pipeline.py --date 2025-09-11 --procedure IV

    # Full pipeline from scratch
    python run_pipeline.py --config config/full_pipeline.json --run-all
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from models.parameters import (
    StagingParameters,
    IntermediateParameters,
    IVAnalysisParameters,
    PlottingParameters,
    PipelineParameters,
)
from pydantic import ValidationError


def run_staging(params: StagingParameters) -> bool:
    """Execute staging pipeline."""
    print("\n" + "="*80)
    print("STEP 1: STAGING - Raw CSV → Parquet")
    print("="*80)

    # Import staging function directly
    sys.path.insert(0, str(Path(__file__).parent / "src" / "staging"))
    from stage_raw_measurements import run_staging_pipeline

    try:
        run_staging_pipeline(params)
        return True
    except Exception as e:
        print(f"\n❌ Staging failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_intermediate(params: IntermediateParameters) -> bool:
    """Execute intermediate preprocessing pipeline."""
    print("\n" + "="*80)
    print("STEP 2: INTERMEDIATE - Segment Detection & Preprocessing")
    print("="*80)

    output_dir = params.get_output_dir()

    print(f"\nConfiguration:")
    print(f"  Stage root:        {params.stage_root}")
    print(f"  Output dir:        {output_dir}")
    print(f"  Procedure:         {params.procedure}")
    print(f"  Voltage column:    {params.voltage_col}")
    print(f"  dV threshold:      {params.dv_threshold}")
    print(f"  Min segment pts:   {params.min_segment_points}")
    print(f"  Workers:           {params.workers}")

    # Import preprocessing function directly
    sys.path.insert(0, str(Path(__file__).parent / "src" / "intermediate" / "IV"))
    from iv_preprocessing_script import run_iv_preprocessing

    try:
        run_iv_preprocessing(params)
        print("\n✓ Intermediate preprocessing completed successfully")
        return True
    except Exception as e:
        print(f"\n❌ Intermediate preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_iv_analysis(params: IVAnalysisParameters) -> bool:
    """Execute IV analysis pipeline."""
    print("\n" + "="*80)
    print("STEP 3: IV ANALYSIS")
    print("="*80)

    stats_dir = params.get_stats_dir()
    hysteresis_dir = params.get_hysteresis_dir()
    peaks_dir = params.get_peaks_dir()

    print(f"\nConfiguration:")
    print(f"  Date:              {params.date}")
    print(f"  Procedure:         {params.procedure}")
    print(f"  Polynomial orders: {params.poly_orders}")
    print(f"  Stats output:      {stats_dir}")
    print(f"  Hysteresis output: {hysteresis_dir}")
    print(f"  Peaks output:      {peaks_dir}")

    # Import analysis functions directly
    sys.path.insert(0, str(Path(__file__).parent / "src" / "analysis" / "IV"))

    try:
        # Step 3.1: Aggregate IV statistics
        print("\n[3.1] Aggregating IV statistics...")
        from aggregate_iv_stats import run_iv_aggregation
        run_iv_aggregation(params)

        # Step 3.2: Compute hysteresis
        if params.compute_hysteresis:
            print("\n[3.2] Computing hysteresis...")
            cmd = [
                sys.executable, "src/analysis/IV/compute_hysteresis.py",
                "--stats-dir", str(stats_dir),
                "--output-dir", str(hysteresis_dir),
            ]
            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"❌ Hysteresis computation failed")
                return False

        # Step 3.3: Analyze peaks
        if params.analyze_peaks and params.compute_hysteresis:
            print("\n[3.3] Analyzing hysteresis peaks...")
            cmd = [
                sys.executable, "src/analysis/IV/analyze_hysteresis_peaks.py",
                "--hysteresis-dir", str(hysteresis_dir),
                "--output-dir", str(peaks_dir),
            ]
            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"❌ Peak analysis failed")
                return False

        print("\n✓ IV analysis completed successfully")
        return True

    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_plotting(params: PlottingParameters, analysis_params: IVAnalysisParameters) -> bool:
    """Execute plotting."""
    print("\n" + "="*80)
    print("STEP 4: PLOTTING")
    print("="*80)

    hysteresis_dir = analysis_params.get_hysteresis_dir()
    output_dir = params.output_dir

    print(f"\nConfiguration:")
    print(f"  Hysteresis data:   {hysteresis_dir}")
    print(f"  Output dir:        {output_dir}")
    print(f"  DPI:               {params.dpi}")
    print(f"  Format:            {params.format}")
    print(f"  Style:             {params.style}")

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Create comparison figure
        print("\n[4.1] Creating polynomial comparison figure...")
        cmd = [
            sys.executable, "src/ploting/IV/compare_polynomial_orders.py",
            "--hysteresis-dir", str(hysteresis_dir),
            "--output-dir", str(output_dir),
        ]

        if params.compact_layout:
            cmd.append("--compact")

        if params.show_residuals:
            cmd.append("--residuals")

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"❌ Plotting failed")
            return False

        print("\n✓ Plotting completed successfully")
        return True

    except Exception as e:
        print(f"\n❌ Plotting failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Unified pipeline runner with Pydantic configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From JSON config (recommended)
  python run_pipeline.py --config config/my_pipeline.json

  # Quick analysis + plotting (assumes data already staged)
  python run_pipeline.py --date 2025-09-11 --procedure IV

  # Specify steps to run
  python run_pipeline.py --config config/pipeline.json --skip-staging

  # Custom output location
  python run_pipeline.py --date 2025-09-11 --output-dir plots/my_analysis
        """
    )

    # Config file option
    parser.add_argument("--config", type=Path, help="Path to JSON configuration file")

    # Quick analysis options
    parser.add_argument("--date", help="Date in YYYY-MM-DD format")
    parser.add_argument("--procedure", default="IV", help="Procedure name (IV, IVg, etc.)")
    parser.add_argument("--stage-root", type=Path, help="Staged data root")
    parser.add_argument("--output-base-dir", type=Path, help="Analysis output base directory")
    parser.add_argument("--output-dir", type=Path, help="Plotting output directory")

    # Pipeline control
    parser.add_argument("--run-all", action="store_true", help="Run all steps (staging, analysis, plotting)")
    parser.add_argument("--skip-staging", action="store_true", help="Skip staging step")
    parser.add_argument("--skip-analysis", action="store_true", help="Skip analysis step")
    parser.add_argument("--skip-plotting", action="store_true", help="Skip plotting step")

    # Analysis options
    parser.add_argument("--poly-orders", nargs="+", type=int, help="Polynomial orders")
    parser.add_argument("--no-hysteresis", action="store_true", help="Skip hysteresis computation")
    parser.add_argument("--no-peaks", action="store_true", help="Skip peak analysis")

    # Plotting options
    parser.add_argument("--dpi", type=int, default=300, help="Plot DPI")
    parser.add_argument("--compact", action="store_true", help="Compact plot layout")
    parser.add_argument("--residuals", action="store_true", help="Show residuals plots")

    args = parser.parse_args()

    print("\n" + "="*80)
    print("NANOLAB DATA PIPELINE - PYDANTIC CONFIGURATION")
    print("="*80)

    try:
        # Mode 1: Load from JSON config
        if args.config:
            if not args.config.exists():
                print(f"\n❌ Config file not found: {args.config}")
                sys.exit(1)

            print(f"\n[info] Loading configuration from: {args.config}")
            params = PipelineParameters.from_json(args.config)
            print("[info] ✓ Configuration validated successfully")

            # Override with command-line flags
            if args.run_all:
                params.run_staging = True
                params.run_analysis = True
                params.run_plotting = True
            if args.skip_staging:
                params.run_staging = False
            if args.skip_analysis:
                params.run_analysis = False
            if args.skip_plotting:
                params.run_plotting = False

        # Mode 2: Quick analysis from command-line
        elif args.date:
            print("\n[info] Creating configuration from command-line arguments")

            stage_root = args.stage_root or Path("data/02_stage/raw_measurements")
            output_base = args.output_base_dir or Path("data/04_analysis")
            plot_output = args.output_dir or Path("plots") / f"{args.date}_{args.procedure}"
            poly_orders = args.poly_orders or [1, 3, 5, 7]

            # Minimal staging config (won't be used unless --run-all)
            staging = StagingParameters(
                raw_root=Path("data/01_raw"),
                stage_root=stage_root,
                procedures_yaml=Path("config/procedures.yml"),
            )

            analysis = IVAnalysisParameters(
                stage_root=stage_root,
                date=args.date,
                output_base_dir=output_base,
                procedure=args.procedure,
                poly_orders=poly_orders,
                compute_hysteresis=not args.no_hysteresis,
                analyze_peaks=not args.no_peaks,
            )

            plotting = PlottingParameters(
                output_dir=plot_output,
                dpi=args.dpi,
                compact_layout=args.compact,
                show_residuals=args.residuals,
            )

            params = PipelineParameters(
                staging=staging,
                analysis=analysis,
                plotting=plotting,
                run_staging=args.run_all and not args.skip_staging,
                run_analysis=not args.skip_analysis,
                run_plotting=not args.skip_plotting,
            )

            print("[info] ✓ Configuration created and validated")

        else:
            parser.print_help()
            print("\n❌ Error: Must provide either --config or --date")
            sys.exit(1)

        # Print pipeline summary
        print("\n" + "-"*80)
        print("PIPELINE CONFIGURATION SUMMARY:")
        print("-"*80)
        print(f"Steps to run:")
        print(f"  Staging:       {'YES' if params.run_staging else 'NO'}")
        print(f"  Intermediate:  {'YES' if params.run_intermediate else 'NO'}")
        print(f"  Analysis:      {'YES' if params.run_analysis else 'NO'}")
        print(f"  Plotting:      {'YES' if params.run_plotting else 'NO'}")
        print(f"\nAnalysis date:      {params.analysis.date}")
        print(f"Procedure:          {params.analysis.procedure}")
        print(f"Polynomial orders:  {params.analysis.poly_orders}")
        print(f"Output directory:   {params.plotting.output_dir}")
        print("-"*80)

        # Validate 4-layer requirements
        if params.run_analysis and not params.run_intermediate:
            if params.analysis.intermediate_root is None:
                print("\n⚠️  WARNING: Analysis requires intermediate_root but intermediate step is not enabled.")
                print("   Either:")
                print("   1. Enable intermediate: set run_intermediate=true")
                print("   2. Run preprocessing separately before analysis")
                sys.exit(1)

        # Run pipeline steps
        success = True

        if params.run_staging:
            success = success and run_staging(params.staging)

        if success and params.run_intermediate:
            success = success and run_intermediate(params.intermediate)

        if success and params.run_analysis:
            success = success and run_iv_analysis(params.analysis)

        if success and params.run_plotting:
            success = success and run_plotting(params.plotting, params.analysis)

        # Final summary
        print("\n" + "="*80)
        if success:
            print("✓ PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*80)
            print(f"\nResults saved to:")
            if params.run_analysis:
                print(f"  Analysis:  {params.analysis.get_stats_dir()}")
                if params.analysis.compute_hysteresis:
                    print(f"  Hysteresis: {params.analysis.get_hysteresis_dir()}")
            if params.run_plotting:
                print(f"  Plots:     {params.plotting.output_dir}")
        else:
            print("❌ PIPELINE FAILED")
        print("="*80 + "\n")

        sys.exit(0 if success else 1)

    except ValidationError as e:
        print(f"\n❌ Configuration validation failed:")
        print(e)
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

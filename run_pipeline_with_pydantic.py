#!/usr/bin/env python3
"""
Run the complete pipeline using Pydantic configuration.

This script demonstrates how to use Pydantic models to run the full pipeline
with validated parameters, either from JSON config or programmatically.

Usage:
    # From JSON config
    python run_pipeline_with_pydantic.py --config config/examples/pipeline_config.json

    # Programmatically
    python run_pipeline_with_pydantic.py --date 2025-09-11 --procedure IV --workers 8
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
    IVAnalysisParameters,
    PlottingParameters,
    PipelineParameters,
)
from pydantic import ValidationError


def run_staging(params: StagingParameters) -> bool:
    """
    Run staging pipeline using validated parameters.

    Calls the existing stage_raw_measurements.py script with parameters.
    """
    print("\n" + "="*70)
    print("STEP 1: STAGING (Raw CSV → Parquet)")
    print("="*70)

    cmd = [
        "python", "src/staging/stage_raw_measurements.py",
        "--raw-root", str(params.raw_root),
        "--stage-root", str(params.stage_root),
        "--procedures-yaml", str(params.procedures_yaml),
        "--workers", str(params.workers),
        "--polars-threads", str(params.polars_threads),
        "--local-tz", params.local_tz,
    ]

    if params.force:
        cmd.append("--force")

    if params.only_yaml_data:
        cmd.append("--only-yaml-data")

    print(f"\nConfiguration:")
    print(f"  Raw root:         {params.raw_root}")
    print(f"  Stage root:       {params.stage_root}")
    print(f"  Procedures YAML:  {params.procedures_yaml}")
    print(f"  Workers:          {params.workers}")
    print(f"  Polars threads:   {params.polars_threads}")
    print(f"  Force overwrite:  {params.force}")

    print(f"\nExecuting: {' '.join(cmd)}\n")

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\n❌ Staging failed with exit code {result.returncode}")
        return False

    print("\n✓ Staging completed successfully")
    return True


def run_iv_analysis(params: IVAnalysisParameters) -> bool:
    """
    Run IV analysis pipeline using validated parameters.

    Calls the IV analysis scripts (aggregate, compute hysteresis, analyze peaks).
    """
    print("\n" + "="*70)
    print("STEP 2: IV ANALYSIS")
    print("="*70)

    stats_dir = params.get_stats_dir()
    hysteresis_dir = params.get_hysteresis_dir()
    peaks_dir = params.get_peaks_dir()

    print(f"\nConfiguration:")
    print(f"  Stage root:       {params.stage_root}")
    print(f"  Date:             {params.date}")
    print(f"  Procedure:        {params.procedure}")
    print(f"  Polynomial orders: {params.poly_orders}")
    print(f"  Stats output:     {stats_dir}")
    print(f"  Hysteresis output: {hysteresis_dir}")

    # Step 2.1: Aggregate IV statistics
    print("\n[2.1] Aggregating IV statistics...")
    cmd = [
        "python", "src/analysis/IV/aggregate_iv_stats.py",
        "--stage-root", str(params.stage_root),
        "--date", params.date,
        "--output-dir", str(stats_dir),
        "--procedure", params.procedure,
    ]

    if params.chip_number:
        cmd.extend(["--chip-number", params.chip_number])

    if params.v_max is not None:
        cmd.extend(["--v-max", str(params.v_max)])

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n❌ IV aggregation failed with exit code {result.returncode}")
        return False

    # Step 2.2: Compute hysteresis (if enabled)
    if params.compute_hysteresis:
        print("\n[2.2] Computing hysteresis...")
        cmd = [
            "python", "src/analysis/IV/compute_hysteresis.py",
            "--stats-dir", str(stats_dir),
            "--output-dir", str(hysteresis_dir),
        ]

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"\n❌ Hysteresis computation failed with exit code {result.returncode}")
            return False

    # Step 2.3: Analyze peaks (if enabled)
    if params.analyze_peaks and params.compute_hysteresis:
        print("\n[2.3] Analyzing hysteresis peaks...")
        cmd = [
            "python", "src/analysis/IV/analyze_hysteresis_peaks.py",
            "--hysteresis-dir", str(hysteresis_dir),
            "--output-dir", str(peaks_dir),
        ]

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"\n❌ Peak analysis failed with exit code {result.returncode}")
            return False

    print("\n✓ IV analysis completed successfully")
    return True


def run_plotting(params: PlottingParameters, analysis_params: IVAnalysisParameters) -> bool:
    """
    Run plotting using validated parameters.

    Calls the plotting scripts to generate publication-quality figures.
    """
    print("\n" + "="*70)
    print("STEP 3: PLOTTING")
    print("="*70)

    hysteresis_dir = analysis_params.get_hysteresis_dir()

    print(f"\nConfiguration:")
    print(f"  Output dir:       {params.output_dir}")
    print(f"  DPI:              {params.dpi}")
    print(f"  Format:           {params.format}")
    print(f"  Style:            {params.style}")
    print(f"  Compact layout:   {params.compact_layout}")
    print(f"  Show residuals:   {params.show_residuals}")

    # Create comprehensive polynomial comparison figure
    print("\n[3.1] Creating polynomial comparison figure...")
    cmd = [
        "python", "src/ploting/IV/compare_polynomial_orders.py",
        "--hysteresis-dir", str(hysteresis_dir),
        "--output-dir", str(params.output_dir),
    ]

    if params.compact_layout:
        cmd.append("--compact")

    if params.show_residuals:
        cmd.append("--residuals")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n❌ Plotting failed with exit code {result.returncode}")
        return False

    print("\n✓ Plotting completed successfully")
    return True


def run_pipeline_from_config(config_path: Path) -> bool:
    """Load configuration from JSON and run full pipeline."""
    print("\n" + "="*70)
    print(f"LOADING CONFIGURATION: {config_path}")
    print("="*70)

    try:
        params = PipelineParameters.from_json(config_path)
        print("\n✓ Configuration loaded and validated successfully!")

        # Validate all paths exist / create output directories
        params.validate_all_paths()
        print("✓ All paths validated")

    except ValidationError as e:
        print(f"\n❌ Configuration validation failed:")
        print(e)
        return False
    except Exception as e:
        print(f"\n❌ Error loading configuration: {e}")
        return False

    # Run pipeline steps
    success = True

    if params.run_staging:
        success = success and run_staging(params.staging)

    if success and params.run_analysis:
        success = success and run_iv_analysis(params.analysis)

    if success and params.run_plotting:
        success = success and run_plotting(params.plotting, params.analysis)

    return success


def run_pipeline_programmatic(
    date: str,
    procedure: str,
    raw_root: Optional[Path] = None,
    stage_root: Optional[Path] = None,
    procedures_yaml: Optional[Path] = None,
    workers: int = 8,
    poly_orders: Optional[list[int]] = None,
    compute_hysteresis: bool = True,
    analyze_peaks: bool = True,
    output_base: Optional[Path] = None,
    plot_output: Optional[Path] = None,
    dpi: int = 300,
) -> bool:
    """Run pipeline with programmatically created parameters."""
    print("\n" + "="*70)
    print("CREATING PIPELINE CONFIGURATION PROGRAMMATICALLY")
    print("="*70)

    # Set defaults
    raw_root = raw_root or Path("data/01_raw")
    stage_root = stage_root or Path("data/02_stage/raw_measurements")
    procedures_yaml = procedures_yaml or Path("config/procedures.yml")
    output_base = output_base or Path("data/04_analysis")
    plot_output = plot_output or Path("plots") / f"{date}_{procedure}"
    poly_orders = poly_orders or [1, 3, 5, 7]

    try:
        # Create parameter objects with validation
        staging = StagingParameters(
            raw_root=raw_root,
            stage_root=stage_root,
            procedures_yaml=procedures_yaml,
            workers=workers,
        )

        analysis = IVAnalysisParameters(
            stage_root=stage_root,
            date=date,
            output_base_dir=output_base,
            procedure=procedure,
            poly_orders=poly_orders,
            compute_hysteresis=compute_hysteresis,
            analyze_peaks=analyze_peaks,
        )

        plotting = PlottingParameters(
            output_dir=plot_output,
            dpi=dpi,
            compact_layout=True,
            show_residuals=True,
        )

        params = PipelineParameters(
            staging=staging,
            analysis=analysis,
            plotting=plotting,
            run_staging=False,  # Usually data already staged
            run_analysis=True,
            run_plotting=True,
        )

        print("\n✓ Configuration created and validated successfully!")

        # Save config for reference
        config_file = Path("config/examples/last_run_config.json")
        params.to_json(config_file)
        print(f"✓ Configuration saved to: {config_file}")

        # Validate paths
        params.validate_all_paths()
        print("✓ All paths validated")

    except ValidationError as e:
        print(f"\n❌ Configuration validation failed:")
        print(e)
        return False

    # Run pipeline steps
    success = True

    if params.run_staging:
        success = success and run_staging(params.staging)

    if success and params.run_analysis:
        success = success and run_iv_analysis(params.analysis)

    if success and params.run_plotting:
        success = success and run_plotting(params.plotting, params.analysis)

    return success


def main():
    parser = argparse.ArgumentParser(
        description="Run complete pipeline with Pydantic-validated configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From JSON config file
  python run_pipeline_with_pydantic.py --config config/examples/pipeline_config.json

  # Programmatically (analysis + plotting only)
  python run_pipeline_with_pydantic.py --date 2025-09-11 --procedure IV --workers 8

  # With custom parameters
  python run_pipeline_with_pydantic.py --date 2025-09-11 --procedure IV --poly-orders 1 3 5 7 --dpi 600
        """
    )

    # Config file option
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to JSON configuration file"
    )

    # Programmatic options
    parser.add_argument("--date", help="Date in YYYY-MM-DD format")
    parser.add_argument("--procedure", default="IV", help="Procedure name (IV, IVg, etc.)")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    parser.add_argument("--poly-orders", nargs="+", type=int, default=[1, 3, 5, 7], help="Polynomial orders")
    parser.add_argument("--no-hysteresis", action="store_true", help="Skip hysteresis computation")
    parser.add_argument("--no-peaks", action="store_true", help="Skip peak analysis")
    parser.add_argument("--dpi", type=int, default=300, help="Plot DPI")

    args = parser.parse_args()

    print("\n" + "="*70)
    print("NANOLAB PIPELINE - PYDANTIC CONFIGURATION")
    print("="*70)

    if args.config:
        # Run from JSON config
        if not args.config.exists():
            print(f"\n❌ Config file not found: {args.config}")
            sys.exit(1)

        success = run_pipeline_from_config(args.config)

    elif args.date:
        # Run programmatically
        success = run_pipeline_programmatic(
            date=args.date,
            procedure=args.procedure,
            workers=args.workers,
            poly_orders=args.poly_orders,
            compute_hysteresis=not args.no_hysteresis,
            analyze_peaks=not args.no_peaks,
            dpi=args.dpi,
        )

    else:
        parser.print_help()
        print("\n❌ Error: Must provide either --config or --date")
        sys.exit(1)

    # Final summary
    print("\n" + "="*70)
    if success:
        print("✓ PIPELINE COMPLETED SUCCESSFULLY!")
    else:
        print("❌ PIPELINE FAILED")
    print("="*70 + "\n")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Example script demonstrating Pydantic parameter usage.

Shows how to:
1. Create parameters programmatically with validation
2. Load parameters from JSON config
3. Save parameters to JSON
4. Handle validation errors gracefully
5. Use parameter helper methods
"""

from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.parameters import (
    StagingParameters,
    IVAnalysisParameters,
    PlottingParameters,
    PipelineParameters,
)
from pydantic import ValidationError


def example_1_create_staging_params():
    """Example 1: Create staging parameters with validation."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Create Staging Parameters")
    print("=" * 70)

    try:
        params = StagingParameters(
            raw_root=Path("data/01_raw"),
            stage_root=Path("data/02_stage/raw_measurements"),
            procedures_yaml=Path("config/procedures.yml"),
            workers=8,
            polars_threads=2,
            force=True,
        )

        print("\n✓ Parameters created successfully!")
        print(f"  Workers: {params.workers}")
        print(f"  Polars threads: {params.polars_threads}")
        print(f"  Force overwrite: {params.force}")
        print(f"  Default rejects dir: {params.rejects_dir}")
        print(f"  Default events dir: {params.events_dir}")

    except ValidationError as e:
        print(f"\n❌ Validation failed:\n{e}")


def example_2_validation_errors():
    """Example 2: Demonstrate validation catching bad values."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Validation Errors")
    print("=" * 70)

    print("\n[Attempt 1] Invalid worker count (0):")
    try:
        StagingParameters(
            raw_root=Path("data/01_raw"),
            stage_root=Path("data/02_stage/raw_measurements"),
            procedures_yaml=Path("config/procedures.yml"),
            workers=0,  # Invalid!
        )
        print("  ❌ Should have failed!")
    except ValidationError as e:
        print(f"  ✓ Caught: {e.errors()[0]['msg']}")

    print("\n[Attempt 2] Invalid date format:")
    try:
        IVAnalysisParameters(
            stage_root=Path("data/02_stage/raw_measurements"),
            date="2025-9-11",  # Should be 2025-09-11
            output_base_dir=Path("data/04_analysis"),
        )
        print("  ❌ Should have failed!")
    except ValidationError as e:
        print(f"  ✓ Caught: {e.errors()[0]['msg']}")

    print("\n[Attempt 3] Even polynomial order:")
    try:
        IVAnalysisParameters(
            stage_root=Path("data/02_stage/raw_measurements"),
            date="2025-09-11",
            output_base_dir=Path("data/04_analysis"),
            poly_orders=[1, 2, 3],  # 2 is even - invalid!
        )
        print("  ❌ Should have failed!")
    except ValidationError as e:
        print(f"  ✓ Caught: {e.errors()[0]['msg']}")


def example_3_json_config():
    """Example 3: Load/save JSON configuration."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: JSON Configuration")
    print("=" * 70)

    config_path = Path("config/examples/pipeline_config.json")

    if config_path.exists():
        print(f"\nLoading configuration from: {config_path}")
        try:
            params = PipelineParameters.from_json(config_path)
            print("\n✓ Configuration loaded successfully!")
            print(f"  Staging workers: {params.staging.workers}")
            print(f"  Analysis date: {params.analysis.date}")
            print(f"  Analysis polynomial orders: {params.analysis.poly_orders}")
            print(f"  Plotting DPI: {params.plotting.dpi}")
            print(f"  Plotting format: {params.plotting.format}")

            # Save to a new location
            output_path = Path("config/examples/pipeline_config_copy.json")
            params.to_json(output_path)
            print(f"\n✓ Saved copy to: {output_path}")

        except ValidationError as e:
            print(f"\n❌ Validation failed:\n{e}")
        except Exception as e:
            print(f"\n❌ Error: {e}")
    else:
        print(f"\n⚠ Config file not found: {config_path}")


def example_4_helper_methods():
    """Example 4: Use parameter helper methods."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Helper Methods")
    print("=" * 70)

    # Create analysis parameters
    analysis_params = IVAnalysisParameters(
        stage_root=Path("data/02_stage/raw_measurements"),
        date="2025-09-11",
        output_base_dir=Path("data/04_analysis"),
        procedure="IVg",
        poly_orders=[1, 3, 5, 7],
    )

    print("\nAnalysis parameter helper methods:")
    print(f"  Stats directory: {analysis_params.get_stats_dir()}")
    print(f"  Hysteresis directory: {analysis_params.get_hysteresis_dir()}")
    print(f"  Peaks directory: {analysis_params.get_peaks_dir()}")

    # Create plotting parameters
    plotting_params = PlottingParameters(
        output_dir=Path("plots/analysis"),
        figure_width=15.0,
        figure_height=10.0,
        style="publication",
    )

    print("\nPlotting parameter helper methods:")
    print(f"  Figure size: {plotting_params.get_figsize()}")
    print(f"  Style params: {plotting_params.get_style_params()}")


def example_5_pipeline_validation():
    """Example 5: Pipeline-level validation."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Pipeline Validation")
    print("=" * 70)

    print("\n[Attempt] Inconsistent stage_root paths:")
    try:
        PipelineParameters(
            staging=StagingParameters(
                raw_root=Path("data/01_raw"),
                stage_root=Path("data/02_stage/path1"),  # Different!
                procedures_yaml=Path("config/procedures.yml"),
            ),
            analysis=IVAnalysisParameters(
                stage_root=Path("data/02_stage/path2"),  # Different!
                date="2025-09-11",
                output_base_dir=Path("data/04_analysis"),
            ),
            plotting=PlottingParameters(
                output_dir=Path("plots"),
            ),
        )
        print("  ❌ Should have failed!")
    except ValidationError as e:
        print(f"  ✓ Caught cross-parameter validation error:")
        print(f"     {e.errors()[0]['msg']}")


def example_6_programmatic_config():
    """Example 6: Build configuration programmatically."""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Programmatic Configuration")
    print("=" * 70)

    # Build step-by-step
    staging = StagingParameters(
        raw_root=Path("data/01_raw"),
        stage_root=Path("data/02_stage/raw_measurements"),
        procedures_yaml=Path("config/procedures.yml"),
        workers=12,  # High parallelism
        polars_threads=1,
        force=True,  # Overwrite existing
    )

    analysis = IVAnalysisParameters(
        stage_root=staging.stage_root,  # Reference staging output
        date="2025-09-11",
        output_base_dir=Path("data/04_analysis"),
        procedure="IV",
        poly_orders=[1, 3, 5, 7],  # All polynomial orders
        fit_backward=True,
        compute_hysteresis=True,
        analyze_peaks=True,
    )

    plotting = PlottingParameters(
        output_dir=Path("plots") / f"{analysis.date}_{analysis.procedure}",
        dpi=300,  # High quality
        format="png",
        style="publication",
        compact_layout=True,
        show_residuals=True,
    )

    # Combine into pipeline
    try:
        pipeline = PipelineParameters(
            staging=staging,
            analysis=analysis,
            plotting=plotting,
            run_staging=True,
            run_analysis=True,
            run_plotting=True,
        )

        print("\n✓ Pipeline configuration created!")
        print(f"  Staging: {pipeline.staging.workers} workers")
        print(f"  Analysis: {pipeline.analysis.procedure} on {pipeline.analysis.date}")
        print(f"  Plotting: {pipeline.plotting.dpi} DPI, {pipeline.plotting.format} format")
        print(f"  Steps enabled: staging={pipeline.run_staging}, analysis={pipeline.run_analysis}, plotting={pipeline.run_plotting}")

        # Save for later use
        output_file = Path("config/examples/programmatic_config.json")
        pipeline.to_json(output_file)
        print(f"\n✓ Saved to: {output_file}")

    except ValidationError as e:
        print(f"\n❌ Validation failed:\n{e}")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("PYDANTIC PARAMETER MODELS - USAGE EXAMPLES")
    print("=" * 70)

    example_1_create_staging_params()
    example_2_validation_errors()
    example_3_json_config()
    example_4_helper_methods()
    example_5_pipeline_validation()
    example_6_programmatic_config()

    print("\n" + "=" * 70)
    print("✓ All examples completed!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

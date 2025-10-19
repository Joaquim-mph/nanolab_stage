"""
Preprocess Command

Detect and segment voltage sweeps in IV measurements.
"""

import sys
import typer
from pathlib import Path
from typing import Optional

from ..utils.console import console, print_step, print_success, print_error, print_config
from ..utils.logging import setup_rich_logging

def preprocess(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="JSON configuration file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    stage_root: Optional[Path] = typer.Option(
        None,
        "--stage-root",
        help="Directory containing staged Parquet files",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    output_root: Optional[Path] = typer.Option(
        None,
        "--output-root",
        help="Output directory for segmented data",
        file_okay=False,
        dir_okay=True,
    ),
    procedure: str = typer.Option(
        "IV",
        "--procedure",
        "-p",
        help="Procedure type to process",
    ),
    voltage_col: str = typer.Option(
        "Vsd (V)",
        "--voltage-col",
        help="Name of voltage column",
    ),
    dv_threshold: float = typer.Option(
        0.001,
        "--dv-threshold",
        help="Voltage change threshold for segment detection",
        min=0.0,
    ),
    min_segment_points: int = typer.Option(
        5,
        "--min-segment-points",
        help="Minimum points per segment",
        min=1,
    ),
    workers: int = typer.Option(
        8,
        "--workers",
        "-w",
        min=1,
        max=32,
        help="Number of parallel worker processes",
    ),
    polars_threads: int = typer.Option(
        2,
        "--polars-threads",
        "-t",
        min=1,
        max=16,
        help="Number of Polars threads per worker",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Reprocess existing segments",
    ),
):
    """
    Preprocess and segment voltage sweeps.

    \b
    This command:
      • Detects voltage sweep segments (forward/return, positive/negative)
      • Saves each segment to separate Parquet files
      • Adds metadata: segment_id, segment_type, segment_direction
      • Run once per date, read many times (10x performance!)

    \b
    Example using config file:
      nanolab-pipeline preprocess --config config/examples/intermediate_config.json

    \b
    Example using CLI options:
      nanolab-pipeline preprocess \\
        --stage-root data/02_stage/raw_measurements \\
        --output-root data/03_intermediate \\
        --procedure IV \\
        --workers 8
    """
    # Print step header
    print_step(
        2,
        "PREPROCESSING",
        "Detecting and segmenting voltage sweeps"
    )

    # Import required modules
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    try:
        from models.parameters import IntermediateParameters
        from intermediate.IV.iv_preprocessing_script import discover_iv_runs
    except ImportError as e:
        print_error(
            f"Failed to import preprocessing module: {e}",
            "Make sure you're running from the project root directory."
        )
        raise typer.Exit(1)

    # Create parameters from config or CLI options
    try:
        if config:
            console.print(f"[info]Loading configuration from:[/info] [path]{config}[/path]")
            params = IntermediateParameters.from_json(config)
        else:
            # Check required parameters
            if stage_root is None or output_root is None:
                print_error(
                    "Missing required parameters",
                    "Either provide --config or specify both --stage-root and --output-root"
                )
                raise typer.Exit(1)

            params = IntermediateParameters(
                stage_root=stage_root,
                output_root=output_root,
                procedure=procedure,
                voltage_col=voltage_col,
                dv_threshold=dv_threshold,
                min_segment_points=min_segment_points,
                workers=workers,
                polars_threads=polars_threads,
                force=force,
            )
    except Exception as e:
        print_error(
            f"Parameter validation failed: {e}",
            "Check your configuration file or input parameters."
        )
        raise typer.Exit(1)

    # Display configuration
    output_dir = params.get_output_dir()
    print_config({
        "stage_root": params.stage_root,
        "output_dir": output_dir,
        "procedure": params.procedure,
        "voltage_col": params.voltage_col,
        "dv_threshold": params.dv_threshold,
        "min_segment_points": params.min_segment_points,
        "workers": params.workers,
        "polars_threads": params.polars_threads,
        "force": params.force,
    })

    # Set up Rich logging
    logger = setup_rich_logging("INFO")

    # Run preprocessing pipeline with Rich integration
    console.print()
    logger.info("[bold cyan]Starting preprocessing pipeline...[/bold cyan]")
    console.print()

    try:
        # Discover IV runs
        logger.info(f"Scanning for IV runs in [path]{params.stage_root}[/path]...")
        iv_runs = discover_iv_runs(params.stage_root)

        if not iv_runs:
            print_error(
                "No IV sweep runs found",
                f"Check that {params.stage_root} contains proc=IV data"
            )
            raise typer.Exit(1)

        logger.info(f"Found [bold]{len(iv_runs):,}[/bold] IV runs")
        console.print()

        # Use our Rich wrapper for the preprocessing pipeline
        from ..utils.preprocessing_wrapper import run_preprocessing_with_progress

        results = run_preprocessing_with_progress(
            params=params,
            iv_runs=iv_runs,
        )

        # Success message
        total_processed = results["ok"] + results["skipped"]

        if results["error"] == 0 and results["ok"] > 0:
            # All new runs processed successfully
            print_success(
                "Preprocessing completed successfully",
                f"Segmented data saved to: [path]{output_dir}[/path]\n"
                f"Processed: {results['ok']:,} runs\n"
                f"Total segments: {results['total_segments']:,}"
            )
        elif results["skipped"] > 0 and results["ok"] == 0 and results["error"] < len(iv_runs):
            # All runs already exist (skipped) - this is OK!
            console.print()
            console.print(
                "[cyan]All runs already preprocessed (skipped)[/cyan]"
            )
            console.print(f"Segmented data location: [path]{output_dir}[/path]")
            console.print(f"Total runs: {results['skipped']:,}")
            console.print(f"Total segments: {results['total_segments']:,}")
            if results["error"] > 0:
                console.print(f"[yellow]Note: {results['error']} runs had errors[/yellow]")
        elif total_processed > 0:
            # Partial success (some processed, some skipped, some errors)
            console.print()
            console.print(
                f"[yellow]Preprocessing completed with {results['error']} errors[/yellow]"
            )
            console.print(f"Segmented data saved to: [path]{output_dir}[/path]")
            console.print(f"Processed: {results['ok']:,} runs")
            console.print(f"Skipped: {results['skipped']:,} runs")
            console.print(f"Total segments: {results['total_segments']:,}")
        else:
            # Complete failure - all failed
            print_error(
                "All runs failed",
                "Check the error messages above for details"
            )
            raise typer.Exit(1)

    except Exception as e:
        print_error(
            f"Preprocessing failed: {e}",
            "Check the error message above for details."
        )
        if console.is_terminal:
            console.print("\n[dim]Full traceback:[/dim]")
            console.print_exception()
        raise typer.Exit(1)



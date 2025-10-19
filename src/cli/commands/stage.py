"""
Stage Command

Stage raw CSV files to schema-validated Parquet format.
"""

import sys
import typer
from pathlib import Path
from typing import Optional

from ..utils.console import console, print_step, print_success, print_error, print_config
from ..utils.logging import setup_rich_logging

def stage(
    raw_root: Path = typer.Option(
        ...,
        "--raw-root",
        help="Directory containing raw CSV files",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    stage_root: Path = typer.Option(
        ...,
        "--stage-root",
        help="Output directory for staged Parquet files",
        file_okay=False,
        dir_okay=True,
    ),
    procedures_yaml: Path = typer.Option(
        "config/procedures.yml",
        "--procedures-yaml",
        help="YAML file defining measurement procedures",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
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
        help="Overwrite existing Parquet files",
    ),
    only_yaml_data: bool = typer.Option(
        False,
        "--only-yaml-data",
        help="Drop columns not defined in YAML schema",
    ),
    local_tz: str = typer.Option(
        "America/Santiago",
        "--local-tz",
        help="Local timezone for date partitioning",
    ),
):
    """
    Stage raw CSV measurements to Parquet format.

    \b
    This command:
      • Parses structured CSV headers (Procedure/Parameters/Metadata/Data)
      • Validates and casts types using procedures.yml schema
      • Normalizes column names via tolerant matching
      • Derives metadata (run_id, with_light flag, etc.)
      • Writes Parquet files partitioned by proc/date/run_id

    \b
    Example:
      nanolab-pipeline stage \\
        --raw-root data/01_raw \\
        --stage-root data/02_stage/raw_measurements \\
        --workers 8
    """
    # Print step header
    print_step(
        1,
        "STAGING",
        "Converting raw CSV files to Parquet format"
    )

    # Display configuration
    print_config({
        "raw_root": raw_root,
        "stage_root": stage_root,
        "procedures_yaml": procedures_yaml,
        "workers": workers,
        "polars_threads": polars_threads,
        "force": force,
        "only_yaml_data": only_yaml_data,
        "local_tz": local_tz,
    })

    # Import staging function
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "staging"))
    try:
        from models.parameters import StagingParameters
        from staging.stage_raw_measurements import run_staging_pipeline
    except ImportError as e:
        print_error(
            f"Failed to import staging module: {e}",
            "Make sure you're running from the project root directory."
        )
        raise typer.Exit(1)

    # Create parameters
    try:
        params = StagingParameters(
            raw_root=raw_root,
            stage_root=stage_root,
            procedures_yaml=procedures_yaml,
            workers=workers,
            polars_threads=polars_threads,
            force=force,
            only_yaml_data=only_yaml_data,
            local_tz=local_tz,
        )
    except Exception as e:
        print_error(
            f"Parameter validation failed: {e}",
            "Check your input parameters and try again."
        )
        raise typer.Exit(1)

    # Set up Rich logging
    logger = setup_rich_logging("INFO")

    # Run staging pipeline with Rich integration
    console.print()
    logger.info("[bold cyan]Starting staging pipeline...[/bold cyan]")
    console.print()

    try:
        # Import staging utilities
        from pathlib import Path as P
        import glob

        # Find all CSV files
        logger.info(f"Scanning for CSV files in [path]{raw_root}[/path]...")
        csv_pattern = str(raw_root / "**" / "*.csv")
        csvs = [Path(p) for p in glob.glob(csv_pattern, recursive=True)]

        if not csvs:
            print_error(
                "No CSV files found",
                f"Check that {raw_root} contains CSV files"
            )
            raise typer.Exit(1)

        logger.info(f"Found [bold]{len(csvs):,}[/bold] CSV files")

        # Use our Rich wrapper for the staging pipeline
        from ..utils.staging_wrapper import run_staging_with_progress

        results = run_staging_with_progress(
            params=params,
            csvs=csvs,
        )

        # Success message
        total_processed = results["ok"] + results["skipped"]

        if results["rejected"] == 0 and results["ok"] > 0:
            # All new files processed successfully
            print_success(
                "Staging completed successfully",
                f"Staged data saved to: [path]{stage_root}[/path]\n"
                f"Processed: {results['ok']:,} files"
            )
        elif results["skipped"] > 0 and results["ok"] == 0 and results["rejected"] < len(csvs):
            # All files already exist (skipped) - this is OK!
            console.print()
            console.print(
                "[cyan]All files already staged (skipped)[/cyan]"
            )
            console.print(f"Staged data location: [path]{stage_root}[/path]")
            console.print(f"Total files: {results['skipped']:,}")
            if results["rejected"] > 0:
                console.print(f"[yellow]Note: {results['rejected']} files were rejected[/yellow]")
        elif total_processed > 0:
            # Partial success (some processed, some skipped, some rejected)
            console.print()
            console.print(
                f"[yellow]Staging completed with {results['rejected']} rejected files[/yellow]"
            )
            console.print(f"Staged data saved to: [path]{stage_root}[/path]")
            console.print(f"Processed: {results['ok']:,} files")
            console.print(f"Skipped: {results['skipped']:,} files")
        else:
            # Complete failure - all rejected
            print_error(
                "All files were rejected",
                "Check the error messages above for details"
            )
            raise typer.Exit(1)

    except Exception as e:
        print_error(
            f"Staging failed: {e}",
            "Check the error message above for details."
        )
        import traceback
        if console.is_terminal:
            console.print("\n[dim]Full traceback:[/dim]")
            console.print_exception()
        raise typer.Exit(1)



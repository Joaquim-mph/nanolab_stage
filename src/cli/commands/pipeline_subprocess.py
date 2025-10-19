"""
Pipeline Command - Subprocess-Based (Production Grade)

Runs each stage as a separate subprocess to ensure complete isolation.
This is the most reliable approach for sequential pipeline stages.
"""

import subprocess
import sys
from pathlib import Path
from typing import Optional
import typer

from ..utils.console import console, print_success, print_error, print_config
from ..utils.logging import setup_rich_logging
from rich.panel import Panel
from rich.table import Table


def create_pipeline_overview() -> Panel:
    """Create a pipeline overview panel."""
    overview = Table.grid(padding=(0, 2))
    overview.add_column(style="cyan", justify="left")
    overview.add_column(style="white")

    overview.add_row("[bold]Stage 1:[/bold]", "Raw CSV → Staged Parquet")
    overview.add_row("", "Schema validation, partitioning, metadata enrichment")
    overview.add_row("")
    overview.add_row("[bold]Stage 2:[/bold]", "Staged Parquet → Segmented Data")
    overview.add_row("", "Voltage sweep detection, segment classification")

    return Panel(
        overview,
        title="[bold cyan]Nanolab Data Pipeline[/bold cyan]",
        border_style="cyan",
        padding=(1, 2),
    )


def pipeline_subprocess(
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
    output_root: Path = typer.Option(
        ...,
        "--output-root",
        help="Output directory for preprocessed segments",
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
    procedure: str = typer.Option(
        "IV",
        "--procedure",
        "-p",
        help="Procedure type to process",
    ),
    voltage_col: str = typer.Option(
        "Vsd (V)",
        "--voltage-col",
        help="Name of voltage column for segmentation",
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
        help="Overwrite existing files",
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
    Run the full data pipeline using subprocesses for complete isolation.

    This command runs each stage as a separate subprocess to avoid any
    resource conflicts, state pollution, or deadlocks between stages.

    \b
    Example:
      nanolab-pipeline pipeline \\
        --raw-root data/01_raw \\
        --stage-root data/02_stage/raw_measurements \\
        --output-root data/03_intermediate \\
        --workers 8
    """
    logger = setup_rich_logging("INFO")

    # Show pipeline overview
    console.print()
    console.print(create_pipeline_overview())
    console.print()

    # Display configuration
    print_config({
        "raw_root": raw_root,
        "stage_root": stage_root,
        "output_root": output_root,
        "procedure": procedure,
        "workers": workers,
        "polars_threads": polars_threads,
        "force": force,
    })

    # Get the Python executable (same as current process)
    python_exe = sys.executable

    # ========== STAGE 1: STAGING ==========
    console.print()
    console.rule("[bold cyan]Stage 1: Raw CSV → Staged Parquet[/bold cyan]")
    console.print()

    logger.info("Running staging as subprocess...")

    stage_cmd = [
        python_exe,
        "nanolab-pipeline.py",
        "stage",
        "--raw-root", str(raw_root),
        "--stage-root", str(stage_root),
        "--procedures-yaml", str(procedures_yaml),
        "--workers", str(workers),
        "--polars-threads", str(polars_threads),
        "--local-tz", local_tz,
    ]

    if force:
        stage_cmd.append("--force")
    if only_yaml_data:
        stage_cmd.append("--only-yaml-data")

    try:
        result = subprocess.run(
            stage_cmd,
            check=True,
            capture_output=False,  # Let output go to terminal
        )

        logger.info("[green]✓[/green] Stage 1 complete")
        console.print()

    except subprocess.CalledProcessError as e:
        print_error(
            "Stage 1 (staging) failed",
            f"Exit code: {e.returncode}"
        )
        raise typer.Exit(1)

    # ========== STAGE 2: PREPROCESSING ==========
    console.print()
    console.rule("[bold cyan]Stage 2: Staged Parquet → Segments[/bold cyan]")
    console.print()

    logger.info("Running preprocessing as subprocess...")

    preprocess_cmd = [
        python_exe,
        "nanolab-pipeline.py",
        "preprocess",
        "--stage-root", str(stage_root),
        "--output-root", str(output_root),
        "--procedure", procedure,
        "--voltage-col", voltage_col,
        "--dv-threshold", str(dv_threshold),
        "--min-segment-points", str(min_segment_points),
        "--workers", str(workers),
        "--polars-threads", str(polars_threads),
    ]

    if force:
        preprocess_cmd.append("--force")

    try:
        result = subprocess.run(
            preprocess_cmd,
            check=True,
            capture_output=False,  # Let output go to terminal
        )

        logger.info("[green]✓[/green] Stage 2 complete")
        console.print()

    except subprocess.CalledProcessError as e:
        print_error(
            "Stage 2 (preprocessing) failed",
            f"Exit code: {e.returncode}"
        )
        raise typer.Exit(1)

    # ========== FINAL SUMMARY ==========
    console.print()
    console.rule("[bold green]Pipeline Complete[/bold green]")
    console.print()

    print_success(
        "Full pipeline completed successfully",
        f"Staged data: [path]{stage_root}[/path]\n"
        f"Segmented data: [path]{output_root / 'iv_segments'}[/path]"
    )

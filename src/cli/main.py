"""
Nanolab Pipeline CLI - Main Application

Modern command-line interface for the nanolab data processing pipeline.
Built with Typer for CLI framework and Rich for beautiful terminal output.

Usage:
    nanolab-pipeline stage [OPTIONS]      # Stage raw CSV files
    nanolab-pipeline preprocess [OPTIONS]  # Segment voltage sweeps
    nanolab-pipeline --help                # Show help message
"""

import typer
from pathlib import Path
from typing import Optional

# Import command functions
from .commands.stage import stage
from .commands.preprocess import preprocess
from .commands.pipeline_subprocess import pipeline_subprocess as pipeline
from .utils.console import console

# Create main Typer app
app = typer.Typer(
    name="nanolab-pipeline",
    help="🔬 Nanolab Data Processing Pipeline - 4-Layer Architecture",
    add_completion=True,
    rich_markup_mode="rich",
    no_args_is_help=True,
)


def version_callback(value: bool):
    """Print version and exit."""
    if value:
        from . import __version__
        console.print(f"[bold cyan]nanolab-pipeline[/bold cyan] version [green]{__version__}[/green]")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit",
    ),
):
    """
    🔬 Nanolab Data Processing Pipeline

    A modern 4-layer medallion architecture for processing semiconductor
    measurement data from raw CSV files to analysis-ready datasets.

    \b
    Architecture:
      Layer 1: Raw CSV files (data/01_raw/)
      Layer 2: Staged Parquet (data/02_stage/)
      Layer 3: Intermediate segments (data/03_intermediate/)
      Layer 4: Analysis results (data/04_analysis/)

    \b
    Quick Start:
      • Full pipeline:       nanolab-pipeline pipeline --raw-root data/01_raw --stage-root data/02_stage --output-root data/03_intermediate
      • Stage only:          nanolab-pipeline stage --raw-root data/01_raw --stage-root data/02_stage
      • Preprocess only:     nanolab-pipeline preprocess --stage-root data/02_stage --output-root data/03_intermediate

    For detailed help on any command, run:
      nanolab-pipeline <command> --help
    """
    pass


# Register commands
app.command(name="pipeline", help="Run full pipeline: Stage + Preprocess")(pipeline)
app.command(name="stage", help="Stage raw CSV files to Parquet")(stage)
app.command(name="preprocess", help="Preprocess and segment voltage sweeps")(preprocess)


if __name__ == "__main__":
    app()

"""
Preprocessing Pipeline Wrapper with Rich Integration

Wraps the preprocessing pipeline to add Rich progress bars and logging.
"""

import time
import os
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Any, List

from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
)

from .console import console
from .logging import setup_rich_logging, create_results_table

# Import preprocessing functions at module level for multiprocessing compatibility
# Worker processes need to be able to import these functions
# Note: The functions themselves handle path setup, so we just need module-level import
_INTERMEDIATE_PATH = str(Path(__file__).parent.parent.parent / "intermediate" / "IV")
_STAGING_PATH = str(Path(__file__).parent.parent.parent / "staging")

if _INTERMEDIATE_PATH not in sys.path:
    sys.path.insert(0, _INTERMEDIATE_PATH)
if _STAGING_PATH not in sys.path:
    sys.path.insert(0, _STAGING_PATH)

from iv_preprocessing_script import process_iv_run, merge_events_to_manifest
from stage_utils import ensure_dir


def create_preprocessing_results_table(
    runs_processed: int,
    runs_ok: int,
    runs_skipped: int,
    runs_error: int,
    total_segments: int,
    elapsed_time: float = None,
) -> Any:
    """
    Create a results summary table for preprocessing operations.

    Args:
        runs_processed: Total runs processed
        runs_ok: Successfully processed runs
        runs_skipped: Skipped runs (already exist)
        runs_error: Error runs
        total_segments: Total segments created
        elapsed_time: Optional elapsed time in seconds

    Returns:
        Rich Table instance
    """
    from rich.table import Table

    table = Table(title="📊 Preprocessing Results", show_header=False, box=None)
    table.add_column("Metric", style="cyan", width=25)
    table.add_column("Count", style="green", justify="right", width=15)
    table.add_column("Percentage", style="dim", justify="right", width=10)

    # Calculate percentages
    total = runs_processed
    ok_pct = (runs_ok / total * 100) if total > 0 else 0
    skip_pct = (runs_skipped / total * 100) if total > 0 else 0
    error_pct = (runs_error / total * 100) if total > 0 else 0

    # Color code based on status
    ok_style = "bold green" if runs_ok > 0 else "dim"
    skip_style = "yellow" if runs_skipped > 0 else "dim"
    error_style = "bold red" if runs_error > 0 else "dim"

    table.add_row(
        "Total Runs",
        f"[bold]{total:,}[/bold]",
        ""
    )
    table.add_row("", "", "")  # Spacer
    table.add_row(
        "✓ Processed Successfully",
        f"[{ok_style}]{runs_ok:,}[/{ok_style}]",
        f"[dim]{ok_pct:.1f}%[/dim]"
    )
    table.add_row(
        "⊝ Skipped (exists)",
        f"[{skip_style}]{runs_skipped:,}[/{skip_style}]",
        f"[dim]{skip_pct:.1f}%[/dim]"
    )
    table.add_row(
        "✗ Errors",
        f"[{error_style}]{runs_error:,}[/{error_style}]",
        f"[dim]{error_pct:.1f}%[/dim]"
    )

    # Add total segments created
    if total_segments > 0:
        table.add_row("", "", "")  # Spacer
        table.add_row(
            "Total Segments Created",
            f"[bold magenta]{total_segments:,}[/bold magenta]",
            ""
        )

    if elapsed_time is not None:
        table.add_row("", "", "")  # Spacer
        minutes = int(elapsed_time // 60)
        seconds = elapsed_time % 60
        if minutes > 0:
            time_str = f"{minutes}m {seconds:.1f}s"
        else:
            time_str = f"{seconds:.1f}s"
        table.add_row(
            "Elapsed Time",
            f"[bold magenta]{time_str}[/bold magenta]",
            ""
        )

        # Runs per second
        if elapsed_time > 0:
            rps = runs_processed / elapsed_time
            table.add_row(
                "Processing Speed",
                f"[dim]{rps:.1f} runs/s[/dim]",
                ""
            )

    return table


def run_preprocessing_with_progress(
    params,
    iv_runs: List[Path],
) -> Dict[str, int]:
    """
    Run preprocessing pipeline with Rich progress bar.

    Args:
        params: IntermediateParameters instance
        iv_runs: List of IV run Parquet files to process

    Returns:
        Dictionary with counts: {ok, skipped, error, total_segments}
    """
    # Set up logging
    logger = setup_rich_logging("INFO")

    # Track statistics
    ok = 0
    skipped = 0
    error = 0
    total_segments = 0

    total_runs = len(iv_runs)

    # Create progress bar with NEW console instance to avoid interference with staging
    # Using the global console singleton can cause deadlocks when stages run sequentially
    from rich.console import Console
    preprocessing_console = Console()

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
        console=preprocessing_console,  # Use dedicated console
    )

    start_time = time.time()

    # Create output directories
    # Note: preprocessing functions already imported at module level
    output_root = params.get_output_dir()
    ensure_dir(output_root)
    ensure_dir(params.events_dir)
    ensure_dir(params.manifest.parent)

    # Set Polars thread count
    os.environ["POLARS_MAX_THREADS"] = str(params.polars_threads)

    # Use context manager for executor to guarantee cleanup
    # Note: ProcessPoolExecutor creation can hang if previous executors
    # haven't fully released resources. The pipeline.py cleanup should
    # handle this, but using context manager ensures proper cleanup.
    logger.info(f"Initializing worker pool with {params.workers} workers...")

    # Create executor and submit all tasks BEFORE starting progress bar
    # This avoids potential deadlock from nested context managers with Rich
    executor = ProcessPoolExecutor(max_workers=params.workers)
    try:
        logger.info("Worker pool initialized successfully")

        # Submit all runs (no verbose logging)
        futures = {}
        for run_path in iv_runs:
            future = executor.submit(
                process_iv_run,
                run_path,
                output_root,
                params.voltage_col,
                params.dv_threshold,
                params.min_segment_points,
                params.force,
                params.events_dir,
            )
            futures[future] = run_path

        # Start progress bar
        progress.start()
        task = progress.add_task(
            "[cyan]Processing IV runs...",
            total=total_runs
        )

        try:
            for future in as_completed(futures):
                run_path = futures[future]

                try:
                    # Add timeout to prevent infinite hang on individual tasks
                    event = future.result(timeout=60)  # 60 second timeout per task

                    status = event.get("status")
                    if status == "ok":
                        ok += 1
                        seg_count = event.get("segments_detected", 0)
                        total_segments += seg_count
                    elif status == "skipped":
                        skipped += 1
                        seg_count = event.get("segments_detected", 0)
                        total_segments += seg_count
                    else:  # error
                        error += 1

                except TimeoutError:
                    error += 1
                    logger.error(f"Task timeout: {run_path}")

                except Exception as e:
                    error += 1
                    logger.error(f"Task error: {e}")

                # Update progress bar
                progress.update(task, advance=1)

        finally:
            progress.stop()

    finally:
        # Explicitly shutdown executor
        logger.info("Shutting down worker pool...")
        executor.shutdown(wait=True)
        logger.info("Worker pool shut down successfully")

    elapsed_time = time.time() - start_time

    # Merge events into manifest
    try:
        merge_events_to_manifest(params.events_dir, params.manifest)
    except Exception as e:
        logger.warning(f"Failed to merge manifest: {e}")

    # Show results table
    console.print()
    results_table = create_preprocessing_results_table(
        runs_processed=total_runs,
        runs_ok=ok,
        runs_skipped=skipped,
        runs_error=error,
        total_segments=total_segments,
        elapsed_time=elapsed_time,
    )
    console.print(results_table)
    console.print()

    return {
        "ok": ok,
        "skipped": skipped,
        "error": error,
        "total_segments": total_segments,
    }

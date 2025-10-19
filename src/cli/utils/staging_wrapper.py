"""
Staging Pipeline Wrapper with Rich Integration

Wraps the staging pipeline to add Rich progress bars and logging.
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

# Import staging functions at module level for multiprocessing compatibility
# Worker processes need to be able to import these functions
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "staging"))
from stage_raw_measurements import ingest_file_task, get_procs_cached, merge_events_to_manifest
from stage_utils import ensure_dir


def run_staging_with_progress(
    params,
    csvs: List[Path],
) -> Dict[str, int]:
    """
    Run staging pipeline with Rich progress bar.

    Args:
        params: StagingParameters instance
        csvs: List of CSV files to process

    Returns:
        Dictionary with counts: {ok, skipped, rejected, submitted}
    """
    # Set up logging
    logger = setup_rich_logging("INFO")

    # Track statistics
    ok = 0
    skipped = 0
    rejected = 0
    submitted = 0

    total_files = len(csvs)

    # Create progress bar with NEW console instance
    # Using dedicated consoles prevents interference between pipeline stages
    from rich.console import Console
    staging_console = Console()

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
        console=staging_console,  # Use dedicated console
    )

    start_time = time.time()

    # Note: staging functions already imported at module level

    # Create output directories
    ensure_dir(params.stage_root)
    ensure_dir(params.rejects_dir)
    ensure_dir(params.events_dir)
    ensure_dir(params.manifest.parent)

    # Set Polars thread count
    os.environ["POLARS_MAX_THREADS"] = str(params.polars_threads)

    # Validate YAML schema (fail fast)
    _ = get_procs_cached(params.procedures_yaml)

    # Use context manager for executor to guarantee cleanup
    logger.info(f"Initializing worker pool with {params.workers} workers...")

    with ProcessPoolExecutor(max_workers=params.workers) as executor:
        logger.info("Worker pool initialized successfully")

        with progress:
            task = progress.add_task(
                "[cyan]Processing CSV files...",
                total=total_files
            )

            # Submit all files
            futures = {}
            for src in csvs:
                future = executor.submit(
                    ingest_file_task,
                    str(src),
                    str(params.stage_root),
                    str(params.procedures_yaml),
                    params.local_tz,
                    params.force,
                    str(params.events_dir),
                    str(params.rejects_dir),
                    params.only_yaml_data,
                )
                futures[future] = src
                submitted += 1

            # Process results as they complete
            for future in as_completed(futures):
                src = futures[future]
                try:
                    result = future.result()

                    status = result.get("status")
                    if status == "ok":
                        ok += 1
                    elif status == "skipped":
                        skipped += 1
                    else:  # reject
                        rejected += 1

                except Exception as e:
                    rejected += 1

                # Update progress bar
                progress.update(task, advance=1)

    # Executor is now guaranteed to be shutdown and cleaned up

    elapsed_time = time.time() - start_time

    # Merge events into manifest
    try:
        merge_events_to_manifest(params.events_dir, params.manifest)
    except Exception as e:
        logger.warning(f"Failed to merge manifest: {e}")

    # Show results table
    console.print()
    results_table = create_results_table(
        files_processed=total_files,
        files_ok=ok,
        files_skipped=skipped,
        files_rejected=rejected,
        elapsed_time=elapsed_time,
    )
    console.print(results_table)
    console.print()

    return {
        "ok": ok,
        "skipped": skipped,
        "rejected": rejected,
        "submitted": submitted,
    }

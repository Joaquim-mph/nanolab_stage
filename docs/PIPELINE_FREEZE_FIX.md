# Pipeline Freeze Fix - Multiprocessing Import & Pickle Issues

## Problem

When running the full pipeline command:
```bash
python nanolab-pipeline.py pipeline \
  --raw-root data/01_raw \
  --stage-root data/02_stage/raw_measurements \
  --output-root data/03_intermediate \
  --workers 8
```

The process would **freeze at the beginning of Stage 2 (preprocessing)** after successfully completing Stage 1 (staging).

### Symptoms
- Stage 1 completes successfully
- Console shows: "Stage 2: Staged Parquet → Segments"
- Console shows: "Scanning for IV runs..." and "Found 7,636 IV runs"
- Console shows: "Initializing worker pool with 8 workers..." and "Worker pool initialized successfully"
- Progress bar appears but never advances: "⠼ Processing IV runs... ━━━━━━━━━━━━━━━━━━━━━━━━━   0%    0/7636"
- **Process freezes indefinitely** - no progress, no CPU activity
- Running preprocessing alone (not in pipeline) works fine

### Root Cause

The issue was a **pickle/multiprocessing import problem**:

Worker functions (`process_iv_run`, `ingest_file_task`) were imported **inside the wrapper function** instead of at module level:

**Before (preprocessing_wrapper.py:164-167):**
```python
def run_preprocessing_with_progress(params, iv_runs):
    # ... setup code ...

    # Import preprocessing internals
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "intermediate" / "IV"))
    from iv_preprocessing_script import process_iv_run, merge_events_to_manifest

    # Create executor and submit tasks
    with ProcessPoolExecutor(max_workers=params.workers) as executor:
        for run_path in iv_runs:
            future = executor.submit(process_iv_run, ...)  # ← This is the problem!
```

**Why this causes a freeze:**

1. `ProcessPoolExecutor` uses **pickle** to serialize functions and send them to worker processes
2. When you submit `executor.submit(process_iv_run, ...)`, Python tries to pickle `process_iv_run`
3. **Pickle can't serialize locally-imported functions** - it needs module-level imports
4. The main process **hangs waiting** for worker processes to import the function
5. Worker processes **can't find the function** because:
   - They don't have the same `sys.path` modifications
   - They don't know about the local function import
   - They try to import from the wrong module path

6. Result: **Deadlock** - main process waits for workers, workers wait for imports

**Why it worked when running preprocessing alone:**

When running `python src/intermediate/IV/iv_preprocessing_script.py` directly:
- The script imports `process_iv_run` at module level (correct)
- Worker processes can find and import the function
- No pickle issues

## Solution

### Fix #1: Move Imports to Module Level (Primary Fix)

Moved all worker function imports to module level so they can be properly pickled:

**After (preprocessing_wrapper.py:1-33):**
```python
"""
Preprocessing Pipeline Wrapper with Rich Integration
"""

import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
# ... other imports ...

# Import preprocessing functions at module level for multiprocessing compatibility
# Worker processes need to be able to import these functions
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "intermediate" / "IV"))
from iv_preprocessing_script import process_iv_run, merge_events_to_manifest
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "staging"))
from stage_utils import ensure_dir


def run_preprocessing_with_progress(params, iv_runs):
    # ... setup code ...

    # Create executor and submit tasks
    with ProcessPoolExecutor(max_workers=params.workers) as executor:
        for run_path in iv_runs:
            future = executor.submit(process_iv_run, ...)  # ✓ Now works!
```

**Why this works:**
- Functions are imported at module level (when the module loads)
- Pickle can serialize module-level functions by reference
- Worker processes can import from the same module path
- No deadlock

Applied the same fix to **staging_wrapper.py** (lines 28-32).

### Fix #2: Use Context Managers for Executors

Changed both wrappers from manual try/finally to context managers:

**Before:**
```python
executor = ProcessPoolExecutor(max_workers=params.workers)
try:
    # ... work ...
finally:
    executor.shutdown(wait=True)
    del executor
```

**After:**
```python
with ProcessPoolExecutor(max_workers=params.workers) as executor:
    # ... work ...
# Executor guaranteed to be cleaned up here
```

This ensures proper cleanup between stages and prevents resource leaks.

### Fix #3: Enhanced Logging

Added detailed logging to track exactly where the freeze occurs:

**preprocessing_wrapper.py (lines 196-246):**
```python
logger.info(f"Submitting {total_runs} tasks to worker pool...")
for i, run_path in enumerate(iv_runs):
    if i == 0:
        logger.info(f"Submitting first task: {run_path}")
    future = executor.submit(process_iv_run, ...)

    if (i + 1) % 1000 == 0:
        logger.info(f"Submitted {i + 1}/{total_runs} tasks...")

logger.info(f"All {total_runs} tasks submitted, waiting for results...")

# Process results
for future in as_completed(futures):
    event = future.result()
    # ...
    if ok + skipped + error == 1:
        logger.info("First task completed successfully - workers are functioning")
```

This helps diagnose:
- Whether task submission completes
- Whether workers start processing
- Where exactly the freeze occurs

### Fix #4: Extended Cleanup Delay

Increased inter-stage cleanup delay from 0.5s to 2.0s:

**pipeline.py (lines 335-345):**
```python
# ========== CLEANUP BETWEEN STAGES ==========
logger.info("Cleaning up Stage 1 resources...")
gc.collect()
time.sleep(2.0)  # Increased from 0.5s
gc.collect()
```

Ensures OS-level resources are fully released between stages.

## Files Modified

1. **src/cli/utils/staging_wrapper.py**
   - Lines 28-32: Moved imports to module level
   - Lines 85-134: Changed to context manager pattern

2. **src/cli/utils/preprocessing_wrapper.py**
   - Lines 28-33: Moved imports to module level
   - Lines 186-246: Changed to context manager + added logging

3. **src/cli/commands/pipeline.py**
   - Lines 335-345: Extended cleanup delay

## Testing

To verify the fix works:

```bash
# Full pipeline (should no longer freeze)
python nanolab-pipeline.py pipeline \
  --raw-root data/01_raw \
  --stage-root data/02_stage/raw_measurements \
  --output-root data/03_intermediate \
  --workers 8

# You should see:
# ✓ Stage 1 complete
# "Cleaning up Stage 1 resources..."
# "Initializing worker pool with 8 workers..."
# "Worker pool initialized successfully"
# "Submitting 7636 tasks to worker pool..."
# "Submitting first task: ..."
# "Submitted 1000/7636 tasks..."
# ...
# "All 7636 tasks submitted, waiting for results..."
# "First task completed successfully - workers are functioning"
# ⠼ Processing IV runs... [progress bar advances]
```

## Technical Deep Dive

### Python Multiprocessing & Pickle

When you call `executor.submit(func, arg1, arg2)`:

1. **Main process** serializes the function and arguments using `pickle`
2. **Pickled data** is sent to worker process via pipe/queue
3. **Worker process** unpickles the function and arguments
4. **Worker** executes `func(arg1, arg2)`
5. **Result** is pickled and sent back to main process

### Pickle Requirements for Functions

For pickle to serialize a function, it must be:

1. **Defined at module level** (not inside another function)
2. **Importable** by the worker process
3. **Available in the same module path** on both main and worker

### What Went Wrong

Our code violated requirement #1:

```python
def run_preprocessing_with_progress(...):
    from iv_preprocessing_script import process_iv_run  # ← Local import

    executor.submit(process_iv_run, ...)  # ← Pickle fails!
```

Pickle tried to serialize `process_iv_run` but couldn't because:
- It's not a module-level function from the worker's perspective
- The `from iv_preprocessing_script import` is local to `run_preprocessing_with_progress`
- Worker processes don't execute `run_preprocessing_with_progress`, so they never see the import

### How Module-Level Imports Fix It

```python
# Module level (runs when module loads)
from iv_preprocessing_script import process_iv_run

def run_preprocessing_with_progress(...):
    executor.submit(process_iv_run, ...)  # ✓ Pickle succeeds!
```

Now pickle can serialize `process_iv_run` as:
```python
("iv_preprocessing_script", "process_iv_run")  # Module name + function name
```

Worker processes can unpickle by:
```python
import iv_preprocessing_script
func = getattr(iv_preprocessing_script, "process_iv_run")
```

This works because `sys.path.insert()` at module level ensures both main and workers can find `iv_preprocessing_script`.

## Lessons Learned

1. **Always import worker functions at module level** when using multiprocessing
2. **Never import worker functions inside the function that calls `executor.submit()`**
3. **Use context managers for ProcessPoolExecutor** - guarantees cleanup
4. **Add logging to diagnose multiprocessing issues** - they're hard to debug
5. **Test integration scenarios** - bugs often only appear when components combine

## References

- Python docs: [multiprocessing - Programming guidelines](https://docs.python.org/3/library/multiprocessing.html#programming-guidelines)
- Python docs: [pickle - What can be pickled](https://docs.python.org/3/library/pickle.html#what-can-be-pickled-and-unpickled)
- PEP 3148: [futures - ProcessPoolExecutor](https://peps.python.org/pep-3148/)

# CLI Quick Reference

## File Structure (Clean)

```
src/cli/
├── main.py                          # Entry point, command registration
├── commands/                        # CLI commands
│   ├── stage.py                     # Stage: CSV → Parquet
│   ├── preprocess.py                # Preprocess: Parquet → Segments
│   └── pipeline_subprocess.py       # Full pipeline (subprocess-based)
└── utils/                           # Shared utilities
    ├── console.py                   # Rich console + helpers
    ├── logging.py                   # Rich logging
    ├── staging_wrapper.py           # Staging with progress bars
    └── preprocessing_wrapper.py     # Preprocessing with progress bars
```

**Total**: 11 files (3 commands, 4 utilities, 4 package files)

## Commands

### Full Pipeline
```bash
python nanolab-pipeline.py pipeline \
  --raw-root data/01_raw \
  --stage-root data/02_stage/raw_measurements \
  --output-root data/03_intermediate \
  --workers 8
```

### Stage Only
```bash
python nanolab-pipeline.py stage \
  --raw-root data/01_raw \
  --stage-root data/02_stage/raw_measurements \
  --workers 8 \
  --polars-threads 2
```

### Preprocess Only
```bash
python nanolab-pipeline.py preprocess \
  --stage-root data/02_stage/raw_measurements \
  --output-root data/03_intermediate \
  --workers 8 \
  --polars-threads 2
```

### Common Options
- `--workers N`: Number of parallel workers (default: 8)
- `--polars-threads N`: Polars threads per worker (default: 2)
- `--force`: Overwrite existing files
- `--help`: Show detailed help

## Key Components

### Console Helpers (`console.py`)
```python
from cli.utils.console import (
    console,          # Global Rich console
    print_success,    # Green success panel
    print_error,      # Red error panel
    print_config,     # Configuration table
)
```

### Logging (`logging.py`)
```python
from cli.utils.logging import setup_rich_logging

logger = setup_rich_logging("INFO")
logger.info("Processing...")
```

### Wrappers
- `staging_wrapper.py`: Wraps staging with Rich progress bars
- `preprocessing_wrapper.py`: Wraps preprocessing with Rich progress bars

## Design Patterns Used

1. **Subprocess Isolation**: Each pipeline stage runs in separate process
2. **Dedicated Consoles**: Each wrapper has its own Console instance
3. **Context Managers**: Proper resource cleanup with `with` statements
4. **Idempotent Operations**: Skipping existing files = success
5. **Module-Level Imports**: Worker functions imported at top of file

## Files Removed

- ❌ `pipeline.py` - OLD single-process version (replaced by `pipeline_subprocess.py`)
- ❌ `preprocessing_wrapper_v2.py` - Experimental callback version (unused)
- ❌ `progress.py` - Utility functions (not needed, duplicated Rich imports)

## Testing

```bash
# Test idempotency (should succeed both times)
python nanolab-pipeline.py pipeline ... # First run
python nanolab-pipeline.py pipeline ... # Second run (all skipped)

# Test with force flag
python nanolab-pipeline.py pipeline ... --force
```

## Common Issues & Solutions

### Issue: Pipeline freezes at preprocessing
**Solution**: Already fixed! Uses subprocess-based pipeline.

### Issue: "All files were rejected" error
**Solution**: Already fixed! Treats "all skipped" as success.

### Issue: Workers not pickling functions
**Solution**: Already fixed! Functions imported at module level.

## Documentation

- **CLI_ARCHITECTURE.md**: Complete architecture documentation
- **PIPELINE_FREEZE_FIX.md**: Multiprocessing debugging journey
- **CLAUDE.md**: Project overview and data flow

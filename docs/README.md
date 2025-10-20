# Nanolab Pipeline Documentation

This directory contains all project documentation organized by topic.

## Quick Start

- **[QUICK_START.md](QUICK_START.md)** - Get started in 5 minutes

## Core Documentation

### Architecture
- **[4LAYER_COMPLETE.md](4LAYER_COMPLETE.md)** - Complete 4-layer architecture guide
  - Medallion architecture: Raw → Stage → Intermediate → Analysis
  - Data flow and transformations
  - Performance optimizations

- **[ADDING_NEW_PROCEDURES.md](ADDING_NEW_PROCEDURES.md)** - Professional guide for extending the pipeline
  - Methodology for adding new measurement procedures (IVg, ITt, etc.)
  - Step-by-step implementation guide
  - Best practices and examples
  - Testing and validation strategies

- **[PIPELINE_VISUALIZATION.md](PIPELINE_VISUALIZATION.md)** - Visual documentation guide
  - Generate professional Graphviz diagrams
  - 5 different diagram types (overview, detailed, CLI, IV analysis, templates)
  - Multiple output formats (PNG, SVG, PDF)
  - Customization and usage examples

- **[PLOTTING_CONFIGURATION.md](PLOTTING_CONFIGURATION.md)** - Publication-ready plotting guide
  - Matplotlib configuration for scientific figures
  - 4 professional themes (default, prism_rain, minimal, presentation)
  - Colorblind-friendly palettes
  - Helper functions and examples
  - Integration with existing scripts

### CLI Documentation
- **[CLI_ARCHITECTURE.md](CLI_ARCHITECTURE.md)** - Complete CLI technical documentation
  - Technology stack (Typer + Rich)
  - Architecture decisions
  - Component breakdown
  - Design patterns

- **[CLI_QUICK_REFERENCE.md](CLI_QUICK_REFERENCE.md)** - Quick reference guide
  - Common commands
  - File structure
  - Troubleshooting

- **[CLI_IMPLEMENTATION_SUMMARY.md](CLI_IMPLEMENTATION_SUMMARY.md)** - Implementation summary
  - What was built
  - Key achievements
  - Performance metrics

### Technical Deep Dives
- **[PIPELINE_FREEZE_FIX.md](PIPELINE_FREEZE_FIX.md)** - Multiprocessing debugging journey
  - Problem diagnosis
  - Root causes (pickle, Rich conflicts, ProcessPoolExecutor)
  - Solutions implemented
  - Lessons learned

## Historical Documentation

See [archive/](archive/) for historical milestone documents and superseded documentation.

## Main Project Files

Located in the root directory:
- **[README.md](../README.md)** - Main project README
- **[CLAUDE.md](../CLAUDE.md)** - Project instructions for Claude Code
- **[nanolab-pipeline.py](../nanolab-pipeline.py)** - CLI entry point

## Directory Structure

```
nanolab_stage/
├── README.md                    # Main project README
├── CLAUDE.md                    # Claude instructions
├── nanolab-pipeline.py          # CLI entry point
│
├── docs/                        # Documentation (you are here)
│   ├── README.md                # This file
│   ├── QUICK_START.md           # Quick start guide
│   ├── 4LAYER_COMPLETE.md       # Architecture guide
│   ├── CLI_ARCHITECTURE.md      # CLI technical docs
│   ├── CLI_QUICK_REFERENCE.md   # CLI reference
│   ├── CLI_IMPLEMENTATION_SUMMARY.md
│   ├── PIPELINE_FREEZE_FIX.md   # Debugging history
│   └── archive/                 # Historical docs
│
├── src/                         # Source code
│   ├── cli/                     # CLI implementation
│   ├── staging/                 # Layer 2: Staging
│   ├── intermediate/            # Layer 3: Preprocessing
│   ├── analysis/                # Layer 4: Analysis
│   ├── ploting/                 # Visualization
│   └── models/                  # Pydantic models
│
├── scripts/                     # Analysis scripts
│   └── explore_mean_traces.py
│
├── config/                      # Configuration
│   └── procedures.yml           # Schema definitions
│
└── data/                        # Data directories
    ├── 01_raw/                  # Layer 1: Raw CSV
    ├── 02_stage/                # Layer 2: Staged Parquet
    ├── 03_intermediate/         # Layer 3: Preprocessed
    └── 04_analysis/             # Layer 4: Analysis results
```

## Documentation by Use Case

### I want to...

#### Run the pipeline
→ Start with [QUICK_START.md](QUICK_START.md)

#### Understand the architecture
→ Read [4LAYER_COMPLETE.md](4LAYER_COMPLETE.md)

#### Add a new measurement procedure
→ Follow [ADDING_NEW_PROCEDURES.md](ADDING_NEW_PROCEDURES.md)

#### Use the CLI
→ Check [CLI_QUICK_REFERENCE.md](CLI_QUICK_REFERENCE.md)

#### Debug multiprocessing issues
→ See [PIPELINE_FREEZE_FIX.md](PIPELINE_FREEZE_FIX.md)

#### Understand CLI implementation
→ Read [CLI_ARCHITECTURE.md](CLI_ARCHITECTURE.md)

#### Generate pipeline diagrams
→ Follow [PIPELINE_VISUALIZATION.md](PIPELINE_VISUALIZATION.md)

#### Create publication-ready plots
→ Follow [PLOTTING_CONFIGURATION.md](PLOTTING_CONFIGURATION.md)

#### See what was accomplished
→ Review [CLI_IMPLEMENTATION_SUMMARY.md](CLI_IMPLEMENTATION_SUMMARY.md)

## Contributing

When adding new documentation:
1. Place in `docs/` if current and relevant
2. Place in `docs/archive/` if historical or superseded
3. Update this README with links
4. Use clear, descriptive filenames

## Version History

- **2025-10-19**: CLI implementation complete, documentation reorganized
- **2025-10-XX**: 4-layer architecture implemented
- **Earlier**: Initial development

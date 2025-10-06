# Intermediate Processing Layer

This directory contains procedure-specific intermediate processing scripts that operate between the staging layer (`src/staging/`) and the analysis layer (`src/analysis/`).

## Structure

Each procedure type has its own subdirectory:

- **`IV/`**: Current-voltage sweep preprocessing (segment detection, filtering, fitting)
- **`IVg/`**: Gate voltage sweep preprocessing (future)

## Data Flow

```
data/02_stage/raw_measurements/  (staged parquet files)
    ↓
src/intermediate/{procedure}/    (procedure-specific transformations)
    ↓
data/04_analysis/                (analysis-ready datasets)
    ↓
src/analysis/                    (statistics, aggregation, plotting)
```

## Purpose

Intermediate scripts handle:
- Procedure-specific data transformations
- Segment detection and classification
- Quality filtering
- Feature extraction
- Intermediate data validation

This separation keeps the staging layer generic (schema validation, partitioning) while allowing procedure-specific logic to live in dedicated modules.

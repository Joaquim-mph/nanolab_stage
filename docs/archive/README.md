# Documentation Archive

This directory contains outdated documentation from previous architecture versions.

## Archived Files

- **`ANALYSIS_4LAYER_STATUS.md`** - Partial migration status (superseded by `4LAYER_COMPLETE.md`)
- **`PIPELINE_USAGE.md`** - Old pipeline usage docs (superseded by `QUICK_START.md`)

## Current Documentation

For up-to-date documentation, see the root directory:

- **`4LAYER_COMPLETE.md`** - Complete 4-layer architecture guide (CURRENT)
- **`QUICK_START.md`** - Quick start guide for 4-layer pipeline
- **`CLAUDE.md`** - Complete implementation reference
- **`FOUR_LAYER_ARCHITECTURE.md`** - Architecture design details
- **`PYDANTIC_MIGRATION.md`** - Pydantic v2 parameter documentation

## Why Archived?

These files document the **3-layer architecture** which has been deprecated. The current codebase uses a **4-layer architecture** with pre-segmented intermediate data for better performance and maintainability.

**Key change:** Analysis scripts now require pre-segmented intermediate data instead of performing segment detection at analysis time.

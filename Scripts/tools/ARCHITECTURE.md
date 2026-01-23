# Architecture: MagNet Data Processing

This document captures the high-level design decisions and architectural principles for the MagNet Codebase.

## Core Principles

1.  **Global Normalization**: All machine learning models must operate on globally normalized data (computed across all experiments and time steps) to maintain consistent scaling.
2.  **Modular Physics**: Physical calculations (Flux Density $B$, Magnetizing Force $H$, Power Loss) are separated from the data loading and normalization logic.
3.  **Config-Driven**: Model inputs, targets, and normalization methods should be defined in `config.yaml`.
4.  **Living Architecture**: The system map (`docs/system_map.mmd`) is the single source of truth for code structure and must be auto-generated after every code change.

## Module Responsibilities

### `src.data.loader`
-   **Responsibility**: Handle low-level file I/O (.mat, .h5).
-   **Constraint**: Must return raw numpy arrays or dictionaries. No physical logic.

### `src.data.preprocessing`
-   **Responsibility**: Transform raw data into model-ready features.
-   **Key Logic**: 
    -   Computes B, H, Loss.
    -   Applies `normalize_data` using Global Statistics.
    -   Splits Train/Test.

### `src.utils.visualization`
-   **Responsibility**: Provide stateless plotting functions for verification and analysis.
-   **Constraint**: Should not depend on the heavier `src.data` modules to avoid circular imports.

## Data Flow
`Raw .mat File` -> `loader.load_full_dataset` -> `preprocessing.process_magnet_dataset` -> `Model`

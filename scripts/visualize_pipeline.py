#!/usr/bin/env python3
"""
Pipeline Visualization Script

Creates visual documentation of the Nanolab 4-layer data pipeline architecture
using Graphviz. Generates both high-level overviews and detailed data flow diagrams.

Usage:
    python scripts/visualize_pipeline.py --output-dir docs/images
    python scripts/visualize_pipeline.py --format png --format svg
    python scripts/visualize_pipeline.py --diagram all

Outputs:
    - pipeline_overview.{png,svg,pdf} - High-level 4-layer architecture
    - pipeline_detailed.{png,svg,pdf} - Detailed data flow with file formats
    - pipeline_cli.{png,svg,pdf} - CLI command flow
    - pipeline_iv_analysis.{png,svg,pdf} - Complete IV analysis workflow
"""

import argparse
from pathlib import Path
from typing import List
import sys

try:
    import graphviz
except ImportError:
    print("Error: graphviz package not installed")
    print("Install with: pip install graphviz")
    print("Also ensure Graphviz system package is installed:")
    print("  Ubuntu/Debian: sudo apt-get install graphviz")
    print("  macOS: brew install graphviz")
    print("  Windows: Download from https://graphviz.org/download/")
    sys.exit(1)


# Color scheme (professional, colorblind-friendly)
COLORS = {
    "layer1": "#E8F4F8",  # Light blue - Raw data
    "layer2": "#B8E6F0",  # Medium blue - Staged
    "layer3": "#88D8E8",  # Darker blue - Intermediate
    "layer4": "#58CAE0",  # Deep blue - Analysis
    "layer5": "#28BCD8",  # Deepest blue - Visualization
    "cli": "#FFE0B2",     # Light orange - CLI
    "script": "#FFF9C4",  # Light yellow - Scripts
    "config": "#E1BEE7",  # Light purple - Config
    "edge": "#546E7A",    # Blue-grey - Arrows
}


def create_overview_diagram(output_dir: Path, formats: List[str]):
    """
    Create high-level 4-layer architecture overview.

    Shows:
    - 4 main layers
    - Data flow direction
    - Key transformations
    """
    dot = graphviz.Digraph(
        name='pipeline_overview',
        comment='Nanolab Pipeline - 4-Layer Architecture',
        format='png',
    )

    # Graph attributes
    dot.attr(
        rankdir='TB',  # Top to bottom
        splines='ortho',  # Orthogonal edges
        nodesep='0.8',
        ranksep='1.2',
        fontname='Arial',
        fontsize='12',
    )

    # Node defaults
    dot.attr('node',
        shape='box',
        style='filled,rounded',
        fontname='Arial',
        fontsize='11',
        margin='0.3,0.2',
        width='3.5',
        height='0.8',
    )

    # Edge defaults
    dot.attr('edge',
        color=COLORS['edge'],
        penwidth='2.0',
        fontname='Arial',
        fontsize='10',
        arrowsize='0.8',
    )

    # Layer 1: Raw
    with dot.subgraph(name='cluster_0') as c:
        c.attr(label='Layer 1: Raw Data', style='filled', color='#E0E0E0')
        c.node('raw',
               'Raw CSV Files\n\n'
               '• Structured headers\n'
               '• Procedure metadata\n'
               '• Time-series/sweep data',
               fillcolor=COLORS['layer1'])

    # Layer 2: Stage
    with dot.subgraph(name='cluster_1') as c:
        c.attr(label='Layer 2: Staged (Bronze)', style='filled', color='#E0E0E0')
        c.node('staged',
               'Staged Parquet Files\n\n'
               '• Schema-validated\n'
               '• Type-cast columns\n'
               '• Partitioned: proc/date/run_id',
               fillcolor=COLORS['layer2'])

    # Layer 3: Intermediate
    with dot.subgraph(name='cluster_2') as c:
        c.attr(label='Layer 3: Intermediate (Silver)', style='filled', color='#E0E0E0')
        c.node('intermediate',
               'Preprocessed Segments\n\n'
               '• Voltage sweep detection\n'
               '• Segment classification\n'
               '• Analysis-ready format',
               fillcolor=COLORS['layer3'])

    # Layer 4: Analysis
    with dot.subgraph(name='cluster_3') as c:
        c.attr(label='Layer 4: Analysis (Gold)', style='filled', color='#E0E0E0')
        c.node('analysis',
               'Analysis Results\n\n'
               '• Statistics (mean, std)\n'
               '• Polynomial fits\n'
               '• Derived quantities',
               fillcolor=COLORS['layer4'])

    # Layer 5: Visualization
    with dot.subgraph(name='cluster_4') as c:
        c.attr(label='Output: Visualization', style='filled', color='#E0E0E0')
        c.node('plots',
               'Publication Figures\n\n'
               '• Transfer characteristics\n'
               '• Hysteresis analysis\n'
               '• Multi-panel figures',
               fillcolor=COLORS['layer5'])

    # Data flow
    dot.edge('raw', 'staged', label='  Staging\n  (CLI: stage)  ')
    dot.edge('staged', 'intermediate', label='  Preprocessing\n  (CLI: preprocess)  ')
    dot.edge('intermediate', 'analysis', label='  Analysis\n  (Python scripts)  ')
    dot.edge('analysis', 'plots', label='  Plotting\n  (Python scripts)  ')

    # Add legend
    with dot.subgraph(name='cluster_legend') as c:
        c.attr(label='Key Characteristics', style='filled', color='#F5F5F5')
        c.node('legend',
               '✓ Idempotent: Safe to rerun any stage\n'
               '✓ Incremental: Process only new data\n'
               '✓ Parallel: Multi-core processing\n'
               '✓ Validated: Pydantic type checking',
               shape='note',
               fillcolor='#FFFDE7',
               fontsize='10')

    # Render in all requested formats
    for fmt in formats:
        output_path = output_dir / f'pipeline_overview.{fmt}'
        dot.format = fmt
        dot.render(str(output_path.with_suffix('')), cleanup=True)
        print(f"Created: {output_path}")


def create_detailed_diagram(output_dir: Path, formats: List[str]):
    """
    Create detailed data flow diagram with file formats and schemas.

    Shows:
    - Specific file formats
    - Partitioning schemes
    - Column transformations
    - Metadata enrichment
    """
    dot = graphviz.Digraph(
        name='pipeline_detailed',
        comment='Nanolab Pipeline - Detailed Data Flow',
        format='png',
    )

    dot.attr(
        rankdir='LR',  # Left to right for more detail
        splines='polyline',
        nodesep='0.6',
        ranksep='1.5',
        fontname='Arial',
        fontsize='11',
    )

    dot.attr('node',
        shape='box',
        style='filled',
        fontname='Courier',
        fontsize='9',
        margin='0.2,0.1',
    )

    dot.attr('edge',
        color=COLORS['edge'],
        penwidth='1.5',
        fontname='Arial',
        fontsize='9',
    )

    # Raw CSV
    dot.node('csv',
             'Raw CSV\n\n'
             '# Procedure: IV\n'
             '# Parameters:\n'
             '  Chip number: 71\n'
             '  VSD end: 8.0\n'
             '# Data:\n'
             '  Vsd (V), I (A)',
             fillcolor=COLORS['layer1'],
             shape='note')

    # Staging process
    dot.node('staging',
             'Staging Process\n\n'
             '• Parse headers\n'
             '• Validate schema\n'
             '• Normalize columns\n'
             '• Type casting\n'
             '• Generate run_id',
             fillcolor=COLORS['cli'],
             shape='component')

    # Staged Parquet
    dot.node('parquet',
             'Staged Parquet\n\n'
             'Partitioned:\n'
             'proc=IV/\n'
             '  date=2025-10-18/\n'
             '    run_id=abc123/\n'
             '      part-000.parquet\n\n'
             'Schema:\n'
             '  Vsd (V): float64\n'
             '  I (A): float64\n'
             '  run_id: str\n'
             '  with_light: bool',
             fillcolor=COLORS['layer2'],
             shape='cylinder')

    # Preprocessing
    dot.node('preprocess',
             'Preprocessing\n\n'
             '• Detect segments\n'
             '• Classify direction\n'
             '• Label quadrants\n'
             '• Add metadata',
             fillcolor=COLORS['cli'],
             shape='component')

    # Segmented data
    dot.node('segments',
             'Segmented Parquet\n\n'
             'Partitioned:\n'
             'proc=IV/\n'
             '  date=2025-10-18/\n'
             '    run_id=abc123/\n'
             '      segment=0/\n'
             '        part-000.parquet\n\n'
             'Added columns:\n'
             '  segment_id: int\n'
             '  segment_type: str\n'
             '  segment_direction: str',
             fillcolor=COLORS['layer3'],
             shape='cylinder')

    # Analysis
    dot.node('analyze',
             'Analysis\n\n'
             '• Group by voltage\n'
             '• Compute statistics\n'
             '• Fit polynomials\n'
             '• Calculate hysteresis',
             fillcolor=COLORS['script'],
             shape='component')

    # Results
    dot.node('results',
             'Analysis Results\n\n'
             'CSV files:\n'
             '  forward_stats.csv\n'
             '  return_stats.csv\n'
             '  hysteresis.csv\n\n'
             'JSON:\n'
             '  fit_parameters.json\n'
             '  device_params.json',
             fillcolor=COLORS['layer4'],
             shape='folder')

    # Plotting
    dot.node('plot',
             'Visualization\n\n'
             '• Load statistics\n'
             '• Create figures\n'
             '• Add annotations\n'
             '• Export high-DPI',
             fillcolor=COLORS['script'],
             shape='component')

    # Final output
    dot.node('figures',
             'Publication Figures\n\n'
             'transfer_char.png\n'
             'hysteresis.png\n'
             'all_ranges.png\n\n'
             'Format: PNG/PDF\n'
             'DPI: 300+',
             fillcolor=COLORS['layer5'],
             shape='note')

    # Data flow
    dot.edge('csv', 'staging')
    dot.edge('staging', 'parquet', label='Schema validation')
    dot.edge('parquet', 'preprocess')
    dot.edge('preprocess', 'segments', label='Segment detection')
    dot.edge('segments', 'analyze')
    dot.edge('analyze', 'results', label='Aggregation')
    dot.edge('results', 'plot')
    dot.edge('plot', 'figures', label='Matplotlib')

    # Render
    for fmt in formats:
        output_path = output_dir / f'pipeline_detailed.{fmt}'
        dot.format = fmt
        dot.render(str(output_path.with_suffix('')), cleanup=True)
        print(f"Created: {output_path}")


def create_cli_diagram(output_dir: Path, formats: List[str]):
    """
    Create CLI command flow diagram.

    Shows:
    - Available CLI commands
    - Command relationships
    - Configuration options
    - Entry points
    """
    dot = graphviz.Digraph(
        name='pipeline_cli',
        comment='Nanolab Pipeline - CLI Architecture',
        format='png',
    )

    dot.attr(
        rankdir='TB',
        splines='ortho',
        nodesep='0.7',
        ranksep='0.8',
        fontname='Arial',
        fontsize='11',
    )

    dot.attr('node',
        shape='box',
        style='filled,rounded',
        fontname='Arial',
        fontsize='10',
        margin='0.25,0.15',
    )

    dot.attr('edge',
        color=COLORS['edge'],
        penwidth='1.5',
        fontname='Arial',
        fontsize='9',
    )

    # Entry point
    dot.node('entry',
             'nanolab-pipeline.py\n\n'
             'Typer CLI Application',
             fillcolor=COLORS['cli'],
             shape='box',
             style='filled,rounded,bold',
             penwidth='2.5')

    # Commands
    with dot.subgraph(name='cluster_commands') as c:
        c.attr(label='Available Commands', style='filled', color='#F0F0F0')

        c.node('cmd_pipeline',
               'pipeline\n\n'
               'Run full pipeline\n'
               '(subprocess-based)',
               fillcolor='#C8E6C9')

        c.node('cmd_stage',
               'stage\n\n'
               'Stage raw CSV\n'
               'to Parquet',
               fillcolor='#BBDEFB')

        c.node('cmd_preprocess',
               'preprocess\n\n'
               'Segment voltage\n'
               'sweeps',
               fillcolor='#B2EBF2')

    # Default paths
    with dot.subgraph(name='cluster_defaults') as c:
        c.attr(label='Default Paths', style='filled', color='#F5F5F5')

        c.node('defaults',
               'data/01_raw/\n'
               'data/02_stage/raw_measurements/\n'
               'data/03_intermediate/iv_segments/\n'
               'config/procedures.yml',
               fillcolor='#FFF9C4',
               shape='note',
               fontname='Courier',
               fontsize='9')

    # Config options
    with dot.subgraph(name='cluster_config') as c:
        c.attr(label='Configuration', style='filled', color='#F5F5F5')

        c.node('config',
               'Options:\n'
               '• --workers N (default: 8)\n'
               '• --polars-threads N (default: 2)\n'
               '• --procedure NAME (default: IV)\n'
               '• --force (overwrite existing)\n'
               '• --config PATH (JSON config)',
               fillcolor='#E1BEE7',
               shape='note',
               fontsize='9')

    # Subprocesses (for pipeline command)
    with dot.subgraph(name='cluster_subprocess') as c:
        c.attr(label='Pipeline Command (Subprocess Isolation)',
               style='filled,dashed',
               color='#FFE0B2')

        c.node('sub1',
               'Subprocess 1:\n'
               'nanolab-pipeline stage',
               fillcolor='#FFE0B2',
               shape='component',
               fontsize='9')

        c.node('sub2',
               'Subprocess 2:\n'
               'nanolab-pipeline preprocess',
               fillcolor='#FFE0B2',
               shape='component',
               fontsize='9')

    # Flow
    dot.edge('entry', 'cmd_pipeline', label='Full pipeline')
    dot.edge('entry', 'cmd_stage', label='Stage only')
    dot.edge('entry', 'cmd_preprocess', label='Preprocess only')

    dot.edge('cmd_pipeline', 'sub1', label='spawns', style='dashed')
    dot.edge('sub1', 'sub2', label='then spawns', style='dashed')

    dot.edge('defaults', 'entry', label='reads', style='dotted', constraint='false')
    dot.edge('config', 'entry', label='configures', style='dotted', constraint='false')

    # Render
    for fmt in formats:
        output_path = output_dir / f'pipeline_cli.{fmt}'
        dot.format = fmt
        dot.render(str(output_path.with_suffix('')), cleanup=True)
        print(f"Created: {output_path}")


def create_iv_analysis_diagram(output_dir: Path, formats: List[str]):
    """
    Create complete IV analysis workflow diagram.

    Shows:
    - Full IV analysis pipeline
    - Analysis scripts
    - Data dependencies
    - Output files
    """
    dot = graphviz.Digraph(
        name='pipeline_iv_analysis',
        comment='IV Analysis Complete Workflow',
        format='png',
    )

    dot.attr(
        rankdir='TB',
        splines='ortho',
        nodesep='0.8',
        ranksep='1.0',
        fontname='Arial',
        fontsize='11',
    )

    dot.attr('node',
        shape='box',
        style='filled',
        fontname='Arial',
        fontsize='10',
        margin='0.25,0.15',
    )

    dot.attr('edge',
        color=COLORS['edge'],
        penwidth='1.5',
        fontname='Arial',
        fontsize='9',
    )

    # Input
    dot.node('iv_input',
             'Input: Segmented IV Data\n\n'
             'data/03_intermediate/iv_segments/\n'
             'proc=IV/date=2025-10-18/',
             fillcolor=COLORS['layer3'],
             shape='folder')

    # Analysis scripts
    with dot.subgraph(name='cluster_analysis') as c:
        c.attr(label='Analysis Pipeline (Layer 4)', style='filled', color='#E8F5E9')

        c.node('agg',
               'aggregate_iv_stats.py\n\n'
               '• Load segmented data\n'
               '• Group by voltage\n'
               '• Compute mean, std\n'
               '• Fit polynomials (1,3,5,7)',
               fillcolor=COLORS['script'])

        c.node('hyst',
               'compute_hysteresis.py\n\n'
               '• Load forward/return stats\n'
               '• Align voltages\n'
               '• Compute ΔI = I_fwd - I_ret\n'
               '• Save hysteresis curves',
               fillcolor=COLORS['script'])

        c.node('peaks',
               'analyze_hysteresis_peaks.py\n\n'
               '• Find max hysteresis\n'
               '• Locate voltage peaks\n'
               '• Compute peak statistics\n'
               '• Generate summary',
               fillcolor=COLORS['script'])

    # Intermediate outputs
    with dot.subgraph(name='cluster_stats') as c:
        c.attr(label='Analysis Results', style='filled', color='#E3F2FD')

        c.node('stats_out',
               'iv_stats/\n\n'
               'forward_vmax8p0V.csv\n'
               'return_with_fit_vmax8p0V.csv\n'
               'fit_parameters.json',
               fillcolor=COLORS['layer4'],
               shape='folder',
               fontsize='9')

        c.node('hyst_out',
               'hysteresis/\n\n'
               'hysteresis_vmax8p0V.csv\n'
               'hysteresis_summary.csv',
               fillcolor=COLORS['layer4'],
               shape='folder',
               fontsize='9')

        c.node('peaks_out',
               'hysteresis_peaks/\n\n'
               'peak_analysis.csv\n'
               'peak_locations.json',
               fillcolor=COLORS['layer4'],
               shape='folder',
               fontsize='9')

    # Visualization
    with dot.subgraph(name='cluster_viz') as c:
        c.attr(label='Visualization', style='filled', color='#FFF8E1')

        c.node('plot_compare',
               'compare_polynomial_orders.py\n\n'
               '• Load hysteresis data\n'
               '• Create 8-subplot figure\n'
               '• Plot all voltage ranges\n'
               '• Export publication figure',
               fillcolor=COLORS['script'])

        c.node('plot_explore',
               'explore_hysteresis.py\n\n'
               '• Statistical exploration\n'
               '• Peak distributions\n'
               '• Multi-panel analysis',
               fillcolor=COLORS['script'])

    # Final output
    dot.node('figures_out',
             'Publication Figures\n\n'
             'plots/\n'
             '  all_ranges_all_polynomials.png\n'
             '  hysteresis_with_peaks.png\n'
             '  statistical_overview.png',
             fillcolor=COLORS['layer5'],
             shape='note')

    # Flow
    dot.edge('iv_input', 'agg')
    dot.edge('agg', 'stats_out')
    dot.edge('stats_out', 'hyst')
    dot.edge('hyst', 'hyst_out')
    dot.edge('hyst_out', 'peaks')
    dot.edge('peaks', 'peaks_out')

    # Plotting dependencies
    dot.edge('hyst_out', 'plot_compare', style='dashed', label='reads')
    dot.edge('peaks_out', 'plot_compare', style='dashed', label='reads')
    dot.edge('stats_out', 'plot_explore', style='dashed', label='reads')
    dot.edge('hyst_out', 'plot_explore', style='dashed', label='reads')

    dot.edge('plot_compare', 'figures_out')
    dot.edge('plot_explore', 'figures_out')

    # Add workflow note
    dot.node('note',
             'Run Order:\n'
             '1. CLI pipeline (Layers 1-3)\n'
             '2. aggregate_iv_stats.py\n'
             '3. compute_hysteresis.py\n'
             '4. analyze_hysteresis_peaks.py\n'
             '5. Visualization scripts',
             shape='note',
             fillcolor='#FFFDE7',
             fontsize='9',
             style='filled')

    # Render
    for fmt in formats:
        output_path = output_dir / f'pipeline_iv_analysis.{fmt}'
        dot.format = fmt
        dot.render(str(output_path.with_suffix('')), cleanup=True)
        print(f"Created: {output_path}")


def create_procedure_template_diagram(output_dir: Path, formats: List[str]):
    """
    Create template diagram for adding new procedures.

    Shows the steps to implement a new measurement type.
    """
    dot = graphviz.Digraph(
        name='pipeline_procedure_template',
        comment='Adding New Procedures - Template',
        format='png',
    )

    dot.attr(
        rankdir='TB',
        splines='ortho',
        nodesep='0.6',
        ranksep='0.9',
        fontname='Arial',
        fontsize='11',
    )

    dot.attr('node',
        shape='box',
        style='filled,rounded',
        fontname='Arial',
        fontsize='10',
        margin='0.3,0.2',
    )

    dot.attr('edge',
        color=COLORS['edge'],
        penwidth='2.0',
        fontname='Arial',
        fontsize='9',
    )

    # Steps
    dot.node('step1',
             'Step 1: Define YAML Schema\n\n'
             'config/procedures.yml\n'
             'Add procedure specification',
             fillcolor='#E8EAF6',
             shape='box',
             style='filled,rounded')

    dot.node('step2',
             'Step 2: Create Pydantic Models\n\n'
             'src/models/parameters.py\n'
             'Add validation classes',
             fillcolor='#F3E5F5',
             shape='box',
             style='filled,rounded')

    dot.node('step3',
             'Step 3: Implement Preprocessing\n\n'
             'src/intermediate/{Proc}/\n'
             'Segment detection logic',
             fillcolor='#E0F2F1',
             shape='box',
             style='filled,rounded')

    dot.node('step4',
             'Step 4: Implement Analysis\n\n'
             'src/analysis/{Proc}/\n'
             'Statistics, fits, derived quantities',
             fillcolor='#E8F5E9',
             shape='box',
             style='filled,rounded')

    dot.node('step5',
             'Step 5: Add Visualization\n\n'
             'src/ploting/{Proc}/\n'
             'Publication-ready figures',
             fillcolor='#FFF3E0',
             shape='box',
             style='filled,rounded')

    dot.node('step6',
             'Step 6: Integrate with CLI\n\n'
             'src/cli/\n'
             'Update wrappers, add commands',
             fillcolor='#FCE4EC',
             shape='box',
             style='filled,rounded')

    dot.node('step7',
             'Step 7: Test & Document\n\n'
             'tests/ + docs/\n'
             'Unit tests, integration tests, guides',
             fillcolor='#F1F8E9',
             shape='box',
             style='filled,rounded')

    # Flow
    dot.edge('step1', 'step2', label='1')
    dot.edge('step2', 'step3', label='2')
    dot.edge('step3', 'step4', label='3')
    dot.edge('step4', 'step5', label='4')
    dot.edge('step5', 'step6', label='5')
    dot.edge('step6', 'step7', label='6')

    # Note
    dot.node('ref',
             'See: docs/ADDING_NEW_PROCEDURES.md\n'
             'for complete implementation guide',
             shape='note',
             fillcolor='#FFFDE7',
             fontsize='9',
             style='filled')

    # Render
    for fmt in formats:
        output_path = output_dir / f'pipeline_procedure_template.{fmt}'
        dot.format = fmt
        dot.render(str(output_path.with_suffix('')), cleanup=True)
        print(f"Created: {output_path}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Generate visual documentation of the Nanolab pipeline architecture',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all diagrams in PNG format
  python scripts/visualize_pipeline.py

  # Generate specific diagram in multiple formats
  python scripts/visualize_pipeline.py --diagram overview --format png --format svg --format pdf

  # Save to custom directory
  python scripts/visualize_pipeline.py --output-dir docs/images

  # Generate all diagrams in all formats
  python scripts/visualize_pipeline.py --diagram all --format png --format svg
        """
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('docs/images'),
        help='Output directory for generated diagrams (default: docs/images)'
    )

    parser.add_argument(
        '--format',
        action='append',
        choices=['png', 'svg', 'pdf', 'dot'],
        default=None,
        help='Output format(s) (can specify multiple times, default: png)'
    )

    parser.add_argument(
        '--diagram',
        choices=['overview', 'detailed', 'cli', 'iv_analysis', 'procedure_template', 'all'],
        default='all',
        help='Which diagram(s) to generate (default: all)'
    )

    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='DPI for raster formats (default: 300)'
    )

    args = parser.parse_args()

    # Set default format if none specified
    if args.format is None:
        formats = ['png']
    else:
        formats = args.format

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Nanolab Pipeline Visualization")
    print(f"{'='*70}\n")
    print(f"Output directory: {args.output_dir}")
    print(f"Formats: {', '.join(formats)}")
    print(f"Diagram(s): {args.diagram}")
    print(f"DPI: {args.dpi}\n")

    # Set DPI for raster formats
    if 'png' in formats:
        import os
        os.environ['GRAPHVIZ_DPI'] = str(args.dpi)

    # Generate requested diagrams
    diagrams_to_create = {
        'overview': create_overview_diagram,
        'detailed': create_detailed_diagram,
        'cli': create_cli_diagram,
        'iv_analysis': create_iv_analysis_diagram,
        'procedure_template': create_procedure_template_diagram,
    }

    if args.diagram == 'all':
        diagrams = diagrams_to_create.items()
    else:
        diagrams = [(args.diagram, diagrams_to_create[args.diagram])]

    # Create diagrams
    for name, create_func in diagrams:
        print(f"\n{'-'*70}")
        print(f"Creating {name} diagram...")
        print(f"{'-'*70}")
        try:
            create_func(args.output_dir, formats)
        except Exception as e:
            print(f"Error creating {name} diagram: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*70}")
    print(f"Visualization complete!")
    print(f"{'='*70}\n")
    print(f"Generated files in: {args.output_dir}\n")

    # List created files
    for fmt in formats:
        files = list(args.output_dir.glob(f'*.{fmt}'))
        if files:
            print(f"{fmt.upper()} files:")
            for f in sorted(files):
                print(f"  • {f.name}")
            print()

    print(f"To view diagrams:")
    print(f"  Open {args.output_dir} in your file browser")
    print()
    print(f"To use in documentation:")
    print(f"  ![Pipeline Overview]({args.output_dir}/pipeline_overview.png)")
    print()


if __name__ == '__main__':
    main()

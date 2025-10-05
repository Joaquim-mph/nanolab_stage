"""
IV Segmentation Visualization Tool

Interactive visualization for verifying IV sweep segmentation quality.
Generates plots showing detected segments with color coding and metadata.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import polars as pl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure


# ----------------------------- Config -----------------------------

SEGMENT_COLORS = {
    "forward_negative": "#e74c3c",    # Red
    "return_negative": "#f39c12",     # Orange
    "forward_positive": "#3498db",    # Blue
    "return_positive": "#2ecc71",     # Green
    "unknown": "#95a5a6",             # Gray
}

DEFAULT_VOLTAGE_COL = "Vsd (V)"
DEFAULT_CURRENT_COL = "I (A)"


# ----------------------------- Data Loading -----------------------------

def load_segmented_run(run_dir: Path) -> Optional[pl.DataFrame]:
    """
    Load all segments from a run directory and combine into single DataFrame.
    
    Args:
        run_dir: Path to run directory (e.g., .../run_id=abc123/)
        
    Returns:
        Combined DataFrame with all segments, or None if no data found
        
    Example:
        >>> df = load_segmented_run(Path("03_intermediate/.../run_id=abc123"))
        >>> df.columns
        ['I (A)', 'Vsd (V)', 'segment_id', 'segment_type', ...]
    """
    segment_dirs = sorted(run_dir.glob("segment=*"))
    
    if not segment_dirs:
        print(f"[warn] No segment directories found in {run_dir}")
        return None
    
    dfs = []
    for seg_dir in segment_dirs:
        parquet_files = list(seg_dir.glob("*.parquet"))
        if not parquet_files:
            continue
        
        for pf in parquet_files:
            try:
                df = pl.read_parquet(pf)
                dfs.append(df)
            except Exception as e:
                print(f"[warn] Failed to read {pf}: {e}")
    
    if not dfs:
        return None
    
    # Combine all segments
    combined = pl.concat(dfs, how="vertical_relaxed")
    
    # Sort by original order (if available) or by segment_id and point_in_segment
    if "point_in_segment" in combined.columns and "segment_id" in combined.columns:
        combined = combined.sort(["segment_id", "point_in_segment"])
    
    return combined


def discover_segmented_runs(
    output_root: Path,
    limit: Optional[int] = None
) -> List[Path]:
    """
    Discover all segmented run directories.
    
    Args:
        output_root: Root of segmented data (03_intermediate/iv_segments)
        limit: Maximum number of runs to return (for quick testing)
        
    Returns:
        List of run directory paths
    """
    run_dirs = sorted(output_root.glob("proc=*/date=*/run_id=*"))
    
    # Filter out manifest directories
    run_dirs = [d for d in run_dirs if d.is_dir() and "_manifest" not in d.parts]
    
    if limit:
        run_dirs = run_dirs[:limit]
    
    return run_dirs


# ----------------------------- Visualization -----------------------------

def plot_iv_segments(
    df: pl.DataFrame,
    voltage_col: str = DEFAULT_VOLTAGE_COL,
    current_col: str = DEFAULT_CURRENT_COL,
    title: str = "IV Sweep Segmentation",
    figsize: Tuple[int, int] = (14, 10),
) -> Figure:
    """
    Create comprehensive visualization of segmented IV sweep.
    
    Generates a multi-panel plot showing:
    1. IV curve with segments colored
    2. Voltage vs time with segments
    3. Current vs time with segments
    4. Segment statistics table
    
    Args:
        df: DataFrame with segmented IV data
        voltage_col: Name of voltage column
        current_col: Name of current column
        title: Plot title
        figsize: Figure size (width, height)
        
    Returns:
        Matplotlib Figure object
        
    Example:
        >>> df = load_segmented_run(Path("03_intermediate/.../run_id=abc123"))
        >>> fig = plot_iv_segments(df)
        >>> fig.savefig("iv_segments.png", dpi=150, bbox_inches="tight")
    """
    # Check required columns
    required = [voltage_col, current_col, "segment_id", "segment_type"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Convert to pandas for matplotlib
    pdf = df.to_pandas()
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight="bold")
    
    # Get unique segments
    segments = pdf[["segment_id", "segment_type"]].drop_duplicates().sort_values("segment_id")
    
    # --- Plot 1: I-V Curve with segments ---
    ax1 = axes[0, 0]
    for _, seg in segments.iterrows():
        seg_id = seg["segment_id"]
        seg_type = seg["segment_type"]
        color = SEGMENT_COLORS.get(seg_type, SEGMENT_COLORS["unknown"])
        
        seg_data = pdf[pdf["segment_id"] == seg_id]
        ax1.plot(
            seg_data[voltage_col],
            seg_data[current_col],
            color=color,
            linewidth=2,
            label=f"Seg {seg_id}: {seg_type}",
            marker='o',
            markersize=3,
            alpha=0.7
        )
    
    ax1.set_xlabel(f"{voltage_col}", fontsize=12)
    ax1.set_ylabel(f"{current_col}", fontsize=12)
    ax1.set_title("IV Characteristic Curve", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best", fontsize=9)
    ax1.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    ax1.axvline(x=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # --- Plot 2: Voltage vs Time ---
    ax2 = axes[0, 1]
    
    # Use point_in_segment or create time index
    if "t (s)" in pdf.columns:
        time_col = "t (s)"
        time_label = "Time (s)"
    elif "point_in_segment" in pdf.columns:
        # Create cumulative point index
        pdf["_time_idx"] = range(len(pdf))
        time_col = "_time_idx"
        time_label = "Measurement Point"
    else:
        pdf["_time_idx"] = range(len(pdf))
        time_col = "_time_idx"
        time_label = "Measurement Point"
    
    for _, seg in segments.iterrows():
        seg_id = seg["segment_id"]
        seg_type = seg["segment_type"]
        color = SEGMENT_COLORS.get(seg_type, SEGMENT_COLORS["unknown"])
        
        seg_data = pdf[pdf["segment_id"] == seg_id]
        ax2.plot(
            seg_data[time_col],
            seg_data[voltage_col],
            color=color,
            linewidth=2,
            label=f"Seg {seg_id}",
            alpha=0.7
        )
    
    ax2.set_xlabel(time_label, fontsize=12)
    ax2.set_ylabel(f"{voltage_col}", fontsize=12)
    ax2.set_title("Voltage vs Time", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # --- Plot 3: Current vs Time ---
    ax3 = axes[1, 0]
    for _, seg in segments.iterrows():
        seg_id = seg["segment_id"]
        seg_type = seg["segment_type"]
        color = SEGMENT_COLORS.get(seg_type, SEGMENT_COLORS["unknown"])
        
        seg_data = pdf[pdf["segment_id"] == seg_id]
        ax3.plot(
            seg_data[time_col],
            seg_data[current_col],
            color=color,
            linewidth=2,
            label=f"Seg {seg_id}",
            alpha=0.7
        )
    
    ax3.set_xlabel(time_label, fontsize=12)
    ax3.set_ylabel(f"{current_col}", fontsize=12)
    ax3.set_title("Current vs Time", fontsize=13, fontweight="bold")
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # --- Plot 4: Segment Statistics Table ---
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Calculate segment statistics
    stats_data = []
    for _, seg in segments.iterrows():
        seg_id = seg["segment_id"]
        seg_data = pdf[pdf["segment_id"] == seg_id]
        
        v_start = seg_data[voltage_col].iloc[0] if len(seg_data) > 0 else 0
        v_end = seg_data[voltage_col].iloc[-1] if len(seg_data) > 0 else 0
        i_max = seg_data[current_col].max()
        i_min = seg_data[current_col].min()
        points = len(seg_data)
        
        stats_data.append([
            f"Seg {int(seg_id)}",
            seg["segment_type"],
            f"{v_start:.3f}→{v_end:.3f}",
            f"{i_min:.2e}",
            f"{i_max:.2e}",
            points
        ])
    
    # Create table
    table = ax4.table(
        cellText=stats_data,
        colLabels=["Seg", "Type", "V Range (V)", "I Min", "I Max", "Points"],
        cellLoc='center',
        loc='center',
        colWidths=[0.1, 0.25, 0.2, 0.15, 0.15, 0.1]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Color code segment rows
    for i, (_, seg) in enumerate(segments.iterrows(), start=1):
        seg_type = seg["segment_type"]
        color = SEGMENT_COLORS.get(seg_type, SEGMENT_COLORS["unknown"])
        for j in range(6):
            table[(i, j)].set_facecolor(color)
            table[(i, j)].set_alpha(0.3)
    
    # Style header row
    for j in range(6):
        table[(0, j)].set_facecolor('#34495e')
        table[(0, j)].set_text_props(weight='bold', color='white')
    
    ax4.set_title("Segment Statistics", fontsize=13, fontweight="bold", pad=20)
    
    plt.tight_layout()
    return fig


def plot_iv_simple(
    df: pl.DataFrame,
    voltage_col: str = DEFAULT_VOLTAGE_COL,
    current_col: str = DEFAULT_CURRENT_COL,
    title: str = "IV Sweep",
    figsize: Tuple[int, int] = (10, 6),
) -> Figure:
    """
    Create simple single-panel IV curve plot with segments.
    
    Args:
        df: DataFrame with segmented IV data
        voltage_col: Name of voltage column
        current_col: Name of current column
        title: Plot title
        figsize: Figure size (width, height)
        
    Returns:
        Matplotlib Figure object
    """
    pdf = df.to_pandas()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if "segment_id" in pdf.columns and "segment_type" in pdf.columns:
        segments = pdf[["segment_id", "segment_type"]].drop_duplicates().sort_values("segment_id")
        
        for _, seg in segments.iterrows():
            seg_id = seg["segment_id"]
            seg_type = seg["segment_type"]
            color = SEGMENT_COLORS.get(seg_type, SEGMENT_COLORS["unknown"])
            
            seg_data = pdf[pdf["segment_id"] == seg_id]
            ax.plot(
                seg_data[voltage_col],
                seg_data[current_col],
                color=color,
                linewidth=2.5,
                label=f"{seg_type}",
                marker='o',
                markersize=4,
                alpha=0.8
            )
    else:
        # No segmentation info, plot as single curve
        ax.plot(
            pdf[voltage_col],
            pdf[current_col],
            color='#3498db',
            linewidth=2.5,
            marker='o',
            markersize=4,
            alpha=0.8
        )
    
    ax.set_xlabel(f"{voltage_col}", fontsize=13)
    ax.set_ylabel(f"{current_col}", fontsize=13)
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axvline(x=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    
    if "segment_id" in pdf.columns:
        ax.legend(loc="best", fontsize=10)
    
    plt.tight_layout()
    return fig


# ----------------------------- CLI -----------------------------

def main() -> None:
    """
    Visualization tool for inspecting IV sweep segmentation.
    
    Command-line interface for generating plots to verify segmentation quality.
    Supports multiple visualization modes and batch processing.
    
    Usage examples:
    
    1. Visualize first 5 runs (quick check):
        $ python visualize_iv_segments.py --limit 5
    
    2. Visualize specific run:
        $ python visualize_iv_segments.py --run-id abc123
    
    3. Simple plots only (faster):
        $ python visualize_iv_segments.py --simple --limit 10
    
    4. Custom output directory:
        $ python visualize_iv_segments.py --output-dir ./plots --limit 20
    
    5. All runs (be patient!):
        $ python visualize_iv_segments.py
    """
    ap = argparse.ArgumentParser(
        description="Visualize IV sweep segmentation for quality verification"
    )
    ap.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/03_intermediate/iv_segments"),
        help="Root of segmented data (default: ./03_intermediate)"
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots/iv_segments"),
        help="Output directory for plots (default: ./plots/iv_segments)"
    )
    ap.add_argument(
        "--voltage-col",
        type=str,
        default=DEFAULT_VOLTAGE_COL,
        help=f"Voltage column name (default: {DEFAULT_VOLTAGE_COL})"
    )
    ap.add_argument(
        "--current-col",
        type=str,
        default=DEFAULT_CURRENT_COL,
        help=f"Current column name (default: {DEFAULT_CURRENT_COL})"
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of runs to visualize (useful for testing)"
    )
    ap.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Visualize specific run_id only"
    )
    ap.add_argument(
        "--simple",
        action="store_true",
        help="Generate simple single-panel plots only (faster)"
    )
    ap.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Plot resolution (default: 150)"
    )
    
    args = ap.parse_args()
    
    data_root: Path = args.data_root
    output_dir: Path = args.output_dir
    voltage_col: str = args.voltage_col
    current_col: str = args.current_col
    limit: Optional[int] = args.limit
    run_id_filter: Optional[str] = args.run_id
    simple_mode: bool = args.simple
    dpi: int = args.dpi
    
    # Validate
    if not data_root.exists():
        raise SystemExit(f"[error] Data root does not exist: {data_root}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Discover runs
    print(f"[info] discovering segmented runs in {data_root}")
    run_dirs = discover_segmented_runs(data_root, limit=limit)
    
    # Filter by run_id if specified
    if run_id_filter:
        run_dirs = [r for r in run_dirs if run_id_filter in r.name]
        if not run_dirs:
            raise SystemExit(f"[error] No runs found matching run_id: {run_id_filter}")
    
    print(f"[info] found {len(run_dirs)} runs to visualize")
    
    if not run_dirs:
        print("[done] no data to visualize")
        return
    
    # Process each run
    success = 0
    failed = 0
    
    for i, run_dir in enumerate(run_dirs, 1):
        # Extract metadata from path
        run_id = run_dir.name.split("=")[1] if "=" in run_dir.name else run_dir.name
        date_part = run_dir.parent.name
        
        try:
            # Load data
            df = load_segmented_run(run_dir)
            
            if df is None or df.height == 0:
                print(f"[{i:04d}] SKIP {run_id} - no data")
                failed += 1
                continue
            
            # Generate title
            title = f"IV Sweep: {run_id[:16]}\n{date_part}"
            
            # Create plot
            if simple_mode:
                fig = plot_iv_simple(df, voltage_col, current_col, title)
                suffix = "simple"
            else:
                fig = plot_iv_segments(df, voltage_col, current_col, title)
                suffix = "detailed"
            
            # Save
            output_file = output_dir / f"iv_seg_{run_id}_{suffix}.png"
            fig.savefig(output_file, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            
            segments = df["segment_id"].n_unique() if "segment_id" in df.columns else 0
            print(f"[{i:04d}] OK {run_id} - {segments} segments → {output_file.name}")
            success += 1
            
        except Exception as e:
            print(f"[{i:04d}] FAIL {run_id} - {e}")
            failed += 1
            continue
    
    print(f"\n[done] visualization complete  |  success={success}  failed={failed}  total={len(run_dirs)}")
    print(f"[info] plots saved to: {output_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Visualize IV sweep segments to verify forward/backward detection.
"""

import polars as pl
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import numpy as np


def visualize_segments(stage_root: Path, date: str, procedure: str, output_dir: Path, n_samples: int = 10):
    """
    Visualize individual IV sweeps to check segment detection.

    Args:
        stage_root: Root of staged data
        date: Date in YYYY-MM-DD format
        procedure: Procedure name (IV, IVg, etc.)
        output_dir: Where to save plots
        n_samples: Number of random runs to visualize
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load staged IV data for the date
    pattern = str(stage_root / f"proc={procedure}" / f"date={date}" / "run_id=*" / "part-*.parquet")

    print(f"Scanning: {pattern}")

    try:
        lf = pl.scan_parquet(pattern)
        df_all = lf.collect()
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print(f"Found {len(df_all)} data points across {df_all['run_id'].n_unique()} runs")

    # Detect voltage column
    if "Vg (V)" in df_all.columns:
        v_col = "Vg (V)"
    elif "Vsd (V)" in df_all.columns:
        v_col = "Vsd (V)"
    else:
        raise ValueError(f"No voltage column found")

    print(f"Using voltage column: {v_col}")

    # Sample random runs
    all_run_ids = df_all["run_id"].unique().to_list()
    if len(all_run_ids) > n_samples:
        import random
        sampled_run_ids = random.sample(all_run_ids, n_samples)
    else:
        sampled_run_ids = all_run_ids

    # Calculate V_max per run for grouping
    v_max_per_run = (
        df_all.group_by("run_id")
        .agg(pl.col(v_col).max().alias("v_max"))
    )
    df_all = df_all.join(v_max_per_run, on="run_id")
    df_all = df_all.with_columns((pl.col("v_max").round(0)).alias("v_max_group"))

    # Group by V_max
    v_max_groups = sorted(df_all["v_max_group"].unique().to_list())

    # Visualize a few runs from each V_max group
    for v_max in v_max_groups[:3]:  # Just first 3 V_max groups
        group_df = df_all.filter(pl.col("v_max_group") == v_max)
        group_run_ids = group_df["run_id"].unique().to_list()

        # Take first 4 runs from this group
        runs_to_plot = group_run_ids[:4]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for idx, run_id in enumerate(runs_to_plot):
            ax = axes[idx]

            run_df = group_df.filter(pl.col("run_id") == run_id)
            run_pd = run_df.select([v_col, "I (A)"]).to_pandas()

            v = run_pd[v_col].values
            i = run_pd["I (A)"].values

            # Plot the raw data with point index
            colors = plt.cm.viridis(np.linspace(0, 1, len(v)))
            ax.scatter(v, i * 1e9, c=colors, s=20, alpha=0.7)
            ax.plot(v, i * 1e9, 'k-', alpha=0.3, linewidth=0.5)

            # Detect turning point
            max_idx = np.argmax(v)

            # Mark forward and backward segments
            ax.axvline(v[max_idx], color='red', linestyle='--', linewidth=2,
                      label=f'Turning point (idx={max_idx})')

            # Annotate start and end
            ax.plot(v[0], i[0] * 1e9, 'go', markersize=12, label='Start')
            ax.plot(v[-1], i[-1] * 1e9, 'rs', markersize=12, label='End')

            # Add some point indices
            step = max(1, len(v) // 10)
            for j in range(0, len(v), step):
                ax.annotate(f'{j}', (v[j], i[j] * 1e9),
                           fontsize=7, alpha=0.6,
                           xytext=(3, 3), textcoords='offset points')

            ax.set_xlabel('Voltage (V)', fontsize=11)
            ax.set_ylabel('Current (nA)', fontsize=11)
            ax.set_title(f'Run {idx+1}\nPoints: {len(v)}, Max V: {v[max_idx]:.2f}V',
                        fontsize=10, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.suptitle(f'V_max Group ≈ {v_max:.0f}V - Segment Detection',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()

        plot_file = output_dir / f"segments_vmax{int(v_max)}V.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {plot_file}")
        plt.close()

    # Create a detailed view of a single run showing the 4-segment detection
    # Pick one run with medium V_max
    if len(v_max_groups) >= 4:
        mid_vmax = v_max_groups[3]
    else:
        mid_vmax = v_max_groups[0]

    mid_group_df = df_all.filter(pl.col("v_max_group") == mid_vmax)
    example_run_id = mid_group_df["run_id"].unique()[0]
    example_df = mid_group_df.filter(pl.col("run_id") == example_run_id)
    example_pd = example_df.select([v_col, "I (A)"]).to_pandas()

    v_ex = example_pd[v_col].values
    i_ex = example_pd["I (A)"].values

    # Detect 4 segments using sign changes
    dv = np.diff(v_ex)
    sign_changes = []
    for i_idx in range(len(dv) - 1):
        if dv[i_idx] * dv[i_idx + 1] < 0:  # Sign change
            sign_changes.append(i_idx + 1)

    turning_points = [0] + sign_changes + [len(v_ex)]

    print(f"\nDetailed segment analysis for example run:")
    print(f"  Total points: {len(v_ex)}")
    print(f"  Voltage range: [{v_ex.min():.3f}, {v_ex.max():.3f}]")
    print(f"  Sign changes detected: {len(sign_changes)}")
    print(f"  Turning point indices: {turning_points}")

    fig, axes = plt.subplots(4, 1, figsize=(14, 16))

    # Panel 1: Raw data with indices
    ax1 = axes[0]
    colors = plt.cm.plasma(np.linspace(0, 1, len(v_ex)))
    ax1.scatter(range(len(v_ex)), v_ex, c=colors, s=30, label='Voltage')
    ax1.set_xlabel('Point Index', fontsize=12)
    ax1.set_ylabel('Voltage (V)', fontsize=12, color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_title('Voltage vs Point Index (color = progression)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Panel 2: Voltage derivative
    ax2 = axes[1]
    dv = np.diff(v_ex)
    ax2.plot(range(len(dv)), dv, 'o-', markersize=4)
    ax2.axhline(0, color='red', linestyle='--', linewidth=2, label='Zero crossing')
    ax2.set_xlabel('Point Index', fontsize=12)
    ax2.set_ylabel('dV/dpoint', fontsize=12)
    ax2.set_title('Voltage Derivative (detect direction changes)', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Panel 3: Voltage trace showing segment boundaries
    ax3 = axes[2]
    ax3.plot(v_ex, 'ko-', markersize=3, linewidth=1, alpha=0.5)

    # Mark turning points
    for tp_idx in turning_points[1:-1]:
        ax3.axvline(tp_idx, color='red', linestyle='--', alpha=0.7, linewidth=2)
        ax3.text(tp_idx, v_ex.max(), f' {tp_idx}', fontsize=10, color='red')

    ax3.set_xlabel('Point Index', fontsize=12)
    ax3.set_ylabel('Voltage (V)', fontsize=12)
    ax3.set_title('Voltage Trace with Detected Turning Points', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Panel 4: I-V curve with 4 segments marked
    ax4 = axes[3]

    if len(turning_points) >= 5:
        # Extract 4 segments
        colors = ['blue', 'orange', 'green', 'red']
        labels = ['Seg 1: Fwd Neg (0→-Vmax)', 'Seg 2: Ret Neg (-Vmax→0)',
                 'Seg 3: Fwd Pos (0→+Vmax)', 'Seg 4: Ret Pos (+Vmax→0)']

        for seg_idx in range(4):
            if seg_idx < len(turning_points) - 1:
                start = turning_points[seg_idx]
                end = turning_points[seg_idx + 1]
                ax4.plot(v_ex[start:end], i_ex[start:end] * 1e9,
                        'o-', color=colors[seg_idx], label=labels[seg_idx],
                        markersize=4, linewidth=2, alpha=0.8)
    else:
        # Fallback: simple split
        max_idx = np.argmax(v_ex)
        min_idx = np.argmin(v_ex)

        ax4.plot(v_ex, i_ex * 1e9, 'ko-', markersize=3, alpha=0.5, label='Full trace')
        ax4.axvline(v_ex[min_idx], color='blue', linestyle='--', label=f'Min V at idx {min_idx}')
        ax4.axvline(v_ex[max_idx], color='red', linestyle='--', label=f'Max V at idx {max_idx}')

    ax4.plot(v_ex[0], i_ex[0] * 1e9, 'go', markersize=15, label='Start', zorder=10)
    ax4.plot(v_ex[-1], i_ex[-1] * 1e9, 'm^', markersize=15, label='End', zorder=10)

    ax4.set_xlabel('Voltage (V)', fontsize=12)
    ax4.set_ylabel('Current (nA)', fontsize=12)
    ax4.set_title('I-V Curve with 4-Segment Detection', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=9, loc='best')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_file = output_dir / "segment_detection_detail.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Saved detailed view: {plot_file}")
    plt.close()

    print(f"\n✓ Segment visualization complete. Check {output_dir}")

    # Print statistics about segment detection
    print("\n=== Segment Detection Statistics ===")
    for v_max in v_max_groups:
        group_df = df_all.filter(pl.col("v_max_group") == v_max)
        n_runs = group_df["run_id"].n_unique()

        # Check a few runs
        sample_runs = group_df["run_id"].unique().to_list()[:5]
        fwd_lens = []
        bwd_lens = []

        for run_id in sample_runs:
            run_df = group_df.filter(pl.col("run_id") == run_id)
            v = run_df[v_col].to_numpy()
            max_idx = np.argmax(v)
            fwd_lens.append(max_idx + 1)
            bwd_lens.append(len(v) - max_idx)

        print(f"\nV_max ≈ {v_max:.0f}V ({n_runs} runs):")
        print(f"  Sample forward segment lengths: {fwd_lens}")
        print(f"  Sample backward segment lengths: {bwd_lens}")
        print(f"  Average forward: {np.mean(fwd_lens):.1f} points")
        print(f"  Average backward: {np.mean(bwd_lens):.1f} points")


def main():
    parser = argparse.ArgumentParser(description="Visualize IV segment detection")
    parser.add_argument("--stage-root", type=Path, default=Path("data/02_stage/raw_measurements"),
                       help="Root of staged data")
    parser.add_argument("--date", required=True, help="Date in YYYY-MM-DD format")
    parser.add_argument("--procedure", type=str, default="IV", help="Procedure name")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for plots")
    parser.add_argument("--n-samples", type=int, default=10, help="Number of runs to sample")

    args = parser.parse_args()

    visualize_segments(
        stage_root=args.stage_root,
        date=args.date,
        procedure=args.procedure,
        output_dir=args.output_dir,
        n_samples=args.n_samples
    )


if __name__ == "__main__":
    main()

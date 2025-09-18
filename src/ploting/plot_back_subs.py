#!/usr/bin/env python3
"""
IV Curve Analysis Tool - Optimized Version

Processes IV sweep data to generate:
- Mean curves with hysteresis
- Directional analysis (forward/backward)
- Baseline subtraction with polynomial fits
- Comprehensive diagnostics and visualizations
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import yaml
import scienceplots
from styles import set_plot_style

set_plot_style('prism_rain')

# ============================= CORE UTILITIES ===============================

def ensure_dir(p: Path) -> None:
    """Create directory if it doesn't exist."""
    p.mkdir(parents=True, exist_ok=True)

def most_recent_sept11(tz_name: str = "America/Santiago") -> str:
    """Get most recent September 11th date string."""
    try:
        from zoneinfo import ZoneInfo
        now = dt.datetime.now(tz=ZoneInfo(tz_name))
    except Exception:
        now = dt.datetime.now()
    year = now.year if now >= now.replace(month=9, day=11) else now.year - 1
    return f"{year}-09-11"

def _norm(s: str) -> str:
    """Normalize string for comparison."""
    return s.strip().lower().replace(" ", "")

# ============================= DATA LOADING ===============================

def read_manifest(stage_root: Path) -> Optional[pl.DataFrame]:
    """Read manifest file if it exists."""
    mp = stage_root / "_manifest" / "manifest.parquet"
    if mp.exists():
        try:
            return pl.read_parquet(mp)
        except Exception:
            return None
    return None

def paths_from_manifest(mani: pl.DataFrame, date_str: str, proc_sub: Optional[str]) -> List[Path]:
    """Extract paths from manifest for given date and procedure filter."""
    df = mani
    if "status" in df.columns:
        df = df.filter(pl.col("status").is_in(["ok", "skipped"]))
    if "path" not in df.columns:
        return []
    
    df = df.filter(pl.col("path").str.contains(f"/date={date_str}/"))
    if proc_sub:
        if "proc" in df.columns:
            df = df.filter(pl.col("proc").str.contains(proc_sub, literal=False, strict=False))
        else:
            df = df.filter(pl.col("path").str.contains(f"/proc=.*{proc_sub}.*?/"))
    
    return [Path(p) for p in df.get_column("path").to_list() if isinstance(p, str)]

def glob_date_paths(stage_root: Path, date_str: str, proc_sub: Optional[str]) -> List[Path]:
    """Fallback path discovery using glob."""
    out: List[Path] = []
    for proc_dir in stage_root.glob("proc=*"):
        name = proc_dir.name.split("=", 1)[-1]
        if proc_sub and proc_sub.lower() not in name.lower():
            continue
        d = proc_dir / f"date={date_str}"
        if d.exists():
            out.extend(d.glob("run_id=*/part-000.parquet"))
    return out

def load_procedures_yaml(path: Path) -> Dict[str, Dict[str, Dict[str, str]]]:
    """Load procedures configuration."""
    with path.open("r", encoding="utf-8") as f:
        y = yaml.safe_load(f) or {}
    return y.get("procedures", {}) or {}

def first_val(df: pl.DataFrame, col: str) -> Optional[Any]:
    """Get first value from column if it exists."""
    if col in df.columns:
        try:
            return df.select(pl.first(col)).item()
        except Exception:
            return None
    return None

# ============================= COLUMN DETECTION ===============================

def current_col_for_proc(df: pl.DataFrame, spec: Dict[str, str]) -> Optional[str]:
    """Find current column in DataFrame."""
    if "I (A)" in df.columns:
        return "I (A)"
    
    # Look for columns ending with (A)
    for c in df.columns:
        if c.endswith("(A)") or "(A" in c:
            return c
    
    # Look for common current column names
    for c in df.columns:
        if _norm(c) in {"i", "id", "ids", "current", "draincurrent"}:
            return c
    return None

def voltage_cols_for_proc(df: pl.DataFrame, spec: Dict[str, str]) -> Tuple[Optional[str], Optional[str]]:
    """Find voltage columns (Vsd, Vg) in DataFrame."""
    vsd = "Vsd (V)" if "Vsd (V)" in df.columns else None
    vg = "Vg (V)" if "Vg (V)" in df.columns else None
    
    if vsd is None:
        for c in df.columns:
            lc = _norm(c)
            if lc.startswith(("vds", "vd", "vsd")) and c.endswith("(V)"):
                vsd = c
                break
    
    if vg is None:
        for c in df.columns:
            lc = _norm(c)
            if lc.startswith("vg") and c.endswith("(V)"):
                vg = c
                break
    
    return vsd, vg

# ============================= NUMERIC PROCESSING ===============================

def coerce_xy(df: pl.DataFrame, x_col: str, y_col: str, dec: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Extract and clean x, y arrays from DataFrame."""
    x = df[x_col].cast(pl.Float64, strict=False).to_numpy()
    y = df[y_col].cast(pl.Float64, strict=False).to_numpy()
    m = np.isfinite(x) & np.isfinite(y)
    
    if dec > 1:
        idx = np.arange(m.size)[m][::dec]
        return x[idx], y[idx]
    return x[m], y[m]

def group_unit_from_max_abs(y_max: float) -> Tuple[float, str]:
    """Determine appropriate current unit and scaling factor."""
    if y_max >= 1e-3:
        return 1e3, "mA"
    if y_max >= 1e-6:
        return 1e6, "µA"
    if y_max >= 1e-9:
        return 1e9, "nA"
    return 1.0, "A"

def arc_length_mean(xs: List[np.ndarray], ys: List[np.ndarray], samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """Resample traces by arc length and compute mean."""
    samples = max(2, int(samples))
    tgrid = np.linspace(0.0, 1.0, samples)
    Xi: List[np.ndarray] = []
    Yi: List[np.ndarray] = []
    
    for x, y in zip(xs, ys):
        if x.size < 2 or y.size < 2:
            continue
        
        dx = np.diff(x)
        dy = np.diff(y)
        s = np.concatenate(([0.0], np.cumsum(np.hypot(dx, dy))))
        
        if not np.isfinite(s[-1]) or s[-1] <= 0:
            s = np.linspace(0.0, 1.0, x.size)
        else:
            s = s / s[-1]
        
        Xi.append(np.interp(tgrid, s, x))
        Yi.append(np.interp(tgrid, s, y))
    
    if not Xi:
        return np.array([]), np.array([])
    
    X = np.vstack(Xi)
    Y = np.vstack(Yi)
    return np.nanmean(X, axis=0), np.nanmean(Y, axis=0)

# ============================= DIRECTIONAL ANALYSIS ===============================

def _split_segments(x, y, *, round_dec=2, dv_threshold=0.05, min_seg_pts=4, min_span=0.3):
    """Split sweep into monotonic forward/backward segments."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.size < 2:
        return [], [], [], []

    xr = np.round(x, round_dec)
    dv = np.diff(xr)
    sgn = np.zeros_like(dv, dtype=int)
    sgn[dv > dv_threshold] = 1
    sgn[dv < -dv_threshold] = -1

    # Segment by direction changes
    segs = []
    if sgn.size:
        start = 0
        cur = sgn[0]
        for i in range(1, len(sgn)):
            if sgn[i] != cur:
                if cur != 0:
                    segs.append((start, i, cur))
                start = i
                cur = sgn[i]
        if cur != 0:
            segs.append((start, len(xr)-1, cur))

    fwd_xs, fwd_ys, bwd_xs, bwd_ys = [], [], [], []
    for s, e, sg in segs:
        if (e - s + 1) < min_seg_pts:
            continue
        if abs(xr[e] - xr[s]) < min_span:
            continue
        
        xs = x[s:e+1]
        ys = y[s:e+1]
        if sg > 0:
            fwd_xs.append(xs)
            fwd_ys.append(ys)
        else:
            bwd_xs.append(xs)
            bwd_ys.append(ys)
    
    return fwd_xs, fwd_ys, bwd_xs, bwd_ys

def _mean_from_binned(xs_list, ys_list, *, round_dec=2):
    """Bin x values and average y per bin."""
    bins = defaultdict(list)
    for xs, ys in zip(xs_list, ys_list):
        xb = np.round(xs, round_dec)
        for xv, yv in zip(xb, ys):
            bins[float(xv)].append(float(yv))
    
    if not bins:
        return np.array([]), np.array([])
    
    xs = np.array(sorted(bins.keys()), float)
    ys = np.array([np.mean(bins[v]) for v in xs], float)
    return xs, ys

def compute_directional_means(groups: Dict[int, List[Dict[str, Any]]], 
                            round_dec: int, dv_threshold: float) -> Dict[int, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """Compute forward/backward means for all groups."""
    mean_dir = {}
    
    for g in range(1, 9):
        trs = groups[g]
        if not trs:
            continue
        
        fwd_xs, fwd_ys, bwd_xs, bwd_ys = [], [], [], []
        for t in trs:
            fx, fy, bx, by = _split_segments(
                t["x"], t["y"],
                round_dec=round_dec,
                dv_threshold=dv_threshold,
                min_seg_pts=4,
                min_span=0.3
            )
            fwd_xs += fx
            fwd_ys += fy
            bwd_xs += bx
            bwd_ys += by

        entry = {}
        xf, yf = _mean_from_binned(fwd_xs, fwd_ys, round_dec=round_dec)
        xb, yb = _mean_from_binned(bwd_xs, bwd_ys, round_dec=round_dec)
        
        if xf.size:
            entry["fwd"] = (xf, yf)
        if xb.size:
            entry["bwd"] = (xb, yb)
        if entry:
            mean_dir[g] = entry
    
    return mean_dir

def compute_backtrace_means(mean_dir: Dict[int, Dict[str, Tuple[np.ndarray, np.ndarray]]]) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Compute backtrace means (average of forward/backward)."""
    mean_back = {}
    
    for g in sorted(mean_dir.keys()):
        entry = mean_dir[g]
        if ("fwd" in entry) and ("bwd" in entry):
            xF, yF = entry["fwd"]
            xB, yB = entry["bwd"]
            xC = np.intersect1d(xF, xB)
            if xC.size == 0:
                continue
            
            yF_i = np.interp(xC, xF, yF)
            yB_i = np.interp(xC, xB, yB)
            y_mean = 0.5 * (yF_i + yB_i)
            mean_back[g] = (xC, y_mean)
    
    return mean_back

# ============================= POLYNOMIAL FITTING ===============================

def fit_full_polynomials(x: np.ndarray, y: np.ndarray, 
                                     degrees: Tuple[int, ...] = (1, 3, 5, 7)) -> Tuple[Dict[int, np.ndarray], Dict[int, float]]:
    """Fit full polynomials with no intercept (all powers: V, V², V³, ..., but no constant term)."""
    polyfits = {}
    r2_by_deg = {}
    
    if x.size < 2:
        return polyfits, r2_by_deg
    
    # Sort for stability
    idx = np.argsort(x)
    x_fit, y_fit = x[idx], y[idx]
    y_mean = float(np.mean(y_fit))
    ss_tot = float(np.sum((y_fit - y_mean) ** 2))

    for deg in degrees:
        if x_fit.size < deg:  # Need at least deg points for deg coefficients (no intercept)
            continue
        
        # Create design matrix for all powers from 1 to deg (no intercept term)
        # For degree n: y = a₁x + a₂x² + a₃x³ + ... + aₙxⁿ
        powers = np.arange(1, deg + 1)  # [1, 2, 3, ..., deg]
        X = np.column_stack([x_fit ** p for p in powers])
        
        # Fit using least squares: minimize ||Xβ - y||²
        try:
            coeffs_all = np.linalg.lstsq(X, y_fit, rcond=None)[0]
        except np.linalg.LinAlgError:
            print(f"[polyfit] deg {deg} failed: singular matrix")
            continue
        
        # Convert to full polynomial coefficients (highest degree first for np.polyval)
        # e.g., for deg=5: [a₅, a₄, a₃, a₂, a₁, 0] represents a₅x⁵ + a₄x⁴ + a₃x³ + a₂x² + a₁x
        coeffs = np.zeros(deg + 1)  # +1 for the zero intercept
        for i, power in enumerate(powers):
            coeffs[deg - power] = coeffs_all[i]  # deg-power gives position from start
        # coeffs[-1] remains 0 (no intercept)
        
        polyfits[deg] = coeffs

        # Calculate R²
        yhat = np.polyval(coeffs, x_fit)
        ss_res = float(np.sum((y_fit - yhat) ** 2))
        r2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else np.nan
        r2_by_deg[deg] = r2
        
        # Display equation for clarity
        terms = []
        for i, power in enumerate(powers):
            coeff = coeffs_all[i]
            if abs(coeff) > 1e-12:  # Only show significant terms
                if power == 1:
                    terms.append(f"{coeff:.3e}×V")
                else:
                    terms.append(f"{coeff:.3e}×V^{power}")
        equation = " + ".join(terms) if terms else "0"
        print(f"[polyfit] deg {deg} (no intercept, all powers): R²={r2:.4f}")
        print(f"          I = {equation}")
    
    return polyfits, r2_by_deg

def fit_physical_models(x: np.ndarray, y: np.ndarray) -> Dict[str, Dict[str, Any]]:
    """Fit common physical models to I-V data."""
    from scipy.optimize import curve_fit
    
    models = {}
    
    # Sort data
    idx = np.argsort(x)
    x_fit, y_fit = x[idx], y[idx]
    
    # Remove any points exactly at zero to avoid numerical issues
    nonzero_mask = np.abs(x_fit) > 1e-10
    if np.any(nonzero_mask):
        x_nz, y_nz = x_fit[nonzero_mask], y_fit[nonzero_mask]
    else:
        x_nz, y_nz = x_fit, y_fit
    
    # Model 1: Power law I = a * V^n (good for many devices)
    def power_law(V, a, n):
        return a * np.sign(V) * np.abs(V)**n
    
    try:
        # Initial guess: linear (n=1)
        popt, _ = curve_fit(power_law, x_nz, y_nz, p0=[1e-6, 1.0], maxfev=2000)
        y_pred = power_law(x_fit, *popt)
        r2 = 1 - np.sum((y_fit - y_pred)**2) / np.sum((y_fit - np.mean(y_fit))**2)
        models['power_law'] = {
            'params': {'a': popt[0], 'n': popt[1]}, 
            'r2': float(r2),
            'equation': f"I = {popt[0]:.3e} × sign(V) × |V|^{popt[1]:.3f}"
        }
        print(f"[physical] Power law: R²={r2:.4f}, I = {popt[0]:.3e} × sign(V) × |V|^{popt[1]:.3f}")
    except Exception:
        pass
    
    # Model 2: Hyperbolic tangent I = a * tanh(b * V) (tunnel junctions, etc.)
    def tanh_model(V, a, b):
        return a * np.tanh(b * V)
    
    try:
        # Initial guess based on data range
        a_guess = np.max(np.abs(y_fit))
        b_guess = 1.0 / (np.max(np.abs(x_fit)) + 1e-10)
        popt, _ = curve_fit(tanh_model, x_fit, y_fit, p0=[a_guess, b_guess], maxfev=2000)
        y_pred = tanh_model(x_fit, *popt)
        r2 = 1 - np.sum((y_fit - y_pred)**2) / np.sum((y_fit - np.mean(y_fit))**2)
        models['tanh'] = {
            'params': {'a': popt[0], 'b': popt[1]}, 
            'r2': float(r2),
            'equation': f"I = {popt[0]:.3e} × tanh({popt[1]:.3e} × V)"
        }
        print(f"[physical] Tanh: R²={r2:.4f}, I = {popt[0]:.3e} × tanh({popt[1]:.3e} × V)")
    except Exception:
        pass
    
    # Model 3: Cubic spline for very nonlinear data
    try:
        from scipy.interpolate import UnivariateSpline
        # Use spline with some smoothing
        spline = UnivariateSpline(x_fit, y_fit, s=len(x_fit)*np.var(y_fit)*0.01)
        y_pred = spline(x_fit)
        r2 = 1 - np.sum((y_fit - y_pred)**2) / np.sum((y_fit - np.mean(y_fit))**2)
        models['spline'] = {
            'spline': spline,
            'r2': float(r2),
            'equation': "Cubic spline (adaptive)"
        }
        print(f"[physical] Spline: R²={r2:.4f}")
    except Exception:
        pass
    
    return models

def subtract_baselines(mean_curves: Dict[int, Tuple[np.ndarray, np.ndarray]], 
                      polyfits: Dict[int, np.ndarray]) -> Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray]]]:
    """Subtract polynomial baselines from all curves."""
    mean_sub_by_deg = {}
    
    for deg, coeffs in polyfits.items():
        sub = {}
        for g, (x, y) in mean_curves.items():
            sub[g] = (x, y - np.polyval(coeffs, x))
        mean_sub_by_deg[deg] = sub
    
    return mean_sub_by_deg

# ============================= PLOTTING UTILITIES ===============================

def _save_line_plot(out_png: Path, xlabel: str, ylabel: str, lines: List[Dict[str, Any]],
                   title: str | None = None, dpi: int = 220, alpha: float = 0.95, legend: bool = True):
    """Save line plot with given specifications."""
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=dpi)
    
    for ln in lines:
        x, y, lbl = ln["x"], ln["y"], ln.get("label")
        ax.plot(x, y, alpha=alpha, label=lbl)
    
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    if title:
        ax.set_title(title)
    if legend:
        ax.legend(frameon=False, fontsize=9, loc="best")
    
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

def _save_bar_plot(out_png: Path, cats: List[str], vals: List[float], xlabel: str, ylabel: str,
                  title: str | None = None, dpi: int = 220):
    """Save bar plot with given specifications."""
    fig, ax = plt.subplots(figsize=(10, 4.0), dpi=dpi)
    ax.bar(cats, vals)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    if title:
        ax.set_title(title)
    
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write data rows to CSV file."""
    if not rows:
        return
    
    df = pl.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(path)

def _ylabel_from_unit(unit: str) -> str:
    """Generate y-axis label from current unit."""
    return f"I ({unit})"

def _x_label_from_arg(x_choice: str) -> str:
    """Generate x-axis label from voltage choice."""
    if x_choice == "vsd":
        return "Vsd (V)"
    elif x_choice == "vg":
        return "Vg (V)"
    else:
        return "V (V)"

# ============================= MAIN ANALYSIS CLASS ===============================

class IVAnalyzer:
    """Main IV curve analysis class."""
    
    def __init__(self, args):
        self.args = args
        self.groups = {k: [] for k in range(1, 9)}
        self.mean_full = {}
        self.mean_dir = {}
        self.mean_back = {}
        self.polyfits = {}
        self.r2_by_deg = {}
        self.mean_sub_by_deg = {}
        self.scale = 1.0
        self.unit = "A"
        
    def load_data(self, stage_root: Path, date_str: str, proc_filter: str, procedures: dict):
        """Load and process IV curve data."""
        # Discover parquet files
        mani = read_manifest(stage_root)
        paths = paths_from_manifest(mani, date_str, proc_filter) if mani is not None else []
        if not paths:
            paths = glob_date_paths(stage_root, date_str, proc_filter)
        
        if not paths:
            print(f"[info] no Parquet for date={date_str}")
            return False

        # Process each file
        for p in paths:
            try:
                df = pl.read_parquet(p)
                proc = first_val(df, "proc")
                spec = procedures.get(proc, {}).get("Data", {}) if isinstance(proc, str) else {}
                
                y_col = current_col_for_proc(df, spec)
                vsd_col, vg_col = voltage_cols_for_proc(df, spec)
                x_col = vsd_col if self.args.x == "vsd" else (vg_col if self.args.x == "vg" else (vsd_col or vg_col))
                
                if not (x_col and y_col):
                    continue
                
                x, y = coerce_xy(df, x_col, y_col, self.args.decimate)
                if x.size < 2:
                    continue
                
                vsd_end = first_val(df, "VSD end")
                if vsd_end is None:
                    vsd_end = float(np.nanmax(np.abs(x))) if x.size else None
                
                try:
                    g = int(round(abs(float(vsd_end)))) if vsd_end is not None else None
                except Exception:
                    g = None
                
                if g is None or g < 1 or g > 8:
                    continue
                
                self.groups[g].append({"x": x, "y": y})
                
            except Exception as e:
                print(f"[warn] Error processing {p}: {e}")
                continue
        
        return bool(any(self.groups.values()))
    
    def compute_means(self):
        """Compute all mean curves."""
        # Full means (with hysteresis)
        for g in range(1, 9):
            trs = self.groups[g]
            if not trs:
                continue
            xs = [t["x"] for t in trs]
            ys = [t["y"] for t in trs]
            xm, ym = arc_length_mean(xs, ys, self.args.mean_samples)
            if xm.size:
                self.mean_full[g] = (xm, ym)
        
        # Directional means
        self.mean_dir = compute_directional_means(
            self.groups, self.args.round_dec, self.args.dv_threshold
        )
        
        # Backtrace means
        self.mean_back = compute_backtrace_means(self.mean_dir)
        
        # Determine scaling
        if self.mean_full:
            ymax = max(float(np.nanmax(np.abs(y))) for (_, y) in self.mean_full.values())
            self.scale, self.unit = group_unit_from_max_abs(ymax)
    
    def fit_polynomials(self):
        """Fit polynomial baselines and physical models."""
        base_g = 8 if 8 in self.mean_back else (max(self.mean_back.keys()) if self.mean_back else None)
        
        if base_g is not None:
            xb, yb = self.mean_back[base_g]
            if xb.size >= 2:
                print(f"[info] Fitting models to base backtrace G={base_g} ({xb.size} points)")
                
                # Try both polynomial and physical models
                self.polyfits, self.r2_by_deg = fit_full_polynomials(xb, yb)
                
                # Also try physical models for comparison
                try:
                    physical_models = fit_physical_models(xb, yb)
                    if physical_models:
                        print("[info] Physical model fits:")
                        for name, model in physical_models.items():
                            if 'r2' in model:
                                print(f"  {name}: R²={model['r2']:.4f}")
                        # Store best physical model for potential use
                        best_physical = max(physical_models.items(), key=lambda x: x[1].get('r2', -np.inf))
                        if best_physical[1].get('r2', 0) > max(self.r2_by_deg.values(), default=0):
                            print(f"[info] Best physical model ({best_physical[0]}) outperforms polynomials")
                except ImportError:
                    print("[warn] scipy not available for physical model fitting")
                except Exception as e:
                    print(f"[warn] Physical model fitting failed: {e}")
                
                if self.polyfits:
                    self.mean_sub_by_deg = subtract_baselines(self.mean_full, self.polyfits)
                else:
                    print("[warn] No successful polynomial fits")
            else:
                print("[warn] insufficient data for polynomial fitting")
        else:
            print("[warn] cannot fit polynomials: no suitable backtrace available")
    
    def generate_plots(self, out_dir: Path, date_str: str):
        """Generate all plots and save data."""
        out_date_dir = out_dir / f"date={date_str}"
        ensure_dir(out_date_dir)
        
        xlab = _x_label_from_arg(self.args.x)
        ylab = _ylabel_from_unit(self.unit)
        
        # 1. Mean curves (full) by group
        self._plot_mean_full(out_date_dir, xlab, ylab)
        
        # 2. Directional means overlay
        self._plot_directional_overlay(out_date_dir, xlab, ylab)
        
        # 3. Forward - backward differences
        self._plot_directional_differences(out_date_dir, xlab, ylab)
        
        # 4. Backtrace means
        self._plot_backtrace_means(out_date_dir, xlab, ylab)
        
        # 5. Polynomial fits
        self._plot_polynomial_fits(out_date_dir, xlab, ylab)
        
        # 6. Baseline-subtracted curves
        self._plot_subtracted_curves(out_date_dir, xlab, ylab)
    
    def _plot_mean_full(self, out_dir: Path, xlab: str, ylab: str):
        """Plot full mean curves."""
        lines = []
        rows = []
        
        for g in sorted(self.mean_full.keys()):
            x, y = self.mean_full[g]
            lines.append({"x": x, "y": y * 1e9, "label": f"G={g}"})
            if self.args.save_csv:
                rows += [{"group": g, "x": float(xx), "y": float(yy)} for xx, yy in zip(x, y)]
        
        _save_line_plot(
            out_dir / "mean_full_by_group.png",
            xlab, ylab, lines,
            title="Arc-length mean curves (full, hysteresis preserved)",
            dpi=self.args.dpi, alpha=self.args.alpha
        )
        
        if self.args.save_csv:
            _write_csv(out_dir / "mean_full_by_group.csv", rows)
    
    def _plot_directional_overlay(self, out_dir: Path, xlab: str, ylab: str):
        """Plot directional means overlay."""
        lines = []
        rows = []
        
        for g in sorted(self.mean_dir.keys()):
            ent = self.mean_dir[g]
            if "fwd" in ent:
                xf, yf = ent["fwd"]
                lines.append({"x": xf, "y": yf * 1e9, "label": f"G={g} fwd"})
                if self.args.save_csv:
                    rows += [{"group": g, "direction": "fwd", "x": float(xx), "y": float(yy)}
                            for xx, yy in zip(xf, yf)]
            
            if "bwd" in ent:
                xb, yb = ent["bwd"]
                lines.append({"x": xb, "y": yb * 1e9, "label": f"G={g} bwd"})
                if self.args.save_csv:
                    rows += [{"group": g, "direction": "bwd", "x": float(xx), "y": float(yy)}
                            for xx, yy in zip(xb, yb)]
        
        if lines:
            _save_line_plot(
                out_dir / "mean_forward_backward_overlay.png",
                xlab, ylab, lines,
                title="Directional means (forward/backward) overlay",
                dpi=self.args.dpi, alpha=self.args.alpha
            )
            
            if self.args.save_csv:
                _write_csv(out_dir / "mean_dir_overlay.csv", rows)
    
    def _plot_directional_differences(self, out_dir: Path, xlab: str, ylab: str):
        """Plot forward - backward differences."""
        lines = []
        rows = []
        
        for g in sorted(self.mean_dir.keys()):
            ent = self.mean_dir[g]
            if ("fwd" in ent) and ("bwd" in ent):
                xF, yF = ent["fwd"]
                xB, yB = ent["bwd"]
                xC = np.intersect1d(xF, xB)
                if xC.size == 0:
                    continue
                
                yF_i = np.interp(xC, xF, yF)
                yB_i = np.interp(xC, xB, yB)
                dy = yF_i - yB_i
                lines.append({"x": xC, "y": dy * 1e9, "label": f"G={g}"})
                
                if self.args.save_csv:
                    rows += [{"group": g, "x": float(xx), "delta_y": float(dd)} 
                            for xx, dd in zip(xC, dy)]
        
        if lines:
            _save_line_plot(
                out_dir / "mean_forward_minus_backward.png",
                xlab, f"Δ{ylab}", lines,
                title="Directional mean difference: forward − backward",
                dpi=self.args.dpi, alpha=self.args.alpha
            )
            
            if self.args.save_csv:
                _write_csv(out_dir / "mean_dir_diff.csv", rows)
    
    def _plot_backtrace_means(self, out_dir: Path, xlab: str, ylab: str):
        """Plot backtrace means."""
        lines = []
        rows = []
        
        for g in sorted(self.mean_back.keys()):
            x, y = self.mean_back[g]
            lines.append({"x": x, "y": y * 1e9, "label": f"G={g}"})
            if self.args.save_csv:
                rows += [{"group": g, "x": float(xx), "y_back": float(yy)} 
                        for xx, yy in zip(x, y)]
        
        if lines:
            _save_line_plot(
                out_dir / "backtrace_means.png",
                xlab, ylab, lines,
                title="Backtrace means (avg of fwd/bwd) by group",
                dpi=self.args.dpi, alpha=self.args.alpha
            )
            
            if self.args.save_csv:
                _write_csv(out_dir / "backtrace_means.csv", rows)
    
    def _plot_polynomial_fits(self, out_dir: Path, xlab: str, ylab: str):
        """Plot polynomial fits and diagnostics."""
        base_g = 8 if 8 in self.mean_back else (max(self.mean_back.keys()) if self.mean_back else None)
        
        if base_g is None or base_g not in self.mean_back or not self.polyfits:
            return
        
        xb, yb = self.mean_back[base_g]
        xs = np.linspace(xb.min(), xb.max(), 600)
        
        # Plot base backtrace with polynomial fits
        lines = [{"x": xb, "y": yb * 1e9, "label": f"G={base_g} backtrace"}]
        
        for deg in (1, 3, 5, 7):
            if deg in self.polyfits:
                coeffs = self.polyfits[deg]
                yfit = np.polyval(coeffs, xs)
                r2 = self.r2_by_deg.get(deg, np.nan)
                lines.append({"x": xs, "y": yfit * 1e9, "label": f"deg {deg} (R²={r2:.3f})"})
        
        _save_line_plot(
            out_dir / "backtrace_base_polyfits.png",
            xlab, ylab, lines,
            title=f"Base backtrace (G={base_g}) with odd-degree polynomial fits",
            dpi=self.args.dpi, alpha=0.9
        )
        
        # R² comparison bar chart
        if self.r2_by_deg:
            degs = sorted(self.r2_by_deg.keys())
            vals = [self.r2_by_deg[d] for d in degs]
            _save_bar_plot(
                out_dir / "polyfit_r2.png",
                [str(d) for d in degs], vals,
                "Polynomial degree", "R²",
                "Polynomial fit R² (base backtrace)",
                dpi=self.args.dpi
            )
        
        # Save coefficient and R² data
        if self.args.save_csv:
            self._save_polynomial_data(out_dir, xs, xb, yb)
    
    def _save_polynomial_data(self, out_dir: Path, xs: np.ndarray, xb: np.ndarray, yb: np.ndarray):
        """Save polynomial fit data to CSV files."""
        # Dense curve data
        rows = []
        for deg in (1, 3, 5, 7):
            if deg not in self.polyfits:
                continue
            yfit = np.polyval(self.polyfits[deg], xs)
            rows += [{"x": float(xx), "y_base": float(yyb), "degree": deg, "y_fit": float(yf)}
                    for xx, yyb, yf in zip(xs, np.interp(xs, xb, yb), yfit)]
        _write_csv(out_dir / "polyfit_backtrace_base.csv", rows)
        
        # Coefficients
        coeff_rows = []
        for deg, coeffs in self.polyfits.items():
            row = {"degree": deg}
            for k, c in enumerate(coeffs[::-1], start=0):
                row[f"c{k}"] = float(c)
            coeff_rows.append(row)
        _write_csv(out_dir / "polyfit_coeffs.csv", coeff_rows)
        
        # R² values
        _write_csv(out_dir / "polyfit_r2.csv",
                  [{"degree": int(d), "R2": float(r2)} for d, r2 in self.r2_by_deg.items()])
    
    def _plot_subtracted_curves(self, out_dir: Path, xlab: str, ylab: str):
        """Plot baseline-subtracted curves."""
        for deg in (1, 3, 5, 7):
            if deg not in self.mean_sub_by_deg:
                continue
            
            lines = []
            rows = []
            
            for g in sorted(self.mean_sub_by_deg[deg].keys()):
                x, y = self.mean_sub_by_deg[deg][g]
                lines.append({"x": x, "y": y * 1e9, "label": f"G={g}"})
                
                if self.args.save_csv:
                    rows += [{"degree": deg, "group": g, "x": float(xx), "y_sub": float(yy)}
                            for xx, yy in zip(x, y)]
            
            r2 = self.r2_by_deg.get(deg, float('nan'))
            _save_line_plot(
                out_dir / f"subtracted_deg{deg}.png",
                xlab, ylab, lines,
                title=f"Baseline-subtracted using deg {deg} (R²={r2:.3f})",
                dpi=self.args.dpi, alpha=self.args.alpha
            )
            
            if self.args.save_csv:
                _write_csv(out_dir / f"subtracted_deg{deg}.csv", rows)
            
            # Per-group plots if requested
            if self.args.per_group_plots:
                self._plot_per_group_subtracted(out_dir, deg, xlab, ylab)
    
    def _plot_per_group_subtracted(self, out_dir: Path, deg: int, xlab: str, ylab: str):
        """Plot per-group subtracted curves."""
        for g in sorted(self.mean_sub_by_deg[deg].keys()):
            x, y = self.mean_sub_by_deg[deg][g]
            _save_line_plot(
                out_dir / f"subtracted_deg{deg}_G{g}.png",
                xlab, ylab, [{"x": x, "y": y * 1e9, "label": f"G={g}"}],
                title=f"Subtracted (deg {deg}) — G={g}",
                dpi=self.args.dpi, alpha=self.args.alpha
            )
            
            if self.args.save_csv:
                _write_csv(
                    out_dir / f"subtracted_deg{deg}_G{g}.csv",
                    [{"degree": deg, "group": g, "x": float(xx), "y_sub": float(yy)} 
                     for xx, yy in zip(x, y)]
                )

# ============================= DIAGNOSTICS ===============================

class DiagnosticsGenerator:
    """Generate additional diagnostic plots and analysis."""
    
    def __init__(self, analyzer: IVAnalyzer):
        self.analyzer = analyzer
    
    def generate_all_diagnostics(self, out_dir: Path):
        """Generate all diagnostic plots."""
        self._plot_loop_areas(out_dir)
        self._plot_overlap_diagnostics(out_dir)
        self._plot_differential_conductance(out_dir)
        self._plot_even_odd_decomposition(out_dir)
        self._plot_parity_analysis(out_dir)
        
        if self.analyzer.polyfits:
            self._plot_residual_analysis(out_dir)
            self._plot_model_selection(out_dir)
    
    def _plot_loop_areas(self, out_dir: Path):
        """Plot hysteresis loop areas."""
        loop_rows = []
        
        for g in sorted(self.analyzer.mean_dir.keys()):
            ent = self.analyzer.mean_dir[g]
            if ("fwd" not in ent) or ("bwd" not in ent):
                continue
            
            xF, yF = ent["fwd"]
            xB, yB = ent["bwd"]
            xC = np.intersect1d(xF, xB)
            if xC.size < 2:
                continue
            
            yF_i = np.interp(xC, xF, yF)
            yB_i = np.interp(xC, xB, yB)
            dy = yF_i - yB_i
            area = float(np.trapezoid(dy, xC))
            
            loop_rows.append({
                "group": g, 
                "area_A_V": area, 
                "n_overlap": int(xC.size),
                "x_min": float(xC.min()), 
                "x_max": float(xC.max())
            })
        
        if loop_rows:
            cats = [f"G={r['group']}" for r in loop_rows]
            vals = [r["area_A_V"] for r in loop_rows]
            _save_bar_plot(
                out_dir / "loop_area_by_group.png", 
                cats, vals,
                "Group (range)", "∮ I dV (A·V)",
                "Loop area by group"
            )
            
            if self.analyzer.args.save_csv:
                _write_csv(out_dir / "loop_area_by_group.csv", loop_rows)
    
    def _plot_overlap_diagnostics(self, out_dir: Path):
        """Plot forward/backward overlap diagnostics."""
        ov_rows = []
        
        for g in sorted(self.analyzer.mean_dir.keys()):
            ent = self.analyzer.mean_dir[g]
            if ("fwd" not in ent) or ("bwd" not in ent):
                continue
            
            xF, _ = ent["fwd"]
            xB, _ = ent["bwd"]
            xC = np.intersect1d(xF, xB)
            if xC.size == 0:
                continue
            
            ov_rows.append({
                "group": g, 
                "n_points": int(xC.size),
                "span_V": float(xC.max() - xC.min())
            })
        
        if ov_rows:
            cats = [f"G={r['group']}" for r in ov_rows]
            
            _save_bar_plot(
                out_dir / "overlap_by_group_points.png",
                cats, [r["n_points"] for r in ov_rows],
                "Group", "# common x points",
                "Fwd/Bwd overlap — points"
            )
            
            _save_bar_plot(
                out_dir / "overlap_by_group_span.png",
                cats, [r["span_V"] for r in ov_rows],
                "Group", "Common span (V)",
                "Fwd/Bwd overlap — span"
            )
            
            if self.analyzer.args.save_csv:
                _write_csv(out_dir / "overlap_by_group.csv", ov_rows)
    
    def _plot_differential_conductance(self, out_dir: Path):
        """Plot dI/dV vs V."""
        lines = []
        rows = []
        
        for g in sorted(self.analyzer.mean_full.keys()):
            x, y = self.analyzer.mean_full[g]
            if x.size < 3:
                continue
            
            # Ensure monotonic x for gradient calculation
            idx = np.argsort(x)
            xs, ys = x[idx], y[idx]
            dydx = np.gradient(ys, xs, edge_order=2)
            
            lines.append({"x": xs, "y": dydx * 1e9, "label": f"G={g}"})
            
            if self.analyzer.args.save_csv:
                rows += [{"group": g, "x": float(xx), "dIdV": float(dd)} 
                        for xx, dd in zip(xs, dydx)]
        
        if lines:
            xlab = _x_label_from_arg(self.analyzer.args.x)
            _save_line_plot(
                out_dir / "dIdV_by_group.png",
                xlab, f"dI/dV ({self.analyzer.unit}/V)",
                lines, "Differential conductance dI/dV by group",
                dpi=self.analyzer.args.dpi, alpha=self.analyzer.args.alpha
            )
            
            if self.analyzer.args.save_csv:
                _write_csv(out_dir / "dIdV_by_group.csv", rows)
    
    def _plot_even_odd_decomposition(self, out_dir: Path):
        """Plot even/odd decomposition of curves."""
        odd_lines, even_lines = [], []
        eo_rows = []
        
        for g in sorted(self.analyzer.mean_full.keys()):
            x, y = self.analyzer.mean_full[g]
            if x.size < 2:
                continue
            
            # Find symmetric voltage points
            xs = np.unique(np.abs(x))
            xs = xs[xs > 0]  # Exclude zero to avoid double counting
            
            x_min, x_max = x.min(), x.max()
            xs = xs[(+xs >= x_min) & (+xs <= x_max) & (-xs >= x_min) & (-xs <= x_max)]
            
            if xs.size == 0:
                continue
            
            # Interpolate at ±xs
            y_plus = np.interp(xs, np.sort(x), y[np.argsort(x)])
            y_minus = np.interp(-xs, np.sort(x), y[np.argsort(x)])
            
            # Even/odd decomposition
            y_even = 0.5 * (y_plus + y_minus)
            y_odd = 0.5 * (y_plus - y_minus)
            
            odd_lines.append({"x": xs, "y": y_odd * 1e9, "label": f"G={g}"})
            even_lines.append({"x": xs, "y": y_even * 1e9, "label": f"G={g}"})
            
            if self.analyzer.args.save_csv:
                eo_rows += [
                    {"group": g, "x_abs": float(xx), "I_plus": float(a), "I_minus": float(b),
                     "I_even": float(ev), "I_odd": float(od)}
                    for xx, a, b, ev, od in zip(xs, y_plus, y_minus, y_even, y_odd)
                ]
        
        if odd_lines:
            xlab = "|" + _x_label_from_arg(self.analyzer.args.x) + "|"
            ylab = _ylabel_from_unit(self.analyzer.unit)
            
            _save_line_plot(
                out_dir / "I_odd_overlay.png",
                xlab, ylab, odd_lines,
                "Odd component I_odd(V)",
                dpi=self.analyzer.args.dpi, alpha=self.analyzer.args.alpha
            )
            
            _save_line_plot(
                out_dir / "I_even_overlay.png",
                xlab, ylab, even_lines,
                "Even component I_even(V)",
                dpi=self.analyzer.args.dpi, alpha=self.analyzer.args.alpha
            )
            
            if self.analyzer.args.save_csv:
                _write_csv(out_dir / "even_odd.csv", eo_rows)
    
    def _plot_parity_analysis(self, out_dir: Path):
        """Plot I(+V) vs I(-V) parity scatter."""
        points = []
        rows = []
        
        for g in sorted(self.analyzer.mean_full.keys()):
            x, y = self.analyzer.mean_full[g]
            if x.size < 2:
                continue
            
            xs = np.unique(np.abs(x))
            xs = xs[xs > 0]
            
            x_min, x_max = x.min(), x.max()
            xs = xs[(+xs >= x_min) & (+xs <= x_max) & (-xs >= x_min) & (-xs <= x_max)]
            
            if xs.size == 0:
                continue
            
            y_plus = np.interp(xs, np.sort(x), y[np.argsort(x)])
            y_minus = np.interp(-xs, np.sort(x), y[np.argsort(x)])
            
            points.append({"x": y_minus * 1e9, "y": y_plus * 1e9, "label": f"G={g}"})
            
            if self.analyzer.args.save_csv:
                rows += [{"group": g, "x_abs": float(xx), "I_plus": float(a), "I_minus": float(b)}
                        for xx, a, b in zip(xs, y_plus, y_minus)]
        
        if points:
            # Create scatter plot with diagonal reference
            fig, ax = plt.subplots(figsize=(10, 4.5), dpi=self.analyzer.args.dpi)
            
            for pt in points:
                ax.scatter(pt["x"], pt["y"], alpha=0.7, s=12, label=pt.get("label"))
            
            # Add y=x reference line
            all_vals = np.hstack([np.hstack([pt["x"], pt["y"]]) for pt in points])
            lim = max(np.nanmax(np.abs(all_vals)), 1.0)
            ax.plot([-lim, lim], [-lim, lim], 'k--', lw=1.0, alpha=0.4, label="y=x")
            
            ax.set_xlabel("I(-V) (nA)")
            ax.set_ylabel("I(+V) (nA)")
            ax.grid(True, alpha=0.25)
            ax.legend(frameon=False, fontsize=9, loc="best")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_title("Parity scatter I(+V) vs I(-V)")
            
            fig.tight_layout()
            fig.savefig(out_dir / "parity_scatter.png")
            plt.close(fig)
            
            if self.analyzer.args.save_csv:
                _write_csv(out_dir / "parity_scatter.csv", rows)
    
    def _plot_residual_analysis(self, out_dir: Path):
        """Plot polynomial fit residuals."""
        base_g = 8 if 8 in self.analyzer.mean_back else (max(self.analyzer.mean_back.keys()) if self.analyzer.mean_back else None)
        
        if base_g is None or base_g not in self.analyzer.mean_back:
            return
        
        xb, yb = self.analyzer.mean_back[base_g]
        idx = np.argsort(xb)
        xb0, yb0 = xb[idx], yb[idx]
        
        res_lines = []
        res_rows = []
        
        for deg in (1, 3, 5, 7):
            if deg not in self.analyzer.polyfits:
                continue
            
            yhat = np.polyval(self.analyzer.polyfits[deg], xb0)
            res = yb0 - yhat
            res_lines.append({"x": xb0, "y": res * 1e9, "label": f"deg {deg}"})
            
            if self.analyzer.args.save_csv:
                res_rows += [{"degree": deg, "x": float(xx), "residual": float(rr)} 
                            for xx, rr in zip(xb0, res)]
        
        if res_lines:
            xlab = _x_label_from_arg(self.analyzer.args.x)
            _save_line_plot(
                out_dir / "polyfit_residuals_base.png",
                xlab, f"Residual ({self.analyzer.unit})",
                res_lines, f"Residuals vs V (base G={base_g})",
                dpi=self.analyzer.args.dpi, alpha=self.analyzer.args.alpha
            )
            
            if self.analyzer.args.save_csv:
                _write_csv(out_dir / "polyfit_residuals_base.csv", res_rows)
    
    def _plot_model_selection(self, out_dir: Path):
        """Plot model selection criteria (AIC/BIC, CV)."""
        base_g = 8 if 8 in self.analyzer.mean_back else (max(self.analyzer.mean_back.keys()) if self.analyzer.mean_back else None)
        
        if base_g is None or base_g not in self.analyzer.mean_back:
            return
        
        xb, yb = self.analyzer.mean_back[base_g]
        idx = np.argsort(xb)
        xb0, yb0 = xb[idx], yb[idx]
        
        # AIC/BIC analysis (full polynomial models)
        aicbic_rows = []
        for deg in (1, 3, 5, 7):
            if deg not in self.analyzer.polyfits:
                continue
            
            yhat = np.polyval(self.analyzer.polyfits[deg], xb0)
            n = xb0.size
            k = deg + 1  # Number of parameters (all powers + intercept)
            rss = float(np.sum((yb0 - yhat)**2))
            sigma2 = rss / max(n - k, 1)  # Adjust for degrees of freedom
            
            aic = n * np.log(sigma2 + 1e-300) + 2*k
            bic = n * np.log(sigma2 + 1e-300) + k * np.log(max(n, 1))
            
            aicbic_rows.append({"degree": deg, "RSS": rss, "AIC": aic, "BIC": bic, "n_params": k})
        
        if aicbic_rows:
            cats = [str(r["degree"]) for r in aicbic_rows]
            _save_bar_plot(
                out_dir / "polyfit_aic.png",
                cats, [r["AIC"] for r in aicbic_rows],
                "Polynomial degree", "AIC",
                "AIC by degree (base backtrace)"
            )
            
            if self.analyzer.args.save_csv:
                _write_csv(out_dir / "polyfit_aic_bic.csv", aicbic_rows)
        
        # Cross-validation analysis (full polynomials)
        cv_rows = []
        even_idx = np.arange(xb0.size) % 2 == 0
        odd_idx = ~even_idx
        
        for deg in (1, 3, 5, 7):
            if xb0.size < (deg + 2):  # Need enough points for full polynomial
                continue
            
            try:
                # Fold A: train even, test odd
                pA = np.polynomial.Polynomial.fit(xb0[even_idx], yb0[even_idx], deg).convert().coef[::-1]
                yA = np.polyval(pA, xb0[odd_idx])
                mseA = float(np.mean((yb0[odd_idx] - yA)**2)) if np.any(odd_idx) else np.nan
                
                # Fold B: train odd, test even  
                pB = np.polynomial.Polynomial.fit(xb0[odd_idx], yb0[odd_idx], deg).convert().coef[::-1]
                yB = np.polyval(pB, xb0[even_idx])
                mseB = float(np.mean((yb0[even_idx] - yB)**2)) if np.any(even_idx) else np.nan
                
                mse = float(np.nanmean([mseA, mseB]))
                cv_rows.append({"degree": deg, "MSE_cv": mse})
            except Exception:
                continue
        
        if cv_rows:
            fig, ax = plt.subplots(figsize=(8, 3.2), dpi=self.analyzer.args.dpi)
            ax.plot([r["degree"] for r in cv_rows], [r["MSE_cv"] for r in cv_rows], 'o-')
            ax.set_xlabel("Polynomial degree")
            ax.set_ylabel("2-fold CV MSE")
            ax.set_title("Cross-validated MSE (base backtrace)")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(out_dir / "polyfit_cv_mse.png")
            plt.close(fig)
            
            if self.analyzer.args.save_csv:
                _write_csv(out_dir / "polyfit_cv_mse.csv", cv_rows)

# ============================= MAIN FUNCTION ===============================

def main() -> None:
    """Main analysis pipeline."""
    ap = argparse.ArgumentParser(
        description="Comprehensive IV curve analysis with directional means and polynomial baseline subtraction"
    )
    
    # Input/output arguments
    ap.add_argument("--stage-root", type=Path, required=True, help="Root directory containing stage data")
    ap.add_argument("--procedures-yaml", type=Path, required=True, help="Procedures configuration file")
    ap.add_argument("--out-dir", type=Path, default=None, help="Output directory (default: <stage-root>/_plots)")
    ap.add_argument("--date", type=str, default=None, help="Date string (default: most recent Sept 11)")
    ap.add_argument("--proc-filter", type=str, default="IV", help="Procedure name filter")
    
    # Analysis parameters
    ap.add_argument("--x", type=str, default="vsd", choices=["vsd", "vg", "auto"], help="Voltage axis choice")
    ap.add_argument("--decimate", type=int, default=1, help="Data decimation factor")
    ap.add_argument("--mean-samples", type=int, default=800, help="Number of samples for arc-length mean")
    ap.add_argument("--round-dec", type=int, default=2, help="Decimal places for voltage binning")
    ap.add_argument("--dv-threshold", type=float, default=1e-6, help="Voltage threshold for direction detection")
    
    # Output options
    ap.add_argument("--save-csv", action="store_true", help="Save data to CSV files")
    ap.add_argument("--per-group-plots", action="store_true", help="Generate per-group baseline-subtracted plots")
    ap.add_argument("--diagnostics", action="store_true", help="Generate comprehensive diagnostics")
    
    # Plot formatting
    ap.add_argument("--dpi", type=int, default=220, help="Plot DPI")
    ap.add_argument("--alpha", type=float, default=0.95, help="Plot line transparency")
    ap.add_argument("--figsize", type=str, default="10,4.5", help="Figure size as 'width,height'")
    
    args = ap.parse_args()
    
    # Validate inputs
    if not args.stage_root.exists():
        raise SystemExit(f"[error] stage root not found: {args.stage_root}")
    if not args.procedures_yaml.exists():
        raise SystemExit(f"[error] procedures.yml not found: {args.procedures_yaml}")
    
    # Setup
    date_str = args.date or most_recent_sept11()
    procedures = load_procedures_yaml(args.procedures_yaml)
    out_dir = args.out_dir or (args.stage_root / "_plots")
    
    print(f"[info] Processing date: {date_str}")
    print(f"[info] Output directory: {out_dir}")
    
    # Initialize analyzer
    analyzer = IVAnalyzer(args)
    
    # Load and process data
    if not analyzer.load_data(args.stage_root, date_str, args.proc_filter, procedures):
        print("[info] No data to process")
        return
    
    print(f"[info] Loaded data for {len([g for g, trs in analyzer.groups.items() if trs])} groups")
    
    # Compute means and fits
    analyzer.compute_means()
    analyzer.fit_polynomials()
    
    if not analyzer.mean_full:
        print("[info] No mean curves computed")
        return
    
    # Generate plots
    analyzer.generate_plots(out_dir, date_str)
    
    # Generate diagnostics if requested
    if args.diagnostics:
        print("[info] Generating diagnostics...")
        diagnostics = DiagnosticsGenerator(analyzer)
        diagnostics.generate_all_diagnostics(out_dir / f"date={date_str}")
    
    print(f"[info] Analysis complete. Results saved to {out_dir}/date={date_str}")

if __name__ == "__main__":
    main()
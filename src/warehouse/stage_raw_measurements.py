"""
Parallel staging of raw lab CSV runs into partitioned Parquet (02_stage/raw_measurements),
driven by a YAML procedures schema. Safe for multi-process execution.

Key improvements vs. single-process:
- ProcessPoolExecutor with configurable --workers
- Avoids manifest write races via per-run event files (then a single merge step)
- Unique reject filenames (no collisions across processes)
- Atomic writes to Parquet (tmp rename)
- Optional cap on Polars threads via POLARS_MAX_THREADS

Usage example:
    python stage_raw_measurements_parallel.py \
      --raw-root 01_raw \
      --stage-root 02_stage/raw_measurements \
      --procedures-yaml procedures.yml \
      --workers 6 \
      --polars-threads 1
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import polars as pl
import yaml

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None  # type: ignore


# ----------------------------- Config & Helpers -----------------------------

DEFAULT_LOCAL_TZ = "America/Santiago"  # Controls date partitioning only
DEFAULT_WORKERS = 6                    # good starting point for 6-core CPUs
DEFAULT_POLARS_THREADS = 1             # to avoid oversubscription with processes

# Section detectors: tolerant, case-insensitive
PROC_LINE_RE   = re.compile(r"^#\s*Procedure\s*:\s*<([^>]+)>\s*$", re.I)
PARAMS_LINE_RE = re.compile(r"^#\s*Parameters\s*:\s*$", re.I)   # matches "#Parameters:" too
META_LINE_RE   = re.compile(r"^#\s*Metadata\s*:\s*$", re.I)     # matches "#Metadata:" too
DATA_LINE_RE   = re.compile(r"^#\s*Data\s*:\s*$", re.I)

# Generic "key: value" within header blocks
KV_PAT = re.compile(r"^#\s*([^:]+):\s*(.*)\s*$")

UNIT_VAL_RE = re.compile(
    r"^\s*(?P<num>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*(?P<unit>[^\s]+)?\s*$"
)

SPECIAL_DATA_RENAMES = {
    "t (s)": "t_s",
    "I (A)": "I_A",
    "VL ((V|v))": "VL_V",  # tolerate lowercase v in unit
    "Plate T (degC)": "plate_C",
    "Ambient T (degC)": "ambient_C",
    "Clock (ms)": "clock_ms",
}

UNIT_SUFFIXES = {
    "V": "_V",
    "A": "_A",
    "degC": "_C",
    "C": "_C",
    "s": "_s",
    "ms": "_ms",
    "K": "_K",
}


def warn(msg: str) -> None:
    print(f"[warn] {msg}", file=sys.stderr)


def sha1_short(s: str, n: int = 16) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:n]


def md5_file(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with p.open("rb") as f:
        while (b := f.read(chunk)):
            h.update(b)
    return h.hexdigest()


def parse_number_unit(s: Any) -> Tuple[Optional[float], Optional[str]]:
    """Return (number, unit) if s looks like '3.2 V'. Otherwise (None, None)."""
    if s is None:
        return None, None
    if isinstance(s, (int, float)):
        return float(s), None
    m = UNIT_VAL_RE.match(str(s))
    if not m:
        return None, None
    num = float(m.group("num"))
    unit = m.group("unit")
    return num, unit


def to_bool(s: Any) -> bool:
    return str(s).strip().lower() in {"1", "true", "yes", "on", "y"}


def parse_datetime_any(x: Any) -> Optional[dt.datetime]:
    """
    Accept epoch (int/float/str) or ISO-like string; return aware UTC datetime.
    Returns None if it cannot parse.
    """
    if x is None:
        return None
    if isinstance(x, (int, float)):
        try:
            return dt.datetime.fromtimestamp(float(x), tz=dt.timezone.utc)
        except Exception:
            return None
    s = str(x).strip()
    # Try epoch in string
    try:
        return dt.datetime.fromtimestamp(float(s), tz=dt.timezone.utc)
    except Exception:
        pass
    # Try ISO (allow missing timezone → assume UTC)
    try:
        d = dt.datetime.fromisoformat(s)
        if d.tzinfo is None:
            d = d.replace(tzinfo=dt.timezone.utc)
        return d.astimezone(dt.timezone.utc)
    except Exception:
        return None


def local_date_for_partition(ts_utc: dt.datetime, tz_name: str) -> str:
    if tz_name and ZoneInfo is not None:
        try:
            local = ts_utc.astimezone(ZoneInfo(tz_name))
            return local.date().isoformat()
        except Exception:
            pass
    return ts_utc.date().isoformat()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def extract_date_from_path(p: Path) -> Optional[str]:
    """
    Try to find a YYYY-MM-DD or YYYY_MM_DD or YYYYMMDD anywhere in the path.
    Returns 'YYYY-MM-DD' or None.
    """
    s = str(p)
    # 2025-09-12 or 2025_09_12
    m = re.search(r"\b(\d{4})[-_](\d{2})[-_](\d{2})\b", s)
    if m:
        y, mo, d = m.groups()
        try:
            dt.date(int(y), int(mo), int(d))
            return f"{y}-{mo}-{d}"
        except Exception:
            pass
    # 20250912
    m = re.search(r"\b(\d{8})\b", s)
    if m:
        raw = m.group(1)
        y, mo, d = raw[:4], raw[4:6], raw[6:8]
        try:
            dt.date(int(y), int(mo), int(d))
            return f"{y}-{mo}-{d}"
        except Exception:
            pass
    return None


# ----------------------------- YAML Procedures ------------------------------

@dataclass
class ProcSpec:
    params: Dict[str, str]  # expected types for Parameters
    meta: Dict[str, str]    # expected types for Metadata
    data: Dict[str, str]    # expected types for Data (informational; not enforced here)


# Global per-process cache
_PROC_CACHE: Dict[str, ProcSpec] | None = None
_PROC_YAML_PATH: Path | None = None


def load_procedures_yaml(path: Path) -> Dict[str, ProcSpec]:
    with path.open("r", encoding="utf-8") as f:
        y = yaml.safe_load(f) or {}
    procs = {}
    root = y.get("procedures", {})
    for name, blocks in root.items():
        ps = ProcSpec(
            params=blocks.get("Parameters", {}) or {},
            meta=blocks.get("Metadata", {}) or {},
            data=blocks.get("Data", {}) or {},
        )
        procs[name] = ps
    return procs


def get_procs_cached(path: Path) -> Dict[str, ProcSpec]:
    global _PROC_CACHE, _PROC_YAML_PATH
    if _PROC_CACHE is None or _PROC_YAML_PATH != path:
        _PROC_CACHE = load_procedures_yaml(path)
        _PROC_YAML_PATH = path
    return _PROC_CACHE


# ----------------------------- Header Parsing -------------------------------

@dataclass
class HeaderBlocks:
    proc: Optional[str]
    parameters: Dict[str, str]
    metadata: Dict[str, str]
    data_header_line: Optional[int]  # line index (0-based) for CSV header row after '# Data:'


def parse_header(path: Path) -> HeaderBlocks:
    proc = None
    params: Dict[str, str] = {}
    meta: Dict[str, str] = {}
    data_header_line: Optional[int] = None
    mode: Optional[str] = None

    with path.open("r", errors="ignore", encoding="utf-8") as f:
        for i, raw in enumerate(f):
            line = raw.rstrip("\n")
            s = line.strip()

            if DATA_LINE_RE.match(s):
                data_header_line = i + 1  # next line is header row
                break

            m = PROC_LINE_RE.match(s)
            if m:
                proc = m.group(1).split(".")[-1].strip()
                continue

            if PARAMS_LINE_RE.match(s):
                mode = "params"
                continue
            if META_LINE_RE.match(s):
                mode = "meta"
                continue

            if s.startswith("#"):
                m = KV_PAT.match(s)
                if m:
                    key = m.group(1).strip()
                    val = m.group(2).strip()
                    if mode == "params":
                        params[key] = val
                    elif mode == "meta":
                        meta[key] = val

    # Fallback: some files put Start time under Parameters if '#Metadata:' wasn't detected
    if "Start time" in params and "Start time" not in meta:
        meta["Start time"] = params["Start time"]

    return HeaderBlocks(proc=proc, parameters=params, metadata=meta, data_header_line=data_header_line)


# --------------------------- Casting / Normalizing --------------------------

def cast_block(block: Dict[str, str], spec: Dict[str, str]) -> Dict[str, Any]:
    """
    Cast a header key/value dict according to a type spec (from YAML).
    Supports: int, float, str, bool, datetime, float_no_unit.
    For 'float' values with units '3.2 V', the numeric part is taken.
    """
    out: Dict[str, Any] = {}
    for k, v in block.items():
        t = (spec.get(k) or "str").strip().lower()
        if t == "int":
            num, _ = parse_number_unit(v)
            out[k] = int(num) if num is not None else int(float(str(v)))
        elif t == "float":
            num, _ = parse_number_unit(v)
            out[k] = float(num) if num is not None else float(str(v))
        elif t == "float_no_unit":
            out[k] = float(str(v))
        elif t == "bool":
            out[k] = to_bool(v)
        elif t == "datetime":
            dtv = parse_datetime_any(v)
            if dtv is None:
                continue
            out[k] = dtv
        else:
            out[k] = v
    return out


def standardize_numeric_cols(df: pl.DataFrame) -> pl.DataFrame:
    """
    Rename common instrument columns to canonical names with unit suffixes.
    """
    ren = {}
    for c in df.columns:
        # Exact renames first
        if c in SPECIAL_DATA_RENAMES:
            ren[c] = SPECIAL_DATA_RENAMES[c]
            continue

        # Tolerate case variations like "VL (v)"
        for pat, target in SPECIAL_DATA_RENAMES.items():
            try:
                if re.fullmatch(pat, c):
                    ren[c] = target
                    break
            except re.error:
                pass

        if c in ren:
            continue

        m = re.match(r"^(.*)\s*\(([^)]+)\)\s*$", c)
        if m:
            base = m.group(1).strip()
            unit = m.group(2).strip()
            suffix = UNIT_SUFFIXES.get(unit, f"_{unit}")
            base_clean = re.sub(r"\s+", "_", base).lower()
            ren[c] = f"{base_clean}{suffix}"
            continue

        if c.upper() == "VG":
            ren[c] = "VG_V"
        elif c.upper() in {"VD", "VDS"}:
            ren[c] = "VD_V"
        elif c.upper() == "I":
            ren[c] = "I_A"

    df = df.rename(ren)

    float_cols = [
        "t_s", "I_A", "VG_V", "VD_V", "VL_V", "plate_C", "ambient_C", "clock_ms",
    ]
    for fc in float_cols:
        if fc in df.columns:
            df = df.with_columns(pl.col(fc).cast(pl.Float64, strict=False))

    return df


def derive_light_flags(params: Dict[str, Any]) -> Tuple[bool, Optional[float], Optional[float]]:
    wl = params.get("Laser wavelength")
    lv = params.get("Laser voltage")
    try:
        wl_f = float(wl) if wl is not None else None
    except Exception:
        wl_f = None
    try:
        lv_f = float(lv) if lv is not None else None
    except Exception:
        lv_f = None
    with_light = (wl_f is not None) and (lv_f is not None) and (lv_f != 0.0)
    return with_light, wl_f, lv_f


# ------------------------------- Ingestion ----------------------------------

def read_numeric_table(path: Path, header_line: Optional[int]) -> pl.DataFrame:
    """
    Read the CSV numeric table. Prefer comment filtering;
    fallback to skip header lines if needed.
    """
    try:
        return pl.read_csv(
            path,
            comment_prefix="#",
            has_header=True,
            infer_schema_length=10000,
            try_parse_dates=True,
            low_memory=True,
            truncate_ragged_lines=True,
        )
    except Exception:
        return pl.read_csv(
            path,
            skip_rows=(header_line or 0),
            has_header=True,
            infer_schema_length=10000,
            try_parse_dates=True,
            low_memory=True,
            truncate_ragged_lines=True,
        )


def resolve_start_dt_and_date(
    src: Path,
    meta: Dict[str, Any],
    local_tz: str,
) -> Tuple[dt.datetime, str, str]:
    """
    Determine start_dt (UTC) and the date partition.
    Fallbacks: (a) parse 'Start time' from meta; (b) date from path; (c) file mtime.
    Returns (start_dt_utc, date_part, origin), where origin is 'meta'|'path'|'mtime'.
    """
    st = meta.get("Start time")
    dtv = st if isinstance(st, dt.datetime) else parse_datetime_any(st)
    if isinstance(dtv, dt.datetime):
        return dtv, local_date_for_partition(dtv, local_tz), "meta"

    dpath = extract_date_from_path(src)
    if dpath:
        if ZoneInfo is not None:
            tz = ZoneInfo(local_tz)
            local_midnight = dt.datetime.combine(dt.date.fromisoformat(dpath), dt.time(), tzinfo=tz)
            utc_dt = local_midnight.astimezone(dt.timezone.utc)
        else:
            utc_dt = dt.datetime.fromisoformat(dpath + "T00:00:00").replace(tzinfo=dt.timezone.utc)
        return utc_dt, dpath, "path"

    mtime = dt.datetime.fromtimestamp(src.stat().st_mtime, tz=dt.timezone.utc)
    dpart = local_date_for_partition(mtime, local_tz)
    return mtime, dpart, "mtime"


def atomic_write_parquet(df: pl.DataFrame, out_file: Path) -> None:
    """Write to a temp file in the same dir, then rename atomically."""
    ensure_dir(out_file.parent)
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=out_file.parent) as tmp:
        tmp_path = Path(tmp.name)
    try:
        df.write_parquet(tmp_path)
        tmp_path.replace(out_file)  # atomic on same filesystem
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
        raise


def ingest_file_task(
    src_str: str,
    stage_root_str: str,
    procedures_yaml_str: str,
    local_tz: str,
    force: bool,
    events_dir_str: str,
    rejects_dir_str: str,
) -> Dict[str, Any]:
    """Worker task: process one CSV path, return an event dict and write per-run event file."""
    src = Path(src_str)
    stage_root = Path(stage_root_str)
    procedures_yaml = Path(procedures_yaml_str)
    events_dir = Path(events_dir_str)
    rejects_dir = Path(rejects_dir_str)

    # Load YAML once per process
    procs = get_procs_cached(procedures_yaml)

    try:
        hb = parse_header(src)
        if not hb.proc:
            raise RuntimeError("missing '# Procedure:'")
        proc = hb.proc

        spec = procs.get(proc, ProcSpec({}, {}, {}))

        params = cast_block(hb.parameters, spec.params)
        meta = cast_block(hb.metadata, spec.meta)

        if "Start time" in params and "Start time" not in meta:
            meta["Start time"] = params["Start time"]

        start_dt, date_part, origin = resolve_start_dt_and_date(src, meta, local_tz)

        rid = sha1_short(f"{src.as_posix()}|{start_dt.timestamp()}")

        # Read table & standardize
        df = read_numeric_table(src, hb.data_header_line)
        if df.height == 0:
            raise RuntimeError("empty data table")
        df = standardize_numeric_cols(df)

        with_light, wl_f, lv_f = derive_light_flags(params)

        out_dir = stage_root / f"proc={proc}" / f"date={date_part}" / f"run_id={rid}"
        out_file = out_dir / "part-000.parquet"

        if out_file.exists() and not force:
            event = {
                "ts": dt.datetime.now(tz=dt.timezone.utc),
                "status": "skipped",
                "run_id": rid,
                "proc": proc,
                "rows": df.height,
                "path": str(out_file),
                "source_file": str(src),
                "date_origin": origin,
            }
        else:
            extra_cols = {
                "run_id": rid,
                "proc": proc,
                "start_dt": start_dt,
                "source_file": str(src),
                "with_light": with_light,
                "wavelength_nm": wl_f,
                "laser_voltage_V": lv_f,
                "chip_group": params.get("Chip group name"),
                "chip_number": params.get("Chip number"),
                "sample": params.get("Sample"),
                "procedure_version": params.get("Procedure version"),
            }
            df = df.with_columns([pl.lit(v).alias(k) for k, v in extra_cols.items()])
            atomic_write_parquet(df, out_file)

            event = {
                "ts": dt.datetime.now(tz=dt.timezone.utc),
                "status": "ok",
                "run_id": rid,
                "proc": proc,
                "rows": df.height,
                "path": str(out_file),
                "source_file": str(src),
                "date_origin": origin,
            }

        # Write per-run event JSON (unique by run_id)
        ev_path = Path(events_dir) / f"event-{rid}.json"
        ensure_dir(ev_path.parent)
        with ev_path.open("w", encoding="utf-8") as f:
            json.dump(event, f, ensure_ascii=False, default=str)

        return event

    except Exception as e:
        # Unique reject filename using path hash
        phash = sha1_short(src.as_posix(), 12)
        rej_path = Path(rejects_dir) / f"{src.stem}-{phash}.reject.json"
        ensure_dir(rej_path.parent)
        rec = {"source_file": str(src), "error": str(e), "ts": dt.datetime.now(tz=dt.timezone.utc).isoformat()}
        with rej_path.open("w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)
        return {"status": "reject", "source_file": str(src), "error": str(e)}


def discover_csvs(root: Path) -> list[Path]:
    EXCLUDE_DIRS = {".git", ".venv", "__pycache__", ".ipynb_checkpoints"}
    files: list[Path] = []
    for p in root.rglob("*.csv"):
        if any(part in EXCLUDE_DIRS for part in p.parts):
            continue
        if p.name.startswith("._"):
            continue
        files.append(p)
    files.sort()
    return files


def merge_events_to_manifest(events_dir: Path, manifest_path: Path) -> None:
    """Merge all event JSONs into a single Parquet manifest (idempotent)."""
    ev_files = sorted(events_dir.glob("event-*.json"))
    if not ev_files:
        return
    rows = []
    for e in ev_files:
        try:
            rows.append(json.loads(e.read_text(encoding="utf-8")))
        except Exception:
            continue
    if not rows:
        return
    df = pl.DataFrame(rows)
    ensure_dir(manifest_path.parent)
    if manifest_path.exists():
        # Concatenate and drop duplicates by (run_id, ts, status, path)
        prev = pl.read_parquet(manifest_path)
        all_df = pl.concat([prev, df], how="vertical_relaxed")
        all_df = all_df.unique(subset=["run_id", "ts", "status", "path"], keep="last")
        all_df.write_parquet(manifest_path)
    else:
        df.write_parquet(manifest_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Parallel stage raw CSV runs → Parquet (02_stage/raw_measurements)")
    ap.add_argument("--raw-root", type=Path, required=True, help="Root folder containing dated subfolders with CSVs (01_raw)")
    ap.add_argument("--stage-root", type=Path, required=True, help="Output root (02_stage/raw_measurements)")
    ap.add_argument("--procedures-yaml", type=Path, required=True, help="YAML schema of procedures and types")
    ap.add_argument("--rejects-dir", type=Path, default=None, help="Folder to write reject records (default: {stage_root}/../_rejects)")
    ap.add_argument("--events-dir", type=Path, default=None, help="Folder for per-run event JSON (default: {stage_root}/_manifest/events)")
    ap.add_argument("--manifest", type=Path, default=None, help="Path to merged manifest parquet (default: {stage_root}/_manifest/manifest.parquet)")
    ap.add_argument("--local-tz", type=str, default=DEFAULT_LOCAL_TZ, help="Timezone name for date partitioning (default: America/Santiago)")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help=f"Process workers (default: {DEFAULT_WORKERS})")
    ap.add_argument("--polars-threads", type=int, default=DEFAULT_POLARS_THREADS, help=f"POLARS_MAX_THREADS per worker (default: {DEFAULT_POLARS_THREADS})")
    ap.add_argument("--force", action="store_true", help="Overwrite if a staged Parquet already exists")
    args = ap.parse_args()

    raw_root: Path = args.raw_root
    stage_root: Path = args.stage_root
    rejects_dir: Path = args.rejects_dir or (stage_root.parent / "_rejects")
    events_dir: Path = args.events_dir or (stage_root / "_manifest" / "events")
    manifest_path: Path = args.manifest or (stage_root / "_manifest" / "manifest.parquet")
    local_tz: str = args.local_tz
    workers: int = max(1, args.workers)
    polars_threads: int = max(1, args.polars_threads)
    force: bool = args.force

    if not raw_root.exists():
        raise SystemExit(f"[error] raw root does not exist: {raw_root}")
    ensure_dir(stage_root)
    ensure_dir(rejects_dir)
    ensure_dir(events_dir)
    ensure_dir(manifest_path.parent)

    # Cap per-process Polars threads to avoid oversubscription
    os.environ["POLARS_MAX_THREADS"] = str(polars_threads)

    # Eagerly load procedures once in the parent (helps fail fast if path is wrong)
    _ = get_procs_cached(args.procedures_yaml)

    csvs = discover_csvs(raw_root)
    print(f"[info] discovered {len(csvs)} CSV files under {raw_root}")
    if not csvs:
        print("[done] nothing to do.")
        return

    # Submit tasks
    submitted = 0
    ok = skipped = reject = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = []
        for src in csvs:
            fut = ex.submit(
                ingest_file_task,
                str(src),
                str(stage_root),
                str(args.procedures_yaml),
                local_tz,
                force,
                str(events_dir),
                str(rejects_dir),
            )
            futs.append((src, fut))
            submitted += 1

        # Gather results
        for i, (src, fut) in enumerate(futs, 1):
            try:
                out = fut.result()
            except Exception as e:
                reject += 1
                print(f"[{i:04d}]  REJECT {src} :: {e}")
                continue

            st = out.get("status")
            if st == "ok":
                ok += 1
            elif st == "skipped":
                skipped += 1
            elif st == "reject":
                reject += 1

            if st in {"ok", "skipped"}:
                print(f"[{i:04d}] {st.upper():>7} {out['proc']:<6} rows={out['rows']:<7} → {out['path']}  ({out.get('date_origin','meta')})")
            else:
                print(f"[{i:04d}]  REJECT {src} :: {out.get('error')}")

    # Merge per-run events to a single manifest parquet (single-thread, race-free)
    merge_events_to_manifest(events_dir, manifest_path)

    print(f"[done] staging complete  |  ok={ok}  skipped={skipped}  rejects={reject}  submitted={submitted}")


if __name__ == "__main__":
    main()
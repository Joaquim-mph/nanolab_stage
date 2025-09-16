#!/usr/bin/env python3
"""
Stage raw lab CSV runs into partitioned Parquet (02_stage/raw_measurements),
driven by a YAML procedures schema.

This version (v1.1) fixes:
- Robust detection of header sections: accepts "#Metadata:" and "#Parameters:" without a space,
  is case-insensitive, and trims whitespace.
- If 'Start time' is missing/unparseable, we now fall back to a date parsed from the path
  (e.g., dated parent folders like 2025-09-12 or 20250912). If that fails, we fall back to the file mtime.
- Emits clear warnings when fallbacks are used, preventing everything from landing in 1969.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import polars as pl
import yaml

try:
    # Python 3.9+
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None  # type: ignore


# ----------------------------- Config & Helpers -----------------------------

DEFAULT_LOCAL_TZ = "America/Santiago"  # Controls date partitioning only

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
    "VL (V)": "VL_V",
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
                warn(f"could not parse datetime for {k!r}: {v!r}")
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
        if c in SPECIAL_DATA_RENAMES:
            ren[c] = SPECIAL_DATA_RENAMES[c]
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
    # (a) try Start time
    st = meta.get("Start time")
    dtv = st if isinstance(st, dt.datetime) else parse_datetime_any(st)
    if isinstance(dtv, dt.datetime):
        return dtv, local_date_for_partition(dtv, local_tz), "meta"

    # (b) try path
    dpath = extract_date_from_path(src)
    if dpath:
        if ZoneInfo is not None:
            tz = ZoneInfo(local_tz)
            local_midnight = dt.datetime.combine(dt.date.fromisoformat(dpath), dt.time(), tzinfo=tz)
            utc_dt = local_midnight.astimezone(dt.timezone.utc)
        else:
            # assume UTC if no zoneinfo
            utc_dt = dt.datetime.fromisoformat(dpath + "T00:00:00").replace(tzinfo=dt.timezone.utc)
        warn(f"{src.name}: using date from path: {dpath}")
        return utc_dt, dpath, "path"

    # (c) file mtime
    mtime = dt.datetime.fromtimestamp(src.stat().st_mtime, tz=dt.timezone.utc)
    dpart = local_date_for_partition(mtime, local_tz)
    warn(f"{src.name}: using file mtime for date partition: {dpart}")
    return mtime, dpart, "mtime"


def ingest_file(
    src: Path,
    stage_root: Path,
    procs: Dict[str, "ProcSpec"],
    rejects_dir: Optional[Path],
    local_tz: str = DEFAULT_LOCAL_TZ,
    force: bool = False,
) -> Dict[str, Any]:
    hb = parse_header(src)
    if not hb.proc:
        raise RuntimeError("missing '# Procedure:'")
    proc = hb.proc

    spec = procs.get(proc, ProcSpec({}, {}, {}))

    params = cast_block(hb.parameters, spec.params)
    meta = cast_block(hb.metadata, spec.meta)

    # Ensure Start time isn't trapped in params because of odd headers
    if "Start time" in params and "Start time" not in meta:
        meta["Start time"] = params["Start time"]

    start_dt, date_part, origin = resolve_start_dt_and_date(src, meta, local_tz)

    # run_id and relative source
    rid = sha1_short(f"{src.as_posix()}|{start_dt.timestamp()}")

    # Read table & standardize column names/types
    df = read_numeric_table(src, hb.data_header_line)
    if df.height == 0:
        raise RuntimeError("empty data table")
    df = standardize_numeric_cols(df)

    with_light, wl_f, lv_f = derive_light_flags(params)

    # Prepare output directories & file
    out_dir = stage_root / f"proc={proc}" / f"date={date_part}" / f"run_id={rid}"
    out_file = out_dir / "part-000.parquet"

    if out_file.exists() and not force:
        return {
            "status": "skipped",
            "run_id": rid,
            "proc": proc,
            "rows": df.height,
            "reason": "exists",
            "path": str(out_file),
        }

    ensure_dir(out_dir)

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
    df.write_parquet(out_file)

    return {
        "status": "ok",
        "run_id": rid,
        "proc": proc,
        "rows": df.height,
        "path": str(out_file),
        "date_origin": origin,
    }


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


def write_reject(rejects_dir: Path, src: Path, err: Exception) -> None:
    ensure_dir(rejects_dir)
    rec = {
        "source_file": str(src),
        "error": str(err),
    }
    jpath = rejects_dir / (src.stem + ".reject.json")
    with jpath.open("w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)


def append_manifest(stage_root: Path, record: Dict[str, Any]) -> None:
    man_dir = stage_root / "_manifest"
    ensure_dir(man_dir)
    p = man_dir / "manifest.parquet"
    row = pl.DataFrame([record])
    if p.exists():
        existing = pl.read_parquet(p)
        out = pl.concat([existing, row], how="vertical_relaxed")
        out.write_parquet(p)
    else:
        row.write_parquet(p)


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage raw CSV runs → Parquet (02_stage/raw_measurements)")
    ap.add_argument("--raw-root", type=Path, required=True, help="Root folder containing dated subfolders with CSVs (01_raw)")
    ap.add_argument("--stage-root", type=Path, required=True, help="Output root (02_stage/raw_measurements)")
    ap.add_argument("--procedures-yaml", type=Path, required=True, help="YAML schema of procedures and types")
    ap.add_argument("--rejects-dir", type=Path, default=None, help="Folder to write small JSON reject records (default: {stage_root}/../_rejects)")
    ap.add_argument("--local-tz", type=str, default=DEFAULT_LOCAL_TZ, help="Timezone name for date partitioning (default: America/Santiago)")
    ap.add_argument("--force", action="store_true", help="Overwrite if a staged Parquet already exists")
    args = ap.parse_args()

    raw_root: Path = args.raw_root
    stage_root: Path = args.stage_root
    rejects_dir: Path = args.rejects_dir or (stage_root.parent / "_rejects")
    local_tz: str = args.local_tz
    force: bool = args.force

    if not raw_root.exists():
        raise SystemExit(f"[error] raw root does not exist: {raw_root}")
    ensure_dir(stage_root)
    ensure_dir(rejects_dir)

    procs = load_procedures_yaml(args.procedures_yaml)
    if not procs:
        warn("procedures.yml has no 'procedures' entries; continuing with permissive casting.")

    csvs = discover_csvs(raw_root)
    print(f"[info] discovered {len(csvs)} CSV files under {raw_root}")

    for i, src in enumerate(csvs, 1):
        try:
            out = ingest_file(
                src=src,
                stage_root=stage_root,
                procs=procs,
                rejects_dir=rejects_dir,
                local_tz=local_tz,
                force=force,
            )
            append_manifest(stage_root, {
                "ts": dt.datetime.now(tz=dt.timezone.utc),
                "source_file": str(src),
                "status": out.get("status"),
                "run_id": out.get("run_id"),
                "proc": out.get("proc"),
                "rows": out.get("rows"),
                "path": out.get("path"),
                "date_origin": out.get("date_origin"),
            })
            print(f"[{i:04d}] {out['status']:>7} {out['proc']:<6} rows={out['rows']:<7} → {out['path']}  ({out.get('date_origin','meta')})")
        except Exception as e:
            write_reject(rejects_dir, src, e)
            print(f"[{i:04d}]  REJECT {src} :: {e}")

    print("[done] staging complete")

if __name__ == "__main__":
    main()

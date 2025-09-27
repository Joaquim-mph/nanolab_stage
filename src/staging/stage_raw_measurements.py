# file: stage_raw_measurements_yaml_parallel.py
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import polars as pl
import yaml

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None  # type: ignore


# ----------------------------- Config -----------------------------

DEFAULT_LOCAL_TZ = "America/Santiago"
DEFAULT_WORKERS = 6
DEFAULT_POLARS_THREADS = 1

PROC_LINE_RE   = re.compile(r"^#\s*Procedure\s*:\s*<([^>]+)>\s*$", re.I)
PARAMS_LINE_RE = re.compile(r"^#\s*Parameters\s*:\s*$", re.I)
META_LINE_RE   = re.compile(r"^#\s*Metadata\s*:\s*$", re.I)
DATA_LINE_RE   = re.compile(r"^#\s*Data\s*:\s*$", re.I)
KV_PAT         = re.compile(r"^#\s*([^:]+):\s*(.*)\s*$")

def warn(msg: str) -> None:
    print(f"[warn] {msg}", file=sys.stderr)

def sha1_short(s: str, n: int = 16) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:n]

def to_bool(s: Any) -> bool:
    return str(s).strip().lower() in {"1", "true", "yes", "on", "y"}

def parse_number_unit(s: Any) -> Tuple[Optional[float], Optional[str]]:
    if s is None:
        return None, None
    if isinstance(s, (int, float)):
        return float(s), None
    m = re.match(r"^\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*([^\s]+)?\s*$", str(s))
    if not m:
        return None, None
    return float(m.group(1)), m.group(2)

def parse_datetime_any(x: Any) -> Optional[dt.datetime]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        try:
            return dt.datetime.fromtimestamp(float(x), tz=dt.timezone.utc)
        except Exception:
            return None
    s = str(x).strip()
    try:
        return dt.datetime.fromtimestamp(float(s), tz=dt.timezone.utc)
    except Exception:
        pass
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
            return ts_utc.astimezone(ZoneInfo(tz_name)).date().isoformat()
        except Exception:
            pass
    return ts_utc.date().isoformat()

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def extract_date_from_path(p: Path) -> Optional[str]:
    s = str(p)
    m = re.search(r"\b(\d{4})[-_](\d{2})[-_](\d{2})\b", s)
    if m:
        y, mo, d = m.groups()
        try:
            dt.date(int(y), int(mo), int(d))
            return f"{y}-{mo}-{d}"
        except Exception:
            pass
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


# ----------------------------- YAML ------------------------------

@dataclass
class ProcSpec:
    params: Dict[str, str]
    meta: Dict[str, str]
    data: Dict[str, str]

_PROC_CACHE: Dict[str, ProcSpec] | None = None
_PROC_YAML_PATH: Path | None = None

def load_procedures_yaml(path: Path) -> Dict[str, ProcSpec]:
    with path.open("r", encoding="utf-8") as f:
        y = yaml.safe_load(f) or {}
    procs = {}
    root = y.get("procedures", {}) or {}
    for name, blocks in root.items():
        procs[name] = ProcSpec(
            params=(blocks.get("Parameters") or {}) ,
            meta=(blocks.get("Metadata") or {}) ,
            data=(blocks.get("Data") or {}) ,
        )
    return procs

def get_procs_cached(path: Path) -> Dict[str, ProcSpec]:
    global _PROC_CACHE, _PROC_YAML_PATH
    if _PROC_CACHE is None or _PROC_YAML_PATH != path:
        _PROC_CACHE = load_procedures_yaml(path)
        _PROC_YAML_PATH = path
    return _PROC_CACHE


# ----------------------------- Header parse -------------------------------

@dataclass
class HeaderBlocks:
    proc: Optional[str]
    parameters: Dict[str, str]
    metadata: Dict[str, str]
    data_header_line: Optional[int]

def parse_header(path: Path) -> HeaderBlocks:
    proc = None
    params: Dict[str, str] = {}
    meta: Dict[str, str] = {}
    data_header_line: Optional[int] = None
    mode: Optional[str] = None

    with path.open("r", errors="ignore", encoding="utf-8") as f:
        for i, raw in enumerate(f):
            s = raw.rstrip("\n").strip()

            if DATA_LINE_RE.match(s):
                data_header_line = i + 1
                break

            m = PROC_LINE_RE.match(s)
            if m:
                proc = m.group(1).split(".")[-1].strip()
                continue
            if PARAMS_LINE_RE.match(s):
                mode = "params"; continue
            if META_LINE_RE.match(s):
                mode = "meta"; continue

            if s.startswith("#"):
                m = KV_PAT.match(s)
                if m:
                    key = m.group(1).strip()
                    val = m.group(2).strip()
                    if mode == "params":
                        params[key] = val
                    elif mode == "meta":
                        meta[key] = val

    if "Start time" in params and "Start time" not in meta:
        meta["Start time"] = params["Start time"]

    return HeaderBlocks(proc=proc, parameters=params, metadata=meta, data_header_line=data_header_line)


# --------------------------- Casting / Normalizing --------------------------

def cast_block(block: Dict[str, str], spec: Dict[str, str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in block.items():
        t = (spec.get(k) or "str").strip().lower()
        try:
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
                dtv = v if isinstance(v, dt.datetime) else parse_datetime_any(v)
                if dtv is not None:
                    out[k] = dtv
            else:
                out[k] = v
        except Exception:
            out[k] = v  # why: do not crash staging on bad header value
    return out


# -------- Column matching to YAML "Data" names (tolerant, but target = YAML) --------

def _norm(s: str) -> str:
    # why: robust compare across "Vsd (V)" vs "vd_v" vs "VDS"
    s = s.strip()
    s = s.lower()
    s = re.sub(r"\s+", "", s)
    s = s.replace("_", "")
    s = s.replace("-", "")
    s = s.replace("°", "deg")
    s = s.replace("℃", "degc")
    s = s.replace("(", "").replace(")", "")
    s = s.replace("[", "").replace("]", "")
    return s

# Synonym seeds per YAML key (targets); each value is a list of regexes or literal alt names.
YAML_DATA_SYNONYMS: Dict[str, List[str]] = {
    "I (A)":         [r"^i$", r"^i_a$", r"^id(_a)?$", r"^ids(_a)?$", r"^current(_a)?$"],
    "Vsd (V)":       [r"^vsd(_v)?$", r"^vd(_v)?$", r"^vds(_v)?$", r"^vdv$", r"^vds$"],
    "Vg (V)":        [r"^vg(_v)?$", r"^v_g(_v)?$", r"^gate_v(_v)?$"],
    "VL (V)":        [r"^vl(_v)?$", r"^vlv$"],
    "t (s)":         [r"^t(_s)?$", r"^time(_s)?$"],
    "Plate T (degC)":[r"^platet(degc)?$", r"^platetemp(degc)?$", r"^plate_c$"],
    "Ambient T (degC)":[r"^ambientt(degc)?$", r"^ambienttemp(degc)?$", r"^ambient_c$"],
    "Clock (ms)":    [r"^clock(_ms)?$"],
    "Vg (V)":        [r"^vg(_v)?$"],
}

def build_yaml_rename_map(df_cols: List[str], yaml_data: Dict[str, str]) -> Dict[str, str]:
    """
    Return mapping {src_col -> target_yaml_col}. If multiple df cols match a target, pick first stable order.
    """
    rename: Dict[str, str] = {}
    by_norm = {_norm(c): c for c in df_cols}

    for target in yaml_data.keys():
        target_norm = _norm(target)
        # 1) exact normalized match
        if target_norm in by_norm:
            rename[by_norm[target_norm]] = target
            continue
        # 2) synonym match
        for pat in YAML_DATA_SYNONYMS.get(target, []):
            # normalize pattern "like" DF names as well
            # compare normalized df col against regex sans punctuation
            rx = re.compile(pat, re.I)
            # try direct df columns
            hit = None
            for c in df_cols:
                if rx.fullmatch(c) or rx.fullmatch(_norm(c)):
                    hit = c; break
            if hit:
                rename[hit] = target
                break
        # 3) uppercase version heuristic (e.g., "VSD (V)")
        if target not in rename:
            upper_guess = target.upper().replace(" ", "")
            for c in df_cols:
                if _norm(c) == _norm(upper_guess):
                    rename[c] = target
                    break
    return rename

def cast_df_data_types(df: pl.DataFrame, yaml_data: Dict[str, str]) -> pl.DataFrame:
    """Cast columns present in df to the types declared in YAML Data."""
    casts = []
    for col, typ in yaml_data.items():
        if col not in df.columns:
            continue
        t = typ.strip().lower()
        if t in {"float", "float_no_unit"}:
            casts.append(pl.col(col).cast(pl.Float64, strict=False).alias(col))
        elif t == "int":
            casts.append(pl.col(col).cast(pl.Int64, strict=False).alias(col))
        elif t == "bool":
            casts.append(pl.when(pl.col(col).is_in([True, False]))
                          .then(pl.col(col))
                          .otherwise(pl.col(col).cast(pl.Utf8, strict=False).str.to_lowercase().is_in(["1","true","yes","on","y"]))
                          .alias(col))
        elif t == "datetime":
            # store as Utf8; many data tables won't have datetimes; skip heavy parsing here
            casts.append(pl.col(col).cast(pl.Utf8, strict=False).alias(col))
        else:
            casts.append(pl.col(col).cast(pl.Utf8, strict=False).alias(col))
    if casts:
        df = df.with_columns(casts)
    return df


# ------------------------------- IO ----------------------------------

def read_numeric_table(path: Path, header_line: Optional[int]) -> pl.DataFrame:
    try:
        return pl.read_csv(
            path,
            comment_prefix="#",
            has_header=True,
            infer_schema_length=10000,
            try_parse_dates=False,
            low_memory=True,
            truncate_ragged_lines=True,
        )
    except Exception:
        return pl.read_csv(
            path,
            skip_rows=(header_line or 0),
            has_header=True,
            infer_schema_length=10000,
            try_parse_dates=False,
            low_memory=True,
            truncate_ragged_lines=True,
        )

def resolve_start_dt_and_date(src: Path, meta: Dict[str, Any], local_tz: str) -> Tuple[dt.datetime, str, str]:
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
    ensure_dir(out_file.parent)
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=out_file.parent) as tmp:
        tmp_path = Path(tmp.name)
    try:
        df.write_parquet(tmp_path)
        tmp_path.replace(out_file)
    except Exception:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        raise


# ------------------------------- Worker ----------------------------------

def ingest_file_task(
    src_str: str,
    stage_root_str: str,
    procedures_yaml_str: str,
    local_tz: str,
    force: bool,
    events_dir_str: str,
    rejects_dir_str: str,
    only_yaml_data: bool,
) -> Dict[str, Any]:
    src = Path(src_str)
    stage_root = Path(stage_root_str)
    procedures_yaml = Path(procedures_yaml_str)
    events_dir = Path(events_dir_str)
    rejects_dir = Path(rejects_dir_str)

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

        df = read_numeric_table(src, hb.data_header_line)
        if df.height == 0:
            raise RuntimeError("empty data table")

        # --- NEW: rename data columns to exact YAML "Data" names ---
        if spec.data:
            ren_map = build_yaml_rename_map(df.columns, spec.data)
            if ren_map:
                df = df.rename(ren_map)
            # Optionally drop non-YAML columns
            if only_yaml_data:
                keep = [c for c in spec.data.keys() if c in df.columns]
                df = df.select(keep)
            # Cast per YAML types (for those present)
            df = cast_df_data_types(df, spec.data)

        # Derive light flags (works even with YAML naming; keys come from Parameters)
        with_light = False
        wl_f = None
        lv_f = None
        try:
            wl_f = float(params.get("Laser wavelength")) if params.get("Laser wavelength") is not None else None
        except Exception:
            wl_f = None
        try:
            lv_f = float(params.get("Laser voltage")) if params.get("Laser voltage") is not None else None
        except Exception:
            lv_f = None
        with_light = (wl_f is not None) and (lv_f is not None) and (lv_f != 0.0)

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

        ev_path = events_dir / f"event-{rid}.json"
        ensure_dir(ev_path.parent)
        with ev_path.open("w", encoding="utf-8") as f:
            json.dump(event, f, ensure_ascii=False, default=str)
        return event

    except Exception as e:
        phash = sha1_short(src.as_posix(), 12)
        rej_path = Path(rejects_dir) / f"{src.stem}-{phash}.reject.json"
        ensure_dir(rej_path.parent)
        rec = {"source_file": str(src), "error": str(e), "ts": dt.datetime.now(tz=dt.timezone.utc).isoformat()}
        with rej_path.open("w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)
        return {"status": "reject", "source_file": str(src), "error": str(e)}


# ------------------------------- Orchestration ----------------------------------

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
        prev = pl.read_parquet(manifest_path)
        all_df = pl.concat([prev, df], how="vertical_relaxed")
        all_df = all_df.unique(subset=["run_id", "ts", "status", "path"], keep="last")
        all_df.write_parquet(manifest_path)
    else:
        df.write_parquet(manifest_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage raw CSVs → Parquet using YAML Data names (parallel & atomic).")
    ap.add_argument("--raw-root", type=Path, required=True, help="Root folder with CSVs (01_raw)")
    ap.add_argument("--stage-root", type=Path, required=True, help="Output root (02_stage/raw_measurements)")
    ap.add_argument("--procedures-yaml", type=Path, required=True, help="YAML schema of procedures and types")
    ap.add_argument("--rejects-dir", type=Path, default=None, help="Folder for reject records (default: {stage_root}/../_rejects)")
    ap.add_argument("--events-dir", type=Path, default=None, help="Per-run event JSONs (default: {stage_root}/_manifest/events)")
    ap.add_argument("--manifest", type=Path, default=None, help="Merged manifest parquet (default: {stage_root}/_manifest/manifest.parquet)")
    ap.add_argument("--local-tz", type=str, default=DEFAULT_LOCAL_TZ, help="Timezone for date partitioning")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Process workers")
    ap.add_argument("--polars-threads", type=int, default=DEFAULT_POLARS_THREADS, help="POLARS_MAX_THREADS per worker")
    ap.add_argument("--force", action="store_true", help="Overwrite staged Parquet if exists")
    ap.add_argument("--only-yaml-data", action="store_true", help="Drop non-YAML data columns")
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
    only_yaml_data: bool = args.only_yaml_data

    if not raw_root.exists():
        raise SystemExit(f"[error] raw root does not exist: {raw_root}")
    ensure_dir(stage_root); ensure_dir(rejects_dir); ensure_dir(events_dir); ensure_dir(manifest_path.parent)

    os.environ["POLARS_MAX_THREADS"] = str(polars_threads)

    _ = get_procs_cached(args.procedures_yaml)  # fail fast if invalid

    csvs = discover_csvs(raw_root)
    print(f"[info] discovered {len(csvs)} CSV files under {raw_root}")
    if not csvs:
        print("[done] nothing to do.")
        return

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
                only_yaml_data,
            )
            futs.append((src, fut)); submitted += 1

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
                print(f"[{i:04d}] {st.upper():>7} {out['proc']:<8} rows={out['rows']:<7} → {out['path']}  ({out.get('date_origin','meta')})")
            else:
                print(f"[{i:04d}]  REJECT {src} :: {out.get('error')}")

    merge_events_to_manifest(events_dir, manifest_path)
    print(f"[done] staging complete  |  ok={ok}  skipped={skipped}  rejects={reject}  submitted={submitted}")


if __name__ == "__main__":
    main()

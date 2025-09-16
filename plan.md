# Lab Experiments DB (Polars) — blueprint

This blueprint sets up a pragmatic, fast, and reproducible data system around your CSV‑based lab runs using **Polars**. It’s designed for:

* quick ingestion from a folder tree of dated subfolders
* robust metadata extraction from `#` headers
* typed, unit‑clean columns (SI)
* lazy analytics & plotting at scale
* simple, file‑based warehousing with Parquet

---

## 1) High‑level approach

**Data lake layout (medallion)**

```
./data/
  01_raw/                      # as collected (read‑only)
  02_stage/                    # row‑level cleaned parquet, partitioned
    raw_measurements/          # per‑run data (1 file per run or per proc/day)
  03_curated/                  # star‑schema parquet tables for analysis
    warehouse/
      runs_metadata/           # 1 row = 1 run
      ts_fact/                 # time‑series fact (ITS, time traces)
      sweep_fact/              # sweep fact (IVg/IVd etc.)
      dims/                    # small dimensions (device, procedure, light…)
  _manifest.parquet            # file audit + provenance (optional)
```

**Identifiers & keys**

* `run_id` (str): stable hash of `{relpath}|{start_time_epoch}`
* `device_id` (str): hash of `{chip_group}|{chip_number}` (or any device tuple)
* `proc` (cat): e.g. `IVg`, `ITS`, `ITt` (exactly as in your headers)
* `start_dt` (datetime\[ns, localtz or UTC])

**Partitioning** (Parquet folders)

* `proc={proc}/date={YYYY-MM-DD}` under `02_stage/raw_measurements/`
* `date={YYYY-MM}` under curated tables when helpful

Rationale: Fast pruning on common filters (by procedure, by day/month).

---

## 2) Canonical schema (curated)

### 2.1 `runs_metadata` (one row per CSV/run)

Minimal columns (extend as needed):

* `run_id` (str) — **PK**
* `proc` (cat)
* `source_file` (str) — relative path from project root
* `file_idx` (int) — optional sequential index inside a day
* `start_dt` (datetime) — parsed from header epoch; tz‑aware
* `chip_group` (str) — e.g. *Alisson*
* `chip_number` (float|int|str) — keep numeric if it’s always numeric
* `sample` (str)
* `with_light` (bool)
* `wavelength_nm` (float) — if available
* `laser_voltage_V` (float) — if available
* `info` (str) — free text ("Information")
* `script_version` / `procedure_version` (str)
* `n_rows` (int) — data row count
* `md5` (str) — file checksum for dedupe

Any extra parameters can live in a **struct** column `params` (Polars `Struct`) to avoid exploding columns while keeping type safety.

### 2.2 `ts_fact` (time‑series fact)

For procedures that produce a time axis (e.g., `ITS`, `ITt`):

* `run_id` (str) — **FK** to `runs_metadata`
* `t_s` (float)
* Common signals (wide): `I_A`, `VG_V`, `VD_V`, `VL_V`, `plate_C`, `ambient_C`, `clock_ms`

> If different procedures add more signals, include them; missing columns just stay null for unrelated procs. Keep it **wide** for speed.

### 2.3 `sweep_fact` (sweeps like `IVg`/`IVd`)

* `run_id` (str)
* Sweep axis (one or more): `VG_V`, `VD_V`
* Responses: `I_A`, etc.
* Optional step index: `step_idx` (int)

### 2.4 Dimensions (optional but handy)

* `device_dim(device_id, chip_group, chip_number, …)`
* `procedure_dim(proc, data_schema, expected_axes, …)`
* `light_dim(run_id, wavelength_nm, power_mW, beam_diam_um, …)`

---

## 3) Parsing & typing rules

* Header lines start with `#`. Blocks:

  * `# Procedure: <...>` → `proc`
  * `# Parameters:` → key: value pairs
  * `# Metadata:` → at least `Start time: <epoch or ISO>`
  * `# Data:` → next line is CSV header

* **Type map** (configurable): `int, float, bool, str, datetime, float_no_unit`

* **Units**: store **SI numeric columns**; retain original unit tags only in `params` if needed. Example: "VG: -2 V" → `VG_V = -2.0`.

* Datetime: prefer `UTC` internally; convert to local tz on display.

> Keep a `procedures.yml` that specifies expected fields & types per `proc`. Unknown keys land in `params`.

---

## 4) Ingestion pipeline (script outline)

1. **Discover files** under a given root; exclude junk (`.__`, `.ipynb_checkpoints`, etc.).
2. Build a **manifest** (path, size, mtime, hash, first bytes) to detect duplicates and incremental work.
3. For each new/changed file:

   * parse header → `proc`, `parameters`, `metadata`
   * infer & cast types (using YAML schema)
   * read numeric table with `polars.read_csv(comment_prefix="#", truncate_ragged_lines=True, …)`
   * standardize column names → SI units (`VG_V`, `I_A`, …)
   * write row‑level Parquet to `02_stage/raw_measurements/proc=.../date=.../run_id=.../part.parquet`
   * upsert a row into `03_curated/warehouse/runs_metadata/`
   * if time series → append to `ts_fact`; if sweep → append to `sweep_fact`

Everything can be done with **Polars Lazy** and small batches.

---

## 5) Code skeletons

### 5.1 Constants & helpers

```python
# constants.py
from __future__ import annotations
from pathlib import Path
import hashlib, re, json, datetime as dt
import polars as pl

RAW_ROOT = Path("data/01_raw")
STAGE_ROOT = Path("data/02_stage/raw_measurements")
WH_ROOT = Path("data/03_curated/warehouse")
EXCLUDE_DIRS = {".git", ".venv", "__pycache__", ".ipynb_checkpoints"}

UNIT_PAT = re.compile(r"^(?P<val>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*(?P<u>\w+)?$")

TYPE_MAP = {
    "int": int,
    "float": float,
    "bool": lambda x: str(x).strip().lower() in {"1","true","yes","on"},
    "str": str,
    "datetime": lambda x: dt.datetime.fromtimestamp(float(x), tz=dt.timezone.utc),
    "float_no_unit": float,
}

def md5_file(p: Path, chunk=1<<20) -> str:
    h = hashlib.md5()
    with p.open('rb') as f:
        while (b := f.read(chunk)):
            h.update(b)
    return h.hexdigest()

def parse_number_unit(s: str) -> tuple[float | None, str | None]:
    m = UNIT_PAT.match(str(s).strip())
    if not m:
        return None, None
    v = float(m.group("val"))
    u = m.group("u")
    return v, u
```

### 5.2 Header parser (tolerant)

```python
# header.py
from __future__ import annotations
from pathlib import Path
import re, datetime as dt

PROC_PAT = re.compile(r"^#\s*Procedure:\s*<([^>]+)>")
KV_PAT   = re.compile(r"^#\s*([\w\s()+/.-]+):\s*(.*)$")

class Header:
    def __init__(self):
        self.proc: str | None = None
        self.parameters: dict[str, str] = {}
        self.metadata: dict[str, str] = {}
        self.data_header_line: int | None = None


def parse_header(path: Path) -> Header:
    h = Header()
    mode = None
    with path.open('r', errors='ignore') as f:
        for i, line in enumerate(f):
            if line.startswith("# Data:"):
                h.data_header_line = i + 1  # next line is csv header
                break
            if (m := PROC_PAT.match(line)):
                h.proc = m.group(1).split('.')[-1]  # keep short name
                continue
            if line.startswith("# Parameters:"):
                mode = "params"; continue
            if line.startswith("# Metadata:"):
                mode = "meta"; continue
            if line.startswith("#") and (m := KV_PAT.match(line)):
                k = m.group(1).strip()
                v = m.group(2).strip()
                if mode == "params":
                    h.parameters[k] = v
                elif mode == "meta":
                    h.metadata[k] = v
    return h
```

### 5.3 Type casting & standard columns

```python
# typing_casts.py
from __future__ import annotations
import datetime as dt
import polars as pl
from .constants import TYPE_MAP, parse_number_unit

# Example minimal expectations per proc (extend via YAML)
PROC_CASTS = {
    "ITS": {
        "params": {"VG": "float", "VDS": "float", "Laser wavelength": "float", "Laser voltage": "float"},
        "meta":   {"Start time": "datetime"},
    },
    "IVg": {
        "params": {"VDS": "float", "N_avg": "int"},
        "meta":   {"Start time": "datetime"},
    },
}

SI_RENAMES = {
    "VG": "VG_V", "VDS": "VD_V", "VL": "VL_V",
    "I": "I_A", "t (s)": "t_s", "Plate T (degC)": "plate_C", "Ambient T (degC)": "ambient_C",
}


def cast_param_block(proc: str, params: dict[str,str]) -> dict:
    spec = PROC_CASTS.get(proc, {}).get("params", {})
    out = {}
    for k, v in params.items():
        # drop units into SI numeric when possible
        val, unit = parse_number_unit(v)
        if k in spec:
            caster = TYPE_MAP[ spec[k] ]
            if spec[k] == "float" and val is not None:
                out[k] = float(val)
            else:
                out[k] = caster(v)
        else:
            out[k] = val if val is not None else v
    return out


def cast_meta_block(proc: str, meta: dict[str,str]) -> dict:
    spec = PROC_CASTS.get(proc, {}).get("meta", {})
    out = {}
    for k, v in meta.items():
        if k in spec:
            out[k] = TYPE_MAP[ spec[k] ](v)
        else:
            out[k] = v
    return out


def standardize_numeric_cols(df: pl.DataFrame) -> pl.DataFrame:
    cols = df.columns
    # Basic SI renames when present
    ren = {c: SI_RENAMES[c] for c in cols if c in SI_RENAMES}
    df = df.rename(ren)
    # Ensure numeric dtypes for known signals if present
    enforce = {
        "t_s": pl.Float64, "I_A": pl.Float64, "VG_V": pl.Float64, "VD_V": pl.Float64,
        "VL_V": pl.Float64, "plate_C": pl.Float64, "ambient_C": pl.Float64, "Clock (ms)": pl.Float64,
    }
    for c, dtp in enforce.items():
        if c in df.columns:
            df = df.with_columns(pl.col(c).cast(dtp, strict=False))
    return df
```

### 5.4 Ingest one file → stage + curated

```python
# ingest_one.py
from __future__ import annotations
from pathlib import Path
import os, datetime as dt, hashlib
import polars as pl
from .constants import RAW_ROOT, STAGE_ROOT, WH_ROOT, md5_file
from .header import parse_header
from .typing_casts import cast_param_block, cast_meta_block, standardize_numeric_cols


def run_id_for(path: Path, start_ts: float | None) -> str:
    key = f"{path.as_posix()}|{start_ts or 0}"
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def read_numeric_table(path: Path, header_line: int | None) -> pl.DataFrame:
    try:
        return pl.read_csv(path, comment_prefix="#", has_header=True,
                           infer_schema_length=10000, try_parse_dates=True,
                           low_memory=True, truncate_ragged_lines=True)
    except Exception:
        # fallback: skip known header lines
        return pl.read_csv(path, skip_rows=header_line or 0, has_header=True,
                           infer_schema_length=10000, try_parse_dates=True,
                           low_memory=True, truncate_ragged_lines=True)


def ingest_file(src: Path) -> dict:
    h = parse_header(src)
    if not h.proc:
        raise RuntimeError(f"{src} missing # Procedure")
    params = cast_param_block(h.proc, h.parameters)
    meta   = cast_meta_block(h.proc, h.metadata)

    # Derive standard flags
    wavelength = params.get("Laser wavelength")
    lvolt = params.get("Laser voltage")
    with_light = (wavelength is not None) and (float(lvolt or 0.0) != 0.0)

    # Numeric table
    df = read_numeric_table(src, h.data_header_line)
    df = standardize_numeric_cols(df)
    n_rows = df.height

    # Identify & layout
    start_ts = None
    st = meta.get("Start time")
    if isinstance(st, dt.datetime):
        start_ts = st.timestamp()
        start_dt = st
    else:
        # attempt parse if it was kept as raw string
        try:
            start_ts = float(st)
            start_dt = dt.datetime.fromtimestamp(start_ts, tz=dt.timezone.utc)
        except Exception:
            start_dt = dt.datetime.utcfromtimestamp(0).replace(tzinfo=dt.timezone.utc)

    rid = run_id_for(src.relative_to(RAW_ROOT), start_ts)
    day = start_dt.date().isoformat()

    # Write stage parquet
    stage_dir = STAGE_ROOT / f"proc={h.proc}" / f"date={day}" / f"run_id={rid}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    df.write_parquet(stage_dir / "part-000.parquet")

    # Upsert curated rows (append-only files; compaction later)
    md_row = {
        "run_id": rid,
        "proc": h.proc,
        "source_file": str(src.relative_to(RAW_ROOT)),
        "start_dt": start_dt,
        "chip_group": params.get("Chip group name"),
        "chip_number": params.get("Chip number"),
        "sample": params.get("Sample"),
        "with_light": with_light,
        "wavelength_nm": wavelength,
        "laser_voltage_V": lvolt,
        "info": params.get("Information"),
        "procedure_version": params.get("Procedure version"),
        "n_rows": n_rows,
        "md5": md5_file(src),
        "params": params,  # Polars Struct if created via pl.from_dicts
    }

    pl.DataFrame([md_row]).write_parquet(WH_ROOT / "runs_metadata" / f"date={day}" / f"runs-{rid}.parquet")

    # Route to facts
    if "t_s" in df.columns:  # time‑series like ITS
        df.with_columns(pl.lit(rid).alias("run_id")).write_parquet(WH_ROOT / "ts_fact" / f"date={day}" / f"ts-{rid}.parquet")
    else:  # assume sweep (IVg/IVd). Adjust if needed.
        df.with_columns(pl.lit(rid).alias("run_id")).write_parquet(WH_ROOT / "sweep_fact" / f"date={day}" / f"sw-{rid}.parquet")

    return {"run_id": rid, "proc": h.proc, "rows": n_rows}
```

### 5.5 Batch ingest

```python
# ingest_all.py
from __future__ import annotations
from pathlib import Path
import polars as pl
from .constants import RAW_ROOT, EXCLUDE_DIRS
from . ingest_one import ingest_file


def discover_csvs(root: Path):
    for p in root.rglob("*.csv"):
        if any(part in EXCLUDE_DIRS for part in p.parts):
            continue
        if p.name.startswith("._"):
            continue
        yield p


def main():
    for p in discover_csvs(RAW_ROOT):
        try:
            out = ingest_file(p)
            print("[ok]", out)
        except Exception as e:
            print(f"[warn] {p}: {e}")

if __name__ == "__main__":
    main()
```

---

## 6) Common queries (Polars Lazy)

**Load all metadata & count runs per day**

```python
lf_md = pl.scan_parquet("data/03_curated/warehouse/runs_metadata/date=*/runs-*.parquet")
(
  lf_md
  .group_by_dynamic("start_dt", every="1d")
  .agg(pl.len().alias("n"))
  .sort("start_dt")
  .collect()
)
```

**Filter by procedure + wavelength + chip**

```python
lf_md.filter(
    (pl.col("proc") == "ITS") &
    (pl.col("wavelength_nm").is_between(365, 470)) &
    (pl.col("chip_number") == 71)
).collect()
```

**Join time‑series with metadata for plotting**

```python
lf_ts = pl.scan_parquet("data/03_curated/warehouse/ts_fact/date=*/ts-*.parquet")
lf = (
  lf_ts.join(lf_md.select(["run_id","proc","wavelength_nm","with_light","chip_group","chip_number"]), on="run_id")
)
```

**Get all `ITS` at `VG≈-2V` and overlay**

```python
its = (
  lf.filter((pl.col("proc")=="ITS") & (pl.col("VG_V").abs().sub(2.0).abs() < 1e-3))
    .select(["run_id","t_s","I_A","wavelength_nm","chip_number"])
    .collect()
)
```

**IVg sweeps for a chip, chronological order**

```python
lf_sw = pl.scan_parquet("data/03_curated/warehouse/sweep_fact/date=*/sw-*.parquet")
(
  lf_sw.join(lf_md, on="run_id")
       .filter((pl.col("proc")=="IVg") & (pl.col("chip_number")==71))
       .select(["start_dt","run_id","VG_V","I_A","with_light"])
       .sort(["start_dt","run_id","VG_V"])
       .collect()
)
```

---

## 7) Plotting notes

* Use **matplotlib** / **polars.DataFrame.to\_pandas()** only if a lib requires Pandas; otherwise keep Polars native then feed arrays.
* Time‑series overlays: group by `(VG_V rounded, wavelength_nm)` or metadata tags (dark/light via `VL_V > threshold`).
* Store figure PNGs under `./figures/{proc}/date=YYYY-MM/…` with consistent naming: `{chip}_{proc}_{runid}.png`.

---

## 8) Validation & provenance

* Simple checks (Polars):

  * `pl.any_horizontal(pl.all().is_null())` rate by column
  * ranges: `I_A.is_between(-1, 1)` (example)
* Keep a `provenance.json` per ingest batch with script git SHA and run timestamp.
* Consider adding a `checks/` module for assertions per procedure.

---

## 9) Performance tips

* Always use **Lazy** (`scan_parquet`) for analytics.
* Partition by `proc` and time to prune.
* Avoid tiny Parquet files in curated tables → **compact** occasionally (read many → write fewer larger files per month partition).
* Use `Float64` for currents/voltages, `Categorical` for `proc`.

---

## 10) Next steps

1. Wire a `procedures.yml` to drive the casts (types + unit rules) instead of the hard‑coded `PROC_CASTS`.
2. Add a tiny CLI (`typer`) with commands: `ingest`, `compact`, `describe`, `plot`.
3. Migrate existing ad‑hoc scripts to read from curated Parquet tables (no more CSV parsing during analysis).
4. Add computed columns (e.g., light ON/OFF segments from `VL_V` threshold) in curated facts for convenience.

This gives you a clean spine now, and space to grow (dimensions, richer validation, or a DuckDB/Delta layer later if needed).

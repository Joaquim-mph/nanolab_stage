# nanolab_stage
storage-processing-reduction of data

Raw CSVs (01_raw)
   └─► stage_raw_measurements.py  →  02_stage/raw_measurements/proc=…/date=…/run_id=…/part-000.parquet
                                       + _manifest/events + manifest.parquet
          └─► build_curated_from_stage.py  → 03_curated/warehouse/
                                              ├─ runs_metadata/date=YYYY-MM/runs-<RID>.parquet
                                              ├─ ts_fact/date=YYYY-MM/ts-<RID>.parquet
                                              └─ sweep_fact/date=YYYY-MM/sw-<RID>.parquet



python src/warehouse/build_curated_from_stage_parallel.py --raw-root data/01_raw --stage-root data/02_stage/raw_measurements --procedures-yaml config/procedures.yml --workers 4 --polars-threads 2

python build_curated_from_stage_parallel.py --stage-root 02_stage/raw_measurements --warehouse-root 03_curated/warehouse --workers 4 --polars-threads 2


python src/warehouse/sanity_check_lab.py --raw-root data/01_raw --stage-root data/02_stage/raw_measurements --warehouse-root data/03_curated/warehouse --local-tz America/Santiago --top-n 6 --plots


python src/warehouse/collect_stats.py --repo-root . --warehouse data/03_curated/warehouse --out-stats data/03_curated/warehouse/stats --docs-dir docs --local-tz America/Santiago --print

python src/ploting/collect_stats_with_requested_plots.py --repo-root . --plots --print
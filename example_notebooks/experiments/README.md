# Experiments

Canonical experiment entrypoints now live under:

- `baselines/`
- `rlb/`
- `drlb/`
- `all_comparison/`

Each named run writes artifacts under its family directory, for example:

- `example_notebooks/experiments/drlb/drlb_smooth/metrics.json`
- `example_notebooks/experiments/drlb/drlb_smooth/config/normalized_config.json`
- `example_notebooks/experiments/drlb/drlb_smooth/config/split_manifest.json`
- `example_notebooks/experiments/drlb/drlb_smooth/outputs/run_summary.json`

Default validation uses the permanent subsample split in `data/fpa/subsample/`.
To switch a run to the normal/full split, change only the selected `split_set`.

For notebook-driven analysis, use the in-process helpers from
`example_notebooks.experiments.notebook_api` instead of shelling out to
`runner.py`. They reuse the same family profiles and runner logic, but return
live Python objects:

```python
from example_notebooks.experiments.notebook_api import run_drlb_profile_inprocess

result = run_drlb_profile_inprocess(
    run_name="drlb_smooth",
    split_set="subsample_train_val_holdout",
    n_trials=1,
    max_train_steps=8,
)

summary = result["summary"]
bidder = result["best_run"]["bidder"]
diagnostics = result["best_run"]["diagnostics"]
```

For `baselines` and `rlb`, use `run_baseline_profile_inprocess(...)` and
`run_rlb_profile_inprocess(...)`. CLI runners remain the lightweight path for
artifact generation and quick metrics collection.

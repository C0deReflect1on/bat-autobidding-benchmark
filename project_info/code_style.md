# Code Style

## Scope

Conventions for the `bat-autobidding-benchmark` repository, updated after the v2 refactoring. Follow these when adding new experiment runners, bidders, or state representations.

---

## Imports & Packaging

- The project is installed as an editable package (`pip install -e .`). **Never** add `sys.path` hacks.
- Import order: standard library, third-party, local (`simulator.*`, `config`, `utils`).
- Absolute imports everywhere except within `simulator/` where single-level relative imports are fine.

## Type Hints

All **new** code must use type hints:

```python
def build_bidder_params(
    base_params: dict,
    model_params: dict,
    *,
    state_type: str,
    objective: str = "clicks",
    verbose: bool = False,
) -> dict:
```

- Use `dict`, `list`, `tuple` (not `typing.Dict` etc.) on Python >= 3.10.
- Use `X | None` instead of `Optional[X]`.
- Return types are mandatory for public functions.
- Frozen dataclasses must use `@dataclass(frozen=True)`.

Existing code that already works should **not** be rewritten just to add hints -- do it when you touch the function for another reason.

## Path Handling

- Always use `pathlib.Path`. **No** `os.path.join`.
- Data paths in `config.py` are `Path` objects; keep them that way.

## Logging

- Use `print(...)` with bracketed prefixes: `[autobidder_check]`, `[runner]`, etc.
- Control verbosity via `verbose`, `debug_logs`, `use_tqdm` flags.
- Do **not** introduce the `logging` module; the codebase is small and print-based.

---

## Bidder Conventions

### Interface

All bidders subclass `_Bidder` from `simulator/model/bidder.py`.

Required: `place_bid(bidding_input_params, history) -> float`.

Trainable bidders also implement `fit`, `save_model`, `load_model`.

### Constructor

Bidders accept a flat `params: dict`. Read values with `params.get("key", default)`. Do not introduce typed config objects for the constructor -- keep the dict-based contract.

### Registration

Add one import and one entry to the `BIDDERS` dict in `simulator/model/__init__.py`.

---

## DRLB State Representations

Each state variant is a frozen dataclass in `simulator/model/drlb/state_representations.py`.

Required interface:

| Field / Method | Type |
|---|---|
| `state_size` | `int` |
| `state_action_size` | `int` |
| `reward_net_order` | `str` (`"predict_first"` or `"learn_first"`) |
| `uses_campaign_meta` | `bool` |
| `get_state(agent) -> np.ndarray` | |
| `compute_step_metrics(agent) -> None` | |
| `reset_step_fields(agent) -> None` | |

To add a variant: create a class, register it in `STATE_REPRESENTATIONS` under a string key; use that key as `model.state_type`. No other files need to change.

---

## Experiment Runner Conventions

### Structure

Each experiment lives in `example_notebooks/experiments/exp_<name>/` and has a single `run_<name>.py` that:

1. Defines `STATE_TYPE`, `OBJECTIVE`, `BASE_DRLB_PARAMS`, `BASELINE_MODEL_PARAMS`.
2. Defines a `search_space(trial) -> dict` function.
3. Calls `runner_utils.run_optuna_experiment(...)`.

Keep runners thin (~70 lines). All shared logic lives in `runner_utils.py`.

### Config

Use `ExperimentConfig.train_val(...)` factory for train/val-split experiments. Use explicit `ExperimentConfig` subclasses only when pointing to different data paths (e.g. global train/test split).

### Artifacts

- `run_summary.json` -- standardized per-run output (git hash, data hash, baseline, best trial, all trials).
- `runs_index.jsonl` -- append-only log of runs.
- Per-trial `*_metrics.json` and `*_training_diagnostics.csv` in `outputs/`.
- Optuna trial `.pt` checkpoints go in a `TemporaryDirectory`; only the best is copied out.

### Reproducibility

`run_summary.json` includes:

- `git_hash` (short SHA of HEAD)
- `data_hash` (SHA-256 of train + val campaign CSVs)
- `split_metadata` (seed, val_fraction, row counts)
- `random_seed` in the experiment config

---

## Frozen Zone Rules

**Never modify** (changes break metric comparisons):

- `simulator/simulation/` -- auction loop, CTR/CVR, containers
- `simulator/validation/` -- `autobidder_check`, `compile_metrics`
- `simulator/model/bidder.py` -- `_Bidder` ABC
- `simulator/model/traffic.py` -- traffic share data
- `data/` -- benchmark datasets

---

## Formatting

- Four-space indentation.
- Line length: 120 characters.
- Imports sorted: stdlib, third-party, local (separated by blank lines).
- Trailing commas in multi-line collections.
- Comments only for non-obvious intent. No narration comments.
- English only in comments and docstrings.

## Metric Naming Reference

The canonical `compile_metrics` 4-tuple order is: `(cpc_relative, rmse, clicks_sum, quickspend)`. Runner code wraps this via `score_to_dict(...)`. The `SCR` label in experiment configs refers to `clicks_sum`.

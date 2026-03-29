# Code Style

## Scope

This document describes the current coding and experiment-writing style observed in the repository. It records the existing conventions and inconsistencies so future work can stay compatible with the codebase as it exists today.

## General Python Style

- Imports usually follow the pattern: standard library, third-party packages, then local modules.
- Local imports are mostly absolute from the repo root, for example `from simulator.validation.check_results import autobidder_check`.
- Relative imports are used inside the `simulator` package when staying within nearby modules, for example in `simulate.py`.
- Type hints are used in many newer files, but coverage is mixed and older modules are less strict.
- `pathlib.Path` is preferred in configuration-style code, while `os.path` is still common in utility and experiment code.

## Class and API Style

- Bidders implement a shared `_Bidder` interface defined in `simulator/model/bidder.py`.
- The common bidder API is `place_bid(self, bidding_input_params, history) -> float`.
- Many bidders accept a `params` dictionary rather than a strongly typed config object.
- Newer bidder code often keeps a `default_params` dictionary and reads values with `params.get(...)`.
- State is often stored directly on instances and updated imperatively during simulation.

## Data Containers

- Simple runtime structures use dataclasses, for example `Campaign` and `SimulationResult` in `simulator/simulation/modules.py`.
- Experiment configuration uses frozen dataclasses in `example_notebooks/experiments/base_exp_config.py`.
- `History` is a plain mutable class rather than a dataclass.

## Configuration Style

- Data locations are centralized in the top-level `config.py`.
- Experiment configs are thin wrappers around those paths and add artifact directories.
- Many functions still receive plain dictionaries even when a higher-level config object exists elsewhere.
- Notebook code often reconstructs paths manually and adjusts `sys.path` before imports.

## Logging and Runtime Diagnostics

- The codebase primarily uses `print(...)` instead of the standard `logging` module.
- Log messages often use bracketed prefixes such as:
  - `[autobidder_check]`
  - `[objective_drlb]`
  - `[train_best_drlb]`
  - `[DRLBBidder]`
- Verbosity is usually controlled by flags like `verbose`, `debug_logs`, `use_tqdm`, `fit_log_every`, and `inference_log_every`.
- Progress bars are provided with `tqdm` where runs are long enough to benefit.

## Experiment Style

### Typical structure

The most common experiment pattern is:

1. Create or load an `ExperimentConfig`.
2. Load train data from CSV.
3. Define an Optuna objective that:
   - instantiates a bidder,
   - runs `autobidder_check(...)`,
   - returns one scalar from the metric tuple.
4. Save best parameters with `pickle`.
5. Optionally retrain or reload the best bidder.
6. Evaluate on the configured test split.

### DRLB style

- DRLB has a dedicated helper module: `example_notebooks/experiments/drlb_experiment.py`.
- DRLB training/evaluation is partly centralized in Python code and partly driven from notebooks.
- DRLB artifacts are split between structured experiment directories and a temporary `tmp_models/` area.

### Baseline style

- Baseline tuning is more script-oriented in `example_notebooks/evaluate_baselines/baselines_finetune.py`.
- Multiple objective methods live in a single trainer class.
- Baseline flows tend to save parameter files under `best_params/<subfolder>/...`.

## Metric and Reporting Style

- Multi-campaign evaluation returns a dictionary with timing, status, score tuple, and `all_hist_data`.
- `compile_metrics(...)` currently returns a 4-tuple:
  - `cpc_relative`
  - `rmse`
  - `clicks_sum`
  - `quickspend`
- Objectives usually index into this tuple directly rather than returning a named structure.
- Console output often labels the third metric as `SCR`, even though the implementation currently returns `clicks_sum`.

## Path and Import Conventions

- The repository root acts as the import root.
- Files import `simulator.*`, `experiments.*`, and `config` as top-level modules.
- Because the repository directory contains a hyphen, the repo name itself is not used as an importable package name.
- Notebooks frequently rely on `sys.path.insert(...)` to make those imports work.

## Notebook Conventions

- Notebooks are a first-class part of the workflow, not just examples.
- Notebook cells often contain setup code for import paths.
- Notebook output is often kept in versioned files, including long logs and Optuna traces.
- Duplicate or variant notebooks exist when iterating quickly, for example `drlb_test.ipynb` and `drlb_test copy.ipynb`.

## Style Characteristics Worth Preserving For Compatibility

- Keep the `_Bidder` contract stable.
- Preserve `params` dictionary inputs in bidder constructors unless there is a strong reason to wrap them.
- Preserve existing metric tuple ordering for code that already indexes into `res["score"]`.
- Preserve experiment artifact directory conventions used by `ExperimentConfig`.
- Prefer additive debugging flags over changing default experiment behavior.

## Current Inconsistencies

- Metric naming is inconsistent:
  - some code uses `RMSE`,
  - some code uses `RMSE_T`,
  - some code prints `SCR` while using `clicks_sum`.
- Path handling mixes `Path`, `os.path`, and notebook-local path bootstrapping.
- Typing quality is uneven across files.
- Comments and inline notes appear in both English and Russian.
- There are both structured experiment helpers and notebook-specific one-off flows.

## Practical Guidance For Future Changes

- Match the local style of the file you are editing rather than forcing a repo-wide rewrite.
- If touching shared simulator code, avoid changing default behavior for unrelated bidders.
- If adding new experiment code, prefer reusing `ExperimentConfig`, `autobidder_check(...)`, and existing artifact directories.
- If adding diagnostics, follow the existing flag-based and bracketed-print style unless there is a deliberate logging refactor.

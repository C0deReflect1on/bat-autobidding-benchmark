# Project Structure

## Purpose

`bat-autobidding-benchmark` is a benchmark repository for auto-bidding strategies in ad auctions. The core runtime is a simulator, while most experimentation is driven from notebooks and helper scripts.

## Top-Level Layout

- `README.md`: repository overview, installation notes, benchmark positioning, and a high-level directory tree.
- `requirements.txt`: Python dependencies.
- `config.py`: central data-path constants used by experiment configs.
- `utils.py`: repo-root helper utilities.
- `data/`: benchmark datasets and DVC descriptors.
- `simulator/`: main reusable code for bidders, simulation, and evaluation.
- `example_notebooks/`: notebooks and helper scripts for experiments, tuning, and evaluation.
- `useful_notebooks/`: auxiliary notebooks for data filtering/preparation.
- `project_info/`: repository documentation snapshots about current code style and structure.

## Core Runtime Code

### `simulator/model/`

Contains bidder implementations and model-specific logic.

- `bidder.py`: abstract `_Bidder` interface with `place_bid(...)`.
- `linear_bidder.py`, `ta_pid.py`, `m_pid.py`, `mystique.py`, `broi_bidder.py`: baseline bidders.
- `rlb_dp_bidder.py`: dynamic-programming RLB bidder.
- `drlb_bidder.py`: BAT adapter around the DRLB agent.
- `traffic.py`: traffic-share helper used in metrics and some bidders.
- `drlb/`: DRLB-specific internal implementation (`rl_bid_agent_alibaba.py`, `dqn.py`, `reward_net.py`, `model.py`, `config.cfg` if present).

### `simulator/simulation/`

Contains the simulation engine and runtime containers.

- `modules.py`: `Campaign`, `SimulationResult`, and `History`.
- `simulate.py`: hourly simulation loop and CTR/CVR helper functions.
- `utils.py`: utility conversions such as price/bin mapping.
- `utils_visualization.py`: plotting/data-prep helpers for notebooks.

### `simulator/validation/`

Contains multi-campaign evaluation and metric aggregation.

- `check_results.py`: `autobidder_check(...)`, the main campaign-batch evaluator.
- `metrics.py`: aggregate metrics like `cpc_relative`, traffic-normalized RMSE, click sum, and quickspend.

## Data Layout

- `data/fpa/`: full FPA train/test/final campaign and stats CSVs.
- `data/small_example/fpa/`: subsample campaign/stats CSVs used for quicker debug iterations.
- `data/traffic_share.csv`: traffic distribution used by `metrics.py`.
- `data/*.dvc`: data references for external pulls.

`config.py` exposes these paths as `Path` constants such as:

- `FPA_CAMPAIGNS_TRAIN`
- `FPA_CAMPAIGNS_TEST`
- `FPA_STATS_TRAIN`
- `FPA_STATS_TEST`
- `FPA_SUBSAMPLE_CAMPAIGNS_TRAIN`
- `FPA_SUBSAMPLE_STATS_TRAIN`

## Experiment Layout

### `example_notebooks/experiments/`

This directory acts as the main experiment workspace.

- `base_exp_config.py`: frozen `ExperimentConfig` dataclass with derived artifact directories.
- `exp_configs.py`: concrete experiment presets such as `RND42N10Config` and `SubsampleRND42N10Config`.
- `drlb_experiment.py`: centralized DRLB Optuna/train/evaluate helpers.
- `exp_1_rnd_42_n10/`: saved baseline/RLB experiments and outputs.
- `exp_2/`: DRLB notebook experiments.

### Other experiment entry points

- `example_notebooks/evaluate_baselines/baselines_finetune.py`: baseline tuning helper script.
- `example_notebooks/baseline_bidders.ipynb`: baseline experiment walkthrough.
- `example_notebooks/bidder_example.ipynb`: how to implement a new bidder.
- `example_notebooks/test_rlb_dp.ipynb` and `tune_rlb/tune_rlb.ipynb`: RLB-focused flows.
- `example_notebooks/deprecate_rlb_validation/`: older retained notebooks.

## Experiment Workflow

The common experiment loop is:

1. Select an `ExperimentConfig` with train/test CSV paths and output directories.
2. Tune bidder hyperparameters, usually with Optuna.
3. Evaluate via `autobidder_check(...)`, which:
   - loads campaign and stats CSVs,
   - creates one `Campaign` object per row,
   - instantiates a bidder per campaign,
   - runs `simulate_campaign(...)`,
   - concatenates histories,
   - computes metrics through `compile_metrics(...)`.
4. Save best params and, for learned bidders, save trained weights.
5. Re-run evaluation on the configured test split.

## Artifact Conventions

There are two artifact styles in the current codebase.

- Structured experiment artifacts via `ExperimentConfig`:
  - `config/`
  - `best_models/`
  - `best_params/`
  - `outputs/`
- Ad hoc artifacts in some flows:
  - DRLB trial checkpoints in `tmp_models/`
  - baseline Optuna SQLite files
  - notebook-generated CSVs inside experiment folders

## Import Structure

The repository is not packaged under a single importable top-level Python package name. Instead, code assumes the repository root is on `PYTHONPATH`.

Current import patterns:

- modules import `simulator.*` directly from repo root,
- notebooks often inject paths with `sys.path.insert(...)`,
- `exp_configs.py` includes a fallback `sys.path.insert(...)` when `from config import ...` fails.

This means the practical import root is the repository directory itself, not a package named after the repository folder.

## Current Structural Characteristics To Keep In Mind

- Runtime code is mostly under `simulator/`.
- Experiment orchestration is split between notebooks and small Python helpers.
- `config.py` is a top-level module rather than part of a package.
- Notebook outputs and historical experiment artifacts are committed into the repository.
- Some older experiment flows remain alongside current ones, so there is duplication rather than a single canonical pipeline for all bidders.

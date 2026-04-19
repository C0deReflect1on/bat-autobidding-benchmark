# Project Structure

## Purpose

`bat-autobidding-benchmark` is a benchmark for auto-bidding strategies in ad auctions. The core runtime is a tabular hourly simulator; experimentation is driven from thin CLI scripts and notebooks.

## Installation

```bash
pip install -e .          # editable install; makes simulator + config importable
```

Only editable installs are supported (`__file__`-relative data paths require the repo on disk).

---

## Frozen Core (DO NOT MODIFY -- benchmark semantics)

Changes to these modules invalidate all previous experiment results.

### `simulator/simulation/`

Hourly auction loop and runtime containers.

- `modules.py` -- `Campaign`, `SimulationResult`, `History`
- `simulate.py` -- `simulate_campaign`, `simulate_step`, CTR/CVR helpers
- `utils.py` -- `price2bin`, `bin2price`
- `utils_visualization.py` -- plotting helpers (tied to `History` DataFrame shapes)

### `simulator/validation/`

Multi-campaign evaluation and metric aggregation.

- `check_results.py` -- `autobidder_check(...)` (the main batch evaluator)
- `metrics.py` -- `compile_metrics(...)` returning a frozen 4-tuple: `(cpc_relative, rmse, clicks_sum, quickspend)`

### `simulator/model/bidder.py`

Abstract bidder interface: `_Bidder.place_bid(bidding_input_params, history) -> float`.

### `simulator/model/traffic.py`

Traffic-share data used by the RMSE metric.

### `data/`

Benchmark datasets and DVC descriptors. Never modified by code.

---

## Agent Implementations (add new bidders here)

### `simulator/model/`

One file per bidder family. All concrete bidders subclass `_Bidder`.

| File | Bidder | Trainable |
|------|--------|-----------|
| `linear_bidder.py` | `LinearBidder` | No (params only) |
| `ta_pid.py` | `TAPIDBidder` | No |
| `m_pid.py` | `MPIDBidder` | No |
| `mystique.py` | `Mystique` | No |
| `broi_bidder.py` | `BROI` | No |
| `rlb_dp_bidder.py` | `RLBDPBidder` | Yes (`fit`, `save_model`, `load_model`) |
| `drlb_bidder.py` | `DRLBBidder` | Yes (`fit`, `save_model`, `load_model`) |

The `BIDDERS` dict in `simulator/model/__init__.py` maps short names to classes. Add one import + one line to register a new bidder.

### `simulator/model/drlb/`

DRLB-specific internals -- DQN, RewardNet, and pluggable state representations.

- `state_representations.py` -- all DRLB state vector definitions (`ImprovedState`, `ScaledBudgetState`, `HybridState`, `DefaultState`), plus `EXP_TYPE_TO_STATE_FAMILY` and `get_state_repr()`.
- `rl_bid_agent_alibaba.py` -- `RlBidAgent`, the core DRLB agent. Delegates state construction and metric computation to the active `StateRepresentation`.
- `dqn.py` -- Double-DQN with experience replay.
- `reward_net.py` -- Auxiliary reward network.
- `model.py` -- Shared MLP `Network` backbone.

---

## Experiment Orchestration

### `example_notebooks/experiments/`

- `base_exp_config.py` -- `ExperimentConfig` frozen dataclass with artifact directory helpers and `train_val()` factory.
- `exp_configs.py` -- Concrete experiment configs. Train/val split configs use the `ExperimentConfig.train_val()` factory; configs pointing to `data/` are full dataclasses.
- `runner_utils.py` -- Shared functions: `ensure_train_val_split`, `run_drlb_candidate`, `run_optuna_experiment`, `build_run_summary`, `append_runs_index`, score/diagnostics helpers.
- `exp_*/run_*.py` -- Per-experiment thin CLIs (~70 lines each). Define experiment-specific constants (`BASE_DRLB_PARAMS`, `BASELINE_MODEL_PARAMS`, `search_space`) and call `run_optuna_experiment`.

### Artifact layout per experiment

```
exp_tune_drlb_dqn_hypgrid_v3/
├── config/
│   ├── train_campaigns.csv
│   ├── train_stats.csv
│   ├── val_campaigns.csv
│   ├── val_stats.csv
│   └── train_val_split_metadata.json
├── best_models/
│   └── best_trainval.pt
├── outputs/
│   ├── run_summary.json      (standardized schema, see code_style.md)
│   └── runs_index.jsonl      (append-only log of all runs)
└── run_hypgrid_v3.py
```

All intermediate artifacts (.pt checkpoints, diagnostics CSVs) live in a
`TemporaryDirectory` and are auto-deleted when the run finishes. Only
`best_trainval.pt` is copied out.  Baseline, trial, and best-refit metrics
are all captured in `run_summary.json` -- no per-run files in `outputs/`.

---

## Adding a New DRLB State Representation

1. Add a frozen dataclass to `simulator/model/drlb/state_representations.py` implementing `get_state`, `compute_step_metrics`, `reset_step_fields`.
2. Add an entry to `STATE_REPRESENTATIONS` dict.
3. Map your `exp_type` strings in `EXP_TYPE_TO_STATE_FAMILY`.
4. Create a runner in `example_notebooks/experiments/exp_<name>/run_<name>.py` with the appropriate `BASE_DRLB_PARAMS` and `search_space`.

No other files need to change.

## Adding a New Non-DRLB Bidder

1. Create `simulator/model/my_bidder.py`, subclass `_Bidder`.
2. Implement `place_bid(bidding_input_params, history) -> float`.
3. Add to `BIDDERS` dict in `simulator/model/__init__.py`.
4. Optionally add `fit` / `save_model` / `load_model` for trainable bidders.

---

## Other Directories

- `agents.md` -- Root-level placeholder file for agent notes/metadata (currently empty).
- `example_notebooks/evaluate_baselines/` -- Optuna tuning for baseline (non-DRLB) bidders.
- `example_notebooks/*.ipynb` -- Walkthrough notebooks.
- `useful_notebooks/` -- Data filtering/preparation notebooks.
- `project_info/` -- This document and `code_style.md`.

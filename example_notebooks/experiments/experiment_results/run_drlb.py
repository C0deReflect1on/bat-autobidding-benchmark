from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from example_notebooks.experiments.exp_configs import ExperimentResultsDrlbConfig
from example_notebooks.experiments.experiment_results.common import require_subsample_ready
from example_notebooks.experiments.shared_runner import run_experiment


EXP_TYPE = "improved_hybrid_drlb_smooth_eval"
OBJECTIVE = "clicks"

BASE_DRLB_PARAMS = {
    "max_bid": 100.0,
    "T": 72,
    "lambda_min": 1e-6,
    "lambda_max": 10.0,
    "bids_per_timestep": 1,
    "dqn_soft_update_tau": 0.01,
    "dqn_loss_type": "smooth_l1",
    "dqn_grad_clip_norm": 5.0,
    "dqn_reward_clip_value": 10.0,
}

BASELINE_MODEL_PARAMS = {
    "dqn_gamma": 1.0,
    "dqn_lr": 1e-4,
    "dqn_target_update_interval": 100,
    "reward_net_lr": 1e-3,
}


def search_space(trial):
    return {
        "dqn_gamma": trial.suggest_float("dqn_gamma", 0.90, 1.0),
        "dqn_lr": trial.suggest_float("dqn_lr", 1e-5, 5e-3, log=True),
        "dqn_target_update_interval": trial.suggest_int("dqn_target_update_interval", 10, 100, step=10),
        "reward_net_lr": trial.suggest_float("reward_net_lr", 1e-5, 5e-3, log=True),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the simple DRLB bidder experiment on the permanent subsample.")
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    require_subsample_ready()
    config = ExperimentResultsDrlbConfig()
    if args.n_trials is not None:
        config = replace(config, n_trials=int(args.n_trials))
    summary = run_experiment(
        config,
        verbose=args.verbose,
        base_drlb_params=BASE_DRLB_PARAMS,
        baseline_model_params=BASELINE_MODEL_PARAMS,
        exp_type=EXP_TYPE,
        objective=OBJECTIVE,
        search_space_fn=search_space,
        n_trials=config.n_trials,
        max_train_steps=args.max_train_steps if args.max_train_steps is not None else config.max_steps,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

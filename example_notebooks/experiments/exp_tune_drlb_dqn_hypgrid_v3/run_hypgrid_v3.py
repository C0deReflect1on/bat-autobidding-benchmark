"""Hypgrid-v3 DRLB tuning experiment (thin CLI)."""
import argparse

from experiments.exp_configs import RND42N10Config, RND42TrainValHypgridV3Config
from experiments.runner_utils import run_optuna_experiment


EXP_TYPE = "hypgrid_v3_drlb_eval"
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
    "reward_net_loss_type": "smooth_l1",
    "reward_net_grad_clip_norm": 5.0,
    "reward_net_reward_clip_value": 10.0,
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
        "dqn_target_update_interval": trial.suggest_int("dqn_target_update_interval", 10, 300, step=10),
        "reward_net_lr": trial.suggest_float("reward_net_lr", 1e-5, 5e-2, log=True),
    }


def main():
    parser = argparse.ArgumentParser(description="Run hypgrid v3 train/val experiment.")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--n-trials", type=int, default=1)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    run_optuna_experiment(
        config=RND42TrainValHypgridV3Config(),
        source_config=RND42N10Config(),
        base_drlb_params=BASE_DRLB_PARAMS,
        baseline_model_params=BASELINE_MODEL_PARAMS,
        exp_type=EXP_TYPE,
        objective=OBJECTIVE,
        search_space_fn=search_space,
        val_fraction=args.val_fraction,
        n_trials=args.n_trials,
        max_train_steps=args.max_train_steps,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()

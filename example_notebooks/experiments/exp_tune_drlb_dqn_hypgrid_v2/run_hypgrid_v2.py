import argparse
import json
import sys
from pathlib import Path

import optuna
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
EXAMPLE_NOTEBOOKS_DIR = SCRIPT_DIR.parent.parent
BAT_AUTOBIDDING_DIR = EXAMPLE_NOTEBOOKS_DIR.parent

if str(EXAMPLE_NOTEBOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_NOTEBOOKS_DIR))
if str(BAT_AUTOBIDDING_DIR) not in sys.path:
    sys.path.insert(0, str(BAT_AUTOBIDDING_DIR))

from experiments.exp_configs import RND42N10Config, RND42TrainValHypgridV2Config
from simulator.model.drlb_bidder import DRLBBidder
from simulator.validation.check_results import autobidder_check


OBJECTIVE = "clicks"
EXP_TYPE = "hypgrid_v2_drlb_eval"
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


def ensure_train_val_split(source_config, target_config, val_fraction=0.2, seed=42):
    source_campaigns = pd.read_csv(source_config.data_config["train"]["campaigns_path"])
    source_stats = pd.read_csv(source_config.data_config["train"]["stats_path"])

    n_val = max(1, int(round(len(source_campaigns) * val_fraction)))
    val_campaigns = (
        source_campaigns
        .sample(n=n_val, random_state=seed)
        .sort_values("campaign_id")
        .reset_index(drop=True)
    )
    val_campaign_ids = set(val_campaigns["campaign_id"].astype(int).tolist())

    train_campaigns = (
        source_campaigns[~source_campaigns["campaign_id"].astype(int).isin(val_campaign_ids)]
        .sort_values("campaign_id")
        .reset_index(drop=True)
    )

    train_stats = source_stats[source_stats["campaign_id"].astype(int).isin(
        set(train_campaigns["campaign_id"].astype(int))
    )].copy()
    val_stats = source_stats[source_stats["campaign_id"].astype(int).isin(val_campaign_ids)].copy()

    train_campaigns.to_csv(target_config.data_config["train"]["campaigns_path"], index=False)
    train_stats.to_csv(target_config.data_config["train"]["stats_path"], index=False)
    val_campaigns.to_csv(target_config.data_config["test"]["campaigns_path"], index=False)
    val_stats.to_csv(target_config.data_config["test"]["stats_path"], index=False)

    metadata = {
        "source_experiment": source_config.experiment_name,
        "target_experiment": target_config.experiment_name,
        "seed": int(seed),
        "val_fraction": float(val_fraction),
        "train_campaigns": int(len(train_campaigns)),
        "val_campaigns": int(len(val_campaigns)),
        "train_stats_rows": int(len(train_stats)),
        "val_stats_rows": int(len(val_stats)),
    }
    metadata_path = target_config.config_dir / "train_val_split_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return metadata


def score_to_dict(score, skipped_campaigns, time_inference_sec, time_overall_sec):
    return {
        "cpc_relative": float(score[0]),
        "rmse": float(score[1]),
        "clicks_sum": float(score[2]),
        "quickspend": float(score[3]),
        "skipped_campaigns": int(skipped_campaigns),
        "time_inference_sec": float(time_inference_sec),
        "time_overall_sec": float(time_overall_sec),
    }


def summarize_diagnostics(diagnostics_df):
    if diagnostics_df.empty:
        return {
            "train_steps": 0,
            "last_dqn_loss": None,
            "last_reward_net_loss": None,
            "dqn_loss_mean": None,
            "dqn_loss_p95": None,
            "reward_net_loss_mean": None,
            "reward_net_loss_p95": None,
            "reward_signal_mean": None,
            "lambda_final": None,
        }

    dqn_loss_nonzero = diagnostics_df.loc[diagnostics_df["dqn_loss"] > 0, "dqn_loss"]
    reward_net_nonzero = diagnostics_df.loc[diagnostics_df["reward_net_loss"] > 0, "reward_net_loss"]
    return {
        "train_steps": int(len(diagnostics_df)),
        "last_dqn_loss": float(diagnostics_df["dqn_loss"].iloc[-1]),
        "last_reward_net_loss": float(diagnostics_df["reward_net_loss"].iloc[-1]),
        "dqn_loss_mean": float(dqn_loss_nonzero.mean()) if not dqn_loss_nonzero.empty else None,
        "dqn_loss_p95": float(dqn_loss_nonzero.quantile(0.95)) if not dqn_loss_nonzero.empty else None,
        "reward_net_loss_mean": float(reward_net_nonzero.mean()) if not reward_net_nonzero.empty else None,
        "reward_net_loss_p95": float(reward_net_nonzero.quantile(0.95)) if not reward_net_nonzero.empty else None,
        "reward_signal_mean": float(diagnostics_df["reward_signal"].mean()),
        "lambda_final": float(diagnostics_df["lambda"].iloc[-1]),
    }


def build_bidder_params(base_params, model_params, verbose=False):
    return {
        **base_params,
        **model_params,
        "model_path": None,
        "exp_type": EXP_TYPE,
        "objective": OBJECTIVE,
        "eval_mode": True,
        "verbose": verbose,
        "use_tqdm": verbose,
        "debug_logs": False,
        "fit_log_every": 500,
        "inference_log_every": 24,
    }


def run_drlb_candidate(config, train_stats_df, train_campaigns_df, label, model_params, max_train_steps=None, verbose=False):
    bidder_params = build_bidder_params(BASE_DRLB_PARAMS, model_params, verbose=verbose)
    bidder = DRLBBidder(bidder_params)
    bidder.fit(
        train_stats_df,
        campaigns_df=train_campaigns_df,
        max_steps=max_train_steps,
        objective=OBJECTIVE,
    )

    diagnostics = bidder.get_training_diagnostics().copy()
    diagnostics_path = config.outputs_dir / f"{label}_training_diagnostics.csv"
    diagnostics.to_csv(diagnostics_path, index=False)

    model_path = config.best_models_dir / f"{label}.pt"
    bidder.save_model(str(model_path))

    eval_params = {
        **bidder_params,
        "input_campaigns": config.data_config["test"]["campaigns_path"],
        "input_stats": config.data_config["test"]["stats_path"],
        "model_path": str(model_path),
        "eval_mode": True,
    }
    result = autobidder_check(
        bidder=DRLBBidder,
        params=eval_params,
        auction_mode=config.auction_mode,
        verbose=verbose,
        log_every_campaigns=100,
        use_tqdm=verbose,
    )

    metrics = score_to_dict(
        score=result["score"],
        skipped_campaigns=result["skipped_campaigns"],
        time_inference_sec=result["time_inference_sec"],
        time_overall_sec=result["time_overall_sec"],
    )
    metrics.update({"label": label, **model_params})
    metrics.update(summarize_diagnostics(diagnostics))

    metrics_path = config.outputs_dir / f"{label}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    return {
        "label": label,
        "params": bidder_params,
        "metrics": metrics,
        "diagnostics_path": diagnostics_path,
        "metrics_path": metrics_path,
        "model_path": model_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Run hypgrid v2 train/val experiment.")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--n-trials", type=int, default=1)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    source_config = RND42N10Config()
    config = RND42TrainValHypgridV2Config()
    config.ensure_artifact_dirs()
    config.config_dir.mkdir(parents=True, exist_ok=True)

    split_metadata = ensure_train_val_split(
        source_config=source_config,
        target_config=config,
        val_fraction=args.val_fraction,
        seed=config.random_seed,
    )

    train_stats_df = pd.read_csv(config.data_config["train"]["stats_path"])
    train_campaigns_df = pd.read_csv(config.data_config["train"]["campaigns_path"])

    baseline_run = run_drlb_candidate(
        config=config,
        train_stats_df=train_stats_df,
        train_campaigns_df=train_campaigns_df,
        label="baseline_manual",
        model_params=BASELINE_MODEL_PARAMS,
        max_train_steps=args.max_train_steps,
        verbose=args.verbose,
    )

    def objective(trial):
        model_params = {
            "dqn_gamma": trial.suggest_float("dqn_gamma", 0.90, 1.0),
            "dqn_lr": trial.suggest_float("dqn_lr", 1e-5, 5e-3, log=True),
            "dqn_target_update_interval": trial.suggest_int("dqn_target_update_interval", 10, 300, step=10),
            "reward_net_lr": trial.suggest_float("reward_net_lr", 1e-5, 5e-2, log=True),
        }
        run = run_drlb_candidate(
            config=config,
            train_stats_df=train_stats_df,
            train_campaigns_df=train_campaigns_df,
            label=f"trial_{trial.number:03d}",
            model_params=model_params,
            max_train_steps=args.max_train_steps,
            verbose=args.verbose,
        )
        metrics = run["metrics"]
        trial.set_user_attr("rmse", metrics["rmse"])
        trial.set_user_attr("clicks_sum", metrics["clicks_sum"])
        trial.set_user_attr("cpc_relative", metrics["cpc_relative"])
        trial.set_user_attr("quickspend", metrics["quickspend"])
        trial.set_user_attr("last_dqn_loss", metrics["last_dqn_loss"])
        trial.set_user_attr("last_reward_net_loss", metrics["last_reward_net_loss"])
        trial.set_user_attr("dqn_loss_mean", metrics["dqn_loss_mean"])
        return metrics["clicks_sum"]

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.random_seed),
    )
    study.optimize(objective, n_trials=max(1, args.n_trials), n_jobs=1, show_progress_bar=args.verbose)
    best_model_params = study.best_trial.params

    best_run = run_drlb_candidate(
        config=config,
        train_stats_df=train_stats_df,
        train_campaigns_df=train_campaigns_df,
        label="best_trainval",
        model_params=best_model_params,
        max_train_steps=args.max_train_steps,
        verbose=args.verbose,
    )

    summary = {
        "split_metadata": split_metadata,
        "baseline_manual": baseline_run["metrics"],
        "best_trial_params": best_model_params,
        "best_trainval": best_run["metrics"],
        "study_best_value": float(study.best_trial.value),
    }
    summary_path = config.outputs_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    print(f"summary_path={summary_path}")


if __name__ == "__main__":
    main()

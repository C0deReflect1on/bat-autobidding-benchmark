import os
import json
import math
import pickle
import uuid
from functools import partial
from pathlib import Path
from time import time
from typing import Dict, Optional, Tuple

import optuna
import pandas as pd

from simulator.model.drlb_bidder import DRLBBidder
from simulator.validation.check_results import autobidder_check


def suggest_drlb_params(trial: optuna.trial.Trial) -> Dict[str, float]:
    return {
        "max_bid": trial.suggest_float("max_bid", 10, 500, log=True),
        "T": trial.suggest_int("T", 24, 96),
        "lambda_min": trial.suggest_float("lambda_min", 1e-7, 1e-4, log=True),
        "lambda_max": trial.suggest_float("lambda_max", 1.0, 30.0, log=True),
    }


def _ensure_eval_slice(
    campaigns_path: str,
    stats_path: str,
    output_dir: Path,
    fraction: Optional[float],
    seed: int,
) -> Tuple[str, str]:
    if fraction is None or fraction >= 1.0:
        return campaigns_path, stats_path

    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{int(fraction * 100)}pct_seed{seed}"
    sliced_campaigns_path = output_dir / f"campaigns_{suffix}.csv"
    sliced_stats_path = output_dir / f"stats_{suffix}.csv"

    if sliced_campaigns_path.exists() and sliced_stats_path.exists():
        return str(sliced_campaigns_path), str(sliced_stats_path)

    campaigns_df = pd.read_csv(campaigns_path)
    sample_size = max(1, math.ceil(len(campaigns_df) * fraction))
    sampled_campaigns = (
        campaigns_df
        .sample(n=sample_size, random_state=seed)
        .sort_values("campaign_id")
        .reset_index(drop=True)
    )
    sampled_campaign_ids = set(sampled_campaigns["campaign_id"].astype(int).tolist())
    sampled_stats = pd.read_csv(stats_path)
    sampled_stats = sampled_stats[sampled_stats["campaign_id"].astype(int).isin(sampled_campaign_ids)].copy()

    sampled_campaigns.to_csv(sliced_campaigns_path, index=False)
    sampled_stats.to_csv(sliced_stats_path, index=False)
    return str(sliced_campaigns_path), str(sliced_stats_path)


def _resolve_eval_paths(config) -> Tuple[str, str]:
    campaigns_path = config.data_config["test"]["campaigns_path"]
    stats_path = config.data_config["test"]["stats_path"]
    return _ensure_eval_slice(
        campaigns_path=campaigns_path,
        stats_path=stats_path,
        output_dir=config.config_dir,
        fraction=getattr(config, "eval_campaign_fraction", None),
        seed=int(getattr(config, "eval_slice_seed", config.random_seed)),
    )


def _save_training_artifacts(config, bidder: DRLBBidder, label: str) -> Optional[Path]:
    diagnostics = bidder.get_training_diagnostics()
    if diagnostics.empty:
        return None

    config.outputs_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_path = config.outputs_dir / f"{label}_training_diagnostics.csv"
    diagnostics.to_csv(diagnostics_path, index=False)
    return diagnostics_path


def _save_evaluation_artifacts(config, label: str, result: Dict[str, object]) -> Dict[str, Path]:
    config.outputs_dir.mkdir(parents=True, exist_ok=True)

    metrics = result["score"]
    metrics_payload = {
        "cpc_relative": metrics[0],
        "rmse": metrics[1],
        "clicks_sum": metrics[2],
        "quickspend": metrics[3],
        "skipped_campaigns": result.get("skipped_campaigns", 0),
        "time_inference_sec": result.get("time_inference_sec"),
        "time_overall_sec": result.get("time_overall_sec"),
    }
    metrics_path = config.outputs_dir / f"{label}_metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2))

    hist_data = result.get("all_hist_data") or []
    hist_frames = [frame for frame in hist_data if isinstance(frame, pd.DataFrame) and not frame.empty]
    if not hist_frames:
        return {"metrics": metrics_path}

    full_hist = pd.concat(hist_frames, axis=0, ignore_index=True)
    hist_path = config.outputs_dir / f"{label}_history.csv"
    full_hist.to_csv(hist_path, index=False)

    summary = (
        full_hist
        .groupby("campaign_id", as_index=False)
        .agg(
            total_spend=("spend_history", "sum"),
            total_clicks=("clicks_history", "sum"),
            mean_bid=("bid", "mean"),
            nonzero_bid_rate=("bid", lambda s: float((s > 0).mean())),
        )
    )
    summary_path = config.outputs_dir / f"{label}_campaign_summary.csv"
    summary.to_csv(summary_path, index=False)
    return {"metrics": metrics_path, "history": hist_path, "summary": summary_path}


def objective_drlb(
    trial: optuna.trial.Trial,
    stats_df: pd.DataFrame,
    campaigns_df: pd.DataFrame,
    campaigns_path: str,
    stats_path: str,
    metric: str = "RMSE_T",
    auction_mode: str = "FPA",
    verbose: bool = True,
    trial_model_dir: Optional[str] = None,
):
    t0 = time()
    params = suggest_drlb_params(trial)
    if verbose:
        print(f"[objective_drlb] trial={trial.number} params={params}")
    custom_params = {
        **params,
        "model_path": None,
        "exp_type": "improved_drlb_eval",
        "bids_per_timestep": 1,
        "eval_mode": True,
        "verbose": verbose,
        "use_tqdm": verbose,
        "debug_logs": False,
        "fit_log_every": 500,
        "inference_log_every": 24,
    }

    bidder = DRLBBidder(custom_params)
    bidder.fit(stats_df, campaigns_df=campaigns_df, objective="clicks")
    if verbose:
        print(f"[objective_drlb] trial={trial.number} fit done in {time() - t0:.1f}s")

    trial_model_root = Path(trial_model_dir or "tmp_models")
    trial_model_root.mkdir(parents=True, exist_ok=True)
    tmp_model_path = str(trial_model_root / f"drlb_trial{trial.number}_{uuid.uuid4().hex}.pt")
    bidder.save_model(tmp_model_path)
    if verbose:
        print(f"[objective_drlb] trial={trial.number} model saved -> {tmp_model_path}")

    res = autobidder_check(
        bidder=DRLBBidder,
        params={
            "input_campaigns": campaigns_path,
            "input_stats": stats_path,
            "model_path": tmp_model_path,
            **params,
            "exp_type": "improved_drlb_eval",
            "bids_per_timestep": 1,
            "eval_mode": True,
            "verbose": verbose,
            "use_tqdm": verbose,
            "debug_logs": False,
            "fit_log_every": 500,
            "inference_log_every": 24,
        },
        auction_mode=auction_mode,
        verbose=verbose,
        log_every_campaigns=100,
        use_tqdm=verbose,
    )

    print(
        f"CPC_REL: {res['score'][0]}, rmse: {res['score'][1]}, "
        f"clicks_sum: {res['score'][2]}, quickspend: {res['score'][3]}"
    )
    if verbose:
        print(f"[objective_drlb] trial={trial.number} total elapsed={time() - t0:.1f}s")
    if metric == "RMSE_T":
        return res["score"][1]
    if metric == "CPC_REL":
        return res["score"][0]
    if metric == "SCR":
        return res["score"][2]
    raise ValueError(f"Unsupported metric: {metric}")


def opt_search_drlb(
    config,
    stats_df: pd.DataFrame,
    campaigns_df: pd.DataFrame,
    n_trials: Optional[int] = None,
    metric: Optional[str] = None,
    auction_mode: Optional[str] = None,
    verbose: bool = True,
):
    config.ensure_artifact_dirs()
    metric = metric or config.metric
    auction_mode = auction_mode or config.auction_mode
    n_trials = int(n_trials or config.n_trials)

    campaigns_path = config.data_config["train"]["campaigns_path"]
    stats_path = config.data_config["train"]["stats_path"]
    eval_campaigns_path, eval_stats_path = _resolve_eval_paths(config)
    trial_model_dir = str(config.outputs_dir / "tmp_models")

    study = optuna.create_study(
        direction="maximize" if metric == "SCR" else "minimize",
        sampler=optuna.samplers.TPESampler(seed=config.random_seed),
    )
    study.optimize(
        partial(
            objective_drlb,
            stats_df=stats_df,
            campaigns_df=campaigns_df,
            campaigns_path=eval_campaigns_path,
            stats_path=eval_stats_path,
            metric=metric,
            auction_mode=auction_mode,
            verbose=verbose,
            trial_model_dir=trial_model_dir,
        ),
        n_trials=n_trials,
        n_jobs=1,
        show_progress_bar=verbose,
    )

    best_params_path = config.best_params_dir / f"drlb_{metric.lower()}_{auction_mode}.pkl"
    best_params_path.parent.mkdir(parents=True, exist_ok=True)
    with open(best_params_path, "wb") as f:
        pickle.dump(study.best_trial.params, f)

    print("Best trial:")
    print(f"  Value: {study.best_trial.value}")
    print("  Params:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")

    return study, best_params_path


def train_best_drlb(
    stats_df: pd.DataFrame,
    campaigns_df: pd.DataFrame,
    best_params_path: str,
    model_path: str,
    config=None,
    verbose: bool = True,
):
    t0 = time()
    if config is not None:
        config.ensure_artifact_dirs()
    with open(best_params_path, "rb") as f:
        best_params = pickle.load(f)

    custom_params = {
        **best_params,
        "model_path": None,
        "exp_type": "improved_drlb_eval",
        "bids_per_timestep": 1,
        "eval_mode": True,
        "verbose": verbose,
        "use_tqdm": verbose,
        "debug_logs": False,
        "fit_log_every": 500,
        "inference_log_every": 24,
    }
    bidder = DRLBBidder(custom_params)
    bidder.fit(stats_df, campaigns_df=campaigns_df, objective="clicks")
    bidder.save_model(model_path)
    if config is not None:
        _save_training_artifacts(config, bidder, label="drlb_train")
    if verbose:
        print(f"[train_best_drlb] model saved -> {model_path} elapsed={time() - t0:.1f}s")
    return bidder


def evaluate_drlb(config, model_path: str, best_params_path: Optional[str] = None, verbose: bool = True):
    t0 = time()
    config.ensure_artifact_dirs()
    if best_params_path is None:
        best_params_path = config.best_params_dir / f"drlb_{config.metric.lower()}_{config.auction_mode}.pkl"

    with open(best_params_path, "rb") as f:
        best_params = pickle.load(f)

    eval_campaigns_path, eval_stats_path = _resolve_eval_paths(config)
    res = autobidder_check(
        bidder=DRLBBidder,
        params={
            "input_campaigns": eval_campaigns_path,
            "input_stats": eval_stats_path,
            "model_path": model_path,
            **best_params,
            "exp_type": "improved_drlb_eval",
            "bids_per_timestep": 1,
            "eval_mode": True,
            "verbose": verbose,
            "use_tqdm": verbose,
            "debug_logs": False,
            "fit_log_every": 500,
            "inference_log_every": 24,
        },
        auction_mode=config.auction_mode,
        verbose=verbose,
        log_every_campaigns=100,
        use_tqdm=verbose,
    )
    _save_evaluation_artifacts(config, label="drlb_eval", result=res)
    if verbose:
        print(f"[evaluate_drlb] score={res['score']} elapsed={time() - t0:.1f}s")
    return res

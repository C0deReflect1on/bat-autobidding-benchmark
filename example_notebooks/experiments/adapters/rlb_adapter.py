from __future__ import annotations

import pickle
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional

import optuna
import pandas as pd

from simulator.model.rlb_dp_bidder import RLBDPBidder
from simulator.validation.check_results import autobidder_check

from ..infra.artifacts import (
    append_runs_index,
    build_summary_header,
    score_to_dict,
    write_metrics,
    write_normalized_config,
    write_run_summary,
    write_split_manifest,
)


def run_rlb_experiment(
    config,
    normalized_splits: dict[str, dict[str, str]],
    *,
    search_space_fn: Optional[Callable[[optuna.trial.Trial], dict]] = None,
    n_trials: Optional[int] = None,
    objective: str = "clicks",
    verbose: bool = False,
) -> dict[str, Any]:
    config.ensure_artifact_dirs()
    write_normalized_config(config)
    write_split_manifest(config, normalized_splits)

    base_params = dict(config.model_config.get("base_params", {}))
    train_stats_df = pd.read_csv(normalized_splits["train"]["stats_path"])
    train_campaigns_path = normalized_splits["train"]["campaigns_path"]

    def objective_fn(trial: optuna.trial.Trial) -> float:
        model_params = search_space_fn(trial) if search_space_fn is not None else default_rlb_search_space(trial)
        run = run_rlb_candidate(
            config=config,
            train_stats_df=train_stats_df,
            train_campaigns_path=train_campaigns_path,
            eval_split=normalized_splits["val"],
            label=f"trial_{trial.number:03d}",
            bidder_params={**base_params, **model_params},
            objective=objective,
        )
        metrics = run["metrics"]
        trial.set_user_attr("rmse", metrics["rmse"])
        trial.set_user_attr("clicks_sum", metrics["clicks_sum"])
        trial.set_user_attr("cpc_relative", metrics["cpc_relative"])
        trial.set_user_attr("quickspend", metrics["quickspend"])
        return metrics["clicks_sum"]

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.optuna_seed),
    )
    study.optimize(
        objective_fn,
        n_trials=max(1, int(n_trials or config.n_trials)),
        n_jobs=1,
        show_progress_bar=verbose,
    )

    best_params = {**base_params, **dict(study.best_trial.params)}
    best_params_path = config.best_params_path("rlb_dp")
    best_params_path.parent.mkdir(parents=True, exist_ok=True)
    with open(best_params_path, "wb") as f:
        pickle.dump(best_params, f)

    best_val_run = run_rlb_candidate(
        config=config,
        train_stats_df=train_stats_df,
        train_campaigns_path=train_campaigns_path,
        eval_split=normalized_splits["val"],
        label="best_val",
        bidder_params=best_params,
        objective=objective,
    )

    with tempfile.TemporaryDirectory(prefix="rlb_refit_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        refit_stats_df, refit_campaigns_path = load_rlb_refit_inputs(
            config=config,
            normalized_splits=normalized_splits,
            scratch_dir=tmpdir_path,
        )
        best_holdout_run = run_rlb_candidate(
            config=config,
            train_stats_df=refit_stats_df,
            train_campaigns_path=refit_campaigns_path,
            eval_split=normalized_splits["test_holdout"],
            label="best_refit",
            bidder_params=best_params,
            objective=objective,
        )
        final_model_path = config.best_models_dir / "best_refit.pkl"
        if Path(best_holdout_run["model_path"]) != final_model_path:
            shutil.copy2(best_holdout_run["model_path"], final_model_path)

    summary = build_summary_header(config, normalized_splits)
    summary.update(
        {
            "tuning": {
                "enabled": True,
                "n_trials": max(1, int(n_trials or config.n_trials)),
                "best_trial_number": int(study.best_trial.number),
                "best_params": best_params,
                "study_best_value": float(study.best_trial.value),
                "all_trials_summary": _study_trials_summary(study),
                "best_val_metrics": best_val_run["metrics"],
            },
            "refit": {
                "scope": config.refit_on,
                "model_path": str(config.best_models_dir / "best_refit.pkl"),
            },
            "final_holdout": {
                "metrics": best_holdout_run["metrics"],
            },
        }
    )

    write_metrics(config, best_holdout_run["metrics"])
    write_run_summary(config, summary)
    append_runs_index(
        config,
        best_holdout_run["metrics"],
        stage="final_holdout",
        label="best_refit_holdout",
    )
    return summary


def run_rlb_candidate(
    *,
    config,
    train_stats_df: pd.DataFrame,
    train_campaigns_path: str,
    eval_split: dict[str, str],
    label: str,
    bidder_params: dict[str, Any],
    objective: str = "clicks",
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix=f"rlb_candidate_{label}_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        model_path = tmpdir_path / f"{label}.pkl"
        bidder = RLBDPBidder(bidder_params)
        bidder.fit(
            train_stats_df,
            campaign_path=train_campaigns_path,
            objective=objective,
        )
        bidder.save_model(str(model_path))
        result = autobidder_check(
            bidder=RLBDPBidder,
            params={
                "input_campaigns": eval_split["campaigns_path"],
                "input_stats": eval_split["stats_path"],
                "model_path": str(model_path),
                **bidder_params,
            },
            auction_mode=config.auction_mode,
            verbose=False,
            use_tqdm=False,
        )
        metrics = score_to_dict(
            result["score"],
            skipped_campaigns=result.get("skipped_campaigns"),
            time_inference_sec=result.get("time_inference_sec"),
            time_overall_sec=result.get("time_overall_sec"),
        )
        metrics["label"] = label
        persisted_model_path = config.best_models_dir / f"{label}.pkl"
        config.best_models_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model_path, persisted_model_path)
        return {
            "label": label,
            "params": bidder_params,
            "metrics": metrics,
            "model_path": persisted_model_path,
        }


def default_rlb_search_space(trial: optuna.trial.Trial) -> dict[str, Any]:
    return {
        "max_bid": trial.suggest_int("max_bid", 50, 300, step=25),
        "gamma": trial.suggest_float("gamma", 0.90, 1.0),
        "N_bound": trial.suggest_int("N_bound", 24, 96, step=24),
        "B_bound": trial.suggest_int("B_bound", 2000, 12000, step=1000),
        "use_smoothing": trial.suggest_categorical("use_smoothing", [False, True]),
    }


def load_rlb_refit_inputs(
    *,
    config,
    normalized_splits: dict[str, dict[str, str]],
    scratch_dir: Path,
) -> tuple[pd.DataFrame, str]:
    train_stats_df = pd.read_csv(normalized_splits["train"]["stats_path"])
    train_campaigns_df = pd.read_csv(normalized_splits["train"]["campaigns_path"])
    if config.refit_on == "train":
        campaigns_path = scratch_dir / "train_campaigns.csv"
        train_campaigns_df.to_csv(campaigns_path, index=False)
        return train_stats_df, str(campaigns_path)
    if config.refit_on == "train_plus_val":
        val_stats_df = pd.read_csv(normalized_splits["val"]["stats_path"])
        val_campaigns_df = pd.read_csv(normalized_splits["val"]["campaigns_path"])
        refit_stats_df = pd.concat([train_stats_df, val_stats_df], ignore_index=True)
        refit_campaigns_df = pd.concat([train_campaigns_df, val_campaigns_df], ignore_index=True)
        campaigns_path = scratch_dir / "train_plus_val_campaigns.csv"
        refit_campaigns_df.to_csv(campaigns_path, index=False)
        return refit_stats_df, str(campaigns_path)
    raise ValueError(f"Unsupported refit_on '{config.refit_on}'")


def _study_trials_summary(study) -> list[dict[str, Any]]:
    rows = []
    for trial in study.trials:
        row = {"trial": int(trial.number), **trial.params}
        row.update(trial.user_attrs)
        rows.append(row)
    return rows

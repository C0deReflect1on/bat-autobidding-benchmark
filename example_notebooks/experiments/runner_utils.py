"""
Shared utilities for DRLB experiment runners.

All three runner scripts (hypgrid_v2, hypgrid_v3, smooth) share the same
train/val splitting, candidate evaluation, diagnostics summarisation, and
artifact-writing logic.  This module extracts that logic so each runner
is a thin ~70-line CLI.
"""
import hashlib
import json
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import optuna
import pandas as pd

from simulator.model.drlb_bidder import DRLBBidder
from simulator.validation.check_results import autobidder_check


# ---------------------------------------------------------------------------
# Train / val splitting
# ---------------------------------------------------------------------------

def ensure_train_val_split(
    source_config,
    target_config,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> dict:
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

    train_stats = source_stats[
        source_stats["campaign_id"].astype(int).isin(set(train_campaigns["campaign_id"].astype(int)))
    ].copy()
    val_stats = source_stats[
        source_stats["campaign_id"].astype(int).isin(val_campaign_ids)
    ].copy()

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


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def score_to_dict(
    score: tuple,
    skipped_campaigns: int,
    time_inference_sec: float,
    time_overall_sec: float,
) -> Dict[str, Any]:
    return {
        "cpc_relative": float(score[0]),
        "rmse": float(score[1]),
        "clicks_sum": float(score[2]),
        "quickspend": float(score[3]),
        "skipped_campaigns": int(skipped_campaigns),
        "time_inference_sec": float(time_inference_sec),
        "time_overall_sec": float(time_overall_sec),
    }


def summarize_diagnostics(diagnostics_df: pd.DataFrame) -> Dict[str, Any]:
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


# ---------------------------------------------------------------------------
# Bidder param construction
# ---------------------------------------------------------------------------

def build_bidder_params(
    base_params: dict,
    model_params: dict,
    *,
    exp_type: str,
    objective: str = "clicks",
    verbose: bool = False,
) -> dict:
    return {
        **base_params,
        **model_params,
        "model_path": None,
        "exp_type": exp_type,
        "objective": objective,
        "eval_mode": True,
        "verbose": verbose,
        "use_tqdm": verbose,
        "debug_logs": False,
        "fit_log_every": 500,
        "inference_log_every": 24,
    }


# ---------------------------------------------------------------------------
# Single candidate: train -> evaluate -> persist artifacts
# ---------------------------------------------------------------------------

def run_drlb_candidate(
    config,
    train_stats_df: pd.DataFrame,
    train_campaigns_df: pd.DataFrame,
    label: str,
    bidder_params: dict,
    *,
    objective: str = "clicks",
    max_train_steps: Optional[int] = None,
    verbose: bool = False,
    scratch_dir: Path,
) -> Dict[str, Any]:
    """Train a DRLBBidder, evaluate on the test split, return metrics.

    Ephemeral artifacts (diagnostics CSV, .pt checkpoint used for eval) are
    written to *scratch_dir* and cleaned up by the caller's
    ``TemporaryDirectory``.  Only the returned metrics dict survives; it
    ends up in ``run_summary.json``.

    The final best model .pt is saved separately by the caller after the
    Optuna loop, not here.
    """
    bidder = DRLBBidder(bidder_params)
    bidder.fit(
        train_stats_df,
        campaigns_df=train_campaigns_df,
        max_steps=max_train_steps,
        objective=objective,
    )

    diagnostics = bidder.get_training_diagnostics().copy()

    model_path = scratch_dir / f"{label}.pt"
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
    metrics.update({"label": label})
    metrics.update(summarize_diagnostics(diagnostics))

    return {
        "label": label,
        "params": bidder_params,
        "metrics": metrics,
        "model_path": model_path,
    }


# ---------------------------------------------------------------------------
# Full Optuna tuning loop with temp-file cleanup
# ---------------------------------------------------------------------------

def run_optuna_experiment(
    config,
    source_config,
    *,
    base_drlb_params: dict,
    baseline_model_params: dict,
    exp_type: str,
    objective: str = "clicks",
    search_space_fn: Callable[[optuna.trial.Trial], dict],
    val_fraction: float = 0.2,
    n_trials: int = 1,
    max_train_steps: Optional[int] = None,
    verbose: bool = False,
) -> dict:
    """End-to-end: split data, run baseline, run Optuna, refit best, write summary.

    All intermediate artifacts (diagnostics CSVs, checkpoint .pt files used
    for eval) live in a TemporaryDirectory and are auto-deleted.  Only
    ``run_summary.json``, ``runs_index.jsonl``, and the final best .pt
    survive on disk.
    """
    config.ensure_artifact_dirs()
    config.config_dir.mkdir(parents=True, exist_ok=True)

    split_metadata = ensure_train_val_split(
        source_config=source_config,
        target_config=config,
        val_fraction=val_fraction,
        seed=config.random_seed,
    )

    train_stats_df = pd.read_csv(config.data_config["train"]["stats_path"])
    train_campaigns_df = pd.read_csv(config.data_config["train"]["campaigns_path"])

    baseline_params = build_bidder_params(
        base_drlb_params, baseline_model_params,
        exp_type=exp_type, objective=objective, verbose=verbose,
    )

    with tempfile.TemporaryDirectory(prefix="bat_run_") as tmpdir:
        tmpdir_path = Path(tmpdir)

        baseline_run = run_drlb_candidate(
            config=config,
            train_stats_df=train_stats_df,
            train_campaigns_df=train_campaigns_df,
            label="baseline_manual",
            bidder_params=baseline_params,
            objective=objective,
            max_train_steps=max_train_steps,
            verbose=verbose,
            scratch_dir=tmpdir_path,
        )

        def optuna_objective(trial: optuna.trial.Trial) -> float:
            model_params = search_space_fn(trial)
            trial_bidder_params = build_bidder_params(
                base_drlb_params, model_params,
                exp_type=exp_type, objective=objective, verbose=verbose,
            )
            run = run_drlb_candidate(
                config=config,
                train_stats_df=train_stats_df,
                train_campaigns_df=train_campaigns_df,
                label=f"trial_{trial.number:03d}",
                bidder_params=trial_bidder_params,
                objective=objective,
                max_train_steps=max_train_steps,
                verbose=verbose,
                scratch_dir=tmpdir_path,
            )
            m = run["metrics"]
            trial.set_user_attr("rmse", m["rmse"])
            trial.set_user_attr("clicks_sum", m["clicks_sum"])
            trial.set_user_attr("cpc_relative", m["cpc_relative"])
            trial.set_user_attr("quickspend", m["quickspend"])
            trial.set_user_attr("last_dqn_loss", m["last_dqn_loss"])
            trial.set_user_attr("last_reward_net_loss", m["last_reward_net_loss"])
            trial.set_user_attr("dqn_loss_mean", m["dqn_loss_mean"])
            trial.set_user_attr("reward_net_loss_mean", m.get("reward_net_loss_mean"))
            return m["clicks_sum"]

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=config.random_seed),
        )
        study.optimize(
            optuna_objective,
            n_trials=max(1, n_trials),
            n_jobs=1,
            show_progress_bar=verbose,
        )
        best_model_params = study.best_trial.params

        best_bidder_params = build_bidder_params(
            base_drlb_params, best_model_params,
            exp_type=exp_type, objective=objective, verbose=verbose,
        )
        best_run = run_drlb_candidate(
            config=config,
            train_stats_df=train_stats_df,
            train_campaigns_df=train_campaigns_df,
            label="best_trainval",
            bidder_params=best_bidder_params,
            objective=objective,
            max_train_steps=max_train_steps,
            verbose=verbose,
            scratch_dir=tmpdir_path,
        )

        shutil.copy2(
            best_run["model_path"],
            config.best_models_dir / "best_trainval.pt",
        )

    summary = build_run_summary(
        config=config,
        split_metadata=split_metadata,
        baseline_metrics=baseline_run["metrics"],
        best_model_params=best_model_params,
        best_run_metrics=best_run["metrics"],
        study_best_value=float(study.best_trial.value),
        study=study,
    )
    summary_path = config.outputs_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    append_runs_index(config, best_run["metrics"])

    print(json.dumps(summary, indent=2))
    print(f"summary_path={summary_path}")

    return summary


# ---------------------------------------------------------------------------
# Artifact management
# ---------------------------------------------------------------------------

def _data_hash(config) -> Optional[str]:
    """SHA-256 over the train+val campaign CSVs for reproducibility tracking."""
    h = hashlib.sha256()
    for split in ("train", "test"):
        p = Path(config.data_config[split]["campaigns_path"])
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()


def _git_hash() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip() or None
    except Exception:
        return None


def build_run_summary(
    config,
    split_metadata: dict,
    baseline_metrics: dict,
    best_model_params: dict,
    best_run_metrics: dict,
    study_best_value: float,
    study: optuna.study.Study,
) -> dict:
    all_trials = []
    for t in study.trials:
        row = {"trial": t.number, **t.params}
        row.update(t.user_attrs)
        all_trials.append(row)

    return {
        "experiment_name": config.experiment_name,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_hash": _git_hash(),
        "data_hash": _data_hash(config),
        "split_metadata": split_metadata,
        "baseline": {"metrics": baseline_metrics},
        "best_trial": {
            "trial_number": study.best_trial.number,
            "params": best_model_params,
            "metrics": best_run_metrics,
        },
        "all_trials_summary": all_trials,
        "study_best_value": study_best_value,
    }


def append_runs_index(config, metrics: dict) -> None:
    """Append one JSONL line to the experiment's runs_index.jsonl."""
    index_path = config.outputs_dir / "runs_index.jsonl"
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "label": metrics.get("label", "unknown"),
        "clicks_sum": metrics.get("clicks_sum"),
        "rmse": metrics.get("rmse"),
        "cpc_relative": metrics.get("cpc_relative"),
        "quickspend": metrics.get("quickspend"),
    }
    with open(index_path, "a") as f:
        f.write(json.dumps(entry) + "\n")

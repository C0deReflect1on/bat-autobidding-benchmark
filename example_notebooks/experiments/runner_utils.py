"""
Shared utilities for DRLB experiment runners.

All three runner scripts (hypgrid_v2, hypgrid_v3, smooth) share the same
candidate evaluation, diagnostics summarisation, and artifact-writing logic.
This module extracts that logic so each runner is a thin ~70-line CLI.

The data split contract is fixed and explicit:
- ``train_val`` for training
- ``val_val`` for tuning/selection
- ``holdout_test`` for final holdout reporting

Runners do not generate or rewrite split files.
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
from .shared_runner import run_experiment


# ---------------------------------------------------------------------------
# Data split resolution
# ---------------------------------------------------------------------------

_ROLE_FALLBACKS = {
    "train_val": ("train_val", "train"),
    "val_val": ("val_val", "test"),
    "holdout_test": ("holdout_test", "test"),
}


def _resolve_split_paths(config, role: str) -> tuple[str, str]:
    if role not in _ROLE_FALLBACKS:
        raise ValueError(f"Unknown split role '{role}'. Supported roles: {sorted(_ROLE_FALLBACKS.keys())}")

    data_config = config.data_config
    for key in _ROLE_FALLBACKS[role]:
        split_cfg = data_config.get(key)
        if isinstance(split_cfg, dict):
            campaigns_path = split_cfg.get("campaigns_path")
            stats_path = split_cfg.get("stats_path")
            if campaigns_path and stats_path:
                return str(campaigns_path), str(stats_path)

    available = sorted(data_config.keys())
    expected = list(_ROLE_FALLBACKS[role])
    raise ValueError(
        f"Missing split role '{role}' in config '{config.experiment_name}'. "
        f"Expected one of {expected}; available keys: {available}"
    )


def _ensure_paths_exist(config, role: str) -> tuple[str, str]:
    campaigns_path, stats_path = _resolve_split_paths(config, role)
    campaigns_file = Path(campaigns_path)
    stats_file = Path(stats_path)
    if not campaigns_file.exists() or not stats_file.exists():
        missing = [str(p) for p in (campaigns_file, stats_file) if not p.exists()]
        raise FileNotFoundError(
            f"Split '{role}' is missing required file(s) for config '{config.experiment_name}': {missing}"
        )
    return campaigns_path, stats_path


def resolve_data_splits(config) -> dict[str, dict[str, str]]:
    roles: dict[str, dict[str, str]] = {}
    for role in ("train_val", "val_val", "holdout_test"):
        campaigns_path, stats_path = _ensure_paths_exist(config, role)
        roles[role] = {
            "campaigns_path": campaigns_path,
            "stats_path": stats_path,
        }
    return roles


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
    data_splits: dict[str, dict[str, str]],
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
    """Train a DRLBBidder, evaluate on the validation split, return metrics.

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
        "input_campaigns": data_splits["val_val"]["campaigns_path"],
        "input_stats": data_splits["val_val"]["stats_path"],
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
    *,
    base_drlb_params: dict,
    baseline_model_params: dict,
    exp_type: str,
    objective: str = "clicks",
    search_space_fn: Callable[[optuna.trial.Trial], dict],
    n_trials: int = 1,
    max_train_steps: Optional[int] = None,
    verbose: bool = False,
) -> dict:
    """Compatibility wrapper that dispatches DRLB runs through the shared runner."""
    return run_experiment(
        config,
        verbose=verbose,
        base_drlb_params=base_drlb_params,
        baseline_model_params=baseline_model_params,
        exp_type=exp_type,
        objective=objective,
        search_space_fn=search_space_fn,
        n_trials=n_trials,
        max_train_steps=max_train_steps,
    )


# ---------------------------------------------------------------------------
# Artifact management
# ---------------------------------------------------------------------------

def _data_hash(config) -> Optional[str]:
    """SHA-256 over canonical split campaign CSVs for reproducibility tracking."""
    h = hashlib.sha256()
    for split in ("train_val", "val_val", "holdout_test"):
        p = Path(_resolve_split_paths(config, split)[0])
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
    data_splits: dict[str, dict[str, str]],
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
        "data_splits": data_splits,
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

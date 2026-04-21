from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional

import optuna
import pandas as pd

from simulator.model.drlb_bidder import DRLBBidder
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


def run_drlb_experiment(
    config,
    normalized_splits: dict[str, dict[str, str]],
    *,
    base_drlb_params: dict,
    baseline_model_params: dict,
    exp_type: str,
    objective: str = "clicks",
    search_space_fn: Callable[[optuna.trial.Trial], dict],
    n_trials: Optional[int] = None,
    max_train_steps: Optional[int] = None,
    verbose: bool = False,
) -> dict[str, Any]:
    return run_drlb_experiment_inprocess(
        config,
        normalized_splits,
        base_drlb_params=base_drlb_params,
        baseline_model_params=baseline_model_params,
        exp_type=exp_type,
        objective=objective,
        search_space_fn=search_space_fn,
        n_trials=n_trials,
        max_train_steps=max_train_steps,
        verbose=verbose,
    )["summary"]


def run_drlb_experiment_inprocess(
    config,
    normalized_splits: dict[str, dict[str, str]],
    *,
    base_drlb_params: dict,
    baseline_model_params: dict,
    exp_type: str,
    objective: str = "clicks",
    search_space_fn: Callable[[optuna.trial.Trial], dict],
    n_trials: Optional[int] = None,
    max_train_steps: Optional[int] = None,
    verbose: bool = False,
) -> dict[str, Any]:
    config.ensure_artifact_dirs()
    write_normalized_config(config)
    write_split_manifest(config, normalized_splits)

    train_stats_df = pd.read_csv(normalized_splits["train"]["stats_path"])
    train_campaigns_df = pd.read_csv(normalized_splits["train"]["campaigns_path"])
    trial_runs: list[dict[str, Any]] = []

    baseline_params = build_bidder_params(
        base_drlb_params,
        baseline_model_params,
        exp_type=exp_type,
        objective=objective,
        verbose=verbose,
    )

    with tempfile.TemporaryDirectory(prefix="bat_run_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        baseline_run = run_drlb_candidate(
            config=config,
            data_splits=normalized_splits,
            train_stats_df=train_stats_df,
            train_campaigns_df=train_campaigns_df,
            label="baseline_manual_val",
            bidder_params=baseline_params,
            eval_split_key="val",
            objective=objective,
            max_train_steps=max_train_steps,
            verbose=verbose,
            scratch_dir=tmpdir_path,
            return_bidder=True,
            return_diagnostics=True,
        )

        def optuna_objective(trial: optuna.trial.Trial) -> float:
            model_params = search_space_fn(trial)
            trial_bidder_params = build_bidder_params(
                base_drlb_params,
                model_params,
                exp_type=exp_type,
                objective=objective,
                verbose=verbose,
            )
            run = run_drlb_candidate(
                config=config,
                data_splits=normalized_splits,
                train_stats_df=train_stats_df,
                train_campaigns_df=train_campaigns_df,
                label=f"trial_{trial.number:03d}",
                bidder_params=trial_bidder_params,
                eval_split_key="val",
                objective=objective,
                max_train_steps=max_train_steps,
                verbose=verbose,
                scratch_dir=tmpdir_path,
                return_bidder=True,
                return_diagnostics=True,
            )
            trial_runs.append(run)
            metrics = run["metrics"]
            for key in (
                "rmse",
                "clicks_sum",
                "cpc_relative",
                "quickspend",
                "last_dqn_loss",
                "last_reward_net_loss",
                "dqn_loss_mean",
                "reward_net_loss_mean",
            ):
                trial.set_user_attr(key, metrics.get(key))
            return metrics["clicks_sum"]

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=config.optuna_seed),
        )
        study.optimize(
            optuna_objective,
            n_trials=max(1, int(n_trials or config.n_trials)),
            n_jobs=1,
            show_progress_bar=verbose,
        )
        best_model_params = dict(study.best_trial.params)

        best_val_bidder_params = build_bidder_params(
            base_drlb_params,
            best_model_params,
            exp_type=exp_type,
            objective=objective,
            verbose=verbose,
        )
        best_val_run = run_drlb_candidate(
            config=config,
            data_splits=normalized_splits,
            train_stats_df=train_stats_df,
            train_campaigns_df=train_campaigns_df,
            label="best_val",
            bidder_params=best_val_bidder_params,
            eval_split_key="val",
            objective=objective,
            max_train_steps=max_train_steps,
            verbose=verbose,
            scratch_dir=tmpdir_path,
            return_bidder=True,
            return_diagnostics=True,
        )

        refit_stats_df, refit_campaigns_df = load_refit_training_frames(
            config=config,
            normalized_splits=normalized_splits,
        )
        best_refit_run = run_drlb_candidate(
            config=config,
            data_splits=normalized_splits,
            train_stats_df=refit_stats_df,
            train_campaigns_df=refit_campaigns_df,
            label="best_refit",
            bidder_params=best_val_bidder_params,
            eval_split_key="test_holdout",
            objective=objective,
            max_train_steps=max_train_steps,
            verbose=verbose,
            scratch_dir=tmpdir_path,
            return_bidder=True,
            return_diagnostics=True,
        )

        shutil.copy2(best_refit_run["model_path"], config.best_models_dir / "best_refit.pt")

    summary = build_summary_header(config, normalized_splits)
    summary.update(
        {
            "reference": {
                "baseline_manual_val": {
                    "metrics": baseline_run["metrics"],
                    "diagnostics_path": baseline_run.get("diagnostics_path"),
                    "diagnostics_plot_path": baseline_run.get("diagnostics_plot_path"),
                }
            },
            "tuning": {
                "enabled": True,
                "n_trials": max(1, int(n_trials or config.n_trials)),
                "best_trial_number": int(study.best_trial.number),
                "best_params": best_model_params,
                "study_best_value": float(study.best_trial.value),
                "all_trials_summary": _study_trials_summary(study),
                "best_val_metrics": best_val_run["metrics"],
                "best_val_diagnostics_path": best_val_run.get("diagnostics_path"),
                "best_val_diagnostics_plot_path": best_val_run.get("diagnostics_plot_path"),
            },
            "refit": {
                "scope": config.refit_on,
                "model_path": str(config.best_models_dir / "best_refit.pt"),
                "diagnostics_path": best_refit_run.get("diagnostics_path"),
                "diagnostics_plot_path": best_refit_run.get("diagnostics_plot_path"),
            },
            "final_holdout": {
                "metrics": best_refit_run["metrics"],
            },
        }
    )

    write_metrics(config, best_refit_run["metrics"])
    write_run_summary(config, summary)
    append_runs_index(
        config,
        best_refit_run["metrics"],
        stage="final_holdout",
        label="best_refit_holdout",
    )

    if verbose:
        print(json.dumps(summary, indent=2))
    return {
        "summary": summary,
        "config": config,
        "normalized_splits": normalized_splits,
        "study": study,
        "reference_run": baseline_run,
        "trial_runs": trial_runs,
        "best_val_run": best_val_run,
        "best_refit_run": best_refit_run,
        "best_run": best_refit_run,
    }


def run_drlb_candidate(
    config,
    data_splits: dict[str, dict[str, str]],
    train_stats_df: pd.DataFrame,
    train_campaigns_df: pd.DataFrame,
    label: str,
    bidder_params: dict,
    *,
    eval_split_key: str,
    objective: str = "clicks",
    max_train_steps: Optional[int] = None,
    verbose: bool = False,
    scratch_dir: Path,
    return_bidder: bool = False,
    return_diagnostics: bool = False,
) -> dict[str, Any]:
    bidder = DRLBBidder(bidder_params)
    bidder.fit(
        train_stats_df,
        campaigns_df=train_campaigns_df,
        max_steps=max_train_steps,
        objective=objective,
    )

    diagnostics = bidder.get_training_diagnostics().copy()
    diagnostics_artifacts = write_training_diagnostics_artifacts(
        config=config,
        label=label,
        diagnostics_df=diagnostics,
    )
    model_path = scratch_dir / f"{label}.pt"
    bidder.save_model(str(model_path))

    eval_params = {
        **bidder_params,
        "input_campaigns": data_splits[eval_split_key]["campaigns_path"],
        "input_stats": data_splits[eval_split_key]["stats_path"],
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
        result["score"],
        skipped_campaigns=result.get("skipped_campaigns"),
        time_inference_sec=result.get("time_inference_sec"),
        time_overall_sec=result.get("time_overall_sec"),
    )
    metrics.update({"label": label})
    metrics.update(summarize_diagnostics(diagnostics))

    return {
        "label": label,
        "params": bidder_params,
        "metrics": metrics,
        "model_path": model_path,
        "bidder": bidder if return_bidder else None,
        "diagnostics": diagnostics if return_diagnostics else None,
        **diagnostics_artifacts,
    }


def load_refit_training_frames(
    *,
    config,
    normalized_splits: dict[str, dict[str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_stats_df = pd.read_csv(normalized_splits["train"]["stats_path"])
    train_campaigns_df = pd.read_csv(normalized_splits["train"]["campaigns_path"])
    if config.refit_on == "train":
        return train_stats_df, train_campaigns_df
    if config.refit_on == "train_plus_val":
        val_stats_df = pd.read_csv(normalized_splits["val"]["stats_path"])
        val_campaigns_df = pd.read_csv(normalized_splits["val"]["campaigns_path"])
        return (
            pd.concat([train_stats_df, val_stats_df], ignore_index=True),
            pd.concat([train_campaigns_df, val_campaigns_df], ignore_index=True),
        )
    raise ValueError(f"Unsupported refit_on '{config.refit_on}'")


def summarize_diagnostics(diagnostics_df: pd.DataFrame) -> dict[str, Any]:
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


def write_training_diagnostics_artifacts(
    *,
    config,
    label: str,
    diagnostics_df: pd.DataFrame,
) -> dict[str, str | None]:
    config.ensure_artifact_dirs()

    csv_path = config.outputs_dir / f"{label}_training_diagnostics.csv"
    diagnostics_df.to_csv(csv_path, index=False)

    plot_path = config.outputs_dir / f"{label}_training_diagnostics.png"
    plot_written = plot_training_diagnostics(diagnostics_df, plot_path, title=label)

    return {
        "diagnostics_path": str(csv_path),
        "diagnostics_plot_path": str(plot_path) if plot_written else None,
    }


def plot_training_diagnostics(
    diagnostics_df: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
    smoothing_window: int = 5,
) -> bool:
    if diagnostics_df.empty:
        return False

    required_columns = {"dqn_loss", "reward_net_loss", "reward_signal", "lambda"}
    if not required_columns.issubset(diagnostics_df.columns):
        return False

    from matplotlib import pyplot as plt

    plot_df = diagnostics_df.copy()
    x_col = "global_t" if "global_t" in plot_df.columns else None
    x_values = plot_df[x_col].to_numpy() if x_col is not None else plot_df.index.to_numpy()

    for column in required_columns:
        plot_df[column] = pd.to_numeric(plot_df[column], errors="coerce")
        plot_df[f"{column}_smooth"] = plot_df[column].rolling(smoothing_window, min_periods=1).mean()

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=140, sharex=True)
    axes = axes.ravel()

    loss_ax = axes[0]
    loss_ax.plot(x_values, plot_df["dqn_loss"], alpha=0.25, label="dqn_loss")
    loss_ax.plot(x_values, plot_df["dqn_loss_smooth"], linewidth=2, label="dqn_loss_smooth")
    loss_ax.plot(x_values, plot_df["reward_net_loss"], alpha=0.25, label="reward_net_loss")
    loss_ax.plot(x_values, plot_df["reward_net_loss_smooth"], linewidth=2, label="reward_net_loss_smooth")
    loss_ax.set_title("Losses")
    loss_ax.legend()

    reward_ax = axes[1]
    reward_ax.plot(x_values, plot_df["reward_signal"], alpha=0.35, label="reward_signal")
    reward_ax.plot(x_values, plot_df["reward_signal_smooth"], linewidth=2, label="reward_signal_smooth")
    reward_ax.set_title("Reward Signal")
    reward_ax.legend()

    lambda_ax = axes[2]
    lambda_ax.plot(x_values, plot_df["lambda"], linewidth=1.5, label="lambda")
    if "eps" in plot_df.columns:
        plot_df["eps"] = pd.to_numeric(plot_df["eps"], errors="coerce")
        lambda_ax.plot(x_values, plot_df["eps"], linewidth=1.5, label="eps")
    lambda_ax.set_title("Lambda / Eps")
    lambda_ax.legend()

    action_ax = axes[3]
    if "dqn_action" in plot_df.columns:
        action_ax.step(x_values, plot_df["dqn_action"], where="post")
        action_ax.set_title("DQN Action")
        action_ax.set_ylabel("action")
    else:
        action_ax.plot(x_values, plot_df["dqn_loss_smooth"], linewidth=2)
        action_ax.set_title("Smoothed DQN Loss")
        action_ax.set_ylabel("dqn_loss")

    for ax in axes:
        ax.set_xlabel("global_t" if x_col is not None else "step")

    fig.suptitle(f"DRLB Training Diagnostics: {title}")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return True


def _study_trials_summary(study) -> list[dict[str, Any]]:
    rows = []
    for trial in study.trials:
        row = {"trial": int(trial.number), **trial.params}
        row.update(trial.user_attrs)
        rows.append(row)
    return rows

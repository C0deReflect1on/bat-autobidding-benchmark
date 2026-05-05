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

DRLB_RUNTIME_DEFAULTS = {
    "model_path": None,
    "eval_mode": True,
    "debug_logs": False,
    "fit_log_every": 500,
    "inference_log_every": 24,
}


def add_smoothed_diagnostics(
    diagnostics_df: pd.DataFrame,
    *,
    smoothing_window: int = 500,
) -> pd.DataFrame:
    if diagnostics_df.empty:
        return diagnostics_df.copy()

    plot_df = diagnostics_df.copy()
    window = max(1, int(smoothing_window))
    for column in ("dqn_loss", "reward_net_loss", "reward_signal"):
        if column in plot_df.columns:
            numeric = pd.to_numeric(plot_df[column], errors="coerce")
            plot_df[f"{column}_smooth"] = numeric.rolling(window, min_periods=1).mean()
    return plot_df


def run_drlb_experiment(
    config,
    normalized_splits: dict[str, dict[str, str]],
    *,
    base_drlb_params: dict,
    reference_model_params: dict,
    state_type: str,
    objective: str = "clicks",
    search_space_fn: Callable[[optuna.trial.Trial], dict],
    n_trials: Optional[int] = None,
    max_train_steps: Optional[int] = None,
    verbose: bool = False,
    show_progress: Optional[bool] = None,
) -> dict[str, Any]:
    return run_drlb_experiment_inprocess(
        config,
        normalized_splits,
        base_drlb_params=base_drlb_params,
        reference_model_params=reference_model_params,
        state_type=state_type,
        objective=objective,
        search_space_fn=search_space_fn,
        n_trials=n_trials,
        max_train_steps=max_train_steps,
        verbose=verbose,
        show_progress=show_progress,
    )["summary"]


def run_drlb_experiment_inprocess(
    config,
    normalized_splits: dict[str, dict[str, str]],
    *,
    base_drlb_params: dict,
    reference_model_params: dict,
    state_type: str,
    objective: str = "clicks",
    search_space_fn: Callable[[optuna.trial.Trial], dict],
    n_trials: Optional[int] = None,
    max_train_steps: Optional[int] = None,
    verbose: bool = False,
    show_progress: Optional[bool] = None,
) -> dict[str, Any]:
    config.ensure_artifact_dirs()
    write_normalized_config(config)
    write_split_manifest(config, normalized_splits)

    train_stats_df = pd.read_csv(normalized_splits["train"]["stats_path"])
    train_campaigns_df = pd.read_csv(normalized_splits["train"]["campaigns_path"])
    trial_runs: list[dict[str, Any]] = []
    run_n_trials = max(1, int(n_trials or config.n_trials))
    progress_enabled = verbose if show_progress is None else show_progress

    with tempfile.TemporaryDirectory(prefix="bat_run_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        reference_params = {
            **DRLB_RUNTIME_DEFAULTS,
            **base_drlb_params,
            **reference_model_params,
            "state_type": state_type,
            "objective": objective,
            "verbose": verbose,
            "use_tqdm": progress_enabled,
        }
        reference_run = run_drlb_candidate(
            config=config,
            data_splits=normalized_splits,
            train_stats_df=train_stats_df,
            train_campaigns_df=train_campaigns_df,
            label="baseline_manual_val",
            bidder_params=reference_params,
            eval_split_key="val",
            objective=objective,
            max_train_steps=max_train_steps,
            verbose=verbose,
            show_progress=progress_enabled,
            scratch_dir=tmpdir_path,
            return_bidder=True,
            return_diagnostics=True,
            write_artifacts=False,
        )

        def optuna_objective(trial: optuna.trial.Trial) -> float:
            trial_bidder_params = {
                **DRLB_RUNTIME_DEFAULTS,
                **base_drlb_params,
                **reference_model_params,
                **search_space_fn(trial),
                "state_type": state_type,
                "objective": objective,
                "verbose": verbose,
                "use_tqdm": progress_enabled,
            }
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
                show_progress=progress_enabled,
                scratch_dir=tmpdir_path,
                return_bidder=True,
                return_diagnostics=True,
                write_artifacts=False,
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
            n_trials=run_n_trials,
            n_jobs=1,
            show_progress_bar=progress_enabled,
        )
        best_model_params = dict(study.best_trial.params)
        best_trial_run = trial_runs[int(study.best_trial.number)]
        best_trial_params = dict(best_trial_run["params"])

        best_val_bidder_params = {
            **best_trial_params,
            "state_type": state_type,
            "objective": objective,
            "verbose": verbose,
            "use_tqdm": progress_enabled,
        }
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
            show_progress=progress_enabled,
            scratch_dir=tmpdir_path,
            return_bidder=True,
            return_diagnostics=True,
            write_artifacts=False,
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
            show_progress=progress_enabled,
            scratch_dir=tmpdir_path,
            return_bidder=True,
            return_diagnostics=True,
            write_artifacts=False,
        )

        if "eval_diagnostics" in best_refit_run:
            train_eval_diagnostics = collect_runtime_diagnostics_for_split(
                config=config,
                data_splits=normalized_splits,
                bidder_params=best_val_bidder_params,
                split_key="train",
                model_path=best_refit_run["model_path"],
                verbose=verbose,
                show_progress=progress_enabled,
            )
            val_eval_diagnostics = collect_runtime_diagnostics_for_split(
                config=config,
                data_splits=normalized_splits,
                bidder_params=best_val_bidder_params,
                split_key="val",
                model_path=best_refit_run["model_path"],
                verbose=verbose,
                show_progress=progress_enabled,
            )
            best_refit_artifacts = write_training_diagnostics_artifacts(
                config=config,
                label="best_refit",
                diagnostics_df=best_refit_run["diagnostics"],
                holdout_diagnostics_df=best_refit_run["eval_diagnostics"],
                action_diagnostics_by_split={
                    "train": train_eval_diagnostics,
                    "val": val_eval_diagnostics,
                    "holdout": best_refit_run["eval_diagnostics"],
                },
            )
            best_refit_run.update(best_refit_artifacts)

        shutil.copy2(best_refit_run["model_path"], config.best_models_dir / "best_refit.pt")

    summary = build_summary_header(config, normalized_splits)
    summary.update(
        {
            "reference": {
                "baseline_manual_val": {
                    "metrics": reference_run["metrics"],
                    "diagnostics_path": reference_run.get("diagnostics_path"),
                    "diagnostics_plot_path": reference_run.get("diagnostics_plot_path"),
                    "reward_net_plot_path": reference_run.get("reward_net_plot_path"),
                    "eval_action_distribution_path": reference_run.get("eval_action_distribution_path"),
                }
            },
            "tuning": {
                "enabled": True,
                "n_trials": run_n_trials,
                "best_trial_number": int(study.best_trial.number),
                "best_params": best_model_params,
                "study_best_value": float(study.best_trial.value),
                "all_trials_summary": _study_trials_summary(study),
                "best_val_metrics": best_val_run["metrics"],
                "best_val_diagnostics_path": best_val_run.get("diagnostics_path"),
                "best_val_diagnostics_plot_path": best_val_run.get("diagnostics_plot_path"),
                "best_val_reward_net_plot_path": best_val_run.get("reward_net_plot_path"),
                "best_val_action_distribution_path": best_val_run.get("eval_action_distribution_path"),
            },
            "refit": {
                "scope": config.refit_on,
                "model_path": str(config.best_models_dir / "best_refit.pt"),
                "diagnostics_path": best_refit_run.get("diagnostics_path"),
                "diagnostics_plot_path": best_refit_run.get("diagnostics_plot_path"),
                "reward_net_plot_path": best_refit_run.get("reward_net_plot_path"),
                "eval_action_distribution_path": best_refit_run.get("eval_action_distribution_path"),
                "combined_diagnostics_plot_path": best_refit_run.get("combined_plot_path"),
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
        "reference_run": reference_run,
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
    show_progress: Optional[bool] = None,
    scratch_dir: Path,
    return_bidder: bool = False,
    return_diagnostics: bool = False,
    write_artifacts: bool = True,
) -> dict[str, Any]:
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
        use_tqdm=verbose if show_progress is None else show_progress,
    )
    eval_diagnostics = _concat_runtime_diagnostics(result.get("runtime_diagnostics"))
    diagnostics_artifacts = {
        "diagnostics_path": None,
        "diagnostics_plot_path": None,
        "reward_net_plot_path": None,
        "eval_action_distribution_path": None,
        "combined_plot_path": None,
    }
    if write_artifacts:
        diagnostics_artifacts = write_training_diagnostics_artifacts(
            config=config,
            label=label,
            diagnostics_df=diagnostics,
            holdout_diagnostics_df=eval_diagnostics,
            action_diagnostics_by_split={eval_split_key: eval_diagnostics},
        )
    metrics = score_to_dict(
        result["score"],
        skipped_campaigns=result.get("skipped_campaigns"),
        time_inference_sec=result.get("time_inference_sec"),
        time_overall_sec=result.get("time_overall_sec"),
        average_end_balance_share=result.get("average_end_balance_share"),
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
        "eval_diagnostics": eval_diagnostics,
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


def collect_runtime_diagnostics_for_split(
    *,
    config,
    data_splits: dict[str, dict[str, str]],
    bidder_params: dict,
    split_key: str,
    model_path: Path,
    verbose: bool = False,
    show_progress: Optional[bool] = None,
) -> pd.DataFrame:
    eval_params = {
        **bidder_params,
        "input_campaigns": data_splits[split_key]["campaigns_path"],
        "input_stats": data_splits[split_key]["stats_path"],
        "model_path": str(model_path),
        "eval_mode": True,
    }
    result = autobidder_check(
        bidder=DRLBBidder,
        params=eval_params,
        auction_mode=config.auction_mode,
        verbose=verbose,
        log_every_campaigns=100,
        use_tqdm=verbose if show_progress is None else show_progress,
    )
    return _concat_runtime_diagnostics(result.get("runtime_diagnostics"))


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
    holdout_diagnostics_df: pd.DataFrame | None = None,
    action_diagnostics_by_split: dict[str, pd.DataFrame] | None = None,
) -> dict[str, str | None]:
    config.ensure_artifact_dirs()

    smoothed_diagnostics_df = add_smoothed_diagnostics(diagnostics_df)
    csv_path = config.outputs_dir / f"{label}_training_diagnostics.csv"
    smoothed_diagnostics_df.to_csv(csv_path, index=False)
    combined_plot_path = config.outputs_dir / "drlb_diagnostics.png"
    combined_plot_written = plot_compact_diagnostics(
        diagnostics_df=smoothed_diagnostics_df,
        holdout_diagnostics_df=holdout_diagnostics_df if holdout_diagnostics_df is not None else pd.DataFrame(),
        action_diagnostics_by_split=action_diagnostics_by_split or {},
        output_path=combined_plot_path,
        title=label,
    )

    return {
        "diagnostics_path": str(csv_path),
        "diagnostics_plot_path": str(combined_plot_path) if combined_plot_written else None,
        "reward_net_plot_path": None,
        "eval_action_distribution_path": None,
        "combined_plot_path": str(combined_plot_path) if combined_plot_written else None,
    }


def plot_compact_diagnostics(
    *,
    diagnostics_df: pd.DataFrame,
    holdout_diagnostics_df: pd.DataFrame,
    action_diagnostics_by_split: dict[str, pd.DataFrame],
    output_path: Path,
    title: str,
) -> bool:
    if diagnostics_df.empty:
        return False

    required_columns = {"dqn_loss", "reward_net_loss"}
    if not required_columns.issubset(diagnostics_df.columns):
        return False

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    train_plot_df = add_smoothed_diagnostics(diagnostics_df)
    train_x_col = "global_t" if "global_t" in train_plot_df.columns else None
    train_x = train_plot_df[train_x_col].to_numpy() if train_x_col is not None else train_plot_df.index.to_numpy()

    fig, axes = plt.subplots(3, 2, figsize=(14, 12), dpi=140)
    dqn_ax = axes[0, 0]
    dqn_ax.plot(train_x, train_plot_df["dqn_loss"], linewidth=1.0, alpha=0.22, color="#1f77b4")
    if "dqn_loss_smooth" in train_plot_df.columns:
        dqn_ax.plot(train_x, train_plot_df["dqn_loss_smooth"], linewidth=1.8, color="#1f77b4")
    dqn_ax.set_title("DQN Loss (Train)")
    dqn_ax.set_xlabel("global_t" if train_x_col is not None else "step")
    dqn_ax.set_ylabel("loss")

    reward_ax = axes[0, 1]
    reward_ax.plot(train_x, train_plot_df["reward_net_loss"], linewidth=1.0, alpha=0.22, label="train_raw", color="#2ca02c")
    if "reward_net_loss_smooth" in train_plot_df.columns:
        reward_ax.plot(
            train_x,
            train_plot_df["reward_net_loss_smooth"],
            linewidth=1.8,
            label="train_smooth",
            color="#2ca02c",
        )
    if (not holdout_diagnostics_df.empty) and ("reward_net_loss" in holdout_diagnostics_df.columns):
        holdout_plot_df = add_smoothed_diagnostics(holdout_diagnostics_df, smoothing_window=50)
        holdout_y = holdout_plot_df["reward_net_loss"].to_numpy()
        holdout_x = holdout_diagnostics_df.index.to_numpy()
        reward_ax.plot(holdout_x, holdout_y, linewidth=0.9, alpha=0.18, label="holdout_raw", color="#d62728")
        if "reward_net_loss_smooth" in holdout_plot_df.columns:
            reward_ax.plot(
                holdout_x,
                holdout_plot_df["reward_net_loss_smooth"].to_numpy(),
                linewidth=1.4,
                label="holdout_smooth",
                color="#d62728",
            )
    reward_ax.set_title("RewardNet Loss (Train / Holdout)")
    reward_ax.set_xlabel("step")
    reward_ax.set_ylabel("loss")
    reward_ax.legend()

    _plot_action_distribution_axis(
        axes[1, 0],
        action_diagnostics_by_split.get("train", pd.DataFrame()),
        split_label="Train",
    )
    _plot_action_distribution_axis(
        axes[1, 1],
        action_diagnostics_by_split.get("val", pd.DataFrame()),
        split_label="Val",
    )
    _plot_action_distribution_axis(
        axes[2, 0],
        action_diagnostics_by_split.get("holdout", pd.DataFrame()),
        split_label="Holdout",
    )
    axes[2, 1].axis("off")

    fig.suptitle(f"DRLB Diagnostics: {title}")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_action_distribution_axis(ax, eval_diagnostics_df: pd.DataFrame, *, split_label: str) -> None:
    if eval_diagnostics_df.empty or "dqn_action" not in eval_diagnostics_df.columns:
        ax.set_title(f"{split_label} Action Distribution")
        ax.set_xlabel("action")
        ax.set_ylabel("count")
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return

    action_counts = eval_diagnostics_df["dqn_action"].astype(int).value_counts().sort_index()
    if action_counts.empty:
        ax.set_title(f"{split_label} Action Distribution")
        ax.set_xlabel("action")
        ax.set_ylabel("count")
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return

    ax.bar(action_counts.index.astype(str), action_counts.values, color="#2f6db3")
    ax.set_title(f"{split_label} Action Distribution")
    ax.set_xlabel("action")
    ax.set_ylabel("count")


def plot_training_diagnostics(
    diagnostics_df: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
) -> bool:
    if diagnostics_df.empty:
        return False

    required_columns = {"dqn_loss"}
    if not required_columns.issubset(diagnostics_df.columns):
        return False

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plot_df = diagnostics_df.copy()
    x_col = "global_t" if "global_t" in plot_df.columns else None
    x_values = plot_df[x_col].to_numpy() if x_col is not None else plot_df.index.to_numpy()

    fig, loss_ax = plt.subplots(1, 1, figsize=(13, 5), dpi=140, sharex=True)
    loss_ax.plot(x_values, plot_df["dqn_loss"], linewidth=1.7, label="dqn_loss")
    loss_ax.set_title("DQN Loss")
    loss_ax.legend()
    loss_ax.set_xlabel("global_t" if x_col is not None else "step")

    fig.suptitle(f"DRLB DQN Diagnostics: {title}")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_reward_net_diagnostics(
    diagnostics_df: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
    smoothing_window: int = 5,
) -> bool:
    if diagnostics_df.empty:
        return False

    required_columns = {"reward_net_loss", "reward_signal"}
    if not required_columns.issubset(diagnostics_df.columns):
        return False

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plot_df = diagnostics_df.copy()
    x_col = "global_t" if "global_t" in plot_df.columns else None
    x_values = plot_df[x_col].to_numpy() if x_col is not None else plot_df.index.to_numpy()

    for column in required_columns:
        plot_df[column] = pd.to_numeric(plot_df[column], errors="coerce")
        plot_df[f"{column}_smooth"] = plot_df[column].rolling(smoothing_window, min_periods=1).mean()

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), dpi=140, sharex=True)

    loss_ax = axes[0]
    loss_ax.plot(x_values, plot_df["reward_net_loss"], alpha=0.25, label="reward_net_loss")
    loss_ax.plot(x_values, plot_df["reward_net_loss_smooth"], linewidth=2, label="reward_net_loss_smooth")
    loss_ax.set_title("RewardNet Loss")
    loss_ax.legend()

    reward_ax = axes[1]
    reward_ax.plot(x_values, plot_df["reward_signal"], alpha=0.35, label="reward_signal")
    reward_ax.plot(x_values, plot_df["reward_signal_smooth"], linewidth=2, label="reward_signal_smooth")
    reward_ax.set_title("Reward Signal")
    reward_ax.legend()

    for ax in axes:
        ax.set_xlabel("global_t" if x_col is not None else "step")

    fig.suptitle(f"DRLB RewardNet Diagnostics: {title}")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_action_distribution(
    eval_diagnostics_df: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
    split_label: str,
) -> bool:
    if eval_diagnostics_df.empty or "dqn_action" not in eval_diagnostics_df.columns:
        return False

    action_counts = (
        pd.to_numeric(eval_diagnostics_df["dqn_action"], errors="coerce")
        .dropna()
        .astype(int)
        .value_counts()
        .sort_index()
    )
    if action_counts.empty:
        return False

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5), dpi=140)
    ax.bar(action_counts.index.astype(str), action_counts.values, color="#2f6db3")
    ax.set_title(f"DQN Action Distribution on {split_label}")
    ax.set_xlabel("action")
    ax.set_ylabel("count")
    fig.suptitle(f"DRLB Validation Action Distribution: {title}")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return True


def _concat_runtime_diagnostics(runtime_diagnostics: list[pd.DataFrame] | None) -> pd.DataFrame:
    if not runtime_diagnostics:
        return pd.DataFrame()
    frames = [df for df in runtime_diagnostics if isinstance(df, pd.DataFrame) and not df.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _study_trials_summary(study) -> list[dict[str, Any]]:
    rows = []
    for trial in study.trials:
        row = {"trial": int(trial.number), **trial.params}
        row.update(trial.user_attrs)
        rows.append(row)
    return rows

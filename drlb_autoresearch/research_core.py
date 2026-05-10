from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable

import matplotlib.pyplot as plt
import optuna
import pandas as pd

from example_notebooks.experiments.adapters.drlb_adapter import (
    DRLB_RUNTIME_DEFAULTS,
    add_smoothed_diagnostics,
)
from example_notebooks.experiments.drlb.profiles import get_profile
from example_notebooks.experiments.infra.artifacts import score_to_dict
from example_notebooks.experiments.infra.split_registry import resolve_split_set
from simulator.model.drlb_bidder import DRLBBidder
from simulator.model.linear_bidder import LinearBidder
from simulator.simulation.bat_step_env import BatStepEnv
from simulator.simulation.simulate import simulate_campaign
from simulator.validation.check_results import autobidder_check, create_campaign_instance


AUTORESEARCH_ROOT = Path(__file__).resolve().parent
ARTIFACTS_ROOT = AUTORESEARCH_ROOT / "artifacts"
AUTORESEARCH_LOG_PATH = AUTORESEARCH_ROOT / "autoresearch.md"
LOCKED_LINEAR_MANIFEST = (
    ARTIFACTS_ROOT
    / "00_locked_linear_baseline"
    / "locked_linear_reference_v1"
    / "baseline_manifest.json"
)


def ensure_run_dir(stage_key: str, run_name: str) -> Path:
    run_dir = ARTIFACTS_ROOT / stage_key / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_locked_linear_reference(
    manifest_path: Path = LOCKED_LINEAR_MANIFEST,
) -> dict[str, Any]:
    payload = json.loads(manifest_path.read_text())
    ref = payload["locked_reference"]
    return {
        "notebook_path": ref["notebook_path"],
        "linear_params": dict(ref["linear_params"]),
        "linear_lambda_init": float(ref["linear_lambda_init"]),
    }


def prepare_context(
    *,
    profile_key: str = "drlb_smooth",
    split_set: str = "full_train_val_holdout",
    objective: str = "clicks",
    auction_mode: str = "FPA",
    locked_reference: dict[str, Any] | None = None,
) -> dict[str, Any]:
    locked = load_locked_linear_reference() if locked_reference is None else dict(locked_reference)
    profile = get_profile(profile_key)
    splits = resolve_split_set(split_set)
    train_stats_df = pd.read_csv(splits["train"]["stats_path"])
    train_campaigns_df = pd.read_csv(splits["train"]["campaigns_path"])
    bidder_params = {
        **DRLB_RUNTIME_DEFAULTS,
        **profile["base_drlb_params"],
        **profile["reference_model_params"],
        "state_type": profile["state_type"],
        "objective": objective,
        "auction_mode": auction_mode,
        "verbose": False,
        "use_tqdm": False,
        "eval_mode": True,
        "init_lambda": locked["linear_lambda_init"],
        "init_lambda_mode": "constant",
    }
    return {
        "profile_key": profile_key,
        "split_set": split_set,
        "objective": objective,
        "auction_mode": auction_mode,
        "profile": profile,
        "splits": splits,
        "train_stats_df": train_stats_df,
        "train_campaigns_df": train_campaigns_df,
        "base_bidder_params": bidder_params,
        "locked_reference": locked,
    }


def append_autoresearch_entry(
    *,
    heading: str,
    stage_key: str,
    run_name: str,
    hypothesis: str,
    params: dict[str, Any],
    result_summary: dict[str, Any],
    decision: str,
    notes: str,
) -> None:
    lines = [
        f"## {heading}",
        f"- stage: `{stage_key}`",
        f"- run: `{run_name}`",
        f"- hypothesis: {hypothesis}",
        f"- decision: `{decision}`",
        "",
        "### Params",
        "```json",
        json.dumps(params, indent=2, ensure_ascii=True),
        "```",
        "",
        "### Result",
        "```json",
        json.dumps(result_summary, indent=2, ensure_ascii=True),
        "```",
        "",
        "### Notes",
        notes.strip() or "n/a",
        "",
    ]
    prefix = "" if not AUTORESEARCH_LOG_PATH.exists() or AUTORESEARCH_LOG_PATH.stat().st_size == 0 else "\n"
    AUTORESEARCH_LOG_PATH.write_text(
        AUTORESEARCH_LOG_PATH.read_text() + prefix + "\n".join(lines)
        if AUTORESEARCH_LOG_PATH.exists()
        else "# DRLB Autoresearch Log\n\n" + "\n".join(lines)
    )


def evaluate_linear_single_pass(
    *,
    split: dict[str, str],
    auction_mode: str,
    linear_params: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    params = _build_linear_bidder_params(linear_params)
    eval_params = {
        "input_campaigns": split["campaigns_path"],
        "input_stats": split["stats_path"],
        **params,
    }
    result = autobidder_check(
        bidder=LinearBidder,
        params=eval_params,
        auction_mode=auction_mode,
        verbose=False,
        log_every_campaigns=100,
        use_tqdm=False,
    )
    metrics = score_to_dict(
        result["score"],
        skipped_campaigns=result.get("skipped_campaigns"),
        time_inference_sec=result.get("time_inference_sec"),
        time_overall_sec=result.get("time_overall_sec"),
        average_end_balance_share=result.get("average_end_balance_share"),
    )
    return metrics, params


def _evaluate_linear_with_rollouts(
    *,
    split: dict[str, str],
    auction_mode: str,
    linear_params: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    params = _build_linear_bidder_params(linear_params)
    eval_params = {
        "input_campaigns": split["campaigns_path"],
        "input_stats": split["stats_path"],
        **params,
    }
    result = autobidder_check(
        bidder=LinearBidder,
        params=eval_params,
        auction_mode=auction_mode,
        verbose=False,
        log_every_campaigns=100,
        use_tqdm=False,
    )
    metrics = score_to_dict(
        result["score"],
        skipped_campaigns=result.get("skipped_campaigns"),
        time_inference_sec=result.get("time_inference_sec"),
        time_overall_sec=result.get("time_overall_sec"),
        average_end_balance_share=result.get("average_end_balance_share"),
    )

    campaign_columns = [
        "campaign_id",
        "region_id",
        "logical_category",
        "auction_budget",
        "hours",
        "clicks_total",
        "spend_total",
        "avg_bid",
        "balance_final",
        "initial_balance",
        "end_balance_share",
    ]
    hourly_columns = [
        "campaign_id",
        "region_id",
        "logical_category",
        "auction_budget",
        "hour_index",
        "curr_timestamp",
        "curr_time",
        "bid",
        "spend_history",
        "clicks_history",
        "balance",
        "clicks",
        "end_balance_share",
    ]

    campaigns_df = pd.read_csv(split["campaigns_path"])[
        ["campaign_id", "region_id", "logical_category", "auction_budget"]
    ].drop_duplicates(subset=["campaign_id"])
    campaign_meta = {
        int(row["campaign_id"]): row
        for _, row in campaigns_df.iterrows()
    }

    campaign_rows: list[dict[str, Any]] = []
    hourly_rows: list[pd.DataFrame] = []
    for history_df in result.get("all_hist_data", []):
        if history_df.empty:
            continue
        ordered_history_df = history_df.sort_values("curr_timestamp").reset_index(drop=True)
        last_row = ordered_history_df.iloc[-1]
        campaign_id = int(last_row["campaign_id"])
        meta_row = campaign_meta.get(campaign_id)
        if meta_row is None:
            continue

        initial_balance = float(last_row["initial_balance"])
        final_balance = float(last_row["balance"])
        campaign_rows.append(
            {
                "campaign_id": campaign_id,
                "region_id": int(meta_row["region_id"]),
                "logical_category": meta_row["logical_category"],
                "auction_budget": float(meta_row["auction_budget"]),
                "hours": int(len(ordered_history_df)),
                "clicks_total": float(last_row["clicks"]),
                "spend_total": float(ordered_history_df["spend_history"].sum()),
                "avg_bid": float(ordered_history_df["bid"].mean()),
                "balance_final": final_balance,
                "initial_balance": initial_balance,
                "end_balance_share": final_balance / max(initial_balance, 1e-9),
            }
        )

        hourly_df = ordered_history_df[
            [
                "campaign_id",
                "curr_timestamp",
                "curr_time",
                "bid",
                "spend_history",
                "clicks_history",
                "balance",
                "clicks",
            ]
        ].copy()
        hourly_df["hour_index"] = hourly_df.index.astype(int)
        hourly_df["region_id"] = int(meta_row["region_id"])
        hourly_df["logical_category"] = meta_row["logical_category"]
        hourly_df["auction_budget"] = float(meta_row["auction_budget"])
        hourly_df["end_balance_share"] = hourly_df["balance"] / initial_balance
        hourly_rows.append(
            hourly_df[
                [
                    "campaign_id",
                    "region_id",
                    "logical_category",
                    "auction_budget",
                    "hour_index",
                    "curr_timestamp",
                    "curr_time",
                    "bid",
                    "spend_history",
                    "clicks_history",
                    "balance",
                    "clicks",
                    "end_balance_share",
                ]
            ]
        )

    campaign_rollout_df = pd.DataFrame(campaign_rows, columns=campaign_columns)
    hourly_rollout_df = (
        pd.concat(hourly_rows, axis=0, ignore_index=True)
        if hourly_rows
        else pd.DataFrame(columns=hourly_columns)
    )
    return metrics, campaign_rollout_df, hourly_rollout_df


def _build_linear_bidder_params(linear_params: dict[str, Any]) -> dict[str, Any]:
    return {
        "cold_start_coef": linear_params["coef"],
        "lower_clip": linear_params["lower_clip"],
        "upper_clip": linear_params["upper_clip"],
        "factor": linear_params["factor"],
    }


def evaluate_drlb_checkpoint(
    *,
    model_path: Path,
    bidder_params: dict[str, Any],
    split: dict[str, str],
    auction_mode: str,
) -> dict[str, Any]:
    eval_params = {
        **bidder_params,
        "input_campaigns": split["campaigns_path"],
        "input_stats": split["stats_path"],
        "model_path": str(model_path),
        "eval_mode": True,
    }
    result = autobidder_check(
        bidder=DRLBBidder,
        params=eval_params,
        auction_mode=auction_mode,
        verbose=False,
        log_every_campaigns=100,
        use_tqdm=False,
    )
    return score_to_dict(
        result["score"],
        skipped_campaigns=result.get("skipped_campaigns"),
        time_inference_sec=result.get("time_inference_sec"),
        time_overall_sec=result.get("time_overall_sec"),
        average_end_balance_share=result.get("average_end_balance_share"),
    )


def campaign_rollout_table(
    *,
    bidder_cls: type,
    bidder_params: dict[str, Any],
    split: dict[str, str],
    auction_mode: str,
    max_campaigns: int | None = None,
) -> pd.DataFrame:
    campaigns_df = pd.read_csv(split["campaigns_path"]).reset_index(drop=True)
    stats_df = pd.read_csv(split["stats_path"])
    rows: list[dict[str, Any]] = []

    for idx, (_, campaign_row) in enumerate(campaigns_df.iterrows(), start=1):
        if max_campaigns is not None and idx > max_campaigns:
            break

        campaign_id = int(campaign_row["campaign_id"])
        campaign_stats = stats_df[stats_df["campaign_id"] == campaign_id].copy()
        if campaign_stats.empty:
            continue

        campaign = create_campaign_instance(campaign_row, mean_click_price=5.0)
        bidder = bidder_cls(dict(bidder_params))
        history_df = simulate_campaign(
            campaign=campaign,
            bidder=bidder,
            stats_file=campaign_stats,
            auction_mode=auction_mode,
        ).to_data_frame()
        if history_df.empty:
            continue

        last_row = history_df.iloc[-1]
        initial_balance = float(last_row["initial_balance"])
        final_balance = float(last_row["balance"])
        rows.append(
            {
                "campaign_id": campaign_id,
                "region_id": int(campaign_row["region_id"]),
                "logical_category": campaign_row["logical_category"],
                "auction_budget": float(campaign_row["auction_budget"]),
                "hours": int(len(history_df)),
                "clicks_total": float(last_row["clicks"]),
                "spend_total": float(history_df["spend_history"].sum()),
                "avg_bid": float(history_df["bid"].mean()),
                "balance_final": final_balance,
                "initial_balance": initial_balance,
                "end_balance_share": final_balance / max(initial_balance, 1e-9),
            }
        )

    return pd.DataFrame(rows)


def campaign_hourly_rollout_table(
    *,
    bidder_cls: type,
    bidder_params: dict[str, Any],
    split: dict[str, str],
    auction_mode: str,
    max_campaigns: int | None = None,
    campaign_ids: list[int] | None = None,
) -> pd.DataFrame:
    campaigns_df = pd.read_csv(split["campaigns_path"]).reset_index(drop=True)
    stats_df = pd.read_csv(split["stats_path"])
    if campaign_ids is not None:
        campaigns_df = campaigns_df[campaigns_df["campaign_id"].isin(campaign_ids)].reset_index(drop=True)
    rows: list[pd.DataFrame] = []

    for idx, (_, campaign_row) in enumerate(campaigns_df.iterrows(), start=1):
        if max_campaigns is not None and idx > max_campaigns:
            break

        campaign_id = int(campaign_row["campaign_id"])
        campaign_stats = stats_df[stats_df["campaign_id"] == campaign_id].copy()
        if campaign_stats.empty:
            continue

        campaign = create_campaign_instance(campaign_row, mean_click_price=5.0)
        bidder = bidder_cls(dict(bidder_params))
        history_df = simulate_campaign(
            campaign=campaign,
            bidder=bidder,
            stats_file=campaign_stats,
            auction_mode=auction_mode,
        ).to_data_frame()
        if history_df.empty:
            continue

        history_df = history_df.sort_values("curr_timestamp").reset_index(drop=True)
        history_df["hour_index"] = history_df.index.astype(int)
        history_df["campaign_id"] = campaign_id
        history_df["region_id"] = int(campaign_row["region_id"])
        history_df["logical_category"] = campaign_row["logical_category"]
        history_df["auction_budget"] = float(campaign_row["auction_budget"])
        history_df["end_balance_share"] = history_df["balance"] / history_df["initial_balance"].clip(lower=1e-9)
        rows.append(
            history_df[
                [
                    "campaign_id",
                    "region_id",
                    "logical_category",
                    "auction_budget",
                    "hour_index",
                    "curr_timestamp",
                    "curr_time",
                    "bid",
                    "spend_history",
                    "clicks_history",
                    "balance",
                    "clicks",
                    "end_balance_share",
                ]
            ]
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "campaign_id",
                "region_id",
                "logical_category",
                "auction_budget",
                "hour_index",
                "curr_timestamp",
                "curr_time",
                "bid",
                "spend_history",
                "clicks_history",
                "balance",
                "clicks",
                "end_balance_share",
            ]
        )
    return pd.concat(rows, axis=0, ignore_index=True)


def cache_locked_linear_baseline(
    *,
    context: dict[str, Any],
    run_name: str = "locked_linear_reference_v1",
) -> dict[str, Any]:
    stage_key = "00_locked_linear_baseline"
    run_dir = ensure_run_dir(stage_key, run_name)
    linear_params = dict(context["locked_reference"]["linear_params"])
    metrics_by_split: dict[str, Any] = {}

    for split_name in ("val", "test_holdout"):
        split = context["splits"][split_name]
        metrics, campaign_df, hourly_df = _evaluate_linear_with_rollouts(
            split=split,
            auction_mode=context["auction_mode"],
            linear_params=linear_params,
        )
        metrics_by_split[split_name] = metrics
        campaign_df.to_csv(run_dir / f"{split_name}_campaign_linear.csv", index=False)
        hourly_df.to_csv(run_dir / f"{split_name}_campaign_hourly_linear.csv", index=False)

    manifest = {
        "locked_reference": context["locked_reference"],
        "linear_params": linear_params,
        "metrics_by_split": metrics_by_split,
    }
    write_json(run_dir / "baseline_manifest.json", manifest)
    return {"run_dir": run_dir, "manifest": manifest}


def _total_candidate_steps(campaigns_df: pd.DataFrame, bidder: DRLBBidder) -> int:
    return int(
        campaigns_df.apply(
            lambda row: bidder._campaign_total_steps(row["campaign_start"], row["campaign_end"]),
            axis=1,
        ).sum()
    )


def _window_action_entropy(action_series: pd.Series) -> float:
    counts = action_series.astype(int).value_counts(normalize=True)
    if counts.empty:
        return 0.0
    return float(-(counts * counts.map(math.log)).sum())


def run_train_with_validation_checkpoints(
    *,
    bidder_params: dict[str, Any],
    train_stats_df: pd.DataFrame,
    train_campaigns_df: pd.DataFrame,
    val_split: dict[str, str],
    objective: str,
    auction_mode: str,
    epochs: int,
    max_steps: int | None,
    record_every: int,
    smoothing_window: int,
) -> tuple[DRLBBidder, pd.DataFrame, pd.DataFrame]:
    bidder = DRLBBidder(dict(bidder_params))
    stats = train_stats_df.sort_values(["campaign_id", "period"]).reset_index(drop=True)
    campaigns = train_campaigns_df.sort_values("campaign_id").reset_index(drop=True)

    bidder.train_prior_lambda_init = float(bidder.agent.ctl_lambda)

    candidate_steps = _total_candidate_steps(campaigns, bidder)
    total_steps = candidate_steps * max(1, int(epochs))
    if max_steps is not None:
        total_steps = min(total_steps, int(max_steps))

    fit_steps = 0
    checkpoints: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="drlb_autoresearch_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        for epoch_idx in range(max(1, int(epochs))):
            if fit_steps >= total_steps:
                break
            for _, campaign_row in campaigns.iterrows():
                if fit_steps >= total_steps:
                    break

                campaign_id = int(campaign_row["campaign_id"])
                campaign_stats = stats[stats["campaign_id"] == campaign_id].copy()
                if campaign_stats.empty:
                    continue

                env = BatStepEnv(
                    stats_pdf=campaign_stats,
                    campaign_row=campaign_row,
                    auction_mode=auction_mode,
                )
                campaign_budget = max(1.0, float(campaign_row["auction_budget"]))
                campaign_start = int(campaign_row["campaign_start"])
                campaign_end = int(campaign_row["campaign_end"])
                campaign_region = campaign_row.get("region_id")

                bidder._init_agent_episode(
                    initial_balance=campaign_budget,
                    total_steps=bidder._campaign_total_steps(campaign_start, campaign_end),
                    start_time=campaign_start,
                    end_time=campaign_end,
                    curr_time=env.campaign.curr_time,
                    region_id=campaign_region,
                    lambda_init=bidder.train_prior_lambda_init,
                )

                while (not env.done()) and fit_steps < total_steps:
                    # observe: build BAT observation from the current campaign-hour.
                    step_input = env.get_step_input()
                    obs = bidder._build_agent_obs(
                        time_step_index=max(0, (step_input.period_start_ts - campaign_start) // 3600),
                        ctr_pred=step_input.ctr_pred,
                        balance=step_input.balance,
                        initial_balance=campaign_budget,
                        start_time=campaign_start,
                        end_time=campaign_end,
                        curr_time=step_input.period_start_ts,
                        region_id=campaign_region,
                    )

                    # act: pick a DQN action under the current epsilon schedule.
                    bidder.agent.state_repr.sync_runtime_context(
                        balance=obs["balance"],
                        initial_budget=obs["initialBalance"],
                        elapsed_time_ratio=obs.get("elapsedTimeRatio"),
                        initial_budget_scale=obs.get("initialBudgetScale"),
                        traffic_share=obs.get("trafficShare"),
                    )
                    state_before_action = bidder.agent.state_repr.curr_state.copy()
                    action_idx = bidder.agent.dqn_agent.act(
                        state_before_action,
                        eps=bidder.agent.eps,
                        eval_mode=False,
                    )
                    action_beta = bidder.agent.BETA[action_idx]

                    # step env: execute the simulator step with the clipped BAT bid.
                    raw_bid = bidder.agent.calc_bid(
                        obs["ctr"],
                        action_beta,
                        available_budget=step_input.balance,
                    )
                    bid = bidder._clip_bid_to_budget(
                        raw_bid,
                        prev_bid=env.campaign.prev_bid,
                        balance=step_input.balance,
                    )
                    outcome = env.step(bid)
                    immediate_reward = bidder._resolve_outcome_reward(outcome, objective)
                    spend = max(0.0, outcome.spent)
                    done = env.done()

                    state_after_outcome = bidder.agent.state_repr.update_state(
                        immediate_reward=immediate_reward,
                        spend=spend,
                        win=bool(spend > 0.0),
                    )

                    # learn DQN: use the RewardNet-predicted reward for the Q-update.
                    rnet_reward = bidder.agent.predict_reward(state_before_action, action_beta)
                    bidder.agent.learn_dqn_transition(
                        state_before_action,
                        action_idx,
                        rnet_reward,
                        state_after_outcome,
                        done=done,
                    )

                    # update RewardNet: train on the realized immediate reward target.
                    bidder.agent.record_reward_net_step(state_before_action, action_beta, immediate_reward)

                    fit_steps += 1
                    if (fit_steps % max(1, int(record_every)) == 0) or (fit_steps == total_steps):
                        model_path = tmpdir_path / f"step_{fit_steps:06d}.pt"
                        bidder.save_model(str(model_path))
                        val_metrics = evaluate_drlb_checkpoint(
                            model_path=model_path,
                            bidder_params=bidder_params,
                            split=val_split,
                            auction_mode=auction_mode,
                        )
                        diagnostics_df = add_smoothed_diagnostics(
                            bidder.get_training_diagnostics().copy(),
                            smoothing_window=smoothing_window,
                        )
                        window = diagnostics_df.tail(max(1, int(record_every)))
                        last_row = diagnostics_df.iloc[-1]
                        checkpoints.append(
                            {
                                "step": int(fit_steps),
                                "epoch": int(epoch_idx + 1),
                                "campaign_id": campaign_id,
                                "val_clicks_sum": float(val_metrics["clicks_sum"]),
                                "val_rmse": float(val_metrics["rmse"]),
                                "val_cpc_relative": float(val_metrics["cpc_relative"]),
                                "val_quickspend": float(val_metrics["quickspend"]),
                                "average_end_balance_share": float(
                                    val_metrics["average_end_balance_share"]
                                ),
                                "lambda_last": float(last_row["lambda"]),
                                "eps_last": float(last_row["eps"]),
                                "eps_expected_last": float(
                                    bidder.agent.epsilon_at_step(int(last_row["global_t"]))
                                ),
                                "dqn_loss_last": float(last_row["dqn_loss"]),
                                "reward_net_loss_last": float(last_row["reward_net_loss"]),
                                "dqn_loss_smooth_last": float(
                                    last_row.get("dqn_loss_smooth", last_row["dqn_loss"])
                                ),
                                "reward_net_loss_smooth_last": float(
                                    last_row.get("reward_net_loss_smooth", last_row["reward_net_loss"])
                                ),
                                "edge_action_share_window": float(
                                    window["dqn_action"].isin([0, len(bidder.agent.BETA) - 1]).mean()
                                ),
                                "action_entropy_window": _window_action_entropy(window["dqn_action"]),
                            }
                        )

                bidder.agent.finish_episode()

    diagnostics_df = add_smoothed_diagnostics(
        bidder.get_training_diagnostics().copy(),
        smoothing_window=smoothing_window,
    )
    diagnostics_df["eps_expected"] = diagnostics_df["global_t"].apply(bidder.agent.epsilon_at_step)
    checkpoints_df = pd.DataFrame(checkpoints)
    return bidder, diagnostics_df, checkpoints_df


def save_training_artifacts(
    *,
    run_dir: Path,
    diagnostics_df: pd.DataFrame,
    checkpoints_df: pd.DataFrame,
) -> None:
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    diagnostics_df.to_csv(run_dir / "diagnostics.csv", index=False)
    checkpoints_df.to_csv(run_dir / "checkpoints.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=130)
    if not diagnostics_df.empty:
        axes[0, 0].plot(diagnostics_df["global_t"], diagnostics_df["dqn_loss"], color="#1f77b4", alpha=0.18)
        axes[0, 0].plot(
            diagnostics_df["global_t"],
            diagnostics_df["dqn_loss_smooth"],
            color="#1f77b4",
            linewidth=1.8,
        )
        axes[0, 0].set_title("DQN Loss")

        axes[0, 1].plot(
            diagnostics_df["global_t"],
            diagnostics_df["reward_net_loss"],
            color="#2ca02c",
            alpha=0.18,
        )
        axes[0, 1].plot(
            diagnostics_df["global_t"],
            diagnostics_df["reward_net_loss_smooth"],
            color="#2ca02c",
            linewidth=1.8,
        )
        axes[0, 1].set_title("RewardNet Loss")

    if not checkpoints_df.empty:
        axes[1, 0].plot(checkpoints_df["step"], checkpoints_df["val_clicks_sum"], color="#d62728", marker="o")
        axes[1, 0].set_title("Validation Clicks")
        axes[1, 1].plot(checkpoints_df["step"], checkpoints_df["eps_last"], color="#9467bd", marker="o")
        axes[1, 1].set_title("Epsilon at Checkpoints")

    for axis in axes.ravel():
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(run_dir / "figures" / "training_curves.png")
    plt.close(fig)


def run_pair_comparison(
    *,
    candidate_dir: Path,
    baseline_cache_dir: Path,
    split_name: str,
    split: dict[str, str],
    auction_mode: str,
    drlb_params: dict[str, Any],
    drlb_model_path: Path,
) -> dict[str, Path]:
    linear_campaign_path = baseline_cache_dir / f"{split_name}_campaign_linear.csv"
    linear_hourly_path = baseline_cache_dir / f"{split_name}_campaign_hourly_linear.csv"
    linear_df = pd.read_csv(linear_campaign_path)
    linear_hourly_df = pd.read_csv(linear_hourly_path)

    drlb_eval_params = dict(drlb_params)
    drlb_eval_params.update({"model_path": str(drlb_model_path), "eval_mode": True})
    drlb_df = campaign_rollout_table(
        bidder_cls=DRLBBidder,
        bidder_params=drlb_eval_params,
        split=split,
        auction_mode=auction_mode,
    )
    drlb_hourly_df = campaign_hourly_rollout_table(
        bidder_cls=DRLBBidder,
        bidder_params=drlb_eval_params,
        split=split,
        auction_mode=auction_mode,
    )

    campaign_cmp = linear_df.merge(
        drlb_df,
        on="campaign_id",
        suffixes=("_linear", "_drlb"),
    )
    campaign_cmp["clicks_delta_drlb_minus_linear"] = (
        campaign_cmp["clicks_total_drlb"] - campaign_cmp["clicks_total_linear"]
    )
    campaign_cmp["spend_delta_drlb_minus_linear"] = (
        campaign_cmp["spend_total_drlb"] - campaign_cmp["spend_total_linear"]
    )
    campaign_cmp["avg_bid_delta_drlb_minus_linear"] = (
        campaign_cmp["avg_bid_drlb"] - campaign_cmp["avg_bid_linear"]
    )
    campaign_cmp["end_balance_share_delta_drlb_minus_linear"] = (
        campaign_cmp["end_balance_share_drlb"] - campaign_cmp["end_balance_share_linear"]
    )
    campaign_cmp["underbids_vs_linear"] = campaign_cmp["avg_bid_drlb"] < campaign_cmp["avg_bid_linear"]
    campaign_cmp["overbids_vs_linear"] = campaign_cmp["avg_bid_drlb"] > campaign_cmp["avg_bid_linear"]

    hourly_cmp = linear_hourly_df.merge(
        drlb_hourly_df,
        on=["campaign_id", "curr_timestamp"],
        how="outer",
        suffixes=("_linear", "_drlb"),
    ).sort_values(["campaign_id", "curr_timestamp"])
    hourly_cmp["bid_delta_drlb_minus_linear"] = hourly_cmp["bid_drlb"] - hourly_cmp["bid_linear"]
    hourly_cmp["spend_delta_drlb_minus_linear"] = (
        hourly_cmp["spend_history_drlb"] - hourly_cmp["spend_history_linear"]
    )
    hourly_cmp["clicks_delta_drlb_minus_linear"] = (
        hourly_cmp["clicks_history_drlb"] - hourly_cmp["clicks_history_linear"]
    )
    hourly_cmp["end_balance_share_delta_drlb_minus_linear"] = (
        hourly_cmp["end_balance_share_drlb"] - hourly_cmp["end_balance_share_linear"]
    )

    comparison_dir = candidate_dir / "pair_comparison" / split_name
    comparison_dir.mkdir(parents=True, exist_ok=True)
    campaign_path = comparison_dir / "campaign_comparison.csv"
    hourly_path = comparison_dir / "campaign_hourly_comparison.csv"
    campaign_cmp.sort_values("clicks_delta_drlb_minus_linear", ascending=True).to_csv(campaign_path, index=False)
    hourly_cmp.to_csv(hourly_path, index=False)

    slices = {
        "worst_click_delta.csv": campaign_cmp.nsmallest(25, "clicks_delta_drlb_minus_linear"),
        "best_click_delta.csv": campaign_cmp.nlargest(25, "clicks_delta_drlb_minus_linear"),
        "high_budget.csv": campaign_cmp.nlargest(25, "auction_budget_linear"),
        "high_leftover_budget.csv": campaign_cmp.nlargest(25, "end_balance_share_drlb"),
        "large_overspend_delta.csv": campaign_cmp.nsmallest(25, "end_balance_share_delta_drlb_minus_linear"),
    }
    for filename, frame in slices.items():
        frame.to_csv(comparison_dir / filename, index=False)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=130)
    ax.scatter(campaign_cmp["clicks_total_linear"], campaign_cmp["clicks_total_drlb"], s=20, alpha=0.7)
    max_clicks = max(
        1.0,
        float(max(campaign_cmp["clicks_total_linear"].max(), campaign_cmp["clicks_total_drlb"].max())),
    )
    ax.plot([0, max_clicks], [0, max_clicks], color="#333333", linewidth=1.0, linestyle="--")
    ax.set_title(f"Campaign Clicks: Linear vs DRLB ({split_name})")
    ax.set_xlabel("Linear clicks")
    ax.set_ylabel("DRLB clicks")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(comparison_dir / "campaign_clicks_scatter.png")
    plt.close(fig)

    return {
        "campaign_comparison_path": campaign_path,
        "campaign_hourly_comparison_path": hourly_path,
    }


def run_fixed_candidate(
    *,
    stage_key: str,
    run_name: str,
    context: dict[str, Any],
    candidate_name: str,
    bidder_params: dict[str, Any],
    hypothesis: str,
    epochs: int,
    max_steps: int | None,
    record_every: int,
    smoothing_window: int,
    baseline_cache_dir: Path,
    evaluate_holdout: bool,
) -> dict[str, Any]:
    run_root = ensure_run_dir(stage_key, run_name)
    candidate_dir = run_root / candidate_name
    (candidate_dir / "figures").mkdir(parents=True, exist_ok=True)
    write_json(candidate_dir / "params.json", bidder_params)

    bidder, diagnostics_df, checkpoints_df = run_train_with_validation_checkpoints(
        bidder_params=bidder_params,
        train_stats_df=context["train_stats_df"],
        train_campaigns_df=context["train_campaigns_df"],
        val_split=context["splits"]["val"],
        objective=context["objective"],
        auction_mode=context["auction_mode"],
        epochs=epochs,
        max_steps=max_steps,
        record_every=record_every,
        smoothing_window=smoothing_window,
    )
    save_training_artifacts(
        run_dir=candidate_dir,
        diagnostics_df=diagnostics_df,
        checkpoints_df=checkpoints_df,
    )

    final_model_path = candidate_dir / "final_model.pt"
    bidder.save_model(str(final_model_path))
    val_final = evaluate_drlb_checkpoint(
        model_path=final_model_path,
        bidder_params=bidder_params,
        split=context["splits"]["val"],
        auction_mode=context["auction_mode"],
    )
    holdout_final = None
    if evaluate_holdout:
        holdout_final = evaluate_drlb_checkpoint(
            model_path=final_model_path,
            bidder_params=bidder_params,
            split=context["splits"]["test_holdout"],
            auction_mode=context["auction_mode"],
        )

    pair_paths: dict[str, dict[str, str]] = {}
    pair_paths["val"] = {
        key: str(value)
        for key, value in run_pair_comparison(
            candidate_dir=candidate_dir,
            baseline_cache_dir=baseline_cache_dir,
            split_name="val",
            split=context["splits"]["val"],
            auction_mode=context["auction_mode"],
            drlb_params=bidder_params,
            drlb_model_path=final_model_path,
        ).items()
    }
    if evaluate_holdout:
        pair_paths["test_holdout"] = {
            key: str(value)
            for key, value in run_pair_comparison(
                candidate_dir=candidate_dir,
                baseline_cache_dir=baseline_cache_dir,
                split_name="test_holdout",
                split=context["splits"]["test_holdout"],
                auction_mode=context["auction_mode"],
                drlb_params=bidder_params,
                drlb_model_path=final_model_path,
            ).items()
        }
    metrics_payload = {
        "val_final": val_final,
        "holdout_final": holdout_final,
        "best_val_clicks_sum": (
            None if checkpoints_df.empty else float(checkpoints_df["val_clicks_sum"].max())
        ),
        "best_checkpoint_step": (
            None
            if checkpoints_df.empty
            else int(checkpoints_df.loc[checkpoints_df["val_clicks_sum"].idxmax(), "step"])
        ),
        "pair_comparison": pair_paths,
    }
    write_json(candidate_dir / "metrics.json", metrics_payload)

    append_autoresearch_entry(
        heading=f"{stage_key} / {candidate_name}",
        stage_key=stage_key,
        run_name=run_name,
        hypothesis=hypothesis,
        params=bidder_params,
        result_summary=metrics_payload,
        decision="promote" if holdout_final is not None else "keep",
        notes=(
            "Validation and holdout pair-comparison saved against locked linear baseline."
            if evaluate_holdout
            else "Validation pair-comparison saved against locked linear baseline."
        ),
    )

    return {
        "candidate": candidate_name,
        "candidate_dir": candidate_dir,
        "params": bidder_params,
        "model_path": final_model_path,
        "val_final": val_final,
        "holdout_final": holdout_final,
        "best_val_clicks_sum": metrics_payload["best_val_clicks_sum"],
        "best_checkpoint_step": metrics_payload["best_checkpoint_step"],
    }


def run_candidate_grid(
    *,
    stage_key: str,
    run_name: str,
    context: dict[str, Any],
    candidates: Iterable[dict[str, Any]],
    baseline_cache_dir: Path,
    epochs: int,
    max_steps: int | None,
    record_every: int,
    smoothing_window: int,
    evaluate_holdout: bool = False,
) -> tuple[pd.DataFrame, Path]:
    rows = []
    run_root = ensure_run_dir(stage_key, run_name)
    for candidate in candidates:
        result = run_fixed_candidate(
            stage_key=stage_key,
            run_name=run_name,
            context=context,
            candidate_name=str(candidate["name"]),
            bidder_params={
                **context["base_bidder_params"],
                **candidate["overrides"],
            },
            hypothesis=str(candidate.get("hypothesis", candidate["name"])),
            epochs=int(candidate.get("epochs", epochs)),
            max_steps=max_steps,
            record_every=record_every,
            smoothing_window=smoothing_window,
            baseline_cache_dir=baseline_cache_dir,
            evaluate_holdout=evaluate_holdout,
        )
        rows.append(
            {
                "candidate": result["candidate"],
                "best_val_clicks_sum": result["best_val_clicks_sum"],
                "best_checkpoint_step": result["best_checkpoint_step"],
                "val_final_clicks_sum": result["val_final"]["clicks_sum"],
                "holdout_final_clicks_sum": (
                    None if result["holdout_final"] is None else result["holdout_final"]["clicks_sum"]
                ),
                "average_end_balance_share_val": result["val_final"]["average_end_balance_share"],
            }
        )
    leaderboard_df = pd.DataFrame(rows).sort_values("best_val_clicks_sum", ascending=False)
    leaderboard_df.to_csv(run_root / "leaderboard.csv", index=False)
    return leaderboard_df, run_root


def run_optuna_search(
    *,
    stage_key: str,
    run_name: str,
    context: dict[str, Any],
    search_space_fn: Callable[[optuna.trial.Trial], dict[str, Any]],
    hypothesis_prefix: str,
    n_trials: int,
    baseline_cache_dir: Path,
    epochs: int,
    max_steps: int | None,
    record_every: int,
    smoothing_window: int,
    holdout_top_k: int = 1,
) -> tuple[pd.DataFrame, Path, dict[str, Any]]:
    trial_results: list[dict[str, Any]] = []

    def objective(trial: optuna.trial.Trial) -> float:
        overrides = search_space_fn(trial)
        candidate_name = f"trial_{trial.number:03d}"
        result = run_fixed_candidate(
            stage_key=stage_key,
            run_name=run_name,
            context=context,
            candidate_name=candidate_name,
            bidder_params={**context["base_bidder_params"], **overrides},
            hypothesis=f"{hypothesis_prefix} trial={trial.number}",
            epochs=epochs,
            max_steps=max_steps,
            record_every=record_every,
            smoothing_window=smoothing_window,
            baseline_cache_dir=baseline_cache_dir,
            evaluate_holdout=False,
        )
        trial_results.append(result)
        trial.set_user_attr("best_val_clicks_sum", result["best_val_clicks_sum"])
        trial.set_user_attr("val_final_clicks_sum", result["val_final"]["clicks_sum"])
        trial.set_user_attr(
            "average_end_balance_share_val",
            result["val_final"]["average_end_balance_share"],
        )
        return float(result["best_val_clicks_sum"] or result["val_final"]["clicks_sum"])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=max(1, int(n_trials)), n_jobs=1, show_progress_bar=False)

    run_root = ensure_run_dir(stage_key, run_name)
    leaderboard_rows = []
    sorted_results = sorted(
        trial_results,
        key=lambda row: float(row["best_val_clicks_sum"] or row["val_final"]["clicks_sum"]),
        reverse=True,
    )
    top_for_holdout = {row["candidate"] for row in sorted_results[: max(1, int(holdout_top_k))]}
    for row in sorted_results:
        holdout_final = None
        if row["candidate"] in top_for_holdout:
            holdout_final = evaluate_drlb_checkpoint(
                model_path=row["model_path"],
                bidder_params=row["params"],
                split=context["splits"]["test_holdout"],
                auction_mode=context["auction_mode"],
            )
            metrics_path = row["candidate_dir"] / "metrics.json"
            metrics_payload = read_json(metrics_path)
            metrics_payload["holdout_final"] = holdout_final
            write_json(metrics_path, metrics_payload)
            append_autoresearch_entry(
                heading=f"{stage_key} / {row['candidate']} / holdout",
                stage_key=stage_key,
                run_name=run_name,
                hypothesis=f"{hypothesis_prefix} holdout promotion",
                params=row["params"],
                result_summary={"holdout_final": holdout_final},
                decision="promote",
                notes="Top validation run promoted to holdout evaluation.",
            )

        leaderboard_rows.append(
            {
                "candidate": row["candidate"],
                "best_val_clicks_sum": row["best_val_clicks_sum"],
                "best_checkpoint_step": row["best_checkpoint_step"],
                "val_final_clicks_sum": row["val_final"]["clicks_sum"],
                "holdout_final_clicks_sum": None if holdout_final is None else holdout_final["clicks_sum"],
                "average_end_balance_share_val": row["val_final"]["average_end_balance_share"],
            }
        )

    leaderboard_df = pd.DataFrame(leaderboard_rows).sort_values("best_val_clicks_sum", ascending=False)
    leaderboard_df.to_csv(run_root / "leaderboard.csv", index=False)
    write_json(
        run_root / "study_summary.json",
        {
            "best_trial_number": int(study.best_trial.number),
            "best_value": float(study.best_value),
            "best_params": study.best_trial.params,
            "n_trials": int(n_trials),
        },
    )
    return leaderboard_df, run_root, study.best_trial.params


def build_bid_lr_search_space(trial: optuna.trial.Trial) -> dict[str, Any]:
    min_bid = trial.suggest_float("min_bid", 0.0, 5.0)
    max_bid = trial.suggest_float("max_bid", 40.0, 120.0)
    if max_bid < min_bid:
        min_bid, max_bid = max_bid, min_bid
    return {
        "min_bid": min_bid,
        "max_bid": max_bid,
        "bid_lower_clip": trial.suggest_int("bid_lower_clip", 1, 18),
        "bid_upper_clip": trial.suggest_int("bid_upper_clip", 1, 18),
        "dqn_lr": trial.suggest_float("dqn_lr", 1e-5, 5e-3, log=True),
        "reward_net_lr": trial.suggest_float("reward_net_lr", 1e-5, 5e-2, log=True),
    }


def build_epsilon_search_space(trial: optuna.trial.Trial) -> dict[str, Any]:
    epsilon_start = trial.suggest_float("dqn_epsilon_start", 0.55, 0.98)
    epsilon_end = trial.suggest_float("dqn_epsilon_end", 0.01, 0.10)
    if epsilon_end > epsilon_start:
        epsilon_end = epsilon_start
    return {
        "dqn_epsilon_start": epsilon_start,
        "dqn_epsilon_end": epsilon_end,
        "dqn_epsilon_anneal": trial.suggest_float("dqn_epsilon_anneal", 1e-6, 2e-4, log=True),
    }


def build_manual_epsilon_candidates() -> list[dict[str, Any]]:
    return [
        {
            "name": "eps_slow_high",
            "hypothesis": "Slow decay keeps exploration alive longer without collapsing to edge actions early.",
            "overrides": {
                "dqn_epsilon_start": 0.95,
                "dqn_epsilon_end": 0.05,
                "dqn_epsilon_anneal": 2e-5,
            },
        },
        {
            "name": "eps_medium_balanced",
            "hypothesis": "Medium decay may reduce variance while preserving enough state coverage.",
            "overrides": {
                "dqn_epsilon_start": 0.80,
                "dqn_epsilon_end": 0.05,
                "dqn_epsilon_anneal": 6e-5,
            },
        },
        {
            "name": "eps_fast_low",
            "hypothesis": "Fast decay checks whether the policy is failing because exploration lasts too long.",
            "overrides": {
                "dqn_epsilon_start": 0.65,
                "dqn_epsilon_end": 0.02,
                "dqn_epsilon_anneal": 1e-4,
            },
        },
    ]


def build_epoch_candidates(base_overrides: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "name": "epoch_1",
            "hypothesis": "One full dataset pass is the control.",
            "epochs": 1,
            "overrides": dict(base_overrides),
        },
        {
            "name": "epoch_2",
            "hypothesis": "Two dataset passes test whether DRLB is data-starved rather than architecture-limited.",
            "epochs": 2,
            "overrides": dict(base_overrides),
        },
        {
            "name": "epoch_3",
            "hypothesis": "Three dataset passes test diminishing returns of repeated simulator replay.",
            "epochs": 3,
            "overrides": dict(base_overrides),
        },
    ]

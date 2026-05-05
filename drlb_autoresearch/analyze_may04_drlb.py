from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from example_notebooks.experiments.adapters.baseline_adapter import evaluate_baseline_model_inprocess
from example_notebooks.experiments.drlb.profiles import build_config as build_drlb_config
from example_notebooks.experiments.infra.artifacts import json_ready
from example_notebooks.experiments.infra.split_utils import resolve_normalized_splits


REPORT_DIR = REPO_ROOT / "drlb_autoresearch" / "may04_report"
LINEAR_PARAMS_PATH = (
    REPO_ROOT
    / "example_notebooks"
    / "evaluate_baselines"
    / "best_params"
    / "fpa_baseline_n10_rndm_42"
    / "linear_scr_FPA.pkl"
)

DRLB_RUNS = {
    "default": "may04_default_state_optuna10",
    "ratio_bat": "may04_ratio_bat_state_optuna10",
    "ta_ratio_bat": "may04_ta_ratio_bat_state_optuna10",
    "default_lambda_rule": "may04_default_state_lambda_rule_optuna1",
}


def main() -> None:
    args = parse_args()
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    config_ref = build_drlb_config(
        run_name="may04_compare_states_vs_linear",
        profile="may04_default_linear_lambda_legacy",
        split_set="full_train_val_holdout",
    )
    normalized_splits = resolve_normalized_splits(config_ref)

    summaries = load_drlb_summaries(config_ref.family_dir)
    linear_metrics = load_or_eval_linear_metrics(
        report_dir=report_dir,
        normalized_splits=normalized_splits,
        refresh=args.refresh_linear,
    )

    aggregate_metrics_df = build_aggregate_metrics(summaries, linear_metrics)
    diagnostics_segments_df = build_diagnostics_segments(summaries)
    campaign_stats_df = build_campaign_stats(normalized_splits)
    top_campaigns_df = select_campaign_slices(campaign_stats_df)
    hourly_stats_df = build_hourly_stats_placeholder(top_campaigns_df)

    summary_payload = build_summary_payload(
        aggregate_metrics_df=aggregate_metrics_df,
        diagnostics_segments_df=diagnostics_segments_df,
        top_campaigns_df=top_campaigns_df,
        summaries=summaries,
        linear_metrics=linear_metrics,
    )

    write_csv(report_dir, aggregate_metrics_df, "aggregate_metrics.csv")
    write_csv(report_dir, diagnostics_segments_df, "diagnostics_segments.csv")
    write_csv(report_dir, campaign_stats_df, "campaign_stats.csv")
    write_csv(report_dir, hourly_stats_df, "hourly_stats.csv")
    write_csv(report_dir, top_campaigns_df, "top_divergent_campaigns.csv")
    (report_dir / "summary.json").write_text(json.dumps(json_ready(summary_payload), indent=2))

    plot_diagnostics_segments(report_dir, diagnostics_segments_df)
    report = render_report(
        aggregate_metrics_df=aggregate_metrics_df,
        diagnostics_segments_df=diagnostics_segments_df,
        top_campaigns_df=top_campaigns_df,
        summary_payload=summary_payload,
    )
    (report_dir / "report.md").write_text(report)

    cross_check_drlb_holdout_metrics(summaries, aggregate_metrics_df, report_dir)
    print(f"Wrote May 04 DRLB analysis artifacts to {report_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build May 04 DRLB diagnostics report.")
    parser.add_argument("--report-dir", default=str(REPORT_DIR))
    parser.add_argument(
        "--refresh-linear",
        action="store_true",
        help="Re-evaluate locked LinearBidder val/holdout instead of reusing cached linear_metrics.json.",
    )
    return parser.parse_args()


def load_drlb_summaries(family_dir: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    for run_key, run_name in DRLB_RUNS.items():
        run_dir = family_dir / run_name
        rows[run_key] = {
            "run_key": run_key,
            "run_name": run_name,
            "run_dir": run_dir,
            "summary": json.loads((run_dir / "outputs" / "run_summary.json").read_text()),
            "metrics": json.loads((run_dir / "outputs" / "metrics.json").read_text()),
            "diagnostics_path": run_dir / "outputs" / "best_refit_training_diagnostics.csv",
            "diagnostics_png": run_dir / "outputs" / "drlb_diagnostics.png",
        }
    return rows


def load_or_eval_linear_metrics(
    *,
    report_dir: Path,
    normalized_splits: dict[str, dict[str, str]],
    refresh: bool,
) -> dict[str, Any]:
    cache_path = report_dir / "linear_metrics.json"
    if cache_path.exists() and not refresh:
        return json.loads(cache_path.read_text())

    with LINEAR_PARAMS_PATH.open("rb") as f:
        linear_params = pickle.load(f)

    linear_val = evaluate_baseline_model_inprocess(
        model_name="linear",
        label="linear_val",
        params_dict=linear_params,
        split=normalized_splits["val"],
        auction_mode="FPA",
    )
    linear_holdout = evaluate_baseline_model_inprocess(
        model_name="linear",
        label="linear_holdout",
        params_dict=linear_params,
        split=normalized_splits["test_holdout"],
        auction_mode="FPA",
    )
    payload = {
        "params_path": str(LINEAR_PARAMS_PATH),
        "params": linear_params,
        "val": linear_val["metrics"],
        "holdout": linear_holdout["metrics"],
    }
    cache_path.write_text(json.dumps(json_ready(payload), indent=2))
    return payload


def build_aggregate_metrics(
    summaries: dict[str, dict[str, Any]],
    linear_metrics: dict[str, Any],
) -> pd.DataFrame:
    rows = [
        {
            "model": "linear",
            "run_key": "linear",
            "run_name": "fpa_baseline_n10_rndm_42",
            "state": "baseline",
            "split": "val",
            **metric_subset(linear_metrics["val"]),
        },
        {
            "model": "linear",
            "run_key": "linear",
            "run_name": "fpa_baseline_n10_rndm_42",
            "state": "baseline",
            "split": "holdout",
            **metric_subset(linear_metrics["holdout"]),
        },
    ]
    for run_key, payload in summaries.items():
        summary = payload["summary"]
        rows.append(
            {
                "model": f"drlb_{run_key}",
                "run_key": run_key,
                "run_name": payload["run_name"],
                "state": run_key,
                "split": "val",
                **metric_subset(summary["tuning"]["best_val_metrics"]),
                "best_trial_number": summary["tuning"]["best_trial_number"],
            }
        )
        rows.append(
            {
                "model": f"drlb_{run_key}",
                "run_key": run_key,
                "run_name": payload["run_name"],
                "state": run_key,
                "split": "holdout",
                **metric_subset(summary["final_holdout"]["metrics"]),
                "best_trial_number": summary["tuning"]["best_trial_number"],
            }
        )

    df = pd.DataFrame(rows)
    linear_ref = df[df["run_key"] == "linear"][
        ["split", "clicks_sum", "cpc_relative", "rmse", "quickspend"]
    ].rename(
        columns={
            "clicks_sum": "linear_clicks_sum",
            "cpc_relative": "linear_cpc_relative",
            "rmse": "linear_rmse",
            "quickspend": "linear_quickspend",
        }
    )
    df = df.merge(linear_ref, on="split")
    for metric in ("clicks_sum", "cpc_relative", "rmse", "quickspend"):
        df[f"{metric}_delta_vs_linear"] = df[metric] - df[f"linear_{metric}"]
    return df.sort_values(["split", "clicks_sum"], ascending=[True, False]).reset_index(drop=True)


def metric_subset(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "clicks_sum": metrics["clicks_sum"],
        "cpc_relative": metrics["cpc_relative"],
        "rmse": metrics["rmse"],
        "quickspend": metrics["quickspend"],
        "average_end_balance_share": metrics.get("average_end_balance_share"),
        "train_steps": metrics.get("train_steps"),
        "last_dqn_loss": metrics.get("last_dqn_loss"),
        "last_reward_net_loss": metrics.get("last_reward_net_loss"),
        "dqn_loss_mean": metrics.get("dqn_loss_mean"),
        "reward_net_loss_mean": metrics.get("reward_net_loss_mean"),
        "lambda_final": metrics.get("lambda_final"),
    }


def build_diagnostics_segments(summaries: dict[str, dict[str, Any]]) -> pd.DataFrame:
    edges = [0, 5000, 10000, 15000, 20000, 25000, 30000, 40000, 50000, 60000]
    rows = []
    for run_key, payload in summaries.items():
        df = pd.read_csv(payload["diagnostics_path"])
        for left, right in zip(edges[:-1], edges[1:]):
            segment = df[(df["global_t"] > left) & (df["global_t"] <= right)]
            if segment.empty:
                continue
            dqn_loss = segment.loc[segment["dqn_loss"] > 0, "dqn_loss"]
            reward_net_loss = segment.loc[segment["reward_net_loss"] > 0, "reward_net_loss"]
            action_counts = segment["dqn_action"].value_counts(normalize=True).sort_index()
            row = {
                "run_key": run_key,
                "segment": f"({left},{right}]",
                "step_start": left + 1,
                "step_end": right,
                "rows": len(segment),
                "dqn_loss_mean": dqn_loss.mean(),
                "dqn_loss_p95": dqn_loss.quantile(0.95),
                "reward_net_loss_mean": reward_net_loss.mean(),
                "reward_net_loss_p95": reward_net_loss.quantile(0.95),
                "reward_signal_mean": segment["reward_signal"].mean(),
                "reward_signal_p95": segment["reward_signal"].quantile(0.95),
                "lambda_start": segment["lambda"].iloc[0],
                "lambda_end": segment["lambda"].iloc[-1],
                "lambda_delta": segment["lambda"].iloc[-1] - segment["lambda"].iloc[0],
                "eps_start": segment["eps"].iloc[0],
                "eps_end": segment["eps"].iloc[-1],
            }
            for action, share in action_counts.items():
                row[f"action_{int(action)}_share"] = share
            rows.append(row)
    return pd.DataFrame(rows)


def build_campaign_stats(normalized_splits: dict[str, dict[str, str]]) -> pd.DataFrame:
    rows = []
    for split, split_key in (("train", "train"), ("val", "val"), ("holdout", "test_holdout")):
        campaigns = pd.read_csv(normalized_splits[split_key]["campaigns_path"])
        campaigns = campaigns[
            [
                "campaign_id",
                "campaign_start",
                "campaign_end",
                "campaign_start_date",
                "campaign_end_date",
                "auction_budget",
                "logical_category",
                "region_id",
            ]
        ].copy()
        campaigns["split"] = split
        campaigns["duration_hours"] = (campaigns["campaign_end"] - campaigns["campaign_start"]) / 3600
        rows.append(campaigns)
    return pd.concat(rows, ignore_index=True)


def select_campaign_slices(campaign_stats_df: pd.DataFrame) -> pd.DataFrame:
    slices = []
    earliest = campaign_stats_df.sort_values("campaign_start").head(16).copy()
    earliest["selection_reason"] = "earliest_campaign_start"
    slices.append(earliest)

    holdout = campaign_stats_df[campaign_stats_df["split"] == "holdout"].copy()
    top_budget = holdout.sort_values("auction_budget", ascending=False).head(16).copy()
    top_budget["selection_reason"] = "largest_holdout_budget"
    slices.append(top_budget)

    long_duration = holdout.sort_values("duration_hours", ascending=False).head(16).copy()
    long_duration["selection_reason"] = "longest_holdout_duration"
    slices.append(long_duration)

    return (
        pd.concat(slices, ignore_index=True)
        .drop_duplicates(["selection_reason", "split", "campaign_id"])
        .sort_values(["selection_reason", "split", "campaign_start"])
        .reset_index(drop=True)
    )


def build_hourly_stats_placeholder(top_campaigns_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in top_campaigns_df.iterrows():
        for hour in range(int(min(row["duration_hours"], 72))):
            rows.append(
                {
                    "split": row["split"],
                    "campaign_id": row["campaign_id"],
                    "selection_reason": row["selection_reason"],
                    "hour_index": hour,
                    "campaign_start": row["campaign_start"],
                    "campaign_end": row["campaign_end"],
                    "auction_budget": row["auction_budget"],
                    "logical_category": row["logical_category"],
                    "note": "timeline scaffold only; no model replay in default fast report",
                }
            )
    return pd.DataFrame(rows)


def build_summary_payload(
    *,
    aggregate_metrics_df: pd.DataFrame,
    diagnostics_segments_df: pd.DataFrame,
    top_campaigns_df: pd.DataFrame,
    summaries: dict[str, dict[str, Any]],
    linear_metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "linear": linear_metrics,
        "runs": [
            {
                "run_key": run_key,
                "run_name": payload["run_name"],
                "summary_path": str(payload["run_dir"] / "outputs" / "run_summary.json"),
                "metrics_path": str(payload["run_dir"] / "outputs" / "metrics.json"),
                "diagnostics_path": str(payload["diagnostics_path"]),
                "diagnostics_png": str(payload["diagnostics_png"]),
                "best_params": payload["summary"]["tuning"]["best_params"],
                "best_trial_number": payload["summary"]["tuning"]["best_trial_number"],
            }
            for run_key, payload in summaries.items()
        ],
        "aggregate_metrics": aggregate_metrics_df.to_dict(orient="records"),
        "diagnostics_segments": diagnostics_segments_df.to_dict(orient="records"),
        "selected_campaign_slices": top_campaigns_df.to_dict(orient="records"),
        "note": (
            "Default report is intentionally fast: DRLB metrics are read from May 04 summaries, "
            "linear val/holdout is evaluated or loaded from cache, diagnostics are read from existing CSVs. "
            "No train/full checkpoint replay is performed."
        ),
    }


def plot_diagnostics_segments(report_dir: Path, diagnostics_segments_df: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), dpi=140, sharex=True)
    for run_key, run_df in diagnostics_segments_df.groupby("run_key"):
        axes[0].plot(run_df["step_end"], run_df["dqn_loss_mean"], marker="o", label=run_key)
        axes[1].plot(run_df["step_end"], run_df["reward_net_loss_mean"], marker="o", label=run_key)
    axes[0].set_title("DQN loss mean by training segment")
    axes[1].set_title("RewardNet loss mean by training segment")
    axes[1].set_xlabel("training step")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(report_dir / "diagnostics_segments.png", bbox_inches="tight")
    plt.close(fig)


def render_report(
    *,
    aggregate_metrics_df: pd.DataFrame,
    diagnostics_segments_df: pd.DataFrame,
    top_campaigns_df: pd.DataFrame,
    summary_payload: dict[str, Any],
) -> str:
    holdout = aggregate_metrics_df[aggregate_metrics_df["split"] == "holdout"].sort_values(
        "clicks_sum",
        ascending=False,
    )
    val = aggregate_metrics_df[aggregate_metrics_df["split"] == "val"].sort_values(
        "clicks_sum",
        ascending=False,
    )
    default_pre = diagnostics_segments_df[
        (diagnostics_segments_df["run_key"] == "default")
        & (diagnostics_segments_df["step_end"] == 25000)
    ]
    default_post = diagnostics_segments_df[
        (diagnostics_segments_df["run_key"] == "default")
        & (diagnostics_segments_df["step_end"] == 30000)
    ]
    jump_text = "Default RewardNet has no 25-30k segment in diagnostics."
    if not default_pre.empty and not default_post.empty:
        jump_text = (
            "Default RewardNet loss mean changes from "
            f"{default_pre.iloc[0]['reward_net_loss_mean']:.3f} in `(20000,25000]` "
            f"to {default_post.iloc[0]['reward_net_loss_mean']:.3f} in `(25000,30000]`."
        )

    return "\n".join(
        [
            "# May 04 DRLB Diagnostics Report",
            "",
            f"Generated at `{summary_payload['generated_at']}`.",
            "",
            "This report follows `example_notebooks/experiments/drlb/may_04/04_compare_states_vs_linear.ipynb`: DRLB metrics are read from existing May 04 summaries, and locked LinearBidder val/holdout metrics are evaluated once then cached in `linear_metrics.json`. No full train/val/holdout checkpoint replay is done by default.",
            "",
            "## Holdout Ranking",
            "",
            format_table(
                holdout[
                    [
                        "model",
                        "clicks_sum",
                        "clicks_sum_delta_vs_linear",
                        "rmse",
                        "cpc_relative",
                        "quickspend",
                        "average_end_balance_share",
                    ]
                ]
            ),
            "",
            "Linear is still the strongest holdout policy by clicks. `default` is the best May 04 DRLB holdout variant, while `ta_ratio_bat` wins validation but does not transfer cleanly to holdout.",
            "",
            "## Validation Ranking",
            "",
            format_table(
                val[
                    [
                        "model",
                        "clicks_sum",
                        "clicks_sum_delta_vs_linear",
                        "rmse",
                        "cpc_relative",
                        "quickspend",
                        "average_end_balance_share",
                    ]
                ]
            ),
            "",
            "Validation best scores are selection-time metrics from `run_summary.json`; exact `best_val` checkpoint files were not persisted. The persisted final model is `best_refit.pt`, trained again on train+val, so validation wins can drift after refit.",
            "",
            "## Diagnostics",
            "",
            format_table(
                diagnostics_segments_df[
                    [
                        "run_key",
                        "segment",
                        "dqn_loss_mean",
                        "dqn_loss_p95",
                        "reward_net_loss_mean",
                        "reward_net_loss_p95",
                        "reward_signal_mean",
                        "lambda_start",
                        "lambda_end",
                    ]
                ].head(40)
            ),
            "",
            jump_text,
            "",
            "Likely non-convergence causes: RewardNet is trained on high-variance return targets while DQN learns from RewardNet predictions; May 04 removed finite lambda clipping; `gamma=1.0` preserves long-horizon Q variance; default-state feature scales mix absolute budget-like values with small ratios.",
            "",
            "`ta_ratio_bat` looks promising because traffic-aware pacing is useful signal. It still underperforms on holdout because the target/reward instability and unconstrained lambda movement remain.",
            "",
            "## Campaign Slices",
            "",
            format_table(
                top_campaigns_df[
                    [
                        "selection_reason",
                        "split",
                        "campaign_id",
                        "campaign_start_date",
                        "auction_budget",
                        "duration_hours",
                        "logical_category",
                    ]
                ].head(36)
            ),
            "",
            "These campaign slices are metadata-only in the fast report. They mark where to run manual replay next: earliest campaign starts, largest holdout budgets, and longest holdout durations.",
            "",
            "## Recommendations",
            "",
            "1. Persist exact `best_val` checkpoints.",
            "2. Restore finite lambda bounds for the next controlled wave.",
            "3. Try `gamma < 1.0` and lower-variance RewardNet targets.",
            "4. Normalize default-state feature scales before comparing state families again.",
            "5. Keep `ta_ratio_bat`, but pair it with clipped lambda and a less noisy RewardNet target.",
            "",
        ]
    )


def format_table(df: pd.DataFrame) -> str:
    table_df = df.copy()
    for column in table_df.columns:
        if pd.api.types.is_float_dtype(table_df[column]):
            table_df[column] = table_df[column].map(lambda value: "" if pd.isna(value) else f"{value:.3f}")
    table_df = table_df.fillna("")
    headers = [str(column) for column in table_df.columns]
    rows = [[str(value).replace("|", "\\|") for value in row] for row in table_df.to_numpy()]
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(row) + " |" for row in rows],
        ]
    )


def cross_check_drlb_holdout_metrics(
    summaries: dict[str, dict[str, Any]],
    aggregate_metrics_df: pd.DataFrame,
    report_dir: Path,
) -> None:
    rows = []
    for run_key, payload in summaries.items():
        actual = aggregate_metrics_df[
            (aggregate_metrics_df["run_key"] == run_key)
            & (aggregate_metrics_df["split"] == "holdout")
        ].iloc[0]
        expected = payload["metrics"]
        for metric in ("clicks_sum", "rmse", "cpc_relative", "quickspend"):
            rows.append(
                {
                    "run_key": run_key,
                    "metric": metric,
                    "expected": expected[metric],
                    "actual": actual[metric],
                    "abs_delta": abs(expected[metric] - actual[metric]),
                }
            )
    pd.DataFrame(rows).to_csv(report_dir / "cross_check_holdout_metrics.csv", index=False)


def write_csv(report_dir: Path, df: pd.DataFrame, name: str) -> None:
    df.to_csv(report_dir / name, index=False)


if __name__ == "__main__":
    main()

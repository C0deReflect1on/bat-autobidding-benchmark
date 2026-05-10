from __future__ import annotations

from typing import Any

import pandas as pd

from simulator.model.linear_bidder import LinearBidder
from simulator.simulation.simulate import simulate_campaign
from simulator.validation.check_results import create_campaign_instance


def estimate_linear_lambda_init(
    *,
    normalized_splits: dict[str, dict[str, str]],
    linear_params: dict[str, Any],
    auction_mode: str,
    mean_click_price: float = 5.0,
    split_roles: list[str] | None = None,
) -> dict[str, Any]:
    if split_roles is None:
        split_roles = ["train"]

    campaigns_list = []
    stats_list = []
    for role in split_roles:
        campaigns_list.append(pd.read_csv(normalized_splits[role]["campaigns_path"]))
        stats_list.append(pd.read_csv(normalized_splits[role]["stats_path"]))
    
    train_campaigns = pd.concat(campaigns_list, ignore_index=True)
    train_stats = pd.concat(stats_list, ignore_index=True)

    linear_hist_parts = []
    for _, campaign_row in train_campaigns.iterrows():
        campaign_id = int(campaign_row["campaign_id"])
        campaign_stats = train_stats[train_stats.campaign_id == campaign_id].copy()
        if campaign_stats.empty:
            continue

        campaign = create_campaign_instance(campaign_row, mean_click_price)
        bidder = LinearBidder(linear_params)
        history = simulate_campaign(
            campaign=campaign,
            bidder=bidder,
            stats_file=campaign_stats,
            auction_mode=auction_mode,
        )
        hist_df = history.to_data_frame()
        if not hist_df.empty:
            linear_hist_parts.append(hist_df)

    linear_hist = pd.concat(linear_hist_parts, ignore_index=True)
    ctr_by_period = (
        train_stats
        .groupby(["campaign_id", "period"], as_index=False)
        .agg(ctr_pred=("CTRPredicts", "mean"))
    )
    lambda_df = linear_hist.merge(
        ctr_by_period,
        left_on=["campaign_id", "prev_timestamp"],
        right_on=["campaign_id", "period"],
        how="inner",
    )
    lambda_df = lambda_df[(lambda_df["bid"] > 0) & (lambda_df["ctr_pred"] > 0)].copy()
    lambda_df["linear_lambda"] = lambda_df["ctr_pred"] / lambda_df["bid"]

    return {
        "linear_lambda_init": float(lambda_df["linear_lambda"].mean()),
        "linear_lambda_median": float(lambda_df["linear_lambda"].median()),
        "linear_lambda_summary": lambda_df["linear_lambda"].describe(
            percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]
        ),
        "linear_lambda_examples": lambda_df[
            ["campaign_id", "prev_timestamp", "bid", "ctr_pred", "linear_lambda"]
        ].head(20),
    }

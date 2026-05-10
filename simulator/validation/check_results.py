import pandas as pd
from time import time
from typing import Type, Dict, Any, List
from simulator.simulation.simulate import simulate_campaign
from simulator.simulation.modules import Campaign
from simulator.validation.metrics import compile_metrics
from tqdm import tqdm


def _campaign_end_balance_rows(hist_data_list: List[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for history_df in hist_data_list:
        if history_df.empty:
            continue
        last_row = history_df.iloc[-1]
        initial_balance = float(last_row["initial_balance"])
        final_balance = float(last_row["balance"])
        spent_total = float(history_df["spend_history"].sum())
        rows.append(
            {
                "campaign_id": int(last_row["campaign_id"]),
                "initial_balance": initial_balance,
                "final_balance": final_balance,
                "spent_total": spent_total,
                "end_balance_share": final_balance / max(initial_balance, 1e-9),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=["campaign_id", "initial_balance", "final_balance", "spent_total", "end_balance_share"]
        )
    return pd.DataFrame(rows)


def autobidder_check(
    bidder: Type,
    params: Dict[str, Any],
    auction_mode: str = 'VCG',
    mean_click_price: float = 5.0,
    verbose: bool = False,
    log_every_campaigns: int = 10,
    use_tqdm: bool = False,
) -> Dict[str, Any]:
    """
    Perform an automated check of a bidding strategy across multiple campaigns.

    This function simulates campaigns using the provided bidder and compiles performance metrics.

    Args:
        bidder: The bidder class to be used in simulations.
        params: Dictionary containing simulation parameters.
        auction_mode: The auction mode to use. Defaults to 'VCG'.
        category: The category of campaigns to simulate. Defaults to 'Transport'.

    Returns: A dictionary containing simulation results and performance metrics.
    """
    status = "OK!"
    status_msg = ""
    time_all_start = time()

    data_campaigns = pd.read_csv(params["input_campaigns"]).reset_index()
    data_stats = pd.read_csv(params["input_stats"])

    time_inf_start = time()
    hist_data_list: List[pd.DataFrame] = []
    runtime_diagnostics_list: List[pd.DataFrame] = []
    skipped_campaigns = 0

    total_campaigns = len(data_campaigns)
    campaign_iter = data_campaigns.iterrows()
    if use_tqdm:
        campaign_iter = tqdm(
            campaign_iter,
            total=total_campaigns,
            desc="autobidder_check campaigns",
            unit="campaign",
        )

    for idx, (_, campaign) in enumerate(campaign_iter, start=1):
        if use_tqdm:
            campaign_iter.set_postfix({"campaign_id": int(campaign["campaign_id"])}, refresh=False)
        if verbose and (idx == 1 or idx % max(1, log_every_campaigns) == 0 or idx == total_campaigns):
            elapsed = time() - time_inf_start
            print(
                f"[autobidder_check] campaign {idx}/{total_campaigns} "
                f"(id={int(campaign['campaign_id'])}) elapsed={elapsed:.1f}s"
            )

        campaign_instance = create_campaign_instance(campaign, mean_click_price)
        bidder_instance = bidder(params)

        campaign_stats = data_stats[data_stats.campaign_id == int(campaign['campaign_id'])].copy()
        if campaign_stats.empty:
            skipped_campaigns += 1
            if verbose:
                print(
                    f"[autobidder_check] skip campaign {idx}/{total_campaigns} "
                    f"(id={int(campaign['campaign_id'])}) reason=no_stats"
                )
            continue

        sim_hist = simulate_campaign(
            campaign=campaign_instance,
            bidder=bidder_instance,
            stats_file=campaign_stats,
            auction_mode=auction_mode
        )
        hist_data_list.append(sim_hist.to_data_frame())
        diagnostics_df = None
        if hasattr(bidder_instance, "get_runtime_diagnostics"):
            diagnostics_df = bidder_instance.get_runtime_diagnostics()
        if (not isinstance(diagnostics_df, pd.DataFrame)) or diagnostics_df.empty:
            if hasattr(bidder_instance, "get_training_diagnostics"):
                diagnostics_df = bidder_instance.get_training_diagnostics()
        if isinstance(diagnostics_df, pd.DataFrame) and not diagnostics_df.empty:
            runtime_diagnostics_list.append(
                diagnostics_df.assign(campaign_id=int(campaign["campaign_id"]))
            )
        # break

    time_inf_end = time()

    if not hist_data_list:
        metrics = (float("inf"), float("inf"), 0.0, 0.0)
    else:
        metrics = compile_metrics(
            pd.concat(hist_data_list, axis=0, ignore_index=True),
            traffic_share_path=params.get("traffic_share_path"),
        )
    if verbose:
        print(
            "[autobidder_check] done "
            f"inference={time_inf_end - time_inf_start:.1f}s overall={time() - time_all_start:.1f}s "
            f"score={metrics} skipped_campaigns={skipped_campaigns}"
        )

    time_all_end = time()
    campaign_budget_df = _campaign_end_balance_rows(hist_data_list)
    return {
        "status": status,
        "status_msg": status_msg,
        "time_overall_sec": time_all_end - time_all_start,
        "time_inference_sec": time_inf_end - time_inf_start,
        "score": metrics,
        "skipped_campaigns": skipped_campaigns,
        "all_hist_data": hist_data_list, # TMP
        "campaign_budget_summary": campaign_budget_df,
        "average_end_balance_share": (
            None if campaign_budget_df.empty else float(campaign_budget_df["end_balance_share"].mean())
        ),
        "runtime_diagnostics": runtime_diagnostics_list,
    }


def create_campaign_instance(campaign: pd.Series, mean_click_price: float) -> Campaign:
    """
    Create a Campaign instance from a pandas Series.
    """
    return Campaign(
        item_id=campaign['item_id'],
        campaign_id=int(campaign['campaign_id']),
        loc_id=int(campaign["loc_id"]),
        region_id=int(campaign["region_id"]),
        logical_category=campaign["logical_category"],
        microcat_ext=int(campaign["microcat_ext"]),
        campaign_start=int(campaign["campaign_start"]),
        campaign_end=int(campaign["campaign_end"]),
        initial_balance=campaign['auction_budget'],
        balance=campaign['auction_budget'],
        curr_time=int(campaign["campaign_start"]),
        prev_time=int(campaign["campaign_start"]),
        prev_balance=campaign['auction_budget'],
        prev_bid=0,
        prev_clicks=0,
        desired_clicks=max(1, campaign['auction_budget'] // mean_click_price),
        desired_time=(int(campaign["campaign_end"]) - int(campaign["campaign_start"])) // 3600,
    )

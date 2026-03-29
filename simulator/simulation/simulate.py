from .utils import price2bin, bin2price
from .modules import SimulationResult, Campaign, History
from ..model.bidder import _Bidder
import pandas as pd
from typing import Tuple


def _bidder_uses_lp_features(bidder: _Bidder) -> bool:
    return any(c.__name__ in ['MPIDBidder', 'SlivkinsBidder'] for c in type(bidder).__mro__)


def _bidder_uses_drlb_ctr_feedback(bidder: _Bidder) -> bool:
    return any(c.__name__ == 'DRLBBidder' for c in type(bidder).__mro__)


def _lookup_recent_stats_window(
    stats_file: pd.DataFrame,
    campaign: Campaign,
) -> pd.DataFrame | None:
    stats_window = (
        stats_file
        [
            (stats_file['period'] >= campaign.curr_time - 3600) &
            (stats_file['period'] < campaign.curr_time) &
            (stats_file['campaign_id'] == campaign.campaign_id)
        ]
        .copy()
    )

    # Find the nearest stats window with logs.
    # Guard against campaigns with no stats at all.
    i = 1
    max_lookback = max(1, int((campaign.curr_time - campaign.campaign_start) // 3600) + 1)
    while stats_window.empty and i <= max_lookback:
        stats_window = (
            stats_file[
                (stats_file['period'] >= campaign.curr_time - 3600 * i) &
                (stats_file['period'] < campaign.curr_time - 3600 * (i - 1)) &
                (stats_file['campaign_id'] == campaign.campaign_id)
            ]
            .copy()
        )
        i += 1

    if stats_window.empty:
        return None
    return stats_window


def simulate_step(
    stats_pdf: pd.DataFrame,
    campaign: Campaign,
    bid: float,
    auction_mode: str = 'VCG'
) -> SimulationResult:
    """
    Simulate one step of the auction for a 1-hour period.
    Args:
        stats_pdf: Statistics dataframe
        campaign: Current campaign object
        bid: Bid value
        auction_mode: Auction type, either 'VCG' or 'FPA'. Default is 'VCG'

    Returns:
        SimulationResult: Results of the auction step
    """

    # Convert bid to bin or set to minimum if bid is zero or negative
    bid_price_bin = price2bin(bid) if bid > 0 else -1000

    # Filter stats for the current time window and campaign
    stats_window = stats_pdf[
        (stats_pdf['period'] >= campaign.curr_time) &
        (stats_pdf['period'] < campaign.curr_time + 3600) &
        (stats_pdf['campaign_id'] == campaign.campaign_id)
    ].copy()

    # Aggregate data for bids less than or equal to the current bid
    agg_data = stats_window[
        stats_window['contact_price_bin'] <= bid_price_bin
    ][
        [
            'AuctionWinBidSurplus',
            'AuctionVisibilitySurplus',
            'AuctionClicksSurplus',
            'AuctionContactsSurplus'
        ]
    ].sum()

    # Create SimulationResult based on auction mode
    if auction_mode == 'VCG':
        return SimulationResult(
            spent=agg_data['AuctionWinBidSurplus'],
            visibility=agg_data['AuctionVisibilitySurplus'],
            clicks=agg_data['AuctionClicksSurplus'],
            contacts=agg_data['AuctionContactsSurplus'],
            bid=bid,
        )
    elif auction_mode == 'FPA':
        return SimulationResult(
            spent=agg_data['AuctionContactsSurplus'] * bid,
            visibility=agg_data['AuctionVisibilitySurplus'],
            clicks=agg_data['AuctionClicksSurplus'],
            contacts=agg_data['AuctionContactsSurplus'],
            bid=bid,
        )
    else:
        raise ValueError("auction_mode must be either 'VCG' or 'FPA'")


def simulate_campaign(
    campaign: Campaign,
    bidder: _Bidder,
    stats_file: pd.DataFrame,
    start_time: int = None,
    auction_mode: str = 'VCG'
) -> History:
    """
    Simulate a campaign using historical data.

    Args:
        campaign: Campaign object
        bidder: Bidder object
        stats_file: Historical statistics
        start_time: Start time for simulation. Defaults to None.
        auction_mode: Auction mode ('VCG' or 'FPA'). Defaults to 'VCG'.

    Returns:
        History: Simulation history of spending and clicks
    """
    # DEBUG: состояние campaign при входе (flush=True чтобы было видно в Jupyter)
    # print(
    #     f"[simulate_campaign] ВХОД: campaign_id={campaign.campaign_id} "
    #     f"balance={campaign.balance:.2f} initial_balance={campaign.initial_balance:.2f} "
    #     f"clicks={campaign.clicks:.2f} curr_time={campaign.curr_time}",
    #     flush=True,
    # )
    # if campaign.balance < 0.01 and campaign.initial_balance > 1:
    #     print(
    #         "[simulate_campaign] !!! ОШИБКА: balance≈0 при initial_balance>0 — "
    #         "campaign уже был использован, повторный запуск с тем же объектом даст неверный результат !!!",
    #         flush=True,
    #     )

    if start_time:
        campaign.curr_time = start_time // 3600 * 3600
    else:
        campaign.curr_time = campaign.campaign_start // 3600 * 3600

    simulation_history = History()

    bidder_spend = 0
    bidder_clicks = 0
    # For M-PID only, not used for cold start, so could be set any
    campaign_ctr, campaign_cr = 0.0, 0.0
    wp_for_lp = None
    ctr_for_lp = None
    cr_for_lp = None

    while campaign.curr_time < campaign.campaign_end:
        # Request bid from bidder
        bid = bidder.place_bid(
            history=simulation_history,
            bidding_input_params={
                    'item_id': campaign.item_id,
                    'loc_id': campaign.loc_id,
                    'region_id': campaign.region_id,
                    'logical_category': campaign.logical_category,
                    'microcat_ext': campaign.microcat_ext,
                    'balance': campaign.balance,
                    'initial_balance': campaign.initial_balance,
                    'clicks': campaign.clicks,
                    'campaign_id': campaign.campaign_id,
                    'campaign_start_time': campaign.campaign_start,
                    'campaign_end_time': campaign.campaign_end,
                    'curr_time': campaign.curr_time,
                    'prev_balance': campaign.prev_balance,
                    'prev_bid': campaign.prev_bid,
                    'prev_clicks': campaign.prev_clicks,
                    'prev_contacts': campaign.prev_contacts,
                    'prev_time': campaign.prev_time,
                    'desired_clicks': campaign.desired_clicks,
                    'desired_time': campaign.desired_time,
                    'prev_ctr': campaign_ctr,
                    'prev_cr': campaign_cr,
                    'ctr_for_lp': ctr_for_lp,
                    'cr_for_lp': cr_for_lp,
                    'wp_for_lp': wp_for_lp
                }
        )

        # Simulate auction results for a 1-hour window
        simulation_result = simulate_step(
            stats_pdf=stats_file,
            campaign=campaign,
            bid=bid,
            auction_mode=auction_mode,
        )
        bidder_spend = simulation_result.spent
        bidder_clicks = simulation_result.clicks

        # Adjust results if spend exceeds budget
        coef = 1.0
        if simulation_result.spent > campaign.balance:
            coef = campaign.balance / simulation_result.spent

        # Update campaign status
        campaign.prev_balance = campaign.balance
        campaign.prev_clicks = campaign.clicks
        campaign.prev_time = campaign.curr_time
        campaign.prev_bid = bid
        campaign.balance -= simulation_result.spent * coef
        campaign.clicks += simulation_result.clicks * coef
        campaign.contacts += simulation_result.contacts * coef
        campaign.curr_time += 3600

        needs_lp_features = _bidder_uses_lp_features(bidder)
        needs_drlb_feedback = _bidder_uses_drlb_ctr_feedback(bidder)
        if needs_lp_features or needs_drlb_feedback:
            stats_window = _lookup_recent_stats_window(stats_file, campaign)

            # Calculate LP features once per campaign for bidders that need them.
            if needs_lp_features and ctr_for_lp is None and stats_window is not None:
                ctr_for_lp, cr_for_lp, wp_for_lp = ctr_cvr_count_for_lp(stats_window)

            # Feed CTR/CVR feedback only to explicitly opted-in bidders such as DRLB.
            if needs_drlb_feedback and stats_window is not None:
                campaign_ctr, campaign_cr = ctr_cvr_count(stats_window, bid)

        # Add record to simulation history
        simulation_history.add(
            campaign=campaign,
            bid=bid,
            spend=bidder_spend,
            clicks=bidder_clicks
        )
        if campaign.balance < 0.00001:
            break

    return simulation_history


def ctr_cvr_count(stats_window: pd.DataFrame, bid: float) -> Tuple[float, float]:
    """
    Calculate CTR and CVR based on the closest bin to the given bid.
    """
    filtered_bins = stats_window[stats_window.contact_price_bin > price2bin(bid)]['contact_price_bin']
    if not filtered_bins.empty:
        closest_bin = filtered_bins.min()
    else:
        closest_bin = stats_window['contact_price_bin'].max()
    campaign_ctr = stats_window[stats_window.contact_price_bin == closest_bin]['CTRPredicts'].max()
    campaign_cr = stats_window[stats_window.contact_price_bin == closest_bin]['CRPredicts'].max()
    return campaign_ctr, campaign_cr


def ctr_cvr_count_for_lp(stats_window: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Extract CTR, CVR, and winning prices for LP problem in M-PID.
    """
    campaign_ctr = stats_window['CTRPredicts']
    campaign_cr = stats_window['CRPredicts']
    wp_for_lp = bin2price(stats_window['contact_price_bin'])
    return campaign_ctr, campaign_cr, wp_for_lp

from dataclasses import dataclass

import pandas as pd

from .modules import Campaign, SimulationResult


@dataclass(frozen=True)
class StepInput:
    campaign_id: int
    period_start_ts: int
    ctr_pred: float
    balance: float
    initial_balance: float


class BatStepEnv:
    """Campaign-hour BAT environment used by DRLB fit."""

    def __init__(
        self,
        stats_pdf: pd.DataFrame,
        campaign_row: pd.Series,
        auction_mode: str,
    ):
        self.stats_pdf = stats_pdf.sort_values("period").reset_index(drop=True)
        self.auction_mode = auction_mode
        self.campaign = self._build_campaign(campaign_row)

    @staticmethod
    def _build_campaign(campaign_row: pd.Series) -> Campaign:
        start = int(campaign_row["campaign_start"])
        end = int(campaign_row["campaign_end"])
        budget = max(1.0, float(campaign_row["auction_budget"]))
        aligned_start = (start // 3600) * 3600
        return Campaign(
            campaign_id=int(campaign_row["campaign_id"]),
            campaign_start=start,
            campaign_end=end,
            initial_balance=budget,
            balance=budget,
            curr_time=aligned_start,
            prev_time=aligned_start,
            prev_balance=budget,
            prev_bid=0.0,
            prev_clicks=0.0,
            prev_contacts=0.0,
        )

    def _window(self, start_ts: int, end_ts: int) -> pd.DataFrame:
        return self.stats_pdf[
            (self.stats_pdf["period"] >= start_ts)
            & (self.stats_pdf["period"] < end_ts)
            & (self.stats_pdf["campaign_id"] == self.campaign.campaign_id)
        ]

    def _current_ctr_pred(self) -> float:
        window = self._window(self.campaign.curr_time, self.campaign.curr_time + 3600)
        if window.empty:
            return 0.0
        return float(max(0.0, window["CTRPredicts"].mean()))

    def get_step_input(self) -> StepInput:
        return StepInput(
            campaign_id=self.campaign.campaign_id,
            period_start_ts=int(self.campaign.curr_time),
            ctr_pred=self._current_ctr_pred(),
            balance=float(max(0.0, self.campaign.balance)),
            initial_balance=float(max(1.0, self.campaign.initial_balance)),
        )

    def step(self, bid: float) -> SimulationResult:
        from .simulate import simulate_step

        simulation_result = simulate_step(
            stats_pdf=self.stats_pdf,
            campaign=self.campaign,
            bid=float(bid),
            auction_mode=self.auction_mode,
        )

        coef = 1.0
        if simulation_result.spent > self.campaign.balance and simulation_result.spent > 0:
            coef = self.campaign.balance / simulation_result.spent

        adjusted = SimulationResult(
            bid=float(bid),
            spent=float(simulation_result.spent) * coef,
            visibility=float(simulation_result.visibility) * coef,
            clicks=float(simulation_result.clicks) * coef,
            contacts=float(simulation_result.contacts) * coef,
        )

        self.campaign.prev_balance = self.campaign.balance
        self.campaign.prev_clicks = self.campaign.clicks
        self.campaign.prev_contacts = self.campaign.contacts
        self.campaign.prev_time = self.campaign.curr_time
        self.campaign.prev_bid = float(bid)
        self.campaign.balance -= adjusted.spent
        self.campaign.clicks += adjusted.clicks
        self.campaign.contacts += adjusted.contacts
        self.campaign.curr_time += 3600
        return adjusted

    def done(self) -> bool:
        return (
            self.campaign.curr_time >= self.campaign.campaign_end
            or self.campaign.balance <= 0
        )

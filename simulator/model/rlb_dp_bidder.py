"""
RLB-DP Bidder adapted for bat-autobidding-benchmark

Key adaptations:
1. Bidding is per HOUR (not per impression)
2. State n = remaining HOURS (not impressions)
3. Reward comes from AuctionClicksSurplus (incremental clicks per bin)
4. Value function D(n,b) optimizes budget pacing over remaining hours
"""

import pickle
from typing import Dict, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import DATA_DIR
from simulator.model.bidder import _Bidder
from simulator.model.traffic import Traffic
from simulator.simulation.modules import History
from simulator.simulation.utils import bin2price, price2bin


def _to_traffic_bucket(traffic_share: float, traffic_share_bins: int) -> int:
    clipped = float(np.clip(traffic_share, 0.0, 1.0))
    return int(np.clip(np.floor(clipped * traffic_share_bins), 0, traffic_share_bins - 1))


def _compute_bin_statistics(stats_df: pd.DataFrame, max_bin: int = 60) -> dict:
    """Compute average surplus and cost per bin from historical data."""
    bin_stats = {
        'click_surplus_per_bin': {},
        'cost_per_bin': {},
        'contacts_surplus_per_bin': {},
    }

    for bin_val in range(max_bin + 1):
        bin_data = stats_df[stats_df['contact_price_bin'] == bin_val]

        if len(bin_data) > 0:
            bin_price = bin2price(bin_val)
            bin_stats['click_surplus_per_bin'][bin_val] = bin_data['AuctionClicksSurplus'].mean()
            bin_stats['cost_per_bin'][bin_val] = bin_price * bin_data['AuctionContactsSurplus'].mean()
            bin_stats['contacts_surplus_per_bin'][bin_val] = bin_data['AuctionContactsSurplus'].mean()
        else:
            bin_stats['click_surplus_per_bin'][bin_val] = 0.0
            bin_stats['cost_per_bin'][bin_val] = 0.0
            bin_stats['contacts_surplus_per_bin'][bin_val] = 0.0

    return bin_stats


def _compute_bin_statistics_by_traffic_bucket(
    stats_df: pd.DataFrame,
    max_bin: int,
    traffic_share_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    click_surplus = np.zeros((traffic_share_bins, max_bin + 1), dtype=float)
    contacts_surplus = np.zeros((traffic_share_bins, max_bin + 1), dtype=float)
    costs = np.zeros((traffic_share_bins, max_bin + 1), dtype=float)

    for traffic_bucket in range(traffic_share_bins):
        bucket_df = stats_df[stats_df['traffic_bucket'] == traffic_bucket]
        for bin_val in range(max_bin + 1):
            bin_data = bucket_df[bucket_df['contact_price_bin'] == bin_val]
            if len(bin_data) == 0:
                continue
            bin_price = bin2price(bin_val)
            click_surplus[traffic_bucket, bin_val] = float(bin_data['AuctionClicksSurplus'].mean())
            contacts_surplus[traffic_bucket, bin_val] = float(bin_data['AuctionContactsSurplus'].mean())
            costs[traffic_bucket, bin_val] = float(bin_price * bin_data['AuctionContactsSurplus'].mean())

    return click_surplus, contacts_surplus, costs


def _train_value_function(
    stats_df: pd.DataFrame,
    N: int = 100,
    B_max: int = 10000,
    B_step: int = 100,
    max_bin: int = 60,
    gamma: float = 1.0,
    objective: str = 'clicks',
):
    """Train V(n,b) and policy via backward induction. Returns (V_full, policy_full)."""
    bin_stats = _compute_bin_statistics(stats_df, max_bin)

    if objective == 'clicks':
        surplus_stats = bin_stats['click_surplus_per_bin']
    else:
        surplus_stats = bin_stats['contacts_surplus_per_bin']
    cost_stats = bin_stats['cost_per_bin']

    B_discrete = B_max // B_step + 1
    V = np.zeros((N + 1, B_discrete))
    policy = np.zeros((N + 1, B_discrete), dtype=np.int32)

    for n in tqdm(range(N, 0, -1), desc="Hours"):
        for b_idx in range(B_discrete):
            # observe state (remaining hours, remaining budget)
            b = b_idx * B_step
            if b == 0:
                continue
            best_value, best_target_bin = 0.0, 0
            # act: enumerate all target bins
            for target_bin in range(max_bin + 1):
                total_reward = sum(surplus_stats.get(i, 0) for i in range(target_bin + 1))
                total_cost = sum(cost_stats.get(i, 0) for i in range(target_bin + 1))

                if total_cost > b:
                    continue

                # step env + learn DP backup
                b_next_idx = min(int((b - total_cost) / B_step), B_discrete - 1)
                value = total_reward + gamma * V[n - 1][b_next_idx]

                if value > best_value:
                    best_value, best_target_bin = value, target_bin

            V[n][b_idx] = best_value
            policy[n][b_idx] = best_target_bin

    V_full = np.zeros((N + 1, B_max + 1))
    policy_full = np.zeros((N + 1, B_max + 1), dtype=np.int32)

    for n in range(N + 1):
        for b in range(B_max + 1):
            b_idx = min(b // B_step, B_discrete - 1)
            V_full[n][b] = V[n][b_idx]
            policy_full[n][b] = policy[n][b_idx]
    return V_full, policy_full


def _train_value_function_with_traffic_state(
    stats_df: pd.DataFrame,
    transition_matrix: np.ndarray,
    traffic_share_bins: int,
    N: int,
    B_max: int,
    B_step: int,
    max_bin: int,
    gamma: float,
    objective: str,
) -> tuple[np.ndarray, np.ndarray]:
    click_surplus, contacts_surplus, cost_stats = _compute_bin_statistics_by_traffic_bucket(
        stats_df=stats_df,
        max_bin=max_bin,
        traffic_share_bins=traffic_share_bins,
    )

    if objective == 'clicks':
        surplus_stats = click_surplus
    else:
        surplus_stats = contacts_surplus

    cumulative_surplus = np.cumsum(surplus_stats, axis=1)
    cumulative_cost = np.cumsum(cost_stats, axis=1)

    B_discrete = B_max // B_step + 1
    V = np.zeros((N + 1, B_discrete, traffic_share_bins), dtype=float)
    policy = np.zeros((N + 1, B_discrete, traffic_share_bins), dtype=np.int32)

    for n in tqdm(range(N, 0, -1), desc="HoursTraffic"):
        for b_idx in range(B_discrete):
            # observe state (remaining hours, remaining budget, traffic bucket)
            b = b_idx * B_step
            if b == 0:
                continue
            for traffic_bucket in range(traffic_share_bins):
                best_value = 0.0
                best_target_bin = 0
                # act: enumerate all target bins
                for target_bin in range(max_bin + 1):
                    total_reward = cumulative_surplus[traffic_bucket, target_bin]
                    total_cost = cumulative_cost[traffic_bucket, target_bin]
                    if total_cost > b:
                        continue

                    # step env + learn DP backup with t -> t_next transition
                    b_next_idx = min(int((b - total_cost) / B_step), B_discrete - 1)
                    continuation = float(np.dot(transition_matrix[traffic_bucket], V[n - 1, b_next_idx]))
                    value = total_reward + gamma * continuation

                    if value > best_value:
                        best_value = value
                        best_target_bin = target_bin

                V[n, b_idx, traffic_bucket] = best_value
                policy[n, b_idx, traffic_bucket] = best_target_bin

    V_full = np.zeros((N + 1, B_max + 1, traffic_share_bins), dtype=float)
    policy_full = np.zeros((N + 1, B_max + 1, traffic_share_bins), dtype=np.int32)

    for n in range(N + 1):
        for b in range(B_max + 1):
            b_idx = min(b // B_step, B_discrete - 1)
            V_full[n, b] = V[n, b_idx]
            policy_full[n, b] = policy[n, b_idx]

    return V_full, policy_full


class RLBDPBidder(_Bidder):
    """
    RLB-DP bidder adapted for hourly bidding in bat-autobidding-benchmark.

    Original RLB-DP (WSDM 2017) was designed for impression-level bidding.
    This version adapts it for hourly budget pacing with aggregated statistics.

    State space:
        n: remaining hours in campaign
        b: remaining budget

    Action space:
        a: bid amount for the next hour

    Value function:
        D(n, b) = expected total clicks when optimally bidding
                  for n remaining hours with budget b

    Key difference: Instead of deciding "bid or not" for each impression,
    we decide "how much to bid for this hour" which determines which
    price bins we win (all bins where bin <= our_bid_bin).
    """

    default_params = {
        'max_bid': 300,
        'lower_clip': 5,
        'upper_clip': 5,
        'gamma': 1.0,
        'model_path': None,
        'N_bound': 100,
        'B_bound': 10000,
        'use_traffic_share_state': False,
        'traffic_share_mode': 'next_hour',
        'traffic_share_bins': 20,
        'traffic_share_path': str(DATA_DIR / 'traffic_share.csv'),
    }

    def __init__(self, params: Optional[Dict] = None):
        super().__init__()

        params = params or {}
        self.max_bid = params.get('max_bid', self.default_params['max_bid'])
        self.lower_clip = int(params.get('lower_clip', self.default_params['lower_clip']))
        self.upper_clip = int(params.get('upper_clip', self.default_params['upper_clip']))
        self.gamma = params.get('gamma', self.default_params['gamma'])
        self.N_bound = params.get('N_bound', self.default_params['N_bound'])
        self.B_bound = params.get('B_bound', self.default_params['B_bound'])

        self.use_traffic_share_state = bool(
            params.get('use_traffic_share_state', self.default_params['use_traffic_share_state'])
        )
        self.traffic_share_mode = params.get('traffic_share_mode', self.default_params['traffic_share_mode'])
        self.traffic_share_bins = int(params.get('traffic_share_bins', self.default_params['traffic_share_bins']))
        self.traffic_share_path = str(params.get('traffic_share_path', self.default_params['traffic_share_path']))

        self.traffic = Traffic(path=self.traffic_share_path)
        self.transition_matrix = None

        # Value function and policy (trained offline)
        self.value_table = None
        self.policy_table = None
        self.model_type = None
        self.policy_max_bin = int(params.get('policy_max_bin', 60))

        model_path = params.get('model_path')
        if model_path:
            self.load_model(model_path)

        self.campaign_hours = None
        self.avg_clicks_per_dollar = None

    def _compute_traffic_share_feature(
        self,
        region_id: int,
        campaign_start: int,
        campaign_end: int,
        hour_start: int,
    ) -> float:
        total_traffic = self.traffic.get_traffic_share(region_id, campaign_start, campaign_end)
        if total_traffic <= 0:
            return 0.0
        if self.traffic_share_mode == 'remaining_ratio':
            portion = self.traffic.get_traffic_share(region_id, hour_start, campaign_end)
        else:
            portion = self.traffic.get_traffic_share(region_id, hour_start, hour_start + 3600)
        return float(np.clip(portion / total_traffic, 0.0, 1.0))

    def _resolve_traffic_bucket(
        self,
        region_id: int,
        campaign_start: int,
        campaign_end: int,
        curr_time: int,
    ) -> int:
        feature = self._compute_traffic_share_feature(
            region_id=region_id,
            campaign_start=campaign_start,
            campaign_end=campaign_end,
            hour_start=curr_time,
        )
        return _to_traffic_bucket(feature, self.traffic_share_bins)

    def _build_traffic_aware_training_stats(
        self,
        stats_df: pd.DataFrame,
        campaign_df: pd.DataFrame,
    ) -> pd.DataFrame:
        campaigns_meta = campaign_df[['campaign_id', 'region_id', 'campaign_start', 'campaign_end']].drop_duplicates(
            subset=['campaign_id']
        )
        joined = stats_df.merge(campaigns_meta, on='campaign_id', how='left')

        traffic_features = []
        for row in joined.itertuples(index=False):
            traffic_features.append(
                self._compute_traffic_share_feature(
                    region_id=int(row.region_id),
                    campaign_start=int(row.campaign_start),
                    campaign_end=int(row.campaign_end),
                    hour_start=int(row.period),
                )
            )

        joined['traffic_share_feature'] = traffic_features
        joined['traffic_bucket'] = joined['traffic_share_feature'].apply(
            lambda x: _to_traffic_bucket(x, self.traffic_share_bins)
        )
        return joined

    def _build_traffic_transition_matrix(self, campaign_df: pd.DataFrame, campaign_ids: np.ndarray) -> np.ndarray:
        transitions = np.zeros((self.traffic_share_bins, self.traffic_share_bins), dtype=float)
        train_campaigns = campaign_df[campaign_df['campaign_id'].isin(campaign_ids)]

        for campaign in train_campaigns.itertuples(index=False):
            hour = int(campaign.campaign_start)
            campaign_end = int(campaign.campaign_end)
            region_id = int(campaign.region_id)
            while hour + 3600 < campaign_end:
                traffic_bucket = self._resolve_traffic_bucket(
                    region_id=region_id,
                    campaign_start=int(campaign.campaign_start),
                    campaign_end=campaign_end,
                    curr_time=hour,
                )
                traffic_bucket_next = self._resolve_traffic_bucket(
                    region_id=region_id,
                    campaign_start=int(campaign.campaign_start),
                    campaign_end=campaign_end,
                    curr_time=hour + 3600,
                )
                transitions[traffic_bucket, traffic_bucket_next] += 1.0
                hour += 3600

        for traffic_bucket in range(self.traffic_share_bins):
            total = transitions[traffic_bucket].sum()
            if total > 0:
                transitions[traffic_bucket] = transitions[traffic_bucket] / total
            else:
                transitions[traffic_bucket, traffic_bucket] = 1.0

        return transitions

    def fit(
        self,
        stats_df: pd.DataFrame,
        campaign_path: str = None,
        N: int = None,
        B_max: int = None,
        B_step: int = 100,
        max_bin: int = 60,
        objective: str = 'clicks',
    ):
        """Train value function and policy on stats_df."""
        N_bound = N or self.N_bound
        B_bound = B_max or self.B_bound
        campaign_df = pd.read_csv(campaign_path) if campaign_path else None

        if campaign_df is not None:
            avg_budget = campaign_df['auction_budget'].mean()
            B_bound = int(avg_budget * 3)

        if self.use_traffic_share_state:
            campaign_df = pd.read_csv(campaign_path)
            traffic_stats_df = self._build_traffic_aware_training_stats(stats_df=stats_df, campaign_df=campaign_df)
            self.transition_matrix = self._build_traffic_transition_matrix(
                campaign_df=campaign_df,
                campaign_ids=traffic_stats_df['campaign_id'].unique(),
            )
            self.value_table, self.policy_table = _train_value_function_with_traffic_state(
                stats_df=traffic_stats_df,
                transition_matrix=self.transition_matrix,
                traffic_share_bins=self.traffic_share_bins,
                N=N_bound,
                B_max=B_bound,
                B_step=B_step,
                max_bin=max_bin,
                gamma=self.gamma,
                objective=objective,
            )
        else:
            self.value_table, self.policy_table = _train_value_function(
                stats_df=stats_df,
                N=N_bound,
                B_max=B_bound,
                B_step=B_step,
                max_bin=max_bin,
                gamma=self.gamma,
                objective=objective,
            )

        self.N_bound = N_bound
        self.B_bound = B_bound
        self.policy_max_bin = int(max_bin)
        return self

    def save_model(self, path: str):
        """Save trained model to disk."""
        data = {
            'type': 'table',
            'table': self.value_table,
            'policy_table': self.policy_table,
            'N_bound': self.N_bound,
            'B_bound': self.B_bound,
            'gamma': self.gamma,
            'policy_max_bin': self.policy_max_bin,
            'use_traffic_share_state': self.use_traffic_share_state,
            'traffic_share_mode': self.traffic_share_mode,
            'traffic_share_bins': self.traffic_share_bins,
            'table_shape': None if self.value_table is None else tuple(self.value_table.shape),
            'policy_shape': None if self.policy_table is None else tuple(self.policy_table.shape),
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load_model(self, model_path: str):
        """Load trained value function (and policy if present) from file."""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)

            self.model_type = model_data.get('type', 'table')

            if self.model_type == 'table':
                self.value_table = model_data['table']
                self.policy_table = model_data.get('policy_table')
                self.N_bound = model_data.get('N_bound', self.N_bound)
                self.B_bound = model_data.get('B_bound', self.B_bound)
                self.policy_max_bin = int(model_data.get('policy_max_bin', self.policy_max_bin))
                if 'use_traffic_share_state' in model_data:
                    self.use_traffic_share_state = bool(model_data.get('use_traffic_share_state'))
                else:
                    self.use_traffic_share_state = bool(
                        self.policy_table is not None and len(self.policy_table.shape) == 3
                    )
                self.traffic_share_mode = model_data.get('traffic_share_mode', self.traffic_share_mode)
                self.traffic_share_bins = int(model_data.get('traffic_share_bins', self.traffic_share_bins))
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        except Exception as e:
            print(f"Warning: Could not load model from {model_path}: {e}")
            self.value_table = None
            self.policy_table = None

    def get_value(self, n: int, b: int, traffic_bucket: Optional[int] = None) -> float:
        """
        Get value function D(n, b).

        Returns expected total clicks for optimally bidding
        n remaining hours with budget b.
        """
        if n <= 0 or b <= 0:
            return 0.0

        if self.value_table is None:
            raise Exception("No model provided")

        if n > self.N_bound:
            b_scaled = int(b * self.N_bound / n)
            n_scaled = self.N_bound
            return self.get_value(n_scaled, b_scaled, traffic_bucket=traffic_bucket)

        if b > self.B_bound:
            n_scaled = int(n * self.B_bound / b)
            b_scaled = self.B_bound
            return self.get_value(n_scaled, b_scaled, traffic_bucket=traffic_bucket)

        try:
            if self.use_traffic_share_state and len(self.value_table.shape) == 3:
                traffic_bucket_use = 0 if traffic_bucket is None else int(
                    np.clip(traffic_bucket, 0, self.traffic_share_bins - 1)
                )
                return max(0.0, self.value_table[n][b][traffic_bucket_use])
            return max(0.0, self.value_table[n][b])
        except (IndexError, KeyError):
            return 0.0

    def compute_optimal_bid(self, n: int, b: int, traffic_bucket: Optional[int] = None) -> float:
        """
        Compute optimal bid for current state (n, b).
        Uses policy table if available (from fit), else heuristic grid search.
        """
        if n <= 0 or b <= 0:
            return 0.0

        if self.value_table is None:
            return min(b / n, self.max_bid)

        if self.policy_table is not None:
            n_use = min(n, self.N_bound)
            b_use = min(b, self.B_bound)
            if n > self.N_bound:
                b_use = min(int(b * self.N_bound / n), self.B_bound)
                n_use = self.N_bound
            elif b > self.B_bound:
                n_use = min(int(n * self.B_bound / b), self.N_bound)
                b_use = self.B_bound
            n_use = min(max(0, n_use), self.N_bound)
            b_use = min(max(0, b_use), self.B_bound)

            if self.use_traffic_share_state and len(self.policy_table.shape) == 3:
                traffic_bucket_use = 0 if traffic_bucket is None else int(
                    np.clip(traffic_bucket, 0, self.traffic_share_bins - 1)
                )
                target_bin = int(self.policy_table[n_use][b_use][traffic_bucket_use])
            else:
                target_bin = int(self.policy_table[n_use][b_use])

            return float(min(bin2price(target_bin), self.max_bid, b))

        best_bid = b / n
        best_value = float('-inf')

        bid_candidates = np.linspace(
            max(10, b / (n * 2)),
            min(b, self.max_bid, b / max(1, n - 1)),
            20,
        )
        for bid in bid_candidates:
            if bid > b:
                continue
            immediate_clicks = (bid * self.avg_clicks_per_dollar) if self.avg_clicks_per_dollar else 0
            total_value = immediate_clicks + self.gamma * self.get_value(
                n - 1,
                int(b - bid),
                traffic_bucket=traffic_bucket,
            )
            if total_value > best_value:
                best_value, best_bid = total_value, bid
        return float(best_bid)

    def place_bid(self, bidding_input_params: Dict[str, any], history: History) -> float:
        """
        Place bid for the next hour based on current state.

        State:
            n: remaining hours until campaign end
            b: remaining budget

        Returns:
            Bid amount (determines which price bins we win)
        """
        campaign_start = bidding_input_params['campaign_start_time']
        campaign_end = bidding_input_params['campaign_end_time']
        curr_time = bidding_input_params['curr_time']
        balance = bidding_input_params['balance']
        initial_balance = bidding_input_params['initial_balance']
        prev_bid = bidding_input_params.get('prev_bid', 0)

        if self.campaign_hours is None:
            self.campaign_hours = (campaign_end - campaign_start) / 3600

        if len(history.rows) == 0:
            avg_hourly_budget = initial_balance / max(1, self.campaign_hours)
            return max(10.0, avg_hourly_budget * 0.3)

        self._update_performance_estimate(history, initial_balance, balance)

        time_remaining = max(0, campaign_end - curr_time)
        hours_remaining = max(1, time_remaining / 3600)

        n = int(np.ceil(hours_remaining))
        b = int(balance)

        traffic_bucket = None
        if self.use_traffic_share_state:
            traffic_bucket = self._resolve_traffic_bucket(
                region_id=int(bidding_input_params['region_id']),
                campaign_start=int(campaign_start),
                campaign_end=int(campaign_end),
                curr_time=int(curr_time),
            )

        bid = self.compute_optimal_bid(n, b, traffic_bucket=traffic_bucket)

        prev_bin = price2bin(prev_bid)
        bin_ = price2bin(bid)
        bin_ = np.clip(bin_, prev_bin - self.lower_clip, prev_bin + self.upper_clip)
        bid = bin2price(bin_)

        return float(bid)

    def _update_performance_estimate(self, history: History, initial_balance: float, current_balance: float):
        """Update estimate of clicks per dollar based on history."""
        if len(history.rows) < 2:
            return

        df = history.to_data_frame()

        total_clicks = df['clicks'].iloc[-1] if len(df) > 0 else 0
        total_spend = initial_balance - current_balance

        if total_spend > 0:
            self.avg_clicks_per_dollar = total_clicks / total_spend
        else:
            self.avg_clicks_per_dollar = None

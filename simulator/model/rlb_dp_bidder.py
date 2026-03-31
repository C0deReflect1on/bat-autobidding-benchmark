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

from simulator.model.bidder import _Bidder
from simulator.simulation.modules import History
from simulator.simulation.utils import bin2price


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
            b = b_idx * B_step
            if b == 0:
                continue
            best_value, best_target_bin = 0.0, 0
            for target_bin in range(max_bin + 1):
                # can optimize here redundant re-sum
                total_reward = sum(surplus_stats.get(i, 0) for i in range(target_bin + 1))
                total_cost = sum(cost_stats.get(i, 0) for i in range(target_bin + 1))

                if total_cost > b:
                    continue

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
        'gamma': 1.0,  # Discount factor (typically 1.0 for undiscounted)
        'model_path': None,
        'N_bound': 100,  # Max hours for normalization
        'B_bound': 10000,  # Max budget for normalization
        'use_smoothing': False,
    }
    
    def __init__(self, params: Optional[Dict] = None):
        super().__init__()
        
        params = params or {}
        self.max_bid = params.get('max_bid', self.default_params['max_bid'])
        self.gamma = params.get('gamma', self.default_params['gamma'])
        self.N_bound = params.get('N_bound', self.default_params['N_bound'])
        self.B_bound = params.get('B_bound', self.default_params['B_bound'])
        self.use_smoothing = params.get('use_smoothing', self.default_params['use_smoothing'])
        
        # Value function and policy (trained offline)
        self.value_table = None
        self.policy_table = None
        self.model_type = None # ?

        # Load model if provided
        model_path = params.get('model_path')
        if model_path:
            self.load_model(model_path)
        
        # Campaign state tracking
        self.campaign_hours = None
        self.avg_clicks_per_dollar = None
    
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
        if campaign_path:
            campaign_df = pd.read_csv(campaign_path)
            avg_budget = campaign_df['auction_budget'].mean()
            B_bound = int(avg_budget * 3) # WTF?

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
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        except Exception as e:
            print(f"Warning: Could not load model from {model_path}: {e}")
            self.value_table = None
            self.policy_table = None
    
    def get_value(self, n: int, b: int) -> float:
        """
        Get value function D(n, b).
        
        Returns expected total clicks for optimally bidding 
        n remaining hours with budget b.
        """
        if n <= 0 or b <= 0:
            return 0.0
        
        if self.value_table is None:
            raise Exception("No model provided") # think how to change to print/log
            # No model: simple heuristic based on historical performance
            if self.avg_clicks_per_dollar is not None:
                return b * self.avg_clicks_per_dollar
            return 0.0
        
        # Normalize state if out of bounds
        if n > self.N_bound:
            # Scale down: if we have more hours than trained,
            # estimate by scaling budget proportionally
            b_scaled = int(b * self.N_bound / n)
            n_scaled = self.N_bound
            return self.get_value(n_scaled, b_scaled)
        
        if b > self.B_bound:
            # Scale down: if we have more budget than trained,
            # estimate by scaling hours proportionally
            n_scaled = int(n * self.B_bound / b)
            b_scaled = self.B_bound
            return self.get_value(n_scaled, b_scaled)
        
        # Lookup in table
        try:
            return max(0.0, self.value_table[n][b])
        except (IndexError, KeyError):
            return 0.0
    
    def compute_optimal_bid(self, n: int, b: int) -> float:
        """
        Compute optimal bid for current state (n, b).
        Uses policy table if available (from fit), else heuristic grid search.
        """
        if n <= 0 or b <= 0:
            return 0.0

        if self.value_table is None:
            return min(b / n, self.max_bid)

        # Use trained policy if available
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
            target_bin = int(self.policy_table[n_use][b_use])
            return float(min(bin2price(target_bin), self.max_bid, b))

        # Fallback: heuristic grid search (for models saved without policy)
        best_bid = b / n
        best_value = float('-inf')
        
        # Need to dive into that heuristics, why like that?
        # Хорошо, что сейчас полиси по умолчанию выходит, но в целом тут дурка
        bid_candidates = np.linspace(
            max(10, b / (n * 2)),
            min(b, self.max_bid, b / max(1, n - 1)),
            20,
        )
        for bid in bid_candidates:
            if bid > b:
                continue
            immediate_clicks = (bid * self.avg_clicks_per_dollar) if self.avg_clicks_per_dollar else 0
            total_value = immediate_clicks + self.gamma * self.get_value(n - 1, int(b - bid))
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
        # Extract campaign info
        campaign_start = bidding_input_params['campaign_start_time']
        campaign_end = bidding_input_params['campaign_end_time']
        curr_time = bidding_input_params['curr_time']
        balance = bidding_input_params['balance']
        initial_balance = bidding_input_params['initial_balance']
        prev_bid = bidding_input_params.get('prev_bid', 0)
        
        # Calculate campaign duration
        if self.campaign_hours is None:
            self.campaign_hours = (campaign_end - campaign_start) / 3600
        
        # Cold start: conservative initial bid
        if len(history.rows) == 0:
            # Start with 30% of average hourly budget
            avg_hourly_budget = initial_balance / max(1, self.campaign_hours)
            return max(10.0, avg_hourly_budget * 0.3)
        
        # Update historical performance estimate
        self._update_performance_estimate(history, initial_balance, balance)
        
        # Calculate remaining time
        time_remaining = max(0, campaign_end - curr_time)
        hours_remaining = max(1, time_remaining / 3600)
        
        # State for value function
        n = int(np.ceil(hours_remaining))  # Remaining hours
        b = int(balance)  # Remaining budget
        
        # Compute optimal bid
        bid = self.compute_optimal_bid(n, b)
        
        # Smoothing: don't change bid too drastically
        if self.use_smoothing and prev_bid > 0:
            bid = np.clip(bid, prev_bid * 0.5, prev_bid * 1.5)

        
        # Ensure reasonable bounds
        bid = np.clip(bid, 10.0, min(balance, self.max_bid))
        
        return float(bid)
    
    def _update_performance_estimate(self, history: History, initial_balance: float, current_balance: float):
        """Update estimate of clicks per dollar based on history."""
        if len(history.rows) < 2:
            return
        
        df = history.to_data_frame()
        
        # Total clicks achieved so far
        total_clicks = df['clicks'].iloc[-1] if len(df) > 0 else 0
        
        # Total spend so far
        total_spend = initial_balance - current_balance
        
        if total_spend > 0:
            self.avg_clicks_per_dollar = total_clicks / total_spend
        else:
            self.avg_clicks_per_dollar = None

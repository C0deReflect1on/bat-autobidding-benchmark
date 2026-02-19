"""
RLB-DP Training Script for Bat-Autobidding-Benchmark

Trains the RLB-DP bidder's value function via fit() (training lives in rlb_dp_bidder.py).
"""

import sys
sys.path.append("../")

import pandas as pd

from simulator.model.rlb_dp_bidder import RLBDPBidder


def main():
    print("=" * 60)
    print("RLB-DP Training for Bat-Autobidding-Benchmark")
    print("=" * 60)

    campaigns_path = '../data/small_example/fpa/subsample_campaigns.csv'
    stats_path = '../data/small_example/fpa/subsample_stats.csv'
    model_path = '../data/rlb_dp_model.pkl'

    print("=== Loading Data ===")
    stats_df = pd.read_csv(stats_path)
    print(f"Loaded {len(stats_df):,} stat records")

    # Create bidder and train (fit is inside the model now)
    bidder = RLBDPBidder(params={'max_bid': 300})
    bidder.fit(
        stats_df=stats_df,
        campaign_path=campaigns_path,
    )
    print(f"\nFitted: N_bound={bidder.N_bound}, B_bound={bidder.B_bound}")

    # Sample values
    print("\n=== Sample Value Function Values ===")
    for n in [1, 10, 50, 100]:
        for b in [1000, 5000, 10000]:
            if b <= bidder.B_bound and n <= bidder.N_bound:
                v = bidder.value_table[n][b]
                print(f"  V({n:3d} hours, ${b:5d}) = {v:.2f} expected clicks")

    # Sample predictions
    print("\n=== Sample Predictions ===")
    for n, b in [(10, 5000), (50, 10000)]:
        bid = bidder.compute_optimal_bid(n, b)
        print(f"  compute_optimal_bid(n={n}, b={b}) -> bid={bid:.2f}")

    # Save
    bidder.save_model(model_path)
    print(f"\nModel saved to: {model_path}")
    print(f"  Size: {bidder.value_table.nbytes / 1024:.1f} KB")
    print("Training complete!")
    print(f"Next steps:")
    print(f"  1. Load: RLBDPBidder(params={{'model_path': '{model_path}'}})")
    print(f"  2. Test: simulate_campaign(campaign, bidder, ...)")


if __name__ == '__main__':
    main()

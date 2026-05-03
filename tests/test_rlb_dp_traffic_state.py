import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from simulator.model.rlb_dp_bidder import RLBDPBidder
from simulator.simulation.modules import History
from simulator.simulation.utils import bin2price


def _write_traffic_csv(path: Path, region_id: int, shares_by_hour: dict[int, float]) -> None:
    rows = [
        {
            "region_id": region_id,
            "dow": 4,
            "hour": hour,
            "traffic_share": share,
        }
        for hour, share in sorted(shares_by_hour.items())
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def _make_training_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    stats_df = pd.DataFrame(
        [
            {
                "campaign_id": 1,
                "period": 0,
                "contact_price_bin": 0,
                "AuctionClicksSurplus": 1.0,
                "AuctionContactsSurplus": 1.0,
            },
            {
                "campaign_id": 1,
                "period": 0,
                "contact_price_bin": 1,
                "AuctionClicksSurplus": 50.0,
                "AuctionContactsSurplus": 1.0,
            },
            {
                "campaign_id": 1,
                "period": 3600,
                "contact_price_bin": 0,
                "AuctionClicksSurplus": 10.0,
                "AuctionContactsSurplus": 1.0,
            },
            {
                "campaign_id": 1,
                "period": 3600,
                "contact_price_bin": 1,
                "AuctionClicksSurplus": 1.0,
                "AuctionContactsSurplus": 100.0,
            },
        ]
    )
    campaign_df = pd.DataFrame(
        [
            {
                "campaign_id": 1,
                "region_id": 123,
                "campaign_start": 0,
                "campaign_end": 7200,
                "auction_budget": 100.0,
            }
        ]
    )
    return stats_df, campaign_df


class TestRLBDPTrafficState(unittest.TestCase):
    def test_fit_without_traffic_state_keeps_2d_tables(self):
        stats_df = pd.DataFrame(
            [
                {
                    "campaign_id": 1,
                    "period": 0,
                    "contact_price_bin": 0,
                    "AuctionClicksSurplus": 1.0,
                    "AuctionContactsSurplus": 1.0,
                },
                {
                    "campaign_id": 1,
                    "period": 0,
                    "contact_price_bin": 1,
                    "AuctionClicksSurplus": 2.0,
                    "AuctionContactsSurplus": 1.0,
                },
            ]
        )
        bidder = RLBDPBidder({"use_traffic_share_state": False})
        bidder.fit(stats_df=stats_df, N=2, B_max=20, B_step=10, max_bin=1)

        self.assertEqual(len(bidder.value_table.shape), 2)
        self.assertEqual(len(bidder.policy_table.shape), 2)
        self.assertEqual(bidder.value_table.shape, (3, 21))
        self.assertEqual(bidder.policy_table.shape, (3, 21))

        bid = bidder.compute_optimal_bid(n=1, b=10)
        self.assertGreaterEqual(bid, 0.0)

    def test_fit_with_traffic_state_builds_3d_and_uses_traffic_bucket(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            traffic_path = tmp / "traffic.csv"
            campaigns_path = tmp / "campaigns.csv"
            _write_traffic_csv(traffic_path, region_id=123, shares_by_hour={3: 0.8, 4: 0.2})
            stats_df, campaign_df = _make_training_frames()
            campaign_df.to_csv(campaigns_path, index=False)

            bidder = RLBDPBidder(
                {
                    "use_traffic_share_state": True,
                    "traffic_share_mode": "next_hour",
                    "traffic_share_bins": 2,
                    "traffic_share_path": str(traffic_path),
                }
            )
            bidder.fit(
                stats_df=stats_df,
                campaign_path=str(campaigns_path),
                N=2,
                B_max=100,
                B_step=10,
                max_bin=1,
            )

            self.assertEqual(len(bidder.value_table.shape), 3)
            self.assertEqual(len(bidder.policy_table.shape), 3)
            self.assertEqual(bidder.value_table.shape[0], 3)
            self.assertEqual(bidder.value_table.shape[2], 2)

            bid_bucket_0 = bidder.compute_optimal_bid(n=1, b=100, traffic_bucket=0)
            bid_bucket_1 = bidder.compute_optimal_bid(n=1, b=100, traffic_bucket=1)
            self.assertEqual(bid_bucket_0, float(bin2price(0)))
            self.assertEqual(bid_bucket_1, float(bin2price(1)))

            history = History()
            history.rows.append({"stub": 1})

            captured = {}

            def _capture_compute(n, b, traffic_bucket=None):
                captured["n"] = n
                captured["b"] = b
                captured["traffic_bucket"] = traffic_bucket
                return 12.0

            bidder.compute_optimal_bid = _capture_compute
            bid = bidder.place_bid(
                {
                    "campaign_start_time": 0,
                    "campaign_end_time": 7200,
                    "curr_time": 0,
                    "balance": 100.0,
                    "initial_balance": 100.0,
                    "prev_bid": 0.0,
                    "region_id": 123,
                },
                history,
            )

            self.assertEqual(captured["traffic_bucket"], 1)
            self.assertGreater(bid, 0.0)

    def test_traffic_share_feature_formulas(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            traffic_path = tmp / "traffic.csv"
            _write_traffic_csv(traffic_path, region_id=123, shares_by_hour={3: 0.25, 4: 0.75})

            next_hour_bidder = RLBDPBidder(
                {
                    "use_traffic_share_state": True,
                    "traffic_share_mode": "next_hour",
                    "traffic_share_bins": 20,
                    "traffic_share_path": str(traffic_path),
                }
            )
            remaining_bidder = RLBDPBidder(
                {
                    "use_traffic_share_state": True,
                    "traffic_share_mode": "remaining_ratio",
                    "traffic_share_bins": 20,
                    "traffic_share_path": str(traffic_path),
                }
            )

            next_hour_h0 = next_hour_bidder._compute_traffic_share_feature(123, 0, 7200, 0)
            next_hour_h1 = next_hour_bidder._compute_traffic_share_feature(123, 0, 7200, 3600)
            remaining_h0 = remaining_bidder._compute_traffic_share_feature(123, 0, 7200, 0)
            remaining_h1 = remaining_bidder._compute_traffic_share_feature(123, 0, 7200, 3600)

            self.assertAlmostEqual(next_hour_h0, 0.25, places=6)
            self.assertAlmostEqual(next_hour_h1, 0.75, places=6)
            self.assertAlmostEqual(remaining_h0, 1.0, places=6)
            self.assertAlmostEqual(remaining_h1, 0.75, places=6)

    def test_save_load_3d_and_old_2d_compatibility(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            traffic_path = tmp / "traffic.csv"
            campaigns_path = tmp / "campaigns.csv"
            model_path = tmp / "model.pkl"
            old_model_path = tmp / "old_model.pkl"

            _write_traffic_csv(traffic_path, region_id=123, shares_by_hour={3: 0.8, 4: 0.2})
            stats_df, campaign_df = _make_training_frames()
            campaign_df.to_csv(campaigns_path, index=False)

            bidder = RLBDPBidder(
                {
                    "use_traffic_share_state": True,
                    "traffic_share_mode": "next_hour",
                    "traffic_share_bins": 2,
                    "traffic_share_path": str(traffic_path),
                }
            )
            bidder.fit(
                stats_df=stats_df,
                campaign_path=str(campaigns_path),
                N=2,
                B_max=100,
                B_step=10,
                max_bin=1,
            )
            bidder.save_model(str(model_path))

            loaded = RLBDPBidder(
                {
                    "use_traffic_share_state": False,
                    "traffic_share_path": str(traffic_path),
                }
            )
            loaded.load_model(str(model_path))

            self.assertTrue(loaded.use_traffic_share_state)
            self.assertEqual(loaded.traffic_share_mode, "next_hour")
            self.assertEqual(loaded.traffic_share_bins, 2)
            self.assertEqual(len(loaded.policy_table.shape), 3)

            old_model = {
                "type": "table",
                "table": np.zeros((2, 5), dtype=float),
                "policy_table": np.zeros((2, 5), dtype=np.int32),
                "N_bound": 1,
                "B_bound": 4,
                "gamma": 1.0,
                "policy_max_bin": 1,
            }
            with open(old_model_path, "wb") as f:
                pickle.dump(old_model, f)

            loaded_old = RLBDPBidder(
                {
                    "use_traffic_share_state": True,
                    "traffic_share_path": str(traffic_path),
                }
            )
            loaded_old.load_model(str(old_model_path))

            self.assertFalse(loaded_old.use_traffic_share_state)
            self.assertEqual(len(loaded_old.policy_table.shape), 2)

    def test_run_rlb_experiment_inprocess_smoke_with_traffic_state_toggle(self):
        try:
            from example_notebooks.experiments.adapters.rlb_adapter import run_rlb_experiment_inprocess
            from example_notebooks.experiments.base_exp_config import ExperimentConfig
            from example_notebooks.experiments.infra.split_utils import resolve_normalized_splits
        except ModuleNotFoundError as exc:
            self.skipTest(f"RLB experiment smoke import skipped: {exc}")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            traffic_path = root / "traffic.csv"
            _write_traffic_csv(traffic_path, region_id=123, shares_by_hour={3: 0.8, 4: 0.2})

            campaigns_df = pd.DataFrame(
                [
                    {
                        "campaign_id": 1,
                        "region_id": 123,
                        "campaign_start": 0,
                        "campaign_end": 7200,
                        "auction_budget": 100.0,
                    }
                ]
            )
            stats_df = pd.DataFrame(
                [
                    {
                        "campaign_id": 1,
                        "period": 0,
                        "contact_price_bin": 0,
                        "AuctionClicksSurplus": 1.0,
                        "AuctionContactsSurplus": 1.0,
                    },
                    {
                        "campaign_id": 1,
                        "period": 3600,
                        "contact_price_bin": 1,
                        "AuctionClicksSurplus": 2.0,
                        "AuctionContactsSurplus": 1.0,
                    },
                ]
            )

            data_config = {}
            for split in ("train", "val", "holdout"):
                campaigns_path = root / f"{split}_campaigns.csv"
                stats_path = root / f"{split}_stats.csv"
                campaigns_df.to_csv(campaigns_path, index=False)
                stats_df.to_csv(stats_path, index=False)
                data_config[split if split != "holdout" else "test_holdout"] = {
                    "campaigns_path": str(campaigns_path),
                    "stats_path": str(stats_path),
                }

            def fake_check(*args, **kwargs):
                return {
                    "score": (1.0, 2.0, 3.0, 4.0),
                    "skipped_campaigns": 0,
                    "time_inference_sec": 0.01,
                    "time_overall_sec": 0.02,
                }

            for use_traffic_share_state in (False, True):
                cfg = ExperimentConfig(
                    experiment_name=f"rlb_smoke_{int(use_traffic_share_state)}",
                    run_name=f"rlb_smoke_{int(use_traffic_share_state)}",
                    n_trials=1,
                    random_seed=42,
                    auction_mode="FPA",
                    metric="SCR",
                    family="rlb",
                    data_config=data_config,
                    model_config={
                        "base_params": {
                            "max_bid": 100,
                            "gamma": 1.0,
                            "N_bound": 4,
                            "B_bound": 200,
                            "use_traffic_share_state": use_traffic_share_state,
                            "traffic_share_mode": "next_hour",
                            "traffic_share_bins": 20,
                            "traffic_share_path": str(traffic_path),
                        }
                    },
                    experiments_data_dir=root,
                )
                normalized = resolve_normalized_splits(cfg)
                with patch("example_notebooks.experiments.adapters.rlb_adapter.autobidder_check", side_effect=fake_check):
                    result = run_rlb_experiment_inprocess(
                        cfg,
                        normalized,
                        search_space_fn=lambda trial: {},
                        n_trials=1,
                    )
                self.assertEqual(result["summary"]["final_holdout"]["metrics"]["clicks_sum"], 3.0)


if __name__ == "__main__":
    unittest.main()

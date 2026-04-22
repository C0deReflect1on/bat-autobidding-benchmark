import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pandas as pd

missing = []
if find_spec("torch") is None:
    missing.append("torch")
if find_spec("matplotlib") is None:
    missing.append("matplotlib")
if missing:
    raise unittest.SkipTest(f"Missing runtime deps for DRLB smoke tests: {', '.join(missing)}")

import torch

from simulator.model.drlb.config_types import DrlbConfigParser
from simulator.model.drlb.replay_buffer import (
    QTransition,
    RTransition,
    ReplayBuffer,
    collate_q_transitions,
    collate_reward_transitions,
)
from simulator.model.drlb_bidder import DRLBBidder
from simulator.simulation.modules import History
from simulator.validation.check_results import autobidder_check


def _make_stats_df(campaign_id: int = 1) -> pd.DataFrame:
    rows = []
    for period in (0, 3600):
        for contact_price_bin, ctr, spend, vis, clicks, contacts in (
            (10, 0.01, 1.0, 10.0, 1.0, 0.2),
            (20, 0.02, 2.0, 20.0, 2.0, 0.4),
        ):
            rows.append(
                {
                    "campaign_id": campaign_id,
                    "period": period,
                    "contact_price_bin": contact_price_bin,
                    "CTRPredicts": ctr,
                    "CRPredicts": 0.1,
                    "AuctionWinBidSurplus": spend,
                    "AuctionVisibilitySurplus": vis,
                    "AuctionClicksSurplus": clicks,
                    "AuctionContactsSurplus": contacts,
                }
            )
    return pd.DataFrame(rows)


def _make_train_campaigns_df(campaign_id: int = 1) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "campaign_id": campaign_id,
                "campaign_start": 0,
                "campaign_end": 7200,
                "auction_budget": 10.0,
            }
        ]
    )


def _make_eval_campaigns_df(campaign_id: int = 1) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "item_id": 0,
                "campaign_id": campaign_id,
                "loc_id": 0,
                "region_id": 0,
                "logical_category": "smoke",
                "microcat_ext": 0,
                "campaign_start": 0,
                "campaign_end": 7200,
                "auction_budget": 10.0,
            }
        ]
    )


class TestDrlbApiSmoke(unittest.TestCase):
    def test_config_parser_allows_extra_fields(self):
        cfg = DrlbConfigParser.from_dict(
            {
                "state_type": "improved",
                "T": 8,
                "max_bid": 42.0,
                "unused_key": "ignored",
            }
        )
        self.assertEqual(cfg.model.T, 8)
        self.assertEqual(cfg.runtime.max_bid, 42.0)
        self.assertEqual(len(cfg.model.lambda_action_betas), 7)

    def test_lambda_action_betas_resizes_dqn(self):
        bidder = DRLBBidder(
            {
                "state_type": "improved",
                "lambda_action_betas": [-0.1, 0.0, 0.1],
                "use_tqdm": False,
                "verbose": False,
                "debug_logs": False,
            }
        )
        self.assertEqual(bidder.agent.dqn_agent.action_size, 3)
        self.assertEqual(bidder.agent.BETA, [-0.1, 0.0, 0.1])

    def test_replay_buffer_sample_shapes(self):
        q_buffer = ReplayBuffer(
            buffer_size=10,
            batch_size=2,
            seed=0,
            collate_fn=collate_q_transitions,
        )
        q_buffer.add(QTransition(np.array([1, 2], dtype=np.float32), 0, 1.0, np.array([3, 4], dtype=np.float32), False))
        q_buffer.add(QTransition(np.array([5, 6], dtype=np.float32), 1, 0.5, np.array([7, 8], dtype=np.float32), True))
        states, actions, rewards, next_states, dones = q_buffer.sample()
        self.assertEqual(tuple(states.shape), (2, 2))
        self.assertEqual(tuple(actions.shape), (2, 1))
        self.assertEqual(tuple(rewards.shape), (2, 1))
        self.assertEqual(tuple(next_states.shape), (2, 2))
        self.assertEqual(tuple(dones.shape), (2, 1))

        r_buffer = ReplayBuffer(
            buffer_size=10,
            batch_size=2,
            seed=0,
            collate_fn=collate_reward_transitions,
        )
        r_buffer.add(RTransition(np.array([1, 2, 3], dtype=np.float32), np.array([0.5], dtype=np.float32)))
        r_buffer.add(RTransition(np.array([4, 5, 6], dtype=np.float32), np.array([1.5], dtype=np.float32)))
        state_actions, rewards = r_buffer.sample()
        self.assertEqual(tuple(state_actions.shape), (2, 3))
        self.assertEqual(tuple(rewards.shape), (2, 1))

    def test_tiny_fit_save_load_and_place_bid(self):
        stats_df = _make_stats_df()
        campaigns_df = _make_train_campaigns_df()

        bidder = DRLBBidder(
            {
                "state_type": "improved",
                "use_tqdm": False,
                "verbose": False,
                "debug_logs": False,
            }
        )
        bidder.fit(stats_df, campaigns_df=campaigns_df, max_steps=2, objective="clicks")

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "smoke.pt"
            bidder.save_model(str(model_path))

            loaded = DRLBBidder({"model_path": str(model_path), "use_tqdm": False})
            bid = loaded.place_bid(
                bidding_input_params={
                    "campaign_id": 1,
                    "campaign_start_time": 0,
                    "campaign_end_time": 7200,
                    "curr_time": 0,
                    "balance": 10.0,
                    "initial_balance": 10.0,
                    "ctr_pred": 0.01,
                    "prev_bid": 1.0,
                },
                history=History(),
            )
            self.assertGreaterEqual(bid, 0.0)

    def test_checkpoint_config_round_trips_through_parser(self):
        bidder = DRLBBidder(
            {
                "state_type": "improved",
                "lambda_action_betas": [-0.1, 0.0, 0.1],
                "max_bid": 42.0,
                "use_tqdm": False,
                "verbose": False,
                "debug_logs": False,
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "roundtrip.pt"
            bidder.save_model(str(model_path))

            payload = torch.load(model_path, map_location="cpu", weights_only=False)
            cfg = DrlbConfigParser.from_checkpoint(payload)

        self.assertEqual(cfg.model.state_type, "improved")
        self.assertEqual(cfg.model.lambda_action_betas, (-0.1, 0.0, 0.1))
        self.assertEqual(cfg.runtime.max_bid, 42.0)

    def test_fit_records_train_prior_lambda_init(self):
        stats_df = _make_stats_df()
        campaigns_df = _make_train_campaigns_df()

        bidder = DRLBBidder({"use_tqdm": False, "verbose": False, "debug_logs": False})
        bidder.fit(stats_df, campaigns_df=campaigns_df, max_steps=1, objective="clicks")

        self.assertIsNotNone(bidder.train_prior_lambda_init)
        self.assertAlmostEqual(bidder.train_prior_lambda_init, 0.005, places=6)

    def test_runtime_obs_prefers_ctr_pred(self):
        bidder = DRLBBidder({"use_tqdm": False, "verbose": False, "debug_logs": False})
        bidding_input_params = {
            "campaign_start_time": 0,
            "campaign_end_time": 7200,
            "curr_time": 0,
            "ctr_pred": 0.25,
            "prev_ctr": 0.01,
        }
        start_time = float(bidding_input_params.get("campaign_start_time", 0.0))
        end_time = float(bidding_input_params.get("campaign_end_time", start_time + 3600.0))
        ctr_pred = float(
            bidding_input_params.get("ctr_pred", bidding_input_params.get("prev_ctr", 0.0))
        )
        obs = bidder._build_agent_obs(
            time_step_index=float(bidder._hour_index(bidding_input_params)),
            ctr_pred=ctr_pred,
            start_time=start_time,
            end_time=end_time,
            curr_time=float(bidding_input_params.get("curr_time", start_time)),
        )
        self.assertAlmostEqual(obs["ctr"], 0.25, places=6)
        self.assertAlmostEqual(obs["ctr_pred"], 0.25, places=6)
        self.assertAlmostEqual(obs["balance"], 10000.0, places=6)
        self.assertAlmostEqual(obs["initialBalance"], 10000.0, places=6)

    def test_sync_runtime_context_updates_budget_and_meta(self):
        bidder = DRLBBidder({"use_tqdm": False, "verbose": False, "debug_logs": False})
        agent = bidder.agent

        agent.sync_runtime_context(
            balance=4.0,
            initial_budget=10.0,
            elapsed_time_ratio=1.5,
        )

        self.assertAlmostEqual(agent.budget, 10.0, places=6)
        self.assertAlmostEqual(agent.rem_budget, 4.0, places=6)
        self.assertAlmostEqual(agent.rem_budget_ratio, 0.4, places=6)
        self.assertAlmostEqual(agent.elapsed_time_ratio, 1.0, places=6)

    def test_improved_and_scaled_budget_share_common_ratio_updates(self):
        common_fields = {
            "wins_t": 2.0,
            "cost_t": 12.0,
            "reward_t": 6.0,
            "imp_opps_t": 3.0,
            "ROL": 5.0,
            "episode_steps_total": 10.0,
            "rem_budget": 40.0,
            "budget": 100.0,
        }

        improved_agent = DRLBBidder(
            {"state_type": "improved", "use_tqdm": False, "verbose": False, "debug_logs": False}
        ).agent
        scaled_agent = DRLBBidder(
            {"state_type": "scaled_budget", "use_tqdm": False, "verbose": False, "debug_logs": False}
        ).agent

        for k, v in common_fields.items():
            setattr(improved_agent, k, v)
            setattr(scaled_agent, k, v)

        improved_agent.state_repr.compute_step_metrics(improved_agent)
        scaled_agent.state_repr.compute_step_metrics(scaled_agent)

        self.assertAlmostEqual(improved_agent.rewards_prev_t_ratio, scaled_agent.rewards_prev_t_ratio, places=6)
        self.assertAlmostEqual(improved_agent.ROL_ratio, scaled_agent.ROL_ratio, places=6)
        self.assertAlmostEqual(improved_agent.rem_budget_ratio, scaled_agent.rem_budget_ratio, places=6)
        self.assertAlmostEqual(improved_agent.CPI, scaled_agent.CPI, places=6)

    def test_ingest_history_uses_explicit_reward_field(self):
        bidder = DRLBBidder({"use_tqdm": False, "verbose": False, "debug_logs": False})
        bidder.agent.reset_episode()

        history = History()
        history.rows.append(
            {
                "bid": 1.0,
                "spend_history": 2.0,
                "clicks_history": 5.0,
                "contacts_history": 1.5,
            }
        )
        bidder._ingest_history(history, reward_field="contacts_history")

        self.assertAlmostEqual(bidder.agent.reward_t, 1.5, places=6)
        self.assertAlmostEqual(bidder.agent.cost_t, 2.0, places=6)

    def test_autobidder_check_tiny_smoke(self):
        stats_df = _make_stats_df()
        eval_campaigns_df = _make_eval_campaigns_df()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            eval_campaigns_path = tmpdir_path / "campaigns.csv"
            eval_stats_path = tmpdir_path / "stats.csv"
            eval_campaigns_df.to_csv(eval_campaigns_path, index=False)
            stats_df.to_csv(eval_stats_path, index=False)

            result = autobidder_check(
                bidder=DRLBBidder,
                params={
                    "input_campaigns": str(eval_campaigns_path),
                    "input_stats": str(eval_stats_path),
                    "state_type": "improved",
                    "objective": "clicks",
                    "eval_mode": True,
                    "use_tqdm": False,
                    "verbose": False,
                    "debug_logs": False,
                },
                auction_mode="VCG",
                verbose=False,
            )
            self.assertIn("score", result)
            self.assertEqual(len(result["score"]), 4)


if __name__ == "__main__":
    unittest.main()

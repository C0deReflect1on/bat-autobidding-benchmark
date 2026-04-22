import json
import pickle
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from example_notebooks.experiments.adapters.baseline_adapter import (
    run_baseline_experiment,
    run_baseline_experiment_inprocess,
)
from example_notebooks.experiments.adapters.drlb_adapter import (
    plot_training_diagnostics,
    run_drlb_experiment,
    run_drlb_experiment_inprocess,
    write_training_diagnostics_artifacts,
)
from example_notebooks.experiments.adapters.rlb_adapter import run_rlb_experiment_inprocess
from example_notebooks.experiments.all_comparison.loader import (
    assert_matching_split_fingerprint,
    load_metrics_table,
)
from example_notebooks.experiments.base_exp_config import ExperimentConfig
from example_notebooks.experiments.infra.split_utils import (
    build_split_manifest,
    resolve_normalized_splits,
    split_fingerprint,
)
from example_notebooks.experiments.notebook_api import (
    run_drlb_profile_inprocess,
    run_profile_inprocess,
)


def _write_split_files(root: Path, prefix: str) -> dict[str, str]:
    campaigns_path = root / f"{prefix}_campaigns.csv"
    stats_path = root / f"{prefix}_stats.csv"
    pd.DataFrame([{"campaign_id": 1}]).to_csv(campaigns_path, index=False)
    pd.DataFrame([{"campaign_id": 1, "period": 0}]).to_csv(stats_path, index=False)
    return {
        "campaigns_path": str(campaigns_path),
        "stats_path": str(stats_path),
    }


class _FakeBaselineTrial:
    def __init__(self, number: int, value: float, params: dict):
        self.number = number
        self.value = value
        self.params = params


class _FakeBaselineStudy:
    def __init__(self, params: dict):
        self.best_trial = _FakeBaselineTrial(0, 3.0, params)
        self.trials = [self.best_trial]


class _FakeBaselineTrainer:
    instances = []

    def __init__(self, *args, **kwargs):
        self.__class__.instances.append(self)
        self.params_dir = Path(kwargs["params_dir"])
        self.random_state = kwargs["random_state"]
        self.data_config = kwargs["data_config"]
        self.metric = kwargs["metric"]
        self.auction_mode = kwargs["auction_mode"]

    def get_params_path(self, model_name: str) -> str:
        return str(self.params_dir / f"{model_name}_{self.metric.lower()}_{self.auction_mode}.pkl")

    def opt_search_linear(self, n_trials=1):
        params = {
            "coef": 0.2,
            "lower_clip": 1,
            "upper_clip": 3,
            "factor": 1.5,
        }
        self.params_dir.mkdir(parents=True, exist_ok=True)
        with open(self.get_params_path("linear"), "wb") as f:
            pickle.dump(params, f)
        return _FakeBaselineStudy(params)


class _FakeDrlbTrial:
    def __init__(self, number: int):
        self.number = number
        self.params = {
            "dqn_gamma": 0.95,
            "dqn_lr": 1e-4,
            "dqn_target_update_interval": 100,
            "reward_net_lr": 1e-3,
        }
        self.user_attrs = {}
        self.value = None

    def set_user_attr(self, key, value):
        self.user_attrs[key] = value


class _FakeDrlbStudy:
    def __init__(self):
        self.trials = []
        self.best_trial = None

    def optimize(self, objective, n_trials=1, n_jobs=1, show_progress_bar=False):
        trial = _FakeDrlbTrial(0)
        trial.value = objective(trial)
        self.trials = [trial]
        self.best_trial = trial


class _FakeRlbTrial:
    def __init__(self, number: int):
        self.number = number
        self.params = {
            "max_bid": 125,
            "gamma": 0.95,
            "N_bound": 48,
            "B_bound": 4000,
            "use_smoothing": False,
        }
        self.user_attrs = {}
        self.value = None

    def set_user_attr(self, key, value):
        self.user_attrs[key] = value


class _FakeRlbStudy:
    def __init__(self):
        self.trials = []
        self.best_trial = None

    def optimize(self, objective, n_trials=1, n_jobs=1, show_progress_bar=False):
        trial = _FakeRlbTrial(0)
        trial.value = objective(trial)
        self.trials = [trial]
        self.best_trial = trial


class TestExperimentArchitecture(unittest.TestCase):
    def test_pyproject_includes_example_notebooks_package(self):
        pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
        pyproject = tomllib.loads(pyproject_path.read_text())
        include = pyproject["tool"]["setuptools"]["packages"]["find"]["include"]

        self.assertIn("example_notebooks*", include)

    def test_drlb_profile_runner_builds_run_directory_and_profile(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with patch("example_notebooks.experiments.notebook_api.run_experiment_inprocess", return_value={"ok": True}) as mocked:
                run_profile_inprocess(
                    "drlb",
                    "smoke_custom_dir",
                    split_set="subsample_train_val_holdout",
                    artifacts_root=root,
                    drlb_profile="drlb_smooth",
                )

            cfg = mocked.call_args.args[0]
            self.assertEqual(cfg.run_name, "smoke_custom_dir")
            self.assertEqual(cfg.drlb_profile, "drlb_smooth")
            self.assertEqual(cfg.experiment_dir, root / "drlb" / "smoke_custom_dir")

    def test_experiment_config_uses_family_run_dir_and_split_registry(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cfg = ExperimentConfig(
                experiment_name="drlb_smooth",
                n_trials=1,
                random_seed=42,
                auction_mode="FPA",
                metric="SCR",
                family="drlb",
                split_set="subsample_train_val_holdout",
                experiments_data_dir=root,
            )

            self.assertEqual(cfg.run_name, "drlb_smooth")
            self.assertEqual(cfg.experiment_dir, root / "drlb" / "drlb_smooth")
            self.assertEqual(cfg.split_set, "subsample_train_val_holdout")
            self.assertEqual(set(cfg.data_config.keys()), {"train", "val", "test_holdout"})
            self.assertEqual(cfg.master_seed, 42)
            self.assertEqual(cfg.data_seed, 1042)
            self.assertEqual(cfg.optuna_seed, 2042)

    def test_split_fingerprint_tracks_campaigns_and_stats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cfg = ExperimentConfig(
                experiment_name="baseline_linear",
                n_trials=1,
                random_seed=1,
                auction_mode="FPA",
                metric="SCR",
                family="baselines",
                data_config={
                    "train": _write_split_files(root, "train"),
                    "val": _write_split_files(root, "val"),
                    "test_holdout": _write_split_files(root, "holdout"),
                },
                experiments_data_dir=root,
            )
            normalized = resolve_normalized_splits(cfg)
            first = split_fingerprint(normalized)

            pd.DataFrame([{"campaign_id": 2, "period": 0}]).to_csv(
                normalized["val"]["stats_path"],
                index=False,
            )
            updated = resolve_normalized_splits(cfg)
            second = split_fingerprint(updated)

            self.assertNotEqual(first, second)

    def test_baseline_adapter_writes_metrics_and_split_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = ExperimentConfig(
                experiment_name="linear_default",
                run_name="linear_default",
                n_trials=2,
                random_seed=42,
                auction_mode="FPA",
                metric="SCR",
                family="baselines",
                model_config={"model_name": "linear"},
                data_config={
                    "train": _write_split_files(root, "train"),
                    "val": _write_split_files(root, "val"),
                    "test_holdout": _write_split_files(root, "holdout"),
                },
                experiments_data_dir=root,
            )
            normalized_splits = resolve_normalized_splits(config)

            def fake_check(*args, **kwargs):
                campaigns_path = kwargs["params"]["input_campaigns"]
                if "holdout" in campaigns_path:
                    return {
                        "score": (10.0, 11.0, 12.0, 13.0),
                        "skipped_campaigns": 0,
                        "time_inference_sec": 1.0,
                        "time_overall_sec": 2.0,
                    }
                return {
                    "score": (1.0, 2.0, 3.0, 4.0),
                    "skipped_campaigns": 0,
                    "time_inference_sec": 0.5,
                    "time_overall_sec": 0.7,
                }

            with patch("example_notebooks.experiments.adapters.baseline_adapter.BaseLineTrainer", _FakeBaselineTrainer):
                with patch("example_notebooks.experiments.adapters.baseline_adapter.autobidder_check", side_effect=fake_check):
                    summary = run_baseline_experiment(config, normalized_splits)

            self.assertEqual(summary["final_holdout"]["metrics"]["clicks_sum"], 12.0)
            self.assertIn("split_fingerprint", summary)
            self.assertTrue((config.outputs_dir / "metrics.json").exists())
            self.assertTrue((config.config_dir / "split_manifest.json").exists())
            metrics_payload = json.loads((config.outputs_dir / "metrics.json").read_text())
            manifest_payload = json.loads((config.config_dir / "split_manifest.json").read_text())
            self.assertEqual(metrics_payload["clicks_sum"], 12.0)
            self.assertEqual(manifest_payload["fingerprint"], summary["split_fingerprint"])
            self.assertEqual(
                _FakeBaselineTrainer.instances[-1].data_config["test"]["campaigns_path"],
                normalized_splits["val"]["campaigns_path"],
            )

    def test_baseline_adapter_inprocess_returns_live_objects(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = ExperimentConfig(
                experiment_name="linear_default",
                run_name="linear_default",
                n_trials=2,
                random_seed=42,
                auction_mode="FPA",
                metric="SCR",
                family="baselines",
                model_config={"model_name": "linear"},
                data_config={
                    "train": _write_split_files(root, "train"),
                    "val": _write_split_files(root, "val"),
                    "test_holdout": _write_split_files(root, "holdout"),
                },
                experiments_data_dir=root,
            )
            normalized_splits = resolve_normalized_splits(config)

            def fake_check(*args, **kwargs):
                campaigns_path = kwargs["params"]["input_campaigns"]
                clicks = 3.0 if "val" in campaigns_path else 12.0
                return {
                    "score": (1.0, 2.0, clicks, 4.0),
                    "skipped_campaigns": 0,
                    "time_inference_sec": 0.5,
                    "time_overall_sec": 0.7,
                }

            with patch("example_notebooks.experiments.adapters.baseline_adapter.BaseLineTrainer", _FakeBaselineTrainer):
                with patch("example_notebooks.experiments.adapters.baseline_adapter.autobidder_check", side_effect=fake_check):
                    result = run_baseline_experiment_inprocess(config, normalized_splits)

            self.assertEqual(result["summary"]["final_holdout"]["metrics"]["clicks_sum"], 12.0)
            self.assertEqual(result["best_run"]["metrics"]["clicks_sum"], 12.0)
            self.assertEqual(result["best_val_run"]["metrics"]["clicks_sum"], 3.0)
            self.assertIsNotNone(result["best_run"]["bidder"])
            self.assertIsNotNone(result["tuning_trainer"])

    def test_drlb_adapter_writes_final_metrics_and_preserves_val_vs_holdout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = ExperimentConfig(
                experiment_name="drlb_smooth",
                run_name="drlb_smooth",
                n_trials=1,
                random_seed=42,
                auction_mode="FPA",
                metric="SCR",
                family="drlb",
                data_config={
                    "train": _write_split_files(root, "train"),
                    "val": _write_split_files(root, "val"),
                    "test_holdout": _write_split_files(root, "holdout"),
                },
                experiments_data_dir=root,
            )
            normalized_splits = resolve_normalized_splits(config)

            def fake_candidate(
                config,
                data_splits,
                train_stats_df,
                train_campaigns_df,
                label,
                bidder_params,
                *,
                eval_split_key,
                objective="clicks",
                max_train_steps=None,
                verbose=False,
                scratch_dir,
                return_bidder=False,
                return_diagnostics=False,
            ):
                model_path = Path(scratch_dir) / f"{label}.pt"
                model_path.write_text("fake model")
                clicks = 77.0 if eval_split_key == "val" else 99.0
                return {
                    "label": label,
                    "params": bidder_params,
                    "model_path": model_path,
                    "metrics": {
                        "label": label,
                        "cpc_relative": 1.0,
                        "rmse": 2.0,
                        "clicks_sum": clicks,
                        "quickspend": 3.0,
                        "skipped_campaigns": 0,
                        "time_inference_sec": 0.1,
                        "time_overall_sec": 0.2,
                        "train_steps": 1,
                    },
                }

            with patch("example_notebooks.experiments.adapters.drlb_adapter.run_drlb_candidate", side_effect=fake_candidate):
                with patch("example_notebooks.experiments.adapters.drlb_adapter.optuna.create_study", return_value=_FakeDrlbStudy()):
                    summary = run_drlb_experiment(
                        config,
                        normalized_splits,
                        base_drlb_params={"max_bid": 10.0},
                        reference_model_params={"dqn_gamma": 1.0},
                        exp_type="improved_drlb_eval",
                        search_space_fn=lambda trial: trial.params,
                    )

            self.assertEqual(summary["reference"]["baseline_manual_val"]["metrics"]["clicks_sum"], 77.0)
            self.assertEqual(summary["tuning"]["best_val_metrics"]["clicks_sum"], 77.0)
            self.assertEqual(summary["final_holdout"]["metrics"]["clicks_sum"], 99.0)
            self.assertEqual(
                json.loads((config.outputs_dir / "metrics.json").read_text())["clicks_sum"],
                99.0,
            )
            self.assertTrue((config.best_models_dir / "best_refit.pt").exists())

    def test_drlb_adapter_inprocess_returns_bidder_and_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = ExperimentConfig(
                experiment_name="drlb_smooth",
                run_name="drlb_smooth",
                n_trials=1,
                random_seed=42,
                auction_mode="FPA",
                metric="SCR",
                family="drlb",
                data_config={
                    "train": _write_split_files(root, "train"),
                    "val": _write_split_files(root, "val"),
                    "test_holdout": _write_split_files(root, "holdout"),
                },
                experiments_data_dir=root,
            )
            normalized_splits = resolve_normalized_splits(config)

            def fake_candidate(
                config,
                data_splits,
                train_stats_df,
                train_campaigns_df,
                label,
                bidder_params,
                *,
                eval_split_key,
                objective="clicks",
                max_train_steps=None,
                verbose=False,
                scratch_dir,
                return_bidder=False,
                return_diagnostics=False,
            ):
                model_path = Path(scratch_dir) / f"{label}.pt"
                model_path.write_text("fake model")
                clicks = 77.0 if eval_split_key == "val" else 99.0
                diagnostics_df = pd.DataFrame({"dqn_loss": [0.0], "reward_net_loss": [0.0]})
                return {
                    "label": label,
                    "params": bidder_params,
                    "model_path": model_path,
                    "metrics": {
                        "label": label,
                        "cpc_relative": 1.0,
                        "rmse": 2.0,
                        "clicks_sum": clicks,
                        "quickspend": 3.0,
                        "skipped_campaigns": 0,
                        "time_inference_sec": 0.1,
                        "time_overall_sec": 0.2,
                        "train_steps": 1,
                    },
                    "bidder": object() if return_bidder else None,
                    "diagnostics": diagnostics_df if return_diagnostics else None,
                    "diagnostics_path": str(root / f"{label}.csv"),
                    "diagnostics_plot_path": str(root / f"{label}.png"),
                    "reward_net_plot_path": str(root / f"{label}_reward.png"),
                    "eval_action_distribution_path": str(root / f"{label}_actions.png"),
                }

            with patch("example_notebooks.experiments.adapters.drlb_adapter.run_drlb_candidate", side_effect=fake_candidate):
                with patch("example_notebooks.experiments.adapters.drlb_adapter.optuna.create_study", return_value=_FakeDrlbStudy()):
                    result = run_drlb_experiment_inprocess(
                        config,
                        normalized_splits,
                        base_drlb_params={"max_bid": 10.0},
                        reference_model_params={"dqn_gamma": 1.0},
                        exp_type="improved_drlb_eval",
                        search_space_fn=lambda trial: trial.params,
                    )

            self.assertEqual(result["summary"]["final_holdout"]["metrics"]["clicks_sum"], 99.0)
            self.assertEqual(result["reference_run"]["metrics"]["clicks_sum"], 77.0)
            self.assertIsNotNone(result["best_run"]["bidder"])
            self.assertIsNotNone(result["best_run"]["diagnostics"])
            self.assertEqual(result["best_val_run"]["metrics"]["clicks_sum"], 77.0)
            self.assertEqual(len(result["trial_runs"]), 1)

    def test_rlb_adapter_inprocess_returns_best_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = ExperimentConfig(
                experiment_name="rlb_default",
                run_name="rlb_default",
                n_trials=1,
                random_seed=42,
                auction_mode="FPA",
                metric="SCR",
                family="rlb",
                data_config={
                    "train": _write_split_files(root, "train"),
                    "val": _write_split_files(root, "val"),
                    "test_holdout": _write_split_files(root, "holdout"),
                },
                experiments_data_dir=root,
            )
            normalized_splits = resolve_normalized_splits(config)

            def fake_rlb_candidate(
                *,
                config,
                train_stats_df,
                train_campaigns_path,
                eval_split,
                label,
                bidder_params,
                objective="clicks",
                return_bidder=False,
            ):
                model_path = config.best_models_dir / f"{label}.pkl"
                config.best_models_dir.mkdir(parents=True, exist_ok=True)
                model_path.write_text("fake model")
                clicks = 77.0 if "val" in eval_split["campaigns_path"] else 99.0
                return {
                    "label": label,
                    "params": bidder_params,
                    "metrics": {
                        "label": label,
                        "cpc_relative": 1.0,
                        "rmse": 2.0,
                        "clicks_sum": clicks,
                        "quickspend": 3.0,
                        "skipped_campaigns": 0,
                        "time_inference_sec": 0.1,
                        "time_overall_sec": 0.2,
                    },
                    "model_path": model_path,
                    "bidder": object() if return_bidder else None,
                    "eval_split": dict(eval_split),
                }

            with patch("example_notebooks.experiments.adapters.rlb_adapter.run_rlb_candidate", side_effect=fake_rlb_candidate):
                with patch("example_notebooks.experiments.adapters.rlb_adapter.optuna.create_study", return_value=_FakeRlbStudy()):
                    result = run_rlb_experiment_inprocess(
                        config,
                        normalized_splits,
                        search_space_fn=lambda trial: trial.params,
                    )

            self.assertEqual(result["summary"]["final_holdout"]["metrics"]["clicks_sum"], 99.0)
            self.assertEqual(result["best_val_run"]["metrics"]["clicks_sum"], 77.0)
            self.assertIsNotNone(result["best_run"]["bidder"])
            self.assertEqual(len(result["trial_runs"]), 1)

    def test_drlb_diagnostics_artifacts_are_written(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = ExperimentConfig(
                experiment_name="drlb_smooth",
                run_name="drlb_smooth",
                n_trials=1,
                random_seed=42,
                auction_mode="FPA",
                metric="SCR",
                family="drlb",
                data_config={
                    "train": _write_split_files(root, "train"),
                    "val": _write_split_files(root, "val"),
                    "test_holdout": _write_split_files(root, "holdout"),
                },
                experiments_data_dir=root,
            )
            diagnostics_df = pd.DataFrame(
                {
                    "global_t": [0, 1, 2],
                    "dqn_loss": [0.0, 1.0, 0.5],
                    "reward_net_loss": [0.0, 2.0, 1.0],
                    "reward_signal": [1.0, 1.5, 2.0],
                    "lambda": [0.1, 0.2, 0.3],
                    "eps": [1.0, 0.9, 0.8],
                    "dqn_action": [0, 1, 2],
                }
            )

            artifacts = write_training_diagnostics_artifacts(
                config=config,
                label="best_refit",
                diagnostics_df=diagnostics_df,
                eval_diagnostics_df=diagnostics_df,
                eval_split_key="val",
            )

            self.assertEqual(
                Path(artifacts["diagnostics_path"]),
                config.outputs_dir / "best_refit_training_diagnostics.csv",
            )
            self.assertEqual(
                Path(artifacts["diagnostics_plot_path"]),
                config.outputs_dir / "best_refit_dqn_diagnostics.png",
            )
            self.assertEqual(
                Path(artifacts["reward_net_plot_path"]),
                config.outputs_dir / "best_refit_reward_net_diagnostics.png",
            )
            self.assertEqual(
                Path(artifacts["eval_action_distribution_path"]),
                config.outputs_dir / "best_refit_val_action_distribution.png",
            )
            self.assertTrue((config.outputs_dir / "best_refit_training_diagnostics.csv").exists())
            self.assertTrue((config.outputs_dir / "best_refit_dqn_diagnostics.png").exists())
            self.assertTrue((config.outputs_dir / "best_refit_reward_net_diagnostics.png").exists())
            self.assertTrue((config.outputs_dir / "best_refit_val_action_distribution.png").exists())

    def test_plot_training_diagnostics_skips_empty_frames(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "empty.png"
            written = plot_training_diagnostics(pd.DataFrame(), output_path, title="empty")
            self.assertFalse(written)
            self.assertFalse(output_path.exists())

    def test_notebook_api_builds_drlb_config_with_overrides(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("example_notebooks.experiments.notebook_api.run_experiment_inprocess", return_value={"ok": True}) as mocked:
                run_drlb_profile_inprocess(
                    run_name="drlb_smooth",
                    artifacts_root=tmpdir,
                    n_trials=7,
                    max_train_steps=11,
                )

            config = mocked.call_args.args[0]
            self.assertEqual(config.family, "drlb")
            self.assertEqual(config.n_trials, 7)
            self.assertEqual(config.max_steps, 11)
            self.assertEqual(config.experiments_data_dir, Path(tmpdir))

    def test_notebook_api_dispatches_inprocess_runner_with_profile_kwargs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("example_notebooks.experiments.notebook_api.run_experiment_inprocess", return_value={"ok": True}) as mocked:
                result = run_profile_inprocess(
                    "drlb",
                    "drlb_smooth",
                    artifacts_root=tmpdir,
                    n_trials=5,
                    max_train_steps=9,
                )

            self.assertEqual(result, {"ok": True})
            args, kwargs = mocked.call_args
            self.assertEqual(args[0].family, "drlb")
            self.assertEqual(args[0].n_trials, 5)
            self.assertEqual(args[0].max_steps, 9)
            self.assertEqual(kwargs["n_trials"], 5)
            self.assertEqual(kwargs["max_train_steps"], 9)
            self.assertIn("reference_model_params", kwargs)
            self.assertNotIn("baseline_model_params", kwargs)
            self.assertIn("search_space_fn", kwargs)

    def test_all_comparison_loader_reads_matching_run_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            split_manifest = {
                "family": "drlb",
                "run_name": "drlb_smooth",
                "split_set": "subsample_train_val_holdout",
                "fingerprint": "same-fingerprint",
                "splits": {},
            }
            for family, run_name in (("drlb", "drlb_smooth"), ("rlb", "rlb_default")):
                run_dir = root / family / run_name
                (run_dir / "config").mkdir(parents=True, exist_ok=True)
                (run_dir / "outputs").mkdir(parents=True, exist_ok=True)
                manifest = dict(split_manifest, family=family, run_name=run_name)
                (run_dir / "config" / "split_manifest.json").write_text(json.dumps(manifest))
                (run_dir / "outputs" / "run_summary.json").write_text(
                    json.dumps(
                        {
                            "family": family,
                            "run_name": run_name,
                            "split_set": "subsample_train_val_holdout",
                            "split_fingerprint": "same-fingerprint",
                        }
                    )
                )
                (run_dir / "outputs" / "metrics.json").write_text(
                    json.dumps({"clicks_sum": 1.0, "rmse": 2.0, "cpc_relative": 3.0, "quickspend": 4.0})
                )

            df = load_metrics_table([root / "drlb" / "drlb_smooth", root / "rlb" / "rlb_default"])
            self.assertEqual(len(df), 2)
            self.assertEqual(assert_matching_split_fingerprint([root / "drlb" / "drlb_smooth", root / "rlb" / "rlb_default"]), "same-fingerprint")


if __name__ == "__main__":
    unittest.main()

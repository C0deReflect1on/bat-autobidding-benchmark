from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

try:
    from example_notebooks.evaluate_baselines.baselines_finetune import BaseLineTrainer
except ModuleNotFoundError:  # pragma: no cover - supports notebook-style imports
    from evaluate_baselines.baselines_finetune import BaseLineTrainer
from simulator.model.broi_bidder import BROI
from simulator.model.linear_bidder import LinearBidder
from simulator.model.m_pid import MPIDBidder
from simulator.model.mystique import Mystique
from simulator.model.ta_pid import TAPIDBidder
from simulator.validation.check_results import autobidder_check

from ..infra.artifacts import append_runs_index, build_summary_header, score_to_dict, write_normalized_config, write_run_summary
from ..infra.split_utils import build_trainer_data_config


_MODEL_TO_BIDDER = {
    "linear": LinearBidder,
    "ta_pid": TAPIDBidder,
    "m_pid": MPIDBidder,
    "mystique": Mystique,
    "broi": BROI,
}

_MODEL_TO_TUNING_METHOD = {
    "linear": "opt_search_linear",
    "ta_pid": "opt_search_tapid",
    "m_pid": "opt_search_mpid",
    "mystique": "opt_search_mystique",
    "broi": "opt_search_broi",
}


def run_baseline_experiment(
    config,
    normalized_splits: dict[str, dict[str, str]],
    *,
    verbose: bool = False,
) -> dict[str, Any]:
    model_name = str(config.model_config.get("model_name", "")).strip()
    if model_name not in _MODEL_TO_BIDDER:
        raise ValueError(
            f"Unsupported or missing baseline model_name '{model_name}'. "
            f"Supported values: {sorted(_MODEL_TO_BIDDER.keys())}"
        )

    config.ensure_artifact_dirs()
    write_normalized_config(config)

    tuning_trainer = BaseLineTrainer(
        data_config=build_trainer_data_config(normalized_splits, eval_split="val"),
        metric=config.metric,
        auction_mode=config.auction_mode,
        base_params_subfolder=config.experiment_name,
        random_state=config.optuna_seed,
        params_dir=config.best_params_dir,
        n_jobs=1,
    )
    study = getattr(tuning_trainer, _MODEL_TO_TUNING_METHOD[model_name])(n_trials=config.n_trials)

    params_path = Path(tuning_trainer.get_params_path(model_name))
    with open(params_path, "rb") as f:
        best_params = pickle.load(f)

    best_val_result = evaluate_baseline_model(
        model_name=model_name,
        params_dict=best_params,
        split=normalized_splits["val"],
        auction_mode=config.auction_mode,
    )
    final_holdout_result = evaluate_baseline_model(
        model_name=model_name,
        params_dict=best_params,
        split=normalized_splits["test_holdout"],
        auction_mode=config.auction_mode,
    )

    summary = build_summary_header(config, normalized_splits)
    summary.update(
        {
            "tuning": {
                "enabled": True,
                "n_trials": int(config.n_trials),
                "best_trial_number": int(study.best_trial.number),
                "best_params": best_params,
                "all_trials_summary": _study_trials_summary(study),
                "best_val_metrics": score_to_dict(
                    best_val_result["score"],
                    skipped_campaigns=best_val_result.get("skipped_campaigns"),
                    time_inference_sec=best_val_result.get("time_inference_sec"),
                    time_overall_sec=best_val_result.get("time_overall_sec"),
                ),
            },
            "refit": {
                "scope": config.refit_on,
                "applicable": False,
            },
            "final_holdout": {
                "metrics": score_to_dict(
                    final_holdout_result["score"],
                    skipped_campaigns=final_holdout_result.get("skipped_campaigns"),
                    time_inference_sec=final_holdout_result.get("time_inference_sec"),
                    time_overall_sec=final_holdout_result.get("time_overall_sec"),
                ),
            },
        }
    )

    write_run_summary(config, summary)
    append_runs_index(
        config,
        summary["final_holdout"]["metrics"],
        stage="final_holdout",
        label=f"{model_name}_holdout",
    )
    return summary


def evaluate_baseline_model(
    *,
    model_name: str,
    params_dict: dict[str, Any],
    split: dict[str, str],
    auction_mode: str,
) -> dict[str, Any]:
    bidder_params = _build_bidder_eval_params(model_name, params_dict)
    return autobidder_check(
        bidder=_MODEL_TO_BIDDER[model_name],
        params={
            "input_campaigns": split["campaigns_path"],
            "input_stats": split["stats_path"],
            **bidder_params,
        },
        auction_mode=auction_mode,
    )


def _build_bidder_eval_params(model_name: str, params_dict: dict[str, Any]) -> dict[str, Any]:
    if model_name == "linear":
        return {
            "cold_start_coef": params_dict["coef"],
            "lower_clip": params_dict["lower_clip"],
            "upper_clip": params_dict["upper_clip"],
            "factor": params_dict["factor"],
        }
    if model_name == "ta_pid":
        return {
            "k_dict": {
                "k_p": params_dict["k_p1"],
                "k_i": params_dict["k_i1"],
                "k_d": params_dict["k_d1"],
            },
            "cold_start_coef": params_dict["coef"],
        }
    if model_name == "m_pid":
        return {
            "k_dict": {
                "k_p": (params_dict["k_p1"], params_dict["k_p2"]),
                "k_i": (params_dict["k_i1"], params_dict["k_i2"]),
                "k_d": (params_dict["k_d1"], params_dict["k_d2"]),
            },
            "correction": [params_dict["alpha"], params_dict["beta"]],
            "cold_start_coef": params_dict["coef"],
            "lower_clip": params_dict["lower_clip"],
            "upper_clip": params_dict["upper_clip"],
            "bid_factor": params_dict["bid_factor"],
        }
    if model_name == "mystique":
        return {
            "pf0": params_dict["pf0"],
            "C_max": params_dict["C_max"],
            "C_min": params_dict["C_min"],
            "E_max": params_dict["E_max"],
            "E_gmc": params_dict["E_gmc"],
        }
    if model_name == "broi":
        return {
            "ro": params_dict["ro"],
            "v_bar": params_dict["v_bar"],
        }
    raise ValueError(f"Unsupported baseline model_name '{model_name}'")


def _study_trials_summary(study) -> list[dict[str, Any]]:
    rows = []
    for trial in study.trials:
        rows.append(
            {
                "trial": int(trial.number),
                "value": None if trial.value is None else float(trial.value),
                **trial.params,
            }
        )
    return rows

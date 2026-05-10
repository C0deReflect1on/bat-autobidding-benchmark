from __future__ import annotations

import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from drlb_autoresearch.research_core import (
        append_autoresearch_entry,
        cache_locked_linear_baseline,
        ensure_run_dir,
        load_locked_linear_reference,
        prepare_context,
        run_pair_comparison,
        write_json,
    )
except ModuleNotFoundError:
    from research_core import (
        append_autoresearch_entry,
        cache_locked_linear_baseline,
        ensure_run_dir,
        load_locked_linear_reference,
        prepare_context,
        run_pair_comparison,
        write_json,
    )
from example_notebooks.experiments.adapters.drlb_adapter import DRLB_RUNTIME_DEFAULTS
from example_notebooks.experiments.drlb.profiles import build_config, get_profile
from example_notebooks.experiments.shared_runner import run_experiment_inprocess


RUN_NAME = "drlb_with_linear_lambda_fit_checkpoint_infer_best_eps_train_plus_val_full"
STAGE_KEY = "04_full_trainval_checkpoint_infer"
STAGE_RUN_NAME = "epsilon_optuna_trial001_train_plus_val_full"

BEST_EPS = {
    "dqn_epsilon_start": 0.8610900264673047,
    "dqn_epsilon_end": 0.09801905607969424,
    "dqn_epsilon_anneal": 1.7341839015915597e-05,
}


def _fixed_eps_search_space(_trial) -> dict[str, float]:
    return dict(BEST_EPS)


def main() -> None:
    locked_reference = load_locked_linear_reference()
    context = prepare_context(
        profile_key="drlb_smooth",
        split_set="full_train_val_holdout",
        objective="clicks",
        auction_mode="FPA",
        locked_reference=locked_reference,
    )

    baseline_cache = cache_locked_linear_baseline(
        context=context,
        run_name="locked_linear_reference_v1",
    )
    baseline_manifest = baseline_cache["manifest"]

    config = build_config(
        RUN_NAME,
        profile="drlb_smooth",
        split_set="full_train_val_holdout",
    )
    config = replace(config, n_trials=1, max_steps=None, refit_on="train_plus_val")
    profile = get_profile("drlb_smooth")

    base_drlb_params = {
        **profile["base_drlb_params"],
        "init_lambda": context["locked_reference"]["linear_lambda_init"],
        "init_lambda_mode": "constant",
    }
    reference_model_params = {
        **profile["reference_model_params"],
        **BEST_EPS,
    }

    result = run_experiment_inprocess(
        config,
        verbose=False,
        base_drlb_params=base_drlb_params,
        reference_model_params=reference_model_params,
        state_type=profile["state_type"],
        objective=profile["objective"],
        search_space_fn=_fixed_eps_search_space,
        n_trials=config.n_trials,
        max_train_steps=config.max_steps,
    )

    best_model_path = config.best_models_dir / "best_refit.pt"
    candidate_dir = ensure_run_dir(STAGE_KEY, STAGE_RUN_NAME)

    pair_params = {
        **DRLB_RUNTIME_DEFAULTS,
        **base_drlb_params,
        **reference_model_params,
        "state_type": profile["state_type"],
        "objective": profile["objective"],
        "auction_mode": config.auction_mode,
        "verbose": False,
        "use_tqdm": False,
        "eval_mode": True,
    }

    val_pair = run_pair_comparison(
        candidate_dir=candidate_dir,
        baseline_cache_dir=baseline_cache["run_dir"],
        split_name="val",
        split=context["splits"]["val"],
        auction_mode=config.auction_mode,
        drlb_params=pair_params,
        drlb_model_path=best_model_path,
    )
    holdout_pair = run_pair_comparison(
        candidate_dir=candidate_dir,
        baseline_cache_dir=baseline_cache["run_dir"],
        split_name="test_holdout",
        split=context["splits"]["test_holdout"],
        auction_mode=config.auction_mode,
        drlb_params=pair_params,
        drlb_model_path=best_model_path,
    )

    summary = result["summary"]
    holdout_metrics = summary["final_holdout"]["metrics"]
    val_metrics = summary["tuning"]["best_val_metrics"]
    action_distribution_path = summary["refit"].get("eval_action_distribution_path")
    diagnostics_plot_path = summary["refit"].get("combined_diagnostics_plot_path")

    payload = {
        "run_name": RUN_NAME,
        "best_eps": BEST_EPS,
        "config": {
            "n_trials": config.n_trials,
            "max_steps": config.max_steps,
            "refit_on": config.refit_on,
            "split_set": config.split_set,
        },
        "baseline_manifest_path": str(baseline_cache["run_dir"] / "baseline_manifest.json"),
        "baseline_metrics": baseline_manifest["metrics_by_split"],
        "metrics": {
            "val": val_metrics,
            "holdout": holdout_metrics,
        },
        "artifacts": {
            "experiment_dir": str(config.experiment_dir),
            "outputs_dir": str(config.outputs_dir),
            "best_model_path": str(best_model_path),
            "pair_comparison": {
                "val": {k: str(v) for k, v in val_pair.items()},
                "test_holdout": {k: str(v) for k, v in holdout_pair.items()},
            },
            "action_distribution_path": action_distribution_path,
            "combined_diagnostics_plot_path": diagnostics_plot_path,
        },
    }
    write_json(candidate_dir / "summary.json", payload)

    shutil.copy2(best_model_path, candidate_dir / "best_refit.pt")
    if diagnostics_plot_path:
        src = Path(diagnostics_plot_path)
        if src.exists():
            shutil.copy2(src, candidate_dir / src.name)
    if action_distribution_path:
        src = Path(action_distribution_path)
        if src.exists():
            shutil.copy2(src, candidate_dir / src.name)

    append_autoresearch_entry(
        heading="04_full_trainval_checkpoint_infer / epsilon_optuna_trial001_train_plus_val_full",
        stage_key=STAGE_KEY,
        run_name=STAGE_RUN_NAME,
        hypothesis=(
            "Promote best epsilon from quick-wave fallback into full train+val refit; "
            "per-campaign λ from init_lambda / get_lambda, checkpoint stores final λ in config."
        ),
        params={
            "init_lambda": context["locked_reference"]["linear_lambda_init"],
            "init_lambda_mode": "constant",
            **BEST_EPS,
        },
        result_summary={
            "val_clicks_sum": float(val_metrics["clicks_sum"]),
            "holdout_clicks_sum": float(holdout_metrics["clicks_sum"]),
            "val_average_end_balance_share": float(val_metrics["average_end_balance_share"]),
            "holdout_average_end_balance_share": float(holdout_metrics["average_end_balance_share"]),
            "locked_linear_holdout_clicks_sum": float(
                baseline_manifest["metrics_by_split"]["test_holdout"]["clicks_sum"]
            ),
            "delta_holdout_clicks_vs_locked_linear": float(
                holdout_metrics["clicks_sum"] - baseline_manifest["metrics_by_split"]["test_holdout"]["clicks_sum"]
            ),
            "pair_comparison_hourly_holdout_path": str(holdout_pair["campaign_hourly_comparison_path"]),
            "action_distribution_path": action_distribution_path,
        },
        decision=(
            "promote"
            if holdout_metrics["clicks_sum"] >= baseline_manifest["metrics_by_split"]["test_holdout"]["clicks_sum"]
            else "reject"
        ),
        notes=(
            "Full refit uses train_plus_val and writes val/holdout P2P comparisons "
            "including hourly traces."
        ),
    )

    print(
        json.dumps(
            {
                "baseline_holdout_clicks_sum": baseline_manifest["metrics_by_split"]["test_holdout"]["clicks_sum"],
                "drlb_holdout_clicks_sum": holdout_metrics["clicks_sum"],
                "drlb_val_clicks_sum": val_metrics["clicks_sum"],
                "action_distribution_path": action_distribution_path,
                "holdout_hourly_pair_path": str(holdout_pair["campaign_hourly_comparison_path"]),
                "candidate_summary_path": str(candidate_dir / "summary.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

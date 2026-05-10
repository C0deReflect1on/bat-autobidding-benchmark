from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

try:
    from drlb_autoresearch.research_core import (
        AUTORESEARCH_LOG_PATH,
        build_bid_lr_search_space,
        build_epoch_candidates,
        build_epsilon_search_space,
        build_manual_epsilon_candidates,
        cache_locked_linear_baseline,
        ensure_run_dir,
        load_locked_linear_reference,
        prepare_context,
        read_json,
        run_candidate_grid,
        run_optuna_search,
        write_json,
    )
except ModuleNotFoundError:
    from research_core import (
        AUTORESEARCH_LOG_PATH,
        build_bid_lr_search_space,
        build_epoch_candidates,
        build_epsilon_search_space,
        build_manual_epsilon_candidates,
        cache_locked_linear_baseline,
        ensure_run_dir,
        load_locked_linear_reference,
        prepare_context,
        read_json,
        run_candidate_grid,
        run_optuna_search,
        write_json,
    )


BASELINE_STAGE = "00_locked_linear_baseline"
BID_LR_STAGE = "01_bid_lr_optuna"
EPS_MANUAL_STAGE = "02_epsilon_manual"
EPS_OPTUNA_STAGE = "03_epsilon_optuna"
EPOCH_STAGE = "04_epoch_sweep"
SUMMARY_STAGE = "99_global_summary"

BASELINE_RUN_NAME = "locked_linear_reference_v1"
BID_LR_RUN_NAME = "bid_lr_optuna_v1"
EPS_MANUAL_RUN_NAME = "epsilon_manual_v1"
EPS_OPTUNA_RUN_NAME = "epsilon_optuna_v1"
EPOCH_RUN_NAME = "epoch_sweep_v1"

MAX_STEPS = 15000
RECORD_EVERY = 5000
SMOOTHING_WINDOW = 500
BID_LR_TRIALS = 8
EPSILON_TRIALS = 8


def _candidate_params(run_root: Path, leaderboard_df: pd.DataFrame) -> dict[str, Any]:
    candidate = str(leaderboard_df.iloc[0]["candidate"])
    return read_json(run_root / candidate / "params.json")


def _best_row(leaderboard_df: pd.DataFrame) -> pd.Series:
    return leaderboard_df.sort_values("best_val_clicks_sum", ascending=False).iloc[0]


def _best_params_from_many(
    candidates: list[tuple[pd.DataFrame, Path]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    best_df, best_root = max(
        candidates,
        key=lambda item: float(_best_row(item[0])["best_val_clicks_sum"]),
    )
    best_row = _best_row(best_df)
    best_candidate = str(best_row["candidate"])
    best_params = read_json(best_root / best_candidate / "params.json")
    return best_params, {
        "candidate": best_candidate,
        "best_val_clicks_sum": float(best_row["best_val_clicks_sum"]),
        "best_checkpoint_step": (
            None if pd.isna(best_row["best_checkpoint_step"]) else int(best_row["best_checkpoint_step"])
        ),
        "run_root": str(best_root),
    }


def main() -> None:
    if not AUTORESEARCH_LOG_PATH.exists():
        AUTORESEARCH_LOG_PATH.write_text("# DRLB Autoresearch Log\n")

    locked_reference = load_locked_linear_reference()
    context = prepare_context(
        profile_key="drlb_smooth",
        split_set="full_train_val_holdout",
        objective="clicks",
        auction_mode="FPA",
        locked_reference=locked_reference,
    )

    print("[1/5] Caching locked linear baseline...")
    baseline_cache = cache_locked_linear_baseline(
        context=context,
        run_name=BASELINE_RUN_NAME,
    )

    print("[2/5] Bid / max_bid / min_bid / LR Optuna...")
    bid_lr_leaderboard_df, bid_lr_root, bid_lr_best_overrides = run_optuna_search(
        stage_key=BID_LR_STAGE,
        run_name=BID_LR_RUN_NAME,
        context=context,
        search_space_fn=build_bid_lr_search_space,
        hypothesis_prefix="Bid/LR optuna against locked linear baseline",
        n_trials=BID_LR_TRIALS,
        baseline_cache_dir=baseline_cache["run_dir"],
        epochs=1,
        max_steps=MAX_STEPS,
        record_every=RECORD_EVERY,
        smoothing_window=SMOOTHING_WINDOW,
        holdout_top_k=1,
    )

    bid_lr_context = dict(context)
    bid_lr_context["base_bidder_params"] = {
        **context["base_bidder_params"],
        **bid_lr_best_overrides,
    }

    print("[3/5] Epsilon manual + Optuna...")
    epsilon_manual_df, epsilon_manual_root = run_candidate_grid(
        stage_key=EPS_MANUAL_STAGE,
        run_name=EPS_MANUAL_RUN_NAME,
        context=bid_lr_context,
        candidates=build_manual_epsilon_candidates(),
        baseline_cache_dir=baseline_cache["run_dir"],
        epochs=1,
        max_steps=MAX_STEPS,
        record_every=RECORD_EVERY,
        smoothing_window=SMOOTHING_WINDOW,
        evaluate_holdout=False,
    )

    epsilon_optuna_df, epsilon_optuna_root, epsilon_best_overrides = run_optuna_search(
        stage_key=EPS_OPTUNA_STAGE,
        run_name=EPS_OPTUNA_RUN_NAME,
        context=bid_lr_context,
        search_space_fn=build_epsilon_search_space,
        hypothesis_prefix="Epsilon schedule optuna with continuous decay",
        n_trials=EPSILON_TRIALS,
        baseline_cache_dir=baseline_cache["run_dir"],
        epochs=1,
        max_steps=MAX_STEPS,
        record_every=RECORD_EVERY,
        smoothing_window=SMOOTHING_WINDOW,
        holdout_top_k=1,
    )

    epsilon_params, epsilon_meta = _best_params_from_many(
        [
            (epsilon_manual_df, epsilon_manual_root),
            (epsilon_optuna_df, epsilon_optuna_root),
        ]
    )

    print("[4/5] Epoch sweep...")
    epoch_context = dict(context)
    epoch_context["base_bidder_params"] = dict(epsilon_params)
    epoch_leaderboard_df, epoch_root = run_candidate_grid(
        stage_key=EPOCH_STAGE,
        run_name=EPOCH_RUN_NAME,
        context=epoch_context,
        candidates=build_epoch_candidates(dict(epsilon_params)),
        baseline_cache_dir=baseline_cache["run_dir"],
        epochs=1,
        max_steps=None,
        record_every=RECORD_EVERY,
        smoothing_window=SMOOTHING_WINDOW,
        evaluate_holdout=True,
    )
    best_epoch_params = _candidate_params(epoch_root, epoch_leaderboard_df)

    print("[5/5] Writing global summary...")
    summary_run_dir = ensure_run_dir(SUMMARY_STAGE, "global_summary_v1")
    write_json(
        summary_run_dir / "global_summary.json",
        {
            "locked_reference": locked_reference,
            "baseline_cache_dir": str(baseline_cache["run_dir"]),
            "bid_lr_best_overrides": bid_lr_best_overrides,
            "epsilon_best_params": epsilon_params,
            "epsilon_best_meta": epsilon_meta,
            "epoch_best_params": best_epoch_params,
            "leaderboards": {
                "bid_lr": str(bid_lr_root / "leaderboard.csv"),
                "epsilon_manual": str(epsilon_manual_root / "leaderboard.csv"),
                "epsilon_optuna": str(epsilon_optuna_root / "leaderboard.csv"),
                "epoch": str(epoch_root / "leaderboard.csv"),
            },
        },
    )

    print("Bid/LR leaderboard")
    print(bid_lr_leaderboard_df.to_string(index=False))
    print("\nEpsilon manual leaderboard")
    print(epsilon_manual_df.to_string(index=False))
    print("\nEpsilon optuna leaderboard")
    print(epsilon_optuna_df.to_string(index=False))
    print("\nEpoch leaderboard")
    print(epoch_leaderboard_df.to_string(index=False))


if __name__ == "__main__":
    main()

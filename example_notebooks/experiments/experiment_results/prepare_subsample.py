from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from config import (
    FPA_CAMPAIGNS_HOLDOUT_TEST,
    FPA_CAMPAIGNS_TRAIN_VAL,
    FPA_CAMPAIGNS_VAL_VAL,
    FPA_EXPERIMENT_SUBSAMPLE_CAMPAIGNS_HOLDOUT,
    FPA_EXPERIMENT_SUBSAMPLE_CAMPAIGNS_TRAIN,
    FPA_EXPERIMENT_SUBSAMPLE_CAMPAIGNS_VAL,
    FPA_EXPERIMENT_SUBSAMPLE_METADATA,
    FPA_EXPERIMENT_SUBSAMPLE_STATS_HOLDOUT,
    FPA_EXPERIMENT_SUBSAMPLE_STATS_TRAIN,
    FPA_EXPERIMENT_SUBSAMPLE_STATS_VAL,
    FPA_STATS_HOLDOUT_TEST,
    FPA_STATS_TRAIN_VAL,
    FPA_STATS_VAL_VAL,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a permanent experiment subsample under data/fpa/subsample.")
    parser.add_argument("--train-campaigns", type=int, default=32)
    parser.add_argument("--val-campaigns", type=int, default=16)
    parser.add_argument("--holdout-campaigns", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_root = FPA_EXPERIMENT_SUBSAMPLE_METADATA.parent
    output_root.mkdir(parents=True, exist_ok=True)

    existing_paths = [
        FPA_EXPERIMENT_SUBSAMPLE_CAMPAIGNS_TRAIN,
        FPA_EXPERIMENT_SUBSAMPLE_STATS_TRAIN,
        FPA_EXPERIMENT_SUBSAMPLE_CAMPAIGNS_VAL,
        FPA_EXPERIMENT_SUBSAMPLE_STATS_VAL,
        FPA_EXPERIMENT_SUBSAMPLE_CAMPAIGNS_HOLDOUT,
        FPA_EXPERIMENT_SUBSAMPLE_STATS_HOLDOUT,
        FPA_EXPERIMENT_SUBSAMPLE_METADATA,
    ]
    if not args.overwrite and all(path.exists() for path in existing_paths):
        print(f"Subsample already exists under {output_root}. Reusing existing files.")
        return

    metadata = {
        "seed": int(args.seed),
        "splits": {
            "train": _write_split(
                campaigns_src=FPA_CAMPAIGNS_TRAIN_VAL,
                stats_src=FPA_STATS_TRAIN_VAL,
                campaigns_dst=FPA_EXPERIMENT_SUBSAMPLE_CAMPAIGNS_TRAIN,
                stats_dst=FPA_EXPERIMENT_SUBSAMPLE_STATS_TRAIN,
                n_campaigns=args.train_campaigns,
                seed=args.seed,
            ),
            "val": _write_split(
                campaigns_src=FPA_CAMPAIGNS_VAL_VAL,
                stats_src=FPA_STATS_VAL_VAL,
                campaigns_dst=FPA_EXPERIMENT_SUBSAMPLE_CAMPAIGNS_VAL,
                stats_dst=FPA_EXPERIMENT_SUBSAMPLE_STATS_VAL,
                n_campaigns=args.val_campaigns,
                seed=args.seed + 1,
            ),
            "test_holdout": _write_split(
                campaigns_src=FPA_CAMPAIGNS_HOLDOUT_TEST,
                stats_src=FPA_STATS_HOLDOUT_TEST,
                campaigns_dst=FPA_EXPERIMENT_SUBSAMPLE_CAMPAIGNS_HOLDOUT,
                stats_dst=FPA_EXPERIMENT_SUBSAMPLE_STATS_HOLDOUT,
                n_campaigns=args.holdout_campaigns,
                seed=args.seed + 2,
            ),
        },
    }
    FPA_EXPERIMENT_SUBSAMPLE_METADATA.write_text(json.dumps(metadata, indent=2))
    print(f"Prepared permanent experiment subsample under {output_root}")


def _write_split(
    *,
    campaigns_src: Path,
    stats_src: Path,
    campaigns_dst: Path,
    stats_dst: Path,
    n_campaigns: int,
    seed: int,
) -> dict:
    campaigns_df = pd.read_csv(campaigns_src)
    stats_df = pd.read_csv(stats_src)

    sample_size = min(int(n_campaigns), len(campaigns_df))
    sampled_campaigns = (
        campaigns_df
        .sample(n=sample_size, random_state=seed)
        .sort_values("campaign_id")
        .reset_index(drop=True)
    )
    campaign_ids = sampled_campaigns["campaign_id"].astype(int)
    sampled_stats = (
        stats_df[stats_df["campaign_id"].astype(int).isin(set(campaign_ids.tolist()))]
        .copy()
        .reset_index(drop=True)
    )

    campaigns_dst.parent.mkdir(parents=True, exist_ok=True)
    sampled_campaigns.to_csv(campaigns_dst, index=False)
    sampled_stats.to_csv(stats_dst, index=False)
    return {
        "source_campaigns_path": str(campaigns_src),
        "source_stats_path": str(stats_src),
        "output_campaigns_path": str(campaigns_dst),
        "output_stats_path": str(stats_dst),
        "seed": int(seed),
        "campaign_count": int(len(sampled_campaigns)),
        "stats_rows": int(len(sampled_stats)),
    }


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from example_notebooks.experiments.infra.split_registry import list_split_sets
from example_notebooks.experiments.rlb.profiles import build_config, list_profiles
from example_notebooks.experiments.shared_runner import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a canonical RLB family experiment.")
    parser.add_argument("--run-name", choices=list_profiles(), default="rlb_default")
    parser.add_argument("--split-set", choices=list_split_sets(), default="subsample_train_val_holdout")
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--artifacts-root", type=str, default=None)
    args = parser.parse_args()

    config = build_config(
        args.run_name,
        split_set=args.split_set,
        experiments_data_dir=None if args.artifacts_root is None else Path(args.artifacts_root),
    )
    if args.n_trials is not None:
        config = replace(config, n_trials=int(args.n_trials))
    summary = run_experiment(config, verbose=args.verbose)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

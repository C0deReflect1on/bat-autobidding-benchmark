from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from example_notebooks.experiments.drlb.profiles import build_config, get_profile, list_profiles
from example_notebooks.experiments.infra.split_registry import list_split_sets
from example_notebooks.experiments.shared_runner import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a canonical DRLB family experiment.")
    parser.add_argument(
        "--run-name",
        default="drlb_smooth",
        help="Artifact directory name under drlb/. If --profile is omitted, must be a known profile key.",
    )
    parser.add_argument(
        "--profile",
        choices=list_profiles(),
        default=None,
        help="Built-in hyperparameter profile; defaults to the same value as --run-name.",
    )
    parser.add_argument("--split-set", choices=list_split_sets(), default="subsample_train_val_holdout")
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--artifacts-root", type=str, default=None)
    args = parser.parse_args()

    profile_key = args.profile or args.run_name
    profile = get_profile(profile_key)
    config = build_config(
        args.run_name,
        profile=args.profile,
        split_set=args.split_set,
        experiments_data_dir=None if args.artifacts_root is None else Path(args.artifacts_root),
    )
    if args.n_trials is not None:
        config = replace(config, n_trials=int(args.n_trials))
    if args.max_train_steps is not None:
        config = replace(config, max_steps=int(args.max_train_steps))

    summary = run_experiment(
        config,
        verbose=args.verbose,
        base_drlb_params=profile["base_drlb_params"],
        baseline_model_params=profile["baseline_model_params"],
        exp_type=profile["exp_type"],
        objective=profile["objective"],
        search_space_fn=profile["search_space_fn"],
        n_trials=config.n_trials,
        max_train_steps=config.max_steps,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

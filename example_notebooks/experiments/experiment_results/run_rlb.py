from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from example_notebooks.experiments.exp_configs import ExperimentResultsRlbConfig
from example_notebooks.experiments.experiment_results.common import require_subsample_ready
from example_notebooks.experiments.shared_runner import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the simple RLB bidder experiment on the permanent subsample.")
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    require_subsample_ready()
    config = ExperimentResultsRlbConfig()
    if args.n_trials is not None:
        config = replace(config, n_trials=int(args.n_trials))
    summary = run_experiment(config, verbose=args.verbose)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

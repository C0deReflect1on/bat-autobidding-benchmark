from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd


def discover_run_dirs(root: Path | None = None) -> list[Path]:
    base_root = root or Path(__file__).resolve().parents[1]
    run_dirs: list[Path] = []
    for family in ("baselines", "rlb", "drlb"):
        family_dir = base_root / family
        if not family_dir.exists():
            continue
        for candidate in sorted(family_dir.iterdir()):
            if (candidate / "outputs" / "metrics.json").exists():
                run_dirs.append(candidate)
    return run_dirs


def load_metrics_table(run_dirs: Iterable[Path] | None = None) -> pd.DataFrame:
    rows = []
    for run_dir in run_dirs or discover_run_dirs():
        summary = _load_json(run_dir / "outputs" / "run_summary.json")
        metrics = _load_json(run_dir / "outputs" / "metrics.json")
        rows.append(
            {
                "family": summary["family"],
                "run_name": summary["run_name"],
                "split_set": summary["split_set"],
                "split_fingerprint": summary["split_fingerprint"],
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def assert_matching_split_fingerprint(run_dirs: Iterable[Path] | None = None) -> str:
    fingerprints = {
        _load_json(run_dir / "config" / "split_manifest.json")["fingerprint"]
        for run_dir in (run_dirs or discover_run_dirs())
    }
    if not fingerprints:
        raise ValueError("No completed run directories were found.")
    if len(fingerprints) != 1:
        raise ValueError(f"Runs use different split fingerprints: {sorted(fingerprints)}")
    return next(iter(fingerprints))


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())

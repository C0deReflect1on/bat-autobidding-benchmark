from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .split_utils import build_split_manifest


def score_to_dict(
    score: tuple,
    skipped_campaigns: int | None = None,
    time_inference_sec: float | None = None,
    time_overall_sec: float | None = None,
) -> dict[str, Any]:
    return {
        "cpc_relative": float(score[0]),
        "rmse": float(score[1]),
        "clicks_sum": float(score[2]),
        "quickspend": float(score[3]),
        "skipped_campaigns": None if skipped_campaigns is None else int(skipped_campaigns),
        "time_inference_sec": None if time_inference_sec is None else float(time_inference_sec),
        "time_overall_sec": None if time_overall_sec is None else float(time_overall_sec),
    }


def build_summary_header(
    config,
    data_splits: dict[str, dict[str, str]],
) -> dict[str, Any]:
    split_manifest = build_split_manifest(config, data_splits)
    return {
        "experiment_name": config.experiment_name,
        "family": config.family,
        "run_name": config.run_name,
        "auction_mode": config.auction_mode,
        "objective_metric": config.objective_metric,
        "objective_type": config.objective_type,
        "split_set": config.split_set,
        "split_fingerprint": split_manifest["fingerprint"],
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_hash": _git_hash(),
        "data_splits": json_ready(split_manifest["splits"]),
        "seeds": json_ready(config.seeds),
    }


def write_normalized_config(config) -> Path:
    config.ensure_artifact_dirs()
    path = config.config_dir / "normalized_config.json"
    path.write_text(json.dumps(json_ready(config.to_dict()), indent=2))
    return path


def write_split_manifest(config, data_splits: dict[str, dict[str, str]]) -> Path:
    config.ensure_artifact_dirs()
    path = config.config_dir / "split_manifest.json"
    payload = build_split_manifest(config, data_splits)
    path.write_text(json.dumps(json_ready(payload), indent=2))
    return path


def write_run_summary(config, summary: dict[str, Any]) -> Path:
    config.ensure_artifact_dirs()
    path = config.outputs_dir / "run_summary.json"
    path.write_text(json.dumps(json_ready(summary), indent=2))
    return path


def write_metrics(config, metrics: dict[str, Any]) -> Path:
    config.ensure_artifact_dirs()
    payload = json.dumps(json_ready(metrics), indent=2)
    output_path = config.outputs_dir / "metrics.json"
    root_path = config.experiment_dir / "metrics.json"
    output_path.write_text(payload)
    root_path.write_text(payload)
    return output_path


def append_runs_index(
    config,
    metrics: dict[str, Any],
    *,
    stage: str,
    label: str,
) -> Path:
    config.ensure_artifact_dirs()
    path = config.outputs_dir / "runs_index.jsonl"
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "family": config.family,
        "run_name": config.run_name,
        "stage": stage,
        "label": label,
        "clicks_sum": metrics.get("clicks_sum"),
        "rmse": metrics.get("rmse"),
        "cpc_relative": metrics.get("cpc_relative"),
        "quickspend": metrics.get("quickspend"),
    }
    with open(path, "a") as f:
        f.write(json.dumps(json_ready(entry)) + "\n")
    return path


def json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return json_ready(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def _git_hash() -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
            or None
        )
    except Exception:
        return None

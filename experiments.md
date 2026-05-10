# Experiment Architecture

## Goal

This document defines the target experiment architecture for this repository.
It is the source of truth for how we should run baseline, `rlb`, and `drlb`
experiments in a way that is:

- reproducible
- simple to set up
- low-duplication
- consistent from config to training budget to final evaluation
- compatible with the current BAT simulator concepts

This is a target architecture document, not a claim that every current script
already follows it.

## Current Repo Reality

The current repository already has useful building blocks, but they are split
across different experiment styles.

### What already exists

- `config.py` defines canonical dataset paths.
- `example_notebooks/experiments/base_exp_config.py` defines a reusable
  `ExperimentConfig` with artifact directories and core metadata.
- `example_notebooks/experiments/exp_configs.py` contains named experiment
  presets.
- `example_notebooks/experiments/runner_utils.py` already centralizes some
  DRLB train/evaluate/report logic.
- `example_notebooks/evaluate_baselines/baselines_finetune.py` contains a
  separate baseline tuning pipeline.
- `simulator/model/drlb_bidder.py` and `simulator/model/drlb/config_types.py`
  define a DRLB adapter with a typed config parser.

### Main gaps to close

- Baselines, `rlb`, and `drlb` do not share one experiment contract.
- Split naming is inconsistent: some code uses `train` / `test`, some uses
  `train_val` / `val_val` / `holdout_test`.
- Seeds are not controlled end-to-end from one experiment config.
- DRLB still has internal hard-coded seeds in parts of the implementation.
- Validation and final holdout reporting are not cleanly separated in all
  experiment scripts.
- Artifact schemas differ between older and newer experiment flows.

## Design Principles

### 1. One experiment contract

Every experiment, regardless of family, must run from one normalized Python
config object.

### 2. One data split vocabulary

All experiment code must reason in the same split roles:

- `train`
- `val`
- `test_holdout`

Legacy names may still be accepted during migration, but they must be
normalized immediately at the experiment boundary.

### 3. One orchestration flow

The runner should own split resolution, seed setup, tuning, refit, holdout
evaluation, and artifact writing. Family-specific code should only define model
defaults, search spaces, and train/eval hooks.

### 4. Reproducible by config

Dataset reproducibility comes from fixed imported paths in `config.py`.
Runtime reproducibility comes from one master experiment seed and deterministic
seed derivation for all stochastic stages.

### 5. Validation is not test

The split used for parameter selection must not be presented as final
performance. Final reporting must come from an untouched holdout split.

## Target Architecture

### Core Objects

### `ExperimentConfig`

Every runnable experiment should be expressible as one Python config object
with the following sections.

```python
@dataclass(frozen=True)
class ExperimentConfig:
    experiment_name: str
    family: str                 # baseline | rlb | drlb
    auction_mode: str           # FPA | VCG
    objective_metric: str       # SCR | RMSE | CPC_REL
    objective_type: str         # clicks | contacts
    split_set: str              # name of canonical split bundle
    data_config: dict
    repro: ReproConfig
    training: TrainingConfig
    tuning: TuningConfig
    artifacts: ArtifactConfig
    model: dict
```

The exact dataclass decomposition may vary, but the normalized fields above are
required.

### `ReproConfig`

```python
@dataclass(frozen=True)
class ReproConfig:
    master_seed: int
    data_seed: int
    optuna_seed: int
    model_seed: int
    replay_buffer_seed: int
    train_seed: int
    eval_seed: int
    deterministic: bool = True
```

Policy:

- `master_seed` is the only seed a human must choose.
- All other seeds are deterministic derivatives of `master_seed`.
- The default derivation rule is fixed offsets:
  - `data_seed = master_seed + 1000`
  - `optuna_seed = master_seed + 2000`
  - `model_seed = master_seed + 3000`
  - `replay_buffer_seed = master_seed + 4000`
  - `train_seed = master_seed + 5000`
  - `eval_seed = master_seed + 6000`
- Derived seeds must be stored in run metadata so the run is replayable.

### `TrainingConfig`

Training configuration must make resource usage explicit and comparable.

```python
@dataclass(frozen=True)
class TrainingConfig:
    max_epochs: int | None
    max_steps: int | None
    checkpoint_policy: str      # none | best_val | final
    refit_on: str               # train | train_plus_val
```

Rules:

- At least one of `max_epochs` or `max_steps` must be explicit for trainable
  approaches.
- The same config that defines data and seed must also define training budget.
- Final refit scope must be explicit and cannot be inferred from runner code.

### `TuningConfig`

```python
@dataclass(frozen=True)
class TuningConfig:
    enabled: bool
    n_trials: int
    sampler: str                # tpe | random | grid
    pruner: str | None
    optimize_split: str         # val
    optimize_metric: str        # SCR | RMSE | CPC_REL
```

Rules:

- Optuna settings belong in config, not in each script body.
- `n_trials` must be recorded in the experiment config and run summary.
- Tuning always optimizes on `val`, never on `test_holdout`.

### `ArtifactConfig`

```python
@dataclass(frozen=True)
class ArtifactConfig:
    experiment_root: Path
    write_trial_index: bool = True
    write_best_params: bool = True
    write_best_model: bool = True
    write_diagnostics: bool = True
```

## Canonical Split Layer

### Source of truth

All dataset paths come from `config.py`.

This repo already defines stable constants such as:

- `FPA_CAMPAIGNS_TRAIN`
- `FPA_STATS_TRAIN`
- `FPA_CAMPAIGNS_TEST`
- `FPA_STATS_TEST`
- `FPA_CAMPAIGNS_TRAIN_VAL`
- `FPA_STATS_TRAIN_VAL`
- `FPA_CAMPAIGNS_VAL_VAL`
- `FPA_STATS_VAL_VAL`
- `FPA_CAMPAIGNS_HOLDOUT_TEST`
- `FPA_STATS_HOLDOUT_TEST`

The future experiment system should not invent alternate path sources.

### Canonical split registry

Experiment code should resolve named split bundles to the normalized roles:

```python
{
    "train": {"campaigns_path": ..., "stats_path": ...},
    "val": {"campaigns_path": ..., "stats_path": ...},
    "test_holdout": {"campaigns_path": ..., "stats_path": ...},
}
```

Example mappings:

- Fixed train/val/holdout experiments:
  - `train` -> `FPA_*_TRAIN_VAL`
  - `val` -> `FPA_*_VAL_VAL`
  - `test_holdout` -> `FPA_*_HOLDOUT_TEST`

- Legacy train/test experiments:
  - `train` -> `FPA_*_TRAIN`
  - `val` -> `FPA_*_TEST` only for backward-compat tuning smoke flows
  - `test_holdout` -> `FPA_*_TEST` only when a true holdout does not yet exist

Important:

- The architecture target is to use true `train` / `val` / `test_holdout`.
- Legacy train/test-only experiments are transitional, not the desired steady
  state.

### Shared data configs across families

The same `data_config` structure must be reusable for:

- baseline tuning
- `rlb` tuning/evaluation
- `drlb` tuning/evaluation

This means data config cannot contain family-specific assumptions.

### Subsampling

Subsampling is allowed only as a declared experiment mode.

Rules:

- subsampling must be driven by `data_seed`
- the sampled split is derived from canonical paths in `config.py`
- the sampling fraction must live in config, not hidden in notebook code
- sampled train/val subsets must preserve campaign-to-stats consistency

## Runner Contract

All experiment families should execute through the same high-level flow.

### Stage 1. Normalize config

The runner loads a named experiment config and produces one normalized object.

This stage is responsible for:

- validating required fields
- resolving legacy aliases
- ensuring split roles are present
- ensuring tuning/training settings are explicit

### Stage 2. Resolve splits

The runner resolves the experiment's split bundle into:

- `train`
- `val`
- `test_holdout`

No downstream model code should need to understand legacy split naming.

### Stage 3. Freeze reproducibility

Before model creation or tuning:

- initialize Python random
- initialize NumPy
- initialize torch if used
- initialize Optuna sampler seed
- initialize any replay buffer seed
- initialize any family-specific sampling/randomness

The seed map must come from `ReproConfig`, not from hard-coded `0` values.

### Stage 4. Run tuning

If tuning is enabled:

- fit candidates on `train`
- score candidates on `val`
- record all trial params and metrics
- choose best params strictly by `optimize_metric`

If tuning is disabled:

- treat the provided model params as the selected params

### Stage 5. Refit selected configuration

After selecting params:

- refit once on the configured refit scope
- allowed scopes are `train` and `train_plus_val`

The chosen scope must be explicit in config and in the run summary.

### Stage 6. Final evaluation

Evaluate the refit model exactly once on `test_holdout`.

This stage produces the only metrics that should be described as final test
results.

### Stage 7. Write artifacts

Artifact writing is centralized and family-agnostic.

## Family Adapter Interface

Each family should plug into the shared runner through a small adapter layer.

```python
class ExperimentFamilyAdapter(Protocol):
    def build_baseline_params(self, config) -> dict: ...
    def build_search_space(self, trial, config) -> dict: ...
    def fit(self, train_split, params, config): ...
    def evaluate(self, split, params, config, fitted_model=None) -> dict: ...
    def save_model(self, fitted_model, path): ...
```

The interface can be refined, but the separation must remain:

- shared runner owns orchestration
- adapters own family-specific behavior

### Baseline adapter

Responsibilities:

- map unified config to bidder params
- define Optuna search spaces for baseline bidders
- run `autobidder_check` for validation and holdout evaluation
- save best params in the standard artifact schema

The intended interaction with
`example_notebooks/evaluate_baselines/baselines_finetune.py` is:

- keep `baselines_finetune.py` as the per-baseline tuning backend
- keep its one-baseline-at-a-time workflow because it is useful for
  reproducibility and debugging
- stop letting it own experiment-wide concerns such as split naming,
  artifact layout, and final holdout reporting
- call it through the shared experiment runner via the baseline adapter

In other words, the new architecture should not throw away
`baselines_finetune.py`. It should preserve its focused baseline-specific
search logic while moving cross-family orchestration into shared utilities.

Target boundary:

- shared runner owns split resolution, seed derivation, artifact paths, and
  summary schema
- `baselines_finetune.py` owns baseline-specific search spaces and tuning logic

Preferred future direction:

- refactor `baselines_finetune.py` toward a callable API that accepts
  normalized train/val inputs
- keep the direct one-model-at-a-time entrypoints available for manual
  baseline reproduction

### RLB adapter

Responsibilities:

- expose `rlb` under the same split, seed, and artifact contract
- reuse the same validation/holdout flow as baselines and `drlb`

Even if `rlb` has lighter fitting needs, it should not have a separate runner
architecture.

### DRLB adapter

Responsibilities:

- build normalized DRLB params from experiment config
- call `DRLBBidder.fit`
- save/load model checkpoints
- expose training diagnostics
- evaluate through the same split contract

Important DRLB-specific requirement:

- DRLB must receive experiment-controlled seeds instead of using internal
  hard-coded seeds

## DRLB Requirements

The current DRLB direction is still valid, even if some older experiment
scripts may no longer compile against the latest codebase.

The intended setup should be preserved.

### Config path

DRLB should continue using typed config parsing, but the experiment system must
be the outer source of truth.

Target layering:

1. `ExperimentConfig` defines the experiment.
2. The DRLB adapter translates experiment config into DRLB model/runtime config.
3. `DrlbConfigParser` validates and normalizes DRLB-specific parameters.

This keeps experiment logic out of low-level DRLB modules.

### Seed path

Current issue:

- DRLB internals still seed some components with fixed `0`.

Target:

- `model_seed` initializes networks
- `replay_buffer_seed` initializes replay buffers
- `train_seed` initializes any training-time randomness
- all DRLB stochasticity must be derived from `master_seed`

### Training budget

DRLB training must not hide resource limits in notebook logic.

The experiment config must explicitly define:

- `max_steps`
- optional `max_epochs`
- whether a short smoke budget or full budget is being used

### Evaluation contract

DRLB validation and DRLB final test must use the same evaluation interface as
other families.

That means:

- same split naming
- same metric keys
- same artifact layout
- same final holdout semantics

## Artifact Schema

Every experiment run should write the same core artifacts under its experiment
directory.

### Required files

- `config/normalized_config.json`
- `outputs/run_summary.json`
- `outputs/runs_index.jsonl` or equivalent tabular trial index
- `best_params/<family_or_model>_<metric>_<auction>.pkl` or a normalized
  replacement format
- `best_models/` checkpoint files when the family is trainable

### Optional files

- training diagnostics CSV
- campaign-level evaluation summaries
- evaluation history dumps

### Required run summary sections

The run summary must separate the stages explicitly:

```json
{
  "experiment_name": "...",
  "family": "drlb",
  "auction_mode": "FPA",
  "objective_metric": "SCR",
  "split_set": "...",
  "seeds": {
    "master_seed": 42,
    "data_seed": 1042,
    "optuna_seed": 2042,
    "model_seed": 3042,
    "replay_buffer_seed": 4042,
    "train_seed": 5042,
    "eval_seed": 6042
  },
  "tuning": {
    "enabled": true,
    "n_trials": 10,
    "best_trial_number": 3,
    "best_params": {}
  },
  "refit": {
    "scope": "train_plus_val"
  },
  "final_holdout": {
    "metrics": {}
  }
}
```

Important:

- validation metrics may appear in the summary
- final metrics must be under a clearly separate holdout section
- validation metrics must never be mislabeled as test results

## Simplicity Rules

To keep experiment setup simple, the architecture should follow these rules.

### Rule 1. Named experiments over custom notebook wiring

Most runs should start from a named config preset, not from manual notebook
assembly.

### Rule 2. Shared defaults with shallow overrides

Defaults should be defined once per family, then overridden only where needed.

Good:

- one baseline default block
- one `rlb` default block
- one `drlb` default block
- small named experiment overrides

Bad:

- repeating full param dictionaries in every notebook or runner

### Rule 3. Family-specific code only where behavior differs

Keep duplication out of:

- split resolution
- Optuna setup
- seed initialization
- artifact writing
- summary schema

Keep family-specific logic only in:

- model parameter translation
- search spaces
- fit/eval implementation details

### Rule 4. One place to read what happened

After any run, the operator should be able to inspect one experiment directory
and understand:

- what data split was used
- what seed policy was used
- how many trials ran
- what params were selected
- what holdout result was produced

## Migration Plan

This architecture should be introduced incrementally.

### Phase 1. Normalize experiment concepts

- keep current scripts working
- standardize split role names
- standardize seed names
- standardize artifact section names in summaries

### Phase 2. Unify runner utilities

- move shared logic for split resolution into one utility
- move shared logic for seed setup into one utility
- move shared logic for artifact writing into one utility
- keep adapters thin and family-specific

### Phase 3. Bring baselines onto the shared contract

- replace the separate baseline tuning flow with the shared runner contract
- keep existing bidder implementations, change only orchestration

### Phase 4. Bring `rlb` onto the shared contract

- expose `rlb` through the same config, split, and summary structure

### Phase 5. Finish DRLB seed integration

- remove hard-coded DRLB seeds
- pass experiment-derived seeds through DRLB config/runtime
- verify repeated runs with the same config produce the same result

### Phase 6. Retire legacy experiment wiring

Once parity is confirmed:

- stop adding new notebook-specific experiment pipelines
- keep compatibility loaders only where needed for old artifacts or configs

## Acceptance Criteria

The target architecture is implemented successfully when all of the following
are true.

### Reproducibility

- The same experiment config and the same master seed produce identical trial
  selection and final metrics.
- Changing the master seed changes only stochastic outcomes, not split
  resolution or artifact schema.

### Data consistency

- Baseline, `rlb`, and `drlb` can all consume the same normalized `data_config`.
- Campaign and stats paths remain paired correctly for every split role.

### Evaluation correctness

- Tuning uses `val`.
- Final reporting uses `test_holdout`.
- Final test numbers are never selected on the same split they report.

### Low duplication

- Shared orchestration logic exists in one place.
- Family adapters are small and only contain model-specific behavior.

### Ease of use

- A smoke experiment can be launched from one named config.
- A full experiment can be launched from one named config.
- A new experiment variant requires changing config values, not copying runner
  code.

## Defaults To Adopt

Unless there is a strong reason to override them, the following defaults should
be used.

- Use Python config objects as the experiment source of truth.
- Use canonical paths imported from `config.py`.
- Use normalized split roles: `train`, `val`, `test_holdout`.
- Use one master seed with deterministic derived seeds.
- Store tuning and final holdout metrics separately.
- Reuse one runner architecture for baselines, `rlb`, and `drlb`.
- Treat older DRLB experiment scripts as migration input, not as the final
  architecture.

## Non-Goals

This document does not require:

- file hashing for dataset identity
- changing the dataset path strategy already defined in `config.py`
- rewriting bidder algorithms themselves as part of experiment unification

The focus is experiment architecture and orchestration.

# Core Simulator Refactor Backlog (Extracted Spec)

Date: 2026-04-18
Source: `project_info/code_review_backlog.md`
Purpose: Keep non-DRLB simulator/metrics/orchestration refactors in one focused execution document while preserving global prioritization in the master backlog.

## 1. Scope

This file is the canonical implementation spec for non-DRLB backlog items:

- `CRB-001` (clipping/accounting consistency)
- `CRB-002` (RMSE timeline parity)
- `CRB-004` (RLB-DP cost semantics by auction mode)
- `CRB-005` (BROI objective-direction correctness)
- `CRB-008` (reproducibility fingerprint completeness)
- `CRB-009` (automated regression test baseline)
- `CRB-010` (metric contract/type alignment)
- `CRB-011` (orchestration duplication cleanup)
- `CRB-012` (explicit bidder capability interface)

## 2. Core Master Table


| ID      | Priority | Area                     | Type             | Validation Class              | Problem                                                                                                | Evidence                                                                                                                                                  | Proposed direction                                                                                                            | Dependencies                                | Acceptance criteria                                                                                        |
| ------- | -------- | ------------------------ | ---------------- | ----------------------------- | ------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| CRB-001 | P0       | Simulator/Metrics        | Algorithm risk   | Probable bug                  | Budget clipping is applied but unscaled values are recorded in history                                 | `simulator/simulation/simulate.py:204-214`, `simulator/simulation/simulate.py:231-236`                                                                    | Make clipped accounting canonical for metric-facing history (or persist explicit raw+clipped dual fields with clear contract) | None                                        | Deterministic clipping test proves history totals match budget delta and clipped state updates             |
| CRB-002 | P0       | Metrics                  | Algorithm risk   | Probable bug                  | RMSE time axis uses length semantics inconsistent with simulation loop horizon                         | `simulator/validation/metrics.py:101`, `simulator/validation/metrics.py:105`, `simulator/simulation/simulate.py:160`                                      | Unify step-count/horizon helper shared by simulator and metric computation                                                    | CRB-001                                     | Synthetic 24h/48h fixtures validate expected step counts and RMSE vector alignment                         |
| CRB-004 | P0       | RLB-DP                   | Algorithm risk   | Needs experiment confirmation | RLB-DP cost estimation (`bin_price * contacts`) drifts from evaluator spend semantics per auction mode | `simulator/model/rlb_dp_bidder.py:37`, `simulator/simulation/simulate.py:93`, `simulator/simulation/simulate.py:101`                                      | Add auction-mode-aware cost model in RLB-DP and document mode assumptions                                                     | CRB-001, CRB-002                            | Mode calibration tests on synthetic data pass for both `VCG` and `FPA`                                     |
| CRB-005 | P0       | Baseline tuning          | Algorithm risk   | Probable bug                  | BROI objective direction key mismatch can invert `SCR` optimization direction                          | `example_notebooks/evaluate_baselines/baselines_finetune.py:42`, `example_notebooks/evaluate_baselines/baselines_finetune.py:360`                         | Centralize metric-direction mapping and reuse everywhere in baseline tuning                                                   | None                                        | Tests assert `SCR -> maximize` for all baseline paths                                                      |
| CRB-008 | P1       | Reproducibility          | Test gap         | Probable gap                  | Data hash fingerprints campaigns only                                                                  | `example_notebooks/experiments/runner_utils.py:373-379`                                                                                                   | Include campaigns+stats+config+runtime fingerprint in summary metadata                                                        | None                                        | Summary contains separate train/test campaigns+stats hashes and runtime metadata; stats changes alter hash |
| CRB-009 | P1       | QA                       | Test gap         | Probable gap                  | No automated test suite for critical simulator/metrics/agent pathways                                  | `pyproject.toml:1-15`, `example_notebooks/test_rlb_dp.ipynb`, `example_notebooks/experiments/exp_train_test_drlb_dqn_smooth/drlb_train_test_smooth.ipynb` | Add deterministic unit/integration suite as regression baseline                                                               | CRB-001, CRB-002, CRB-004, CRB-005, CRB-008 | `tests/` suite exists with pass/fail entrypoint and at least one test per P0 core item                     |
| CRB-010 | P2       | Contracts/API            | Abstraction flaw | Probable gap                  | `compile_metrics` annotation/docs drift from actual 4-metric return payload                            | `simulator/validation/metrics.py:131-144`, `simulator/validation/metrics.py:179`                                                                          | Align type hints/docs/runtime schema to one explicit metric contract                                                          | None                                        | Static checks and docs match runtime payload shape                                                         |
| CRB-011 | P2       | Experiment orchestration | Duplication      | Probable gap                  | Legacy DRLB experiment pipeline duplicates shared orchestration logic                                  | `example_notebooks/experiments/exp_1_rnd_42_n10/drlb_experiment.py`, `example_notebooks/experiments/runner_utils.py`                                      | Consolidate to one canonical orchestration path with migration/deprecation policy                                             | CRB-008, CRB-009                            | Legacy path delegates to shared utilities or is deprecated with migration notes and parity check           |
| CRB-012 | P2       | Legacy coupling          | Overengineering  | Probable gap                  | Bidder capability routing depends on class-name string matching                                        | `simulator/simulation/simulate.py:8-13`                                                                                                                   | Replace with explicit capability flags/protocol contract on bidder interface                                                  | CRB-011                                     | No class-name string dispatch remains; compatibility validated with DRLB + non-DRLB bidders                |


## 3. Target Architecture (Core)

### 3.1 Accounting + Metric Consistency

- Simulator history, budget clipping, and KPI aggregation must use one explicit accounting contract.
- Time horizon/step-count definitions must be shared primitives, not duplicated formulas.

### 3.2 Evaluation Correctness

- Auction-mode-sensitive methods (notably RLB-DP) must align with evaluator spend semantics.
- Optimization objective mapping must be centralized to avoid metric-direction drift.

### 3.3 Reliability + Maintainability

- Reproducibility metadata should fingerprint all critical data inputs.
- Orchestration and capability routing should use explicit interfaces, not duplicated scripts or string-coupled checks.

## 4. Core Implementation Waves

### Core Wave A (Correctness)

- Items: `CRB-001`, `CRB-002`, `CRB-004`, `CRB-005`
- Exit gates:
  - accounting + RMSE parity tests pass
  - RLB-DP mode calibration completed
  - baseline objective direction tests pass

### Core Wave B (Reliability)

- Items: `CRB-008`, `CRB-009`
- Exit gates:
  - reproducibility metadata expanded and validated
  - regression test suite established for critical paths

### Core Wave C (Maintainability)

- Items: `CRB-010`, `CRB-011`, `CRB-012`
- Exit gates:
  - metric contract aligned and typed
  - orchestration duplication removed/deprecated
  - explicit bidder capability interface adopted

## 5. Definition of Done (Core Track)

- Simulator accounting and metrics are internally consistent.
- Core evaluation logic is mode-correct and objective-correct.
- Regression tests protect critical core paths.
- Public contracts and orchestration paths are explicit and non-duplicated.


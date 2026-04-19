# DRLB Refactor Backlog (Extracted Spec)

Date: 2026-04-18
Source: `project_info/code_review_backlog.md`
Purpose: Keep DRLB-specific correctness/refactor work in one focused execution document while preserving global prioritization in the master backlog.

## 1. Scope

This file is the canonical implementation spec for DRLB-related backlog items:

- `CRB-003` (simulator-driven DRLB fit transitions)
- `CRB-006` (`leastWinningCost` feature semantics)
- `CRB-007` (`potential_reward`/ratio signal quality)
- `CRB-013` (shared replay buffer for DQN + RewardNet)
- `CRB-014` (typed DRLB config/init/load normalization)

## 2. DRLB Master Table

| ID | Priority | Area | Type | Validation Class | Problem | Evidence | Proposed direction | Dependencies | Acceptance criteria |
|---|---|---|---|---|---|---|---|---|---|
| CRB-003 | P0 | DRLB training | Algorithm risk | Needs experiment confirmation | `fit` updates are action-insensitive relative to deployment dynamics | `simulator/model/drlb_bidder.py:415-424`, `simulator/model/drlb_bidder.py:320-327`, `simulator/simulation/simulate.py:79-88` | Use simulator-driven training transitions via shared step semantics (`simulate_step` + clipping/accounting), exposed through iterator/callback API (avoid direct monolithic `simulate_campaign` call in `fit`) | CRB-001, CRB-002 | Simulator-driven rollout path exists; parity tests confirm training transitions match inference semantics; ablation shows actions change targets |
| CRB-006 | P1 | DRLB features | Abstraction flaw | Needs experiment confirmation | `leastWinningCost` is currently a `prev_bid` proxy | `simulator/model/drlb_bidder.py:260` | Rename to true semantics or provide market-threshold estimate source | CRB-003 | Feature schema/docs/diagnostics align; finalized field is non-null on >95% of steps |
| CRB-007 | P1 | DRLB reward shaping | Algorithm risk | Needs experiment confirmation | `potential_reward == observed reward` in key paths | `simulator/model/drlb_bidder.py:240-244`, `simulator/model/drlb_bidder.py:422-423`, `simulator/model/drlb/state_representations.py:39-42` | Split observed vs potential reward source, or remove dependent ratio feature | CRB-003, CRB-006 | Ratio feature is non-degenerate in diagnostics; regression test covers variability |
| CRB-013 | P2 | DRLB infra | Duplication | Probable gap | Two near-identical `ReplayBuffer` implementations | `simulator/model/drlb/dqn.py:181-221`, `simulator/model/drlb/reward_net.py:125-157` | Shared replay-buffer abstraction with minimal configurable collation/schema | CRB-009 | DQN + RewardNet both import shared replay buffer; parity tests unchanged |
| CRB-014 | P1 | DRLB config | Abstraction flaw | Probable gap | Config parsing duplicated across `__init__` and `load_model` | `simulator/model/drlb_bidder.py:54-106`, `simulator/model/drlb_bidder.py:139-147`, `simulator/model/drlb_bidder.py:505-559` | Typed config dataclasses (`DRLBParams`, `OptimizerParams`, `Runtime/Env/State params`) + one `from_dict/validate/normalize` path | CRB-009 | Init/load use one parser; round-trip config tests pass; invalid values fail fast with explicit errors |

## 3. Target Architecture (DRLB)

### 3.1 Training/Inference Transition Parity

- Introduce shared transition primitive for DRLB loops that uses simulator accounting semantics.
- Keep per-step agent update logic in `fit`; simulator side produces the transition outcome.
- Ensure clipping and spend/click accounting are exactly the same in both training and inference pathways.

### 3.2 Typed Configuration Model

- Create dataclasses that separate concerns:
  - core bidder/model params
  - optimizer/training params
  - runtime/env/state params
- Centralize defaulting + coercion + validation in one path used by both init and checkpoint load.
- Preserve backward compatibility for existing checkpoints with explicit adapter layer.

### 3.3 DRLB Infrastructure Cleanup

- Consolidate replay buffer duplication into shared module with explicit tensor conversion behavior.
- Keep APIs small and explicit to avoid over-generalized abstractions.

## 4. DRLB Implementation Waves

### DRLB Wave A (Correctness First)

- Items: `CRB-003`
- Exit gates:
  - simulator-driven training transitions implemented
  - transition parity tests vs inference pass
  - action-sensitivity ablation documented

### DRLB Wave B (Feature + Config Reliability)

- Items: `CRB-014`, `CRB-006`, `CRB-007`
- Exit gates:
  - typed config lifecycle used by both init/load
  - least-winning-cost semantics aligned with naming
  - reward ratio signal no longer degenerate

### DRLB Wave C (Maintainability)

- Items: `CRB-013`
- Exit gates:
  - shared replay buffer adopted in DQN + RewardNet
  - no training-behavior regression in smoke tests

## 5. Definition of Done (DRLB Track)

- DRLB training uses shared simulator step semantics.
- DRLB configuration is typed, centralized, and validated.
- DRLB feature/reward semantics are documented and test-covered.
- DRLB infra duplication is reduced without behavior drift.

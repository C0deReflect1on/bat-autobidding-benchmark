# Overengineering Audit: `simulator/model/drlb` and `example_notebooks/experiments`

Date: 2026-04-22  
Status: Full source audit focused on readability cost  
Scope: all source files under `simulator/model/drlb` and `example_notebooks/experiments`, plus notebook entrypoints as thin consumers of that stack

## Executive Summary

After a full pass over the targeted source files, the overengineering problem is broader than just `_safe_float` and `_ensure_*`.

The strongest recurring pattern is this:

- the code introduces many small helpers, wrapper entrypoints, normalization steps, and indirection layers
- most of them are locally understandable
- but together they make the code harder to read than the underlying ideas actually are

The biggest sources of readability tax are:

1. `simulator/model/drlb/config_types.py`, which acts like a miniature schema library even though ordinary experiment execution uses a flat config
2. `simulator/model/drlb/rl_bid_agent_alibaba.py` and `state_representations.py`, where simple RL state updates are split across many tiny methods and lookup layers
3. `simulator/model/drlb/reward_net.py`, which still carries dead cache-style machinery that is not meaningfully used
4. `example_notebooks/experiments/base_exp_config.py`, which turns one config object into validator, path builder, seed registry, serializer, and factory
5. the whole experiments routing stack: `README` -> `notebook_api.py` -> `shared_runner.py` -> adapters -> infra helpers
6. `example_notebooks/experiments/adapters/drlb_adapter.py`, which is no longer just a runner and now also contains a diagnostics/reporting mini-framework

There are also a few places that are actually fine and should not be lumped into the same bucket:

- `simulator/model/drlb/replay_buffer.py` is small and readable
- `example_notebooks/experiments/all_comparison/loader.py` is simple and proportional
- most `__init__.py` files are harmless

## What I Reviewed

Reviewed source files under:

- `simulator/model/drlb/*.py`
- `example_notebooks/experiments/**/*.py`
- `example_notebooks/experiments/README.md`

Checked notebooks as consumers:

- they mainly call `example_notebooks.experiments.notebook_api`
- they are not the source of architectural complexity
- they mostly confirm that the abstraction stack is real and user-facing

Ignored generated outputs under `outputs/` and metrics snapshots for the purpose of judging architecture.

## DRLB Core Review

### `config_types.py`: the clearest overengineering hotspot

File: `simulator/model/drlb/config_types.py:63-358`

This file is still the strongest example of overengineering in the repo.

In the current codebase, normal experiment execution appears to use a flat config:

- experiment params are assembled as flat dicts in `example_notebooks/experiments/adapters/drlb_adapter.py:26-46`
- those dicts are passed into `DRLBBidder(...)` in `example_notebooks/experiments/adapters/drlb_adapter.py:298`
- `DRLBBidder.__init__` immediately calls `DrlbConfigParser.from_dict(params)` in `simulator/model/drlb_bidder.py:28-33`

The nested config shape is not the common run path. It is created only in checkpoints:

- checkpoint save writes nested config in `simulator/model/drlb_bidder.py:520-538`
- checkpoint load reads it through `from_checkpoint(...)` in `simulator/model/drlb_bidder.py:541-553`

So the parser currently supports two shapes, but only one of them is the everyday experiment shape.

Helpers that materially unsimplify reading:

- `from_dict(...)` and `from_checkpoint(...)` in `config_types.py:99-114`
- `_validate_checkpoint_config_shape(...)` in `config_types.py:116-159`
- `_flatten_raw(...)` in `config_types.py:161-194`
- `_build(...)` plus nested `get(...)` in `config_types.py:196-288`
- `_parse_lambda_action_betas(...)` in `config_types.py:290-301`
- `_ensure_in(...)`, `_as_float(...)`, `_as_optional_float(...)`, `_as_int(...)`, `_as_bool(...)` in `config_types.py:303-358`

Why this is overengineered:

- the code validates shape, flattens shape, then rebuilds typed nested dataclasses
- many tiny coercion helpers force the reader to jump around for simple fields
- the file behaves like a homegrown schema library, not a project-sized config loader
- `_flatten_raw(...)` is especially hard to justify when the routine path is already flat

This is exactly the kind of code that feels “safe” while making the main idea less visible.

The simplest accurate description of the current checkpoint path is:

`nested checkpoint config -> validate -> flatten -> rebuild typed nested dataclasses`

That is too much transformation for this repo.

### `rl_bid_agent_alibaba.py`: simple stateful logic split into too many hidden mutations

File: `simulator/model/drlb/rl_bid_agent_alibaba.py:10-274`

This file is not as obviously overengineered as `config_types.py`, but it is still a major readability problem.

The core idea is straightforward:

- maintain mutable episode state
- choose DQN action
- update lambda
- update reward model and DQN
- emit bid

Instead, that logic is scattered across many small methods:

- `_scale_budget(...)` and `_default_dqn_action_index(...)` in `rl_bid_agent_alibaba.py:12-18`
- `_get_state(...)` in `rl_bid_agent_alibaba.py:88-89`
- `_reset_episode(...)`, `configure_episode(...)`, `_update_step(...)`, `_reset_step(...)`, `_update_reward_cost(...)`, `_episode_done(...)`, `_record_step_history(...)`, `_model_upd(...)` in `rl_bid_agent_alibaba.py:108-230`
- `finalize_episode(...)`, `act(...)`, `calc_bid(...)` in `rl_bid_agent_alibaba.py:232-274`

Some of these helpers are reasonable. The problem is the total layering:

- `_get_state(...)` is only a thin delegation to `self.state_repr.get_state(self)`
- `_model_upd(...)` hides the main algorithmic step behind an opaque name
- resetting state is split across `_reset_episode(...)` and `_reset_step(...)` with many mutable fields
- `act(...)` is no longer “pick an action and return a bid”; it is an event processor coordinating state sync, timestep rollover, update calls, and bidding

This file suffers from too much state mutation plus too many helper boundaries. The reader has to mentally reconstruct one algorithm from a sequence of private methods and shared mutable fields.

Helpers that are especially questionable:

- `_get_state(...)`: trivial wrapper, low value
- `_default_dqn_action_index(...)`: tiny policy encoded as a helper
- `_scale_budget(...)`: tiny transformation, probably fine alone, but part of the “many tiny helpers” pattern
- `_model_upd(...)`: important logic hidden behind an unclear name

### `state_representations.py`: a framework-shaped solution for a small set of variants

File: `simulator/model/drlb/state_representations.py:18-203`

This file is a classic example of architecture that is clean on paper and heavy in practice.

What it does:

- defines a base class
- defines helper methods for metric computation
- defines four state representation classes
- encodes behavioural flags like `reward_net_order` and `uses_campaign_meta`
- adds lookup tables from `exp_type` to state family to instance

Helpers and abstractions that add reading cost:

- `BaseStateRepresentation` helper methods `_compute_common_ratios`, `_compute_cpi`, `_compute_cpm`, `_compute_reward_density` in `state_representations.py:27-40`
- the subclass split between `CpiRatioState`, `ImprovedState`, `ScaledBudgetState`, `HybridState`, and `DefaultState` in `state_representations.py:49-167`
- lookup tables `STATE_REPRESENTATIONS` and `EXP_TYPE_TO_STATE_FAMILY` in `state_representations.py:174-197`
- `get_state_repr(...)` in `state_representations.py:200-203`

Why this feels overengineered:

- there are only a handful of real variants
- behaviour is controlled partly by subclass methods and partly by configuration fields on those classes
- `reward_net_order` is a string flag living on state objects, but it actually controls learning order in `RlBidAgent._model_upd(...)`
- `uses_campaign_meta` is another flag that influences logic outside the state file

So this is not just “state representation.” It is a small behaviour registry that indirectly controls other modules.

That means the reader must understand:

1. `exp_type`
2. state family lookup
3. subclass fields
4. subclass `get_state(...)`
5. how `RlBidAgent` interprets `reward_net_order` and `uses_campaign_meta`

For this repo size, that is too much mechanism around a few state variants.

### `reward_net.py`: dead abstractions are still present

File: `simulator/model/drlb/reward_net.py:25-130`

This file contains one of the cleanest examples of dead complexity.

The main reward network path is simple:

- `add(...)`
- `step(...)`
- `act(...)`
- `learn(...)`

But the file also contains an extra cache-style subsystem:

- `self.M`, `self.S`, `self.V` in `reward_net.py:68-71`
- `add_to_M(...)` in `reward_net.py:86-90`
- `get_from_M(...)` in `reward_net.py:92-94`

These are effectively dead weight in the current code:

- `add_to_M(...)` and `get_from_M(...)` have no meaningful use sites in the targeted source tree
- `self.V` and `self.S` are only reset in `RlBidAgent._reset_episode(...)` at `rl_bid_agent_alibaba.py:132-133`
- the extra cache logic is not part of the active training story

This is not “maybe useful later” complexity. It is current readability debt.

Other readability issues:

- wildcard import `from .model import *` in `reward_net.py:11`
- global `device = torch.device("cpu")` in `reward_net.py:23`
- `set_seed()` called during object construction in `reward_net.py:54`

The class would be materially clearer if the dead cache machinery were removed and imports/seeding were made explicit.

### `dqn.py`: hidden policy complexity inside `act(...)`

File: `simulator/model/drlb/dqn.py:26-191`

This file is mostly readable, but it still has a few “why is this here?” layers.

The main one is `unimodal_check(...)` in `dqn.py:163-191`.

Instead of straightforward epsilon-greedy behaviour, `act(...)` in `dqn.py:97-127` branches into:

- normal epsilon-greedy if Q-values are “unimodal”
- increased exploration probability if they are not

That may have historical motivation, but from a readability point of view:

- it is hidden policy complexity
- it is not surfaced in config
- it makes `act(...)` significantly harder to scan than standard DQN policy code

Other smaller issues:

- `learn(..., gamma)` takes `gamma` as an argument even though the class already stores `self.gamma`
- `_soft_update(...)` is fine, but only one part of a file that still uses wildcard imports and global device constants
- `set_seed()` is called in `__init__`, adding another hidden global side effect

This is not the worst file, but it still tries to be more special than it needs to be.

### `model.py`: global seeding is hidden inside object construction

File: `simulator/model/drlb/model.py:17-52`

This file is small, but the hidden side effect is important.

`Network.__init__(...)` calls `set_seed()` in `model.py:31-32`, and `set_seed()` itself mutates global randomness state in `model.py:45-52`.

That means creating a network instance silently resets:

- Python random
- NumPy random
- Torch random
- CUDA deterministic flags

This is not just a helper. It is global runtime policy hidden in a model constructor.

That is a strong example of “helper function makes code look simpler locally while making behaviour less obvious globally.”

Also:

- both `dqn.py` and `reward_net.py` use `from .model import *`
- the file is tiny enough that explicit imports would be much clearer

### `replay_buffer.py`: mostly fine

File: `simulator/model/drlb/replay_buffer.py:18-75`

This is one of the few targeted files that is not meaningfully overengineered.

Why it works:

- small API
- direct data structures
- clear responsibility
- no excess helper stack

`collate_q_transitions(...)` and `collate_reward_transitions(...)` are proportionate helpers, not framework creep.

## Experiments Stack Review

### `base_exp_config.py`: one object doing too many jobs

File: `example_notebooks/experiments/base_exp_config.py:11-184`

This file is one of the largest contributors to experiment-stack complexity.

`ExperimentConfig` looks like a plain config object, but it also:

- validates names and modes in `__post_init__` at `base_exp_config.py:48-60`
- resolves split data in `base_exp_config.py:63-72`
- computes artifact directories in `base_exp_config.py:74-81`
- derives seed values in `base_exp_config.py:83-95`
- exposes directory creation in `ensure_artifact_dirs()` at `base_exp_config.py:97-101`
- owns path naming in `best_params_path(...)` at `base_exp_config.py:103-104`
- exposes a no-op alias `objective_metric` at `base_exp_config.py:106-108`
- serializes itself in `to_dict()` at `base_exp_config.py:122-134`
- infers split-set names in `_infer_split_set(...)` at `base_exp_config.py:136-145`
- exposes a factory `train_val(...)` at `base_exp_config.py:147-167`

Then there are more helper-level issues:

- `_json_ready(...)` in `base_exp_config.py:170-177` duplicates functionality also present in `infra/artifacts.py:112-121`
- `assert_unique_experiment_names(...)` in `base_exp_config.py:180-184` does not appear to be part of the active execution path
- `objective_metric` is just `return self.metric`, which adds naming surface without adding meaning

This file is overengineered because it turns “experiment config” into a god object.

### `notebook_api.py`: too many entrypoints for one user story

File: `example_notebooks/experiments/notebook_api.py:14-177`

This file is where notebook convenience turns into routing overhead.

The notebook user story is simple:

- choose a family
- choose a run name
- maybe override a few knobs
- run the experiment in-process

Instead the file introduces:

- `run_profile_inprocess(...)`
- `run_baseline_profile_inprocess(...)`
- `run_rlb_profile_inprocess(...)`
- `run_drlb_profile_inprocess(...)`
- `build_family_config(...)`
- `build_family_runner_kwargs(...)`
- `_normalize_family(...)`

Most of these are thin wrappers around other wrappers.

The main problem is not any single helper. It is that a notebook user now depends on a small routing API framework instead of one straightforward execution function.

Most questionable helpers:

- `_normalize_family(...)` in `notebook_api.py:173-177`: tiny aliasing helper with low value
- `build_family_runner_kwargs(...)` in `notebook_api.py:152-170`: hidden parameter assembly
- family-specific wrapper functions in `notebook_api.py:42-107`: convenience surface area bigger than necessary

### `shared_runner.py`: duplicated dispatch

File: `example_notebooks/experiments/shared_runner.py:12-75`

This file is small, but it duplicates itself almost line-for-line:

- `run_experiment(...)`
- `run_experiment_inprocess(...)`

Both:

- resolve normalized splits
- initialize seeds
- dispatch by family

The only real difference is which family adapter function is called.

That is not catastrophic, but it is still classic “one abstraction layer too many.”

### `infra/artifacts.py`: helper explosion around JSON writing

File: `example_notebooks/experiments/infra/artifacts.py:13-133`

This module is not useless, but it multiplies helper surface area in a way that hurts the common path.

Functions here:

- `score_to_dict(...)`
- `build_summary_header(...)`
- `write_normalized_config(...)`
- `write_split_manifest(...)`
- `write_run_summary(...)`
- `write_metrics(...)`
- `append_runs_index(...)`
- `json_ready(...)`
- `_git_hash(...)`

Problems:

- every `write_*` function repeats `config.ensure_artifact_dirs()`
- `json_ready(...)` duplicates logic already present in `base_exp_config.py`
- building headers, manifests, and JSON-safe serialization is spread across multiple helper boundaries

This is not horrible by itself, but it contributes strongly to the feeling that experiment runs are mediated by plumbing rather than direct code.

### `infra/split_utils.py`: too much machinery for a small split universe

File: `example_notebooks/experiments/infra/split_utils.py:17-128`

This file contains many small helpers:

- `resolve_normalized_splits(...)`
- `build_trainer_data_config(...)`
- `split_fingerprint(...)`
- `build_split_manifest(...)`
- `normalized_split_summary(...)`
- `_resolve_data_config(...)`
- `_resolve_split_paths(...)`
- `_ensure_split_paths_exist(...)`

For a large system this might be normal. In this repo, the split space is small and known.

So the helper density feels high relative to the problem:

- alias normalization for split roles
- summary building
- manifest building
- file existence checks
- full-file fingerprints

The code is not wrong, but it is another example of framework-shaped thinking being applied to a small, mostly fixed workflow.

Specific helpers that feel weak:

- `build_trainer_data_config(...)`: one-line wrapper around a tiny dict reshape
- `normalized_split_summary(...)`: another small structural wrapper
- `_resolve_data_config(...)` and `_resolve_split_paths(...)`: readable, but they add another stage of indirection around already small configs

### `infra/split_registry.py`: mostly okay, but has extra convenience surface

File: `example_notebooks/experiments/infra/split_registry.py:21-76`

This module is comparatively mild.

`resolve_split_set(...)` and `list_split_sets()` are fine. The main questionable bit is `split_registry_summary(...)`, which adds another convenience helper without much evidence that it is central to the runtime path.

This is not a core problem file, but it still participates in the “many little access layers” style.

### `infra/reproducibility.py`: small, but duplicates seeding philosophy elsewhere

File: `example_notebooks/experiments/infra/reproducibility.py:19-37`

This file by itself is okay.

The real issue is architectural inconsistency:

- experiments layer has explicit reproducibility helpers here
- DRLB core also resets seeds inside `model.py`, `dqn.py`, and `reward_net.py`

That means randomness policy is spread across different layers instead of being owned in one clear place.

### Family `profiles.py` files: repetitive scaffolding

Files:

- `example_notebooks/experiments/baselines/profiles.py:8-48`
- `example_notebooks/experiments/rlb/profiles.py:8-55`
- `example_notebooks/experiments/drlb/profiles.py:9-128`

Each family gets:

- a registry dict
- `list_profiles()`
- `get_profile(...)`
- `build_config(...)`

Individually this is fine. Collectively it is repetitive scaffolding.

The DRLB profile file adds even more structure:

- `_COMMON_BASE_PARAMS`
- `_COMMON_MODEL_PARAMS`
- `_WIDE_LAMBDA_ACTION_BETAS`
- `_base_search_space(...)`
- `_DRLB_PROFILES`

This is not the worst problem, but it keeps reinforcing the same pattern: a lot of surface area for what are basically named experiment presets.

### Family `runner.py` files: three very similar CLIs

Files:

- `example_notebooks/experiments/baselines/runner.py:18-38`
- `example_notebooks/experiments/rlb/runner.py:18-39`
- `example_notebooks/experiments/drlb/runner.py:18-66`

These files are simple, but they are also visibly duplicated.

Each one:

- sets up `argparse`
- builds config
- optionally `replace(...)` fields
- calls `run_experiment(...)`
- prints JSON summary

This is not severe overengineering, but it is more template duplication than clarity.

### `adapters/baseline_adapter.py`: too many model-specific routing helpers

File: `example_notebooks/experiments/adapters/baseline_adapter.py:30-257`

This adapter is readable, but still more layered than necessary.

Helpers and structures that add complexity:

- `_MODEL_TO_BIDDER` and `_MODEL_TO_TUNING_METHOD` in `baseline_adapter.py:30-44`
- `run_baseline_experiment(...)` wrapper in `baseline_adapter.py:47-57`
- `evaluate_baseline_model(...)` wrapper in `baseline_adapter.py:149-162`
- `_build_bidder_eval_params(...)` huge model-specific `if` ladder in `baseline_adapter.py:201-244`
- `_study_trials_summary(...)` in `baseline_adapter.py:247-257`

This file duplicates model knowledge across multiple places:

- class registry
- tuning method registry
- parameter conversion helper

That is not elegant abstraction. It is routing code spread across several structures.

### `adapters/rlb_adapter.py`: one more copy of the same orchestration template

File: `example_notebooks/experiments/adapters/rlb_adapter.py:26-266`

This file follows the same pattern as DRLB and baseline adapters:

- thin `run_*` wrapper
- in-process implementation
- candidate runner
- refit loader helper
- study summary helper

Key helpers:

- `default_rlb_search_space(...)` in `rlb_adapter.py:227-234`
- `load_rlb_refit_inputs(...)` in `rlb_adapter.py:237-257`
- `_study_trials_summary(...)` in `rlb_adapter.py:260-266`

The adapter is not terrible, but the architecture keeps repeating the same helper families across modules instead of centralizing the common orchestration pattern.

### `adapters/drlb_adapter.py`: experiment runner turned into diagnostics framework

File: `example_notebooks/experiments/adapters/drlb_adapter.py:26-585`

This is the heaviest experiments file by far.

It contains:

- bidder param assembly in `build_bidder_params(...)`
- wrapper entrypoint `run_drlb_experiment(...)`
- main orchestration `run_drlb_experiment_inprocess(...)`
- candidate execution `run_drlb_candidate(...)`
- refit helper `load_refit_training_frames(...)`
- diagnostics summarization `summarize_diagnostics(...)`
- artifact writing `write_training_diagnostics_artifacts(...)`
- three plotting helpers
- runtime diagnostics concatenation `_concat_runtime_diagnostics(...)`
- study summary helper `_study_trials_summary(...)`

Why this is overengineered:

- the file is doing runner logic, tuning logic, evaluation logic, metrics logic, artifact logic, diagnostics logic, and plotting logic
- `build_bidder_params(...)` in `drlb_adapter.py:26-46` silently injects many runtime defaults instead of making them explicit at the call site
- the diagnostics subsystem is effectively a second framework nested inside the experiment framework

Helpers that most hurt readability:

- `build_bidder_params(...)`
- `run_drlb_candidate(...)`
- `summarize_diagnostics(...)`
- `write_training_diagnostics_artifacts(...)`
- `plot_training_diagnostics(...)`
- `plot_reward_net_diagnostics(...)`
- `plot_action_distribution(...)`
- `_concat_runtime_diagnostics(...)`

None of these helpers are individually outrageous. The problem is that the file is now responsible for too many kinds of work, and helper extraction no longer simplifies that.

### `all_comparison/loader.py`: proportional and fine

File: `example_notebooks/experiments/all_comparison/loader.py:10-53`

This module is simple and proportional:

- `discover_run_dirs(...)`
- `load_metrics_table(...)`
- `assert_matching_split_fingerprint(...)`
- `_load_json(...)`

This is not where the complexity problem lives.

## Helpers That Look Actively Removable

These are the helpers that most strongly look like current readability debt rather than useful abstraction:

- `simulator/model/drlb/config_types.py:_flatten_raw`
- `simulator/model/drlb/config_types.py:_validate_checkpoint_config_shape`
- `simulator/model/drlb/config_types.py:_ensure_in`
- `simulator/model/drlb/config_types.py:_as_float`
- `simulator/model/drlb/config_types.py:_as_optional_float`
- `simulator/model/drlb/config_types.py:_as_int`
- `simulator/model/drlb/config_types.py:_as_bool`
- `simulator/model/drlb/reward_net.py:add_to_M`
- `simulator/model/drlb/reward_net.py:get_from_M`
- `simulator/model/drlb/model.py:set_seed`
- `example_notebooks/experiments/base_exp_config.py:objective_metric`
- `example_notebooks/experiments/base_exp_config.py:_json_ready`
- `example_notebooks/experiments/base_exp_config.py:assert_unique_experiment_names`
- `example_notebooks/experiments/notebook_api.py:_normalize_family`
- `example_notebooks/experiments/notebook_api.py:build_family_runner_kwargs`
- `example_notebooks/experiments/infra/split_utils.py:build_trainer_data_config`
- `example_notebooks/experiments/infra/split_utils.py:normalized_split_summary`
- `example_notebooks/experiments/infra/split_registry.py:split_registry_summary`
- repeated `_study_trials_summary(...)` helpers across adapters

## Helpers That Are Okay or Mostly Okay

These do not look like primary overengineering targets:

- `simulator/model/drlb/replay_buffer.py:ReplayBuffer`
- `simulator/model/drlb/replay_buffer.py:collate_q_transitions`
- `simulator/model/drlb/replay_buffer.py:collate_reward_transitions`
- `example_notebooks/experiments/all_comparison/loader.py:_load_json`
- `example_notebooks/experiments/infra/split_registry.py:resolve_split_set`
- `example_notebooks/experiments/infra/reproducibility.py:derive_seed_map`

They may be imperfect, but they are not the main readability problem.

## Most Important Simplifications

### 1. Collapse DRLB config handling to one visible path

- Keep the typed dataclasses.
- Stop maintaining a mini parser framework around them.
- Treat flat experiment params as the main shape.
- Handle checkpoint nested shape with one explicit converter, not `validate -> flatten -> build`.

### 2. Remove dead and hidden behaviour from the DRLB core

- delete `RewardNet` cache leftovers (`M`, `S`, `V`, `add_to_M`, `get_from_M`)
- stop calling `set_seed()` from model constructors
- replace wildcard imports with explicit ones
- either simplify or explicitly justify the `unimodal_check(...)` exploration logic

### 3. Make `ExperimentConfig` a boring data object

- move path materialization out
- move JSON conversion out
- keep only minimal validation in `__post_init__`
- remove no-op aliases and stale convenience helpers

### 4. Flatten the notebook and experiment routing stack

- one main in-process entrypoint is enough
- one common family-dispatch table is enough
- avoid separate wrapper layers that mainly rename arguments
- merge the duplicated `run_experiment(...)` and `run_experiment_inprocess(...)` structure

### 5. Split diagnostics/reporting out of `drlb_adapter.py`

- if plots are needed, put them in a separate diagnostics module
- keep the runner focused on training/evaluation orchestration
- stop growing experiment adapters into mini application layers

## Bottom Line

The repo does not just have a few ugly helpers. It has a repeated design habit:

- normalize
- wrap
- dispatch
- summarize
- serialize
- then finally do the actual work

That habit shows up in both targeted areas.

The worst files are:

- `simulator/model/drlb/config_types.py`
- `example_notebooks/experiments/base_exp_config.py`
- `example_notebooks/experiments/notebook_api.py`
- `example_notebooks/experiments/shared_runner.py`
- `example_notebooks/experiments/adapters/drlb_adapter.py`

The most obvious dead or stale complexity is in:

- `simulator/model/drlb/reward_net.py`
- repeated JSON/artifact helpers in `example_notebooks/experiments`

Your original instinct was correct, but the full review makes the claim stronger:

the issue is not only messy `_safe_*` and `_ensure_*` helpers. The issue is that many parts of this repo have been turned into helper stacks and routing layers that make the code look more general, more defensive, and more “architected” than the actual problem requires.

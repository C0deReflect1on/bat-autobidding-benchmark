# Agent Rules (Global)

These rules apply to the whole repository.

1. Do not implement input type checks anywhere.
2. Do not implement input validation helpers anywhere.
3. Do not implement numeric/boolean/string coercion helpers anywhere.
4. Do not add defensive parsing wrappers or fallback parsing logic.
5. Read input/config fields directly and keep logic minimal.
6. Do not add pass-through wrapper methods that only proxy another call (for example `func(self): self._func()`).
7. Keep state access explicit through `state_repr` and `curr_state`; do not use dynamic attribute proxying (for example `__getattr__`) to expose all state fields.
8. Use clear transition naming for RL flow (`state_before_action`, `state_after_outcome`) and avoid ambiguous naming like unclear `prev_state`.
9. In key algorithmic loops (especially training/fit loops), keep short step comments that map to algorithm phases (observe, act, step env, learn DQN, update RewardNet, flush episode).
10. Do not split a single logical transition into multiple public method calls unless there is a real runtime need; if split is required, document the reason in code.

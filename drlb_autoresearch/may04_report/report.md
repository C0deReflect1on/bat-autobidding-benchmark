# May 04 DRLB Diagnostics Report

Generated at `2026-05-05T07:27:31+00:00` from saved May 04 run summaries and persisted `best_refit.pt` checkpoints.

## Artifacts

- `summary.json`: aggregate metrics, deltas vs linear, transfer rows, diagnostics segment summaries.
- `aggregate_metrics.csv`: train/val/holdout metrics for linear and all DRLB runs.
- `diagnostics_segments.csv`: DQN, RewardNet, reward, lambda, epsilon, and action statistics by training-step bands.
- `campaign_stats.csv`: per-campaign DRLB vs locked linear deltas.
- `hourly_stats.csv`: campaign-hour trajectories with DRLB, linear, and delta columns.
- `top_divergent_campaigns.csv`: earliest campaigns plus strongest holdout divergences.
- `diagnostics_segments.png`, `action_distribution.png`, `hourly_trajectories.png`: quick visual checks.

## Aggregate Ranking

| run_key | bidder | clicks_sum | clicks_sum_delta_vs_linear | rmse | cpc_relative | quickspend | average_end_balance_share |
| --- | --- | --- | --- | --- | --- | --- | --- |
| linear | linear | 17792.734 | 0.000 | 1.503 | 421.338 | 0.005 | 0.335 |
| default | drlb | 13013.700 | -4779.034 | 1.730 | 401.078 | 0.084 | 0.474 |
| ratio_bat | drlb | 11719.541 | -6073.192 | 1.689 | 719.600 | 0.086 | 0.491 |
| ta_ratio_bat | drlb | 11715.777 | -6076.957 | 1.929 | 519.273 | 0.131 | 0.372 |
| default_lambda_rule | drlb | 9441.814 | -8350.919 | 1.362 | 1721.031 | 0.033 | 0.731 |

The locked `LinearBidder` remains the strongest holdout policy by clicks. Among DRLB variants, `default` produces the most holdout clicks, while `ta_ratio_bat` is the best validation scorer but does not transfer to holdout. All DRLB refit checkpoints trail linear on holdout clicks and carry materially worse pacing/efficiency metrics.

## Validation to Refit Transfer

| run_key | best_val_clicks_sum | refit_val_clicks_sum | refit_holdout_clicks_sum | refit_holdout_delta_clicks_vs_linear | best_val_rmse | best_val_cpc_relative |
| --- | --- | --- | --- | --- | --- | --- |
| default | 2346.814 | 2290.230 | 13013.700 | -4779.034 | 1.340 | 623.641 |
| ratio_bat | 2407.070 | 2242.961 | 11719.541 | -6073.192 | 1.615 | 470.349 |
| ta_ratio_bat | 2637.439 | 2154.233 | 11715.777 | -6076.957 | 1.491 | 81.021 |
| default_lambda_rule | 2176.214 | 1827.071 | 9441.814 | -8350.919 | 1.519 | 821.291 |

The `best_val` rows are validation-time model-selection metrics from `run_summary.json`; exact `best_val` checkpoint files were not persisted. This report therefore uses the persisted `best_refit.pt` checkpoints for train, val, and holdout evaluation. The gap is important: Optuna picked parameters by validation clicks, then the chosen parameters were refit on train+val, so the persisted model is not the exact model that achieved the recorded validation score.

The validation objective is click-heavy and does not explicitly penalize unstable pacing enough. A trial can win validation clicks by bidding aggressively on a favorable validation mix, then refit into a different campaign composition and spend trajectory. The train/val/holdout campaign split also changes start-time windows and budget/category mix, so the same lambda trajectory can land in different traffic regimes.

## Non-Convergence Diagnosis

| run_key | segment | dqn_loss_mean | dqn_loss_p95 | reward_net_loss_mean | reward_net_loss_p95 | reward_signal_mean | lambda_start | lambda_end |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| default | (0,5000] | 5.6153 | 11.4026 | 109.0144 | 389.7863 | 5.2838 | 0.0031 | 0.0030 |
| default | (5000,10000] | 10.2897 | 21.7104 | 1971.4464 | 7061.9166 | 25.3452 | 0.0027 | 0.0027 |
| default | (10000,15000] | 13.9475 | 30.9709 | 1163.7452 | 4191.1296 | 11.3690 | 0.0027 | 0.0035 |
| default | (15000,20000] | 16.9387 | 39.7866 | 920.3248 | 2957.2340 | 8.3565 | 0.0037 | 0.0036 |
| default | (20000,25000] | 18.3437 | 44.9270 | 1002.8099 | 4271.7415 | 16.0686 | 0.0037 | 0.0028 |
| default | (25000,30000] | 20.4128 | 53.8080 | 1330.1651 | 5422.5042 | 9.4918 | 0.0027 | 0.0011 |
| default | (30000,40000] | 21.4371 | 59.6358 | 4738.6762 | 30617.4656 | 14.9529 | 0.0011 | 0.0028 |
| default | (40000,50000] | 21.7540 | 62.3160 | 3726.6840 | 26553.7624 | 13.3435 | 0.0025 | 0.0016 |
| default | (50000,60000] | 20.4368 | 58.8465 | 3262.8419 | 23554.5569 | 13.1948 | 0.0016 | 0.0026 |
| ratio_bat | (0,5000] | 2.0001 | 4.1130 | 60.7145 | 298.9783 | 5.2916 | 0.0031 | 0.0029 |
| ratio_bat | (5000,10000] | 5.9253 | 11.0456 | 5635.3312 | 19220.0270 | 20.0467 | 0.0027 | 0.0031 |
| ratio_bat | (10000,15000] | 10.0443 | 22.4997 | 2456.3948 | 10241.8995 | 15.2094 | 0.0031 | 0.0018 |
| ratio_bat | (15000,20000] | 15.3284 | 45.0606 | 1507.8721 | 6698.5801 | 10.3583 | 0.0019 | 0.0022 |
| ratio_bat | (20000,25000] | 8.9048 | 17.4898 | 993.1823 | 4429.7439 | 11.3453 | 0.0022 | 0.0028 |
| ratio_bat | (25000,30000] | 9.6885 | 18.8156 | 846.6571 | 3540.0916 | 18.5206 | 0.0029 | 0.0004 |
| ratio_bat | (30000,40000] | 12.7288 | 26.6797 | 859.0625 | 3484.8226 | 10.9190 | 0.0004 | 0.0028 |
| ratio_bat | (40000,50000] | 15.1811 | 39.8682 | 714.4382 | 2489.7976 | 11.7003 | 0.0030 | 0.0009 |
| ratio_bat | (50000,60000] | 12.1500 | 23.2886 | 749.0191 | 2721.3415 | 14.3602 | 0.0009 | 0.0025 |
| ta_ratio_bat | (0,5000] | 1.9828 | 3.9375 | 64.5183 | 315.9655 | 5.1575 | 0.0031 | 0.0027 |
| ta_ratio_bat | (5000,10000] | 6.2298 | 9.5567 | 10916.9978 | 35796.5045 | 26.6385 | 0.0025 | 0.0026 |
| ta_ratio_bat | (10000,15000] | 7.0292 | 10.4613 | 6156.0112 | 24261.4655 | 16.6289 | 0.0026 | 0.0025 |
| ta_ratio_bat | (15000,20000] | 6.5036 | 9.4014 | 3998.0536 | 18084.7926 | 13.5402 | 0.0027 | 0.0026 |
| ta_ratio_bat | (20000,25000] | 8.2997 | 13.2231 | 2942.4120 | 14148.5334 | 18.5263 | 0.0026 | 0.0002 |
| ta_ratio_bat | (25000,30000] | 9.4759 | 15.2789 | 2413.8882 | 11515.0657 | 22.8267 | 0.0002 | 0.0002 |
| ta_ratio_bat | (30000,40000] | 11.3439 | 18.6313 | 1910.8721 | 9667.7661 | 15.3794 | 0.0002 | 0.0016 |
| ta_ratio_bat | (40000,50000] | 12.2382 | 20.8428 | 1399.7450 | 6273.7242 | 15.6293 | 0.0015 | 0.0020 |
| ta_ratio_bat | (50000,60000] | 13.8811 | 25.1957 | 1264.6812 | 5236.1676 | 17.6791 | 0.0021 | 0.0015 |
| default_lambda_rule | (0,5000] | 6.2308 | 14.0340 | 61.8969 | 249.3688 | 9.0603 | 0.0047 | 0.0029 |
| default_lambda_rule | (5000,10000] | 9.9242 | 23.2888 | 835.2443 | 2603.3285 | 18.7260 | 0.0027 | 0.0053 |
| default_lambda_rule | (10000,15000] | 11.7170 | 29.7490 | 590.6216 | 2126.6917 | 14.1682 | 0.0052 | 0.0051 |
| default_lambda_rule | (15000,20000] | 16.6634 | 38.3238 | 489.6593 | 1822.5631 | 10.0371 | 0.0055 | 0.0016 |
| default_lambda_rule | (20000,25000] | 21.8661 | 44.2280 | 632.2370 | 2578.2770 | 15.7619 | 0.0016 | 0.0028 |
| default_lambda_rule | (25000,30000] | 21.5829 | 43.9846 | 719.3331 | 2920.1620 | 10.6097 | 0.0028 | 0.0025 |
| default_lambda_rule | (30000,40000] | 27.6545 | 51.6446 | 679.2470 | 2626.1987 | 11.4164 | 0.0025 | 0.0066 |
| default_lambda_rule | (40000,50000] | 33.9269 | 56.5450 | 664.9312 | 2481.4420 | 12.3192 | 0.0066 | 0.0004 |
| default_lambda_rule | (50000,60000] | 30.5847 | 50.1343 | 753.9576 | 2997.1274 | 14.3189 | 0.0004 | 0.0005 |

The diagnostics do not show a converged actor-critic style loop. DQN loss stays active after epsilon has annealed, while RewardNet loss remains large and often spikes late in refit. That is consistent with the current training loop: DQN learns from RewardNet predictions, while RewardNet is trained on high-variance return targets. With `gamma=1.0`, bootstrapped Q targets preserve long-horizon variance instead of damping it.

May 04 also removed finite lambda clipping. That gives the policy enough freedom to drift aggressively when RewardNet predictions shift. In the default state, absolute remaining budget and CPM-like signals sit next to small ratio features, so state scale mismatch can make the reward model and DQN fit sharp local patterns instead of a stable pacing rule.

## Default RewardNet Jump

For `default`, RewardNet mean loss rises from `1002.810` in `(20000,25000]` to `1330.165` in `(25000,30000]`; p95 moves from `4271.741` to `5422.504`.

The jump around 25-30k steps is most likely a data/order and target-regime effect rather than a single coding failure. The refit pass walks campaigns in campaign order; when the loop reaches a budget/category/time slice with larger realized returns, Monte Carlo-style RewardNet targets shift abruptly. DQN is already near low epsilon by then, so it is mostly exploiting predictions from a reward model whose target distribution has just changed. The resulting policy/lambda movement then feeds back into future observed transitions.

## Why `ta_ratio_bat` Looks Promising

| run_key | clicks_sum | clicks_sum_delta_vs_linear | rmse | quickspend | average_end_balance_share |
| --- | --- | --- | --- | --- | --- |
| linear | 3729.288 | 0.000 | 1.407 | 0.004 | 0.329 |
| default | 2290.230 | -1439.057 | 1.549 | 0.066 | 0.503 |
| ratio_bat | 2242.961 | -1486.326 | 1.573 | 0.074 | 0.490 |
| ta_ratio_bat | 2154.233 | -1575.055 | 1.796 | 0.132 | 0.377 |
| default_lambda_rule | 1827.071 | -1902.217 | 1.296 | 0.039 | 0.715 |

`ta_ratio_bat` has the most useful validation signal because traffic-aware pacing features give the policy more context than raw default state or budget-only ratios. But the added signal does not remove the unstable target problem. On holdout it underperforms because RewardNet still learns noisy return targets, the reward remains sparse/high-variance, and unconstrained lambda movement can turn a better state representation into faster policy drift.

## Campaign Slices

| selection_reason | split | run_key | campaign_id | logical_category | auction_budget | clicks_total_delta_vs_linear | spend_total_delta_vs_linear | balance_final_delta_vs_linear | pacing_error_delta_vs_linear |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| earliest_campaign_start | train | default | 72171396 | 1.210 | 96.000 | -1.925 | -1.068 | 0.015 | 0.000 |
| earliest_campaign_start | val | default | 72797337 | 1.200 | 230.400 | 0.503 | 38.440 | -38.440 | -0.167 |
| earliest_campaign_start | train | default | 72799055 | 1.700 | 1877.760 | -0.882 | -175.782 | 175.782 | 0.094 |
| earliest_campaign_start | val | default | 72803871 | 1.100 | 80.640 | 1.005 | 48.938 | -48.938 | -0.607 |
| earliest_campaign_start | val | default | 72804037 | 1.180 | 599.040 | -18.006 | 0.841 | 0.000 | 0.000 |
| earliest_campaign_start | train | default | 72804823 | 1.130 | 322.560 | -2.094 | 5.585 | -5.585 | -0.017 |
| earliest_campaign_start | val | default | 72805975 | 5.360 | 1933.820 | -0.396 | -215.798 | 215.798 | 0.112 |
| earliest_campaign_start | train | default | 72806735 | 1.100 | 49.920 | 0.047 | 0.496 | -0.496 | -0.010 |
| earliest_campaign_start | train | default | 72807079 | 5.260 | 2298.240 | 20.440 | -107.422 | 0.000 | 0.000 |
| earliest_campaign_start | train | default | 72812052 | 1.130 | 1065.600 | -2.374 | -566.594 | 566.594 | 0.532 |
| earliest_campaign_start | val | default | 72813453 | 5.260 | 614.400 | -42.457 | 125.293 | -73.614 | -0.120 |
| earliest_campaign_start | val | default | 72814242 | 2.310 | 201.600 | -1.374 | -118.422 | 118.422 | 0.587 |
| earliest_campaign_start | train | default_lambda_rule | 72171396 | 1.210 | 96.000 | 0.178 | -51.095 | 50.042 | 0.521 |
| earliest_campaign_start | val | default_lambda_rule | 72797337 | 1.200 | 230.400 | 0.014 | 0.889 | -0.889 | -0.004 |
| earliest_campaign_start | train | default_lambda_rule | 72799055 | 1.700 | 1877.760 | -0.897 | -176.275 | 176.275 | 0.094 |
| earliest_campaign_start | val | default_lambda_rule | 72803871 | 1.100 | 80.640 | -0.100 | -7.780 | 7.780 | 0.096 |
| earliest_campaign_start | val | default_lambda_rule | 72804037 | 1.180 | 599.040 | -29.515 | -583.592 | 579.467 | 0.967 |
| earliest_campaign_start | train | default_lambda_rule | 72804823 | 1.130 | 322.560 | -13.676 | -312.260 | 312.260 | 0.968 |
| earliest_campaign_start | val | default_lambda_rule | 72805975 | 5.360 | 1933.820 | -0.557 | -220.326 | 220.326 | 0.114 |
| earliest_campaign_start | train | default_lambda_rule | 72806735 | 1.100 | 49.920 | -0.264 | -4.542 | 4.542 | 0.091 |
| earliest_campaign_start | train | default_lambda_rule | 72807079 | 5.260 | 2298.240 | -6.889 | -2317.278 | 2200.209 | 0.957 |
| earliest_campaign_start | train | default_lambda_rule | 72812052 | 1.130 | 1065.600 | -0.418 | -487.282 | 487.282 | 0.457 |
| earliest_campaign_start | val | default_lambda_rule | 72813453 | 5.260 | 614.400 | -41.231 | 89.163 | -73.614 | -0.120 |
| earliest_campaign_start | val | default_lambda_rule | 72814242 | 2.310 | 201.600 | -1.958 | -130.644 | 130.644 | 0.648 |
| earliest_campaign_start | train | ratio_bat | 72171396 | 1.210 | 96.000 | -1.925 | -1.067 | 0.014 | 0.000 |
| earliest_campaign_start | val | ratio_bat | 72797337 | 1.200 | 230.400 | 0.427 | 21.602 | -21.602 | -0.094 |
| earliest_campaign_start | train | ratio_bat | 72799055 | 1.700 | 1877.760 | -0.840 | -174.191 | 174.191 | 0.093 |
| earliest_campaign_start | val | ratio_bat | 72803871 | 1.100 | 80.640 | -0.074 | -0.375 | 0.375 | 0.005 |
| earliest_campaign_start | val | ratio_bat | 72804037 | 1.180 | 599.040 | -14.009 | -8.792 | 4.667 | 0.008 |
| earliest_campaign_start | train | ratio_bat | 72804823 | 1.130 | 322.560 | -1.030 | 0.360 | -0.360 | -0.001 |
| earliest_campaign_start | val | ratio_bat | 72805975 | 5.360 | 1933.820 | -0.173 | -207.841 | 207.841 | 0.107 |
| earliest_campaign_start | train | ratio_bat | 72806735 | 1.100 | 49.920 | -0.060 | -1.683 | 1.683 | 0.034 |
| earliest_campaign_start | train | ratio_bat | 72807079 | 5.260 | 2298.240 | 20.499 | -114.756 | 0.000 | 0.000 |
| earliest_campaign_start | train | ratio_bat | 72812052 | 1.130 | 1065.600 | -1.206 | -537.275 | 537.275 | 0.504 |
| earliest_campaign_start | val | ratio_bat | 72813453 | 5.260 | 614.400 | -42.520 | 126.970 | -73.614 | -0.120 |
| earliest_campaign_start | val | ratio_bat | 72814242 | 2.310 | 201.600 | -1.958 | -130.644 | 130.644 | 0.648 |

The largest holdout click divergence in the selected table is `default` on campaign `74762502`: clicks delta `438.001`, spend delta `-2247.586`, and final-balance delta `0.000`.

## Recommendations

1. Persist exact `best_val` checkpoints so validation-to-train comparisons do not conflate model selection with refit drift.
2. Restore finite lambda bounds or use a clipped action schedule for the next wave; isolate this from state representation changes.
3. Set `gamma < 1.0` in the next controlled run to damp long-horizon bootstrapped targets.
4. Replace high-variance RewardNet targets with immediate-step or normalized episode-return targets in an ablation.
5. Normalize default-state scale before comparing state families again.
6. Keep `ta_ratio_bat` in the candidate set, but evaluate it with clipped lambda and a lower-variance RewardNet target.

## Verification Notes

The script re-evaluates every saved `best_refit.pt` checkpoint on train, val, and holdout, and evaluates the locked `LinearBidder` on the same splits using `linear_scr_FPA.pkl`. It writes non-empty aggregate, campaign, hourly, diagnostics, and divergent-campaign artifacts. Holdout metrics are cross-checked against each run's existing `outputs/metrics.json`; validation `best_val` metrics are copied directly from `run_summary.json` into `summary.json`.

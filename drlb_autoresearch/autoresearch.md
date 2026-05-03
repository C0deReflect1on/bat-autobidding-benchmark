# DRLB Autoresearch: Текущее Состояние

Обновлено: 2026-05-03

## 1) Текущий срез по score

### Locked Linear baseline (эталон)
- Источник: `drlb_autoresearch/artifacts/00_locked_linear_baseline/locked_linear_reference_v1/baseline_manifest.json`
- `val clicks_sum`: `3729.287567540193`
- `test_holdout clicks_sum`: `17792.73354565583`
- `linear_lambda_init`: `0.0028423174374845716`

### Лучший DRLB на текущий момент (quick-wave)
- Источник: `drlb_autoresearch/artifacts/03b_epsilon_optuna_quick_wave_fallback/epsilon_optuna_parallel_quick5k_manual_signal/summary.json`
- Кандидат: `trial_001`
- `best_val_clicks_sum @ step=5000`: `2437.200300045727`
- `holdout_final_clicks_sum`: `13645.38446573859`
- Дельта к linear:
  - `val`: `-1292.087267494466`
  - `holdout`: `-4147.34907991724`
- Вывод: это лучший DRLB среди проверенных quick-wave кандидатов, но он все еще ниже locked linear.

## 2) Что сейчас реально помогает DRLB

Главный прирост дал тюнинг epsilon:
- `dqn_epsilon_start = 0.8610900264673047`
- `dqn_epsilon_end = 0.09801905607969424`
- `dqn_epsilon_anneal = 1.7341839015915597e-05`

Рабочая база из лучшего кандидата:
- `state_type = improved`
- `objective = clicks`
- `auction_mode = FPA`
- `fit_lambda_init = 0.0028423174374845716` (из locked linear)
- `inference_lambda_init_mode = checkpoint_final` (в autoresearch context)
- `dqn_gamma = 1.0`
- `dqn_lr = 1e-4`
- `reward_net_lr = 1e-3`

Что не сработало:
- Сильно расширенный bid-range + LR sweep (`max_bid` около `238`) дал просадку.

## 3) Общий конфиг, на котором велся autoresearch

- Профиль: `drlb_smooth`
- Сплиты: `full_train_val_holdout`
- Основная цель оптимизации: `best_val_clicks_sum` на чекпоинтах
- Quick-wave режим: `epochs=1`, `max_steps=5000` (в расширенных скриптах бывает `15000`)
- Full refit режим: `refit_on=train_plus_val`, `max_steps=None`
- Сиды: стандартная схема `ExperimentConfig` (`random_seed=42` и производные сиды)

Про Monte Carlo / rollouts:
- Отдельного параметра вида `monte_carlo_n` в текущем коде нет.
- Оценка идет через simulator-проходы и rollout-таблицы (`autobidder_check`, `simulate_campaign`, pair comparison CSV).

Про диагностические картинки:
- Храним только один PNG: `drlb_diagnostics.png`.
- Отдельные `*_action_distribution.png` не используем.

## 4) Какие параметры менять дальше (приоритет)

1. Epsilon рядом с текущим лучшим:
   - `dqn_epsilon_start` в диапазоне `[0.80, 0.92]`
   - `dqn_epsilon_end` в диапазоне `[0.07, 0.10]`
   - `dqn_epsilon_anneal` в диапазоне `[1e-5, 3e-5]`
2. Bid bounds аккуратно:
   - Держать `max_bid` консервативно (не прыгать сразу к `200+`)
   - Тюнить вместе: `min_bid`, `max_bid`, `bid_lower_clip`, `bid_upper_clip`
3. Lambda-inference путь:
   - Сравнить `inference_lambda_init_mode=checkpoint_final` против фиксированного inference lambda
4. LR локально:
   - Делать узкий поиск вокруг текущих `dqn_lr`/`reward_net_lr`

## 5) Инструкция для продолжения autoresearch

1. Всегда использовать locked linear как anchor и считать дельту DRLB к baseline.
2. Прогонять DRLB-кандидаты, ранжировать по `best_val_clicks_sum`, затем топ-кандидатов проверять на `test_holdout`.
3. Для каждого promoted-кандидата сохранять минимум:
   - `params.json`
   - `checkpoints.csv`
   - `diagnostics.csv`
   - `summary.json` (val/holdout и дельты к linear)
4. Обязательно сохранять pair comparison по `val` и `test_holdout` (campaign + hourly).
5. Продвигать гипотезу только если holdout улучшился заметно и воспроизводимо.
6. В этот файл писать только краткие факты, без мусора из ноутбуков и длинных логов.

## 6) Точки входа для следующего шага

- Quick research:
  - `drlb_autoresearch/run_full_15k_autoresearch.py`
- Full train+val refit c лучшим epsilon:
  - `drlb_autoresearch/run_full_trainval_checkpoint_infer_autoresearch.py`

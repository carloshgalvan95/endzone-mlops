# Phase 0 feature input audit

Diagnostic only. Features are unchanged. 2025 appears in the
distribution tables and is not scored. Univariate CV trains on seasons
strictly before the scored season. Permutation importance uses the
current logistic pipeline, fit on the training seasons and scored on
2024.

## Settings

- Train seasons: through 2023
- Validation season: 2024
- Test season (distribution only): 2025
- Univariate CV seasons: 2016 through 2023
- Bootstrap resamples: 1000 (seed 42)
- Spearman redundancy threshold: 0.9
- Permutation repeats: 10

The logistic step is the training pipeline: EPA columns filled with 0,
win-rate columns filled with 0.5, other columns left as-is, then scaled.

## Missingness

| feature | missing_rate | train | val | test |
| --- | --- | --- | --- | --- |
| home_recent_offensive_epa_per_play | 0.0025 | 0.0027 | 0.0000 | 0.0000 |
| home_recent_defensive_epa_per_play | 0.0025 | 0.0027 | 0.0000 | 0.0000 |
| away_recent_offensive_epa_per_play | 0.0022 | 0.0024 | 0.0000 | 0.0000 |
| away_recent_defensive_epa_per_play | 0.0022 | 0.0024 | 0.0000 | 0.0000 |
| home_recent_win_rate | 0.0022 | 0.0024 | 0.0000 | 0.0000 |
| away_recent_win_rate | 0.0022 | 0.0024 | 0.0000 | 0.0000 |
| home_rest_days | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| away_rest_days | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| home_season_offensive_epa_per_play | 0.0612 | 0.0598 | 0.0561 | 0.0561 |
| away_season_offensive_epa_per_play | 0.0608 | 0.0593 | 0.0561 | 0.0561 |
| home_games_played | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| away_games_played | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Distribution

Out of range counts values in that slice outside the closed training
min and max. `time_counter` is yes when season medians are strictly
monotone and 2024 or 2025 sits entirely outside that training range.

| feature | train_min | train_max | val_min | val_max | val_out_of_range | test_min | test_max | test_out_of_range | time_counter |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| home_recent_offensive_epa_per_play | -0.5209 | 0.3346 | -0.2512 | 0.2207 | 0 | -0.2429 | 0.2065 | 0 | no |
| home_recent_defensive_epa_per_play | -0.5152 | 0.2415 | -0.1815 | 0.1798 | 0 | -0.2582 | 0.2228 | 0 | no |
| away_recent_offensive_epa_per_play | -0.8229 | 0.3290 | -0.2668 | 0.2145 | 0 | -0.2446 | 0.2099 | 0 | no |
| away_recent_defensive_epa_per_play | -0.8229 | 0.3346 | -0.1863 | 0.1865 | 0 | -0.2166 | 0.2303 | 0 | no |
| home_recent_win_rate | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0 | 0.0000 | 1.0000 | 0 | no |
| away_recent_win_rate | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0 | 0.0000 | 1.0000 | 0 | no |
| home_rest_days | 4.0000 | 16.0000 | 4.0000 | 14.0000 | 0 | 4.0000 | 15.0000 | 0 | no |
| away_rest_days | 4.0000 | 21.0000 | 4.0000 | 14.0000 | 0 | 4.0000 | 15.0000 | 0 | no |
| home_season_offensive_epa_per_play | -0.8150 | 0.6209 | -0.4584 | 0.3083 | 0 | -0.4370 | 0.3519 | 0 | no |
| away_season_offensive_epa_per_play | -0.8229 | 0.4671 | -0.4809 | 0.2767 | 0 | -0.2632 | 0.2693 | 0 | no |
| home_games_played | 0.0000 | 444.0000 | 368.0000 | 461.0000 | 31 | 387.0000 | 482.0000 | 152 | no |
| away_games_played | 0.0000 | 443.0000 | 367.0000 | 459.0000 | 31 | 386.0000 | 481.0000 | 162 | no |

## Univariate signal

Each number is a one-feature logistic on rolling-origin season folds.
Intervals are the 2.5 and 97.5 percentiles of a bootstrap of the
pooled out-of-fold rows.

| feature | log_loss | log_loss_low | log_loss_high | auc | auc_low | auc_high | folds_used |
| --- | --- | --- | --- | --- | --- | --- | --- |
| home_recent_offensive_epa_per_play | 0.6747 | 0.6650 | 0.6856 | 0.6040 | 0.5799 | 0.6276 | 8 |
| home_recent_defensive_epa_per_play | 0.6836 | 0.6772 | 0.6906 | 0.5501 | 0.5255 | 0.5738 | 8 |
| away_recent_offensive_epa_per_play | 0.6700 | 0.6613 | 0.6783 | 0.6153 | 0.5922 | 0.6377 | 8 |
| away_recent_defensive_epa_per_play | 0.6895 | 0.6822 | 0.6977 | 0.5278 | 0.5028 | 0.5511 | 8 |
| home_recent_win_rate | 0.6694 | 0.6598 | 0.6802 | 0.6124 | 0.5890 | 0.6362 | 8 |
| away_recent_win_rate | 0.6746 | 0.6651 | 0.6851 | 0.5983 | 0.5742 | 0.6219 | 8 |
| home_rest_days | 0.6887 | 0.6831 | 0.6948 | 0.5121 | 0.4885 | 0.5356 | 8 |
| away_rest_days | 0.6886 | 0.6830 | 0.6950 | 0.5033 | 0.4808 | 0.5269 | 8 |
| home_season_offensive_epa_per_play | 0.6790 | 0.6691 | 0.6893 | 0.5896 | 0.5654 | 0.6135 | 8 |
| away_season_offensive_epa_per_play | 0.6765 | 0.6678 | 0.6848 | 0.5996 | 0.5760 | 0.6241 | 8 |
| home_games_played | 0.6887 | 0.6835 | 0.6943 | 0.4976 | 0.4727 | 0.5207 | 8 |
| away_games_played | 0.6881 | 0.6839 | 0.6928 | 0.5106 | 0.4858 | 0.5344 | 8 |

## Correlation and redundancy

Spearman correlation on training rows. A feature is redundant when its
strongest other-feature absolute correlation is at least 0.9.

| feature | home_recent_offensive_epa_per_play | home_recent_defensive_epa_per_play | away_recent_offensive_epa_per_play | away_recent_defensive_epa_per_play | home_recent_win_rate | away_recent_win_rate | home_rest_days | away_rest_days | home_season_offensive_epa_per_play | away_season_offensive_epa_per_play | home_games_played | away_games_played |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| home_recent_offensive_epa_per_play | 1.0000 | 0.0205 | 0.0742 | 0.0320 | 0.6427 | 0.0390 | 0.0268 | 0.0316 | 0.8238 | 0.0667 | 0.1919 | 0.1794 |
| home_recent_defensive_epa_per_play | 0.0205 | 1.0000 | 0.0149 | 0.0749 | -0.4999 | -0.0377 | -0.0240 | -0.0152 | 0.0216 | 0.0329 | 0.1951 | 0.2011 |
| away_recent_offensive_epa_per_play | 0.0742 | 0.0149 | 1.0000 | 0.0315 | 0.0452 | 0.6246 | 0.0237 | 0.0049 | 0.0778 | 0.8304 | 0.1755 | 0.1887 |
| away_recent_defensive_epa_per_play | 0.0320 | 0.0749 | 0.0315 | 1.0000 | -0.0464 | -0.5016 | -0.0233 | 0.0101 | 0.0108 | 0.0252 | 0.2071 | 0.2020 |
| home_recent_win_rate | 0.6427 | -0.4999 | 0.0452 | -0.0464 | 1.0000 | 0.0605 | 0.0388 | 0.0332 | 0.5345 | 0.0224 | 0.0221 | 0.0077 |
| away_recent_win_rate | 0.0390 | -0.0377 | 0.6246 | -0.5016 | 0.0605 | 1.0000 | 0.0220 | -0.0015 | 0.0500 | 0.5323 | -0.0007 | 0.0138 |
| home_rest_days | 0.0268 | -0.0240 | 0.0237 | -0.0233 | 0.0388 | 0.0220 | 1.0000 | 0.3877 | 0.0252 | 0.0122 | -0.0175 | -0.0186 |
| away_rest_days | 0.0316 | -0.0152 | 0.0049 | 0.0101 | 0.0332 | -0.0015 | 0.3877 | 1.0000 | 0.0294 | 0.0010 | 0.0021 | -0.0018 |
| home_season_offensive_epa_per_play | 0.8238 | 0.0216 | 0.0778 | 0.0108 | 0.5345 | 0.0500 | 0.0252 | 0.0294 | 1.0000 | 0.0769 | 0.1838 | 0.1752 |
| away_season_offensive_epa_per_play | 0.0667 | 0.0329 | 0.8304 | 0.0252 | 0.0224 | 0.5323 | 0.0122 | 0.0010 | 0.0769 | 1.0000 | 0.1663 | 0.1763 |
| home_games_played | 0.1919 | 0.1951 | 0.1755 | 0.2071 | 0.0221 | -0.0007 | -0.0175 | 0.0021 | 0.1838 | 0.1663 | 1.0000 | 0.9922 |
| away_games_played | 0.1794 | 0.2011 | 0.1887 | 0.2020 | 0.0077 | 0.0138 | -0.0186 | -0.0018 | 0.1752 | 0.1763 | 0.9922 | 1.0000 |

Redundant pairs:

| left | right | spearman |
| --- | --- | --- |
| home_games_played | away_games_played | 0.9922 |

## Permutation importance

Mean drop in negative log loss on the validation season when the column is shuffled.

| feature | importance_mean | importance_std |
| --- | --- | --- |
| home_recent_offensive_epa_per_play | 0.0187 | 0.0028 |
| home_recent_defensive_epa_per_play | 0.0113 | 0.0049 |
| away_recent_offensive_epa_per_play | 0.0104 | 0.0050 |
| away_recent_defensive_epa_per_play | -0.0019 | 0.0030 |
| home_recent_win_rate | 0.0130 | 0.0034 |
| away_recent_win_rate | 0.0034 | 0.0025 |
| home_rest_days | -0.0019 | 0.0010 |
| away_rest_days | 0.0024 | 0.0026 |
| home_season_offensive_epa_per_play | 0.0184 | 0.0041 |
| away_season_offensive_epa_per_play | 0.0031 | 0.0029 |
| home_games_played | 0.0013 | 0.0011 |
| away_games_played | 0.0003 | 0.0015 |

## Skipped univariate folds

Every configured fold was fit.

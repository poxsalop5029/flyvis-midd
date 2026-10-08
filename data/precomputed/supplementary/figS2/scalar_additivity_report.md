# MIDD additivity report

Inference unit: 10 trained models. Frames, receptors, pairs, and cycles were not treated as independent samples.

## Descriptive classification

```text
  rendering                     descriptive_classification
     boxeye                         weak global additivity
highres_232 gain- or condition-dependent global additivity
```

## Model-level mean ± SEM

```text
                slope                  r2           normalized_rmse           sign_accuracy           cycle_rmse           normalized_intercept          
                 mean       sem      mean       sem            mean       sem          mean       sem       mean       sem                 mean       sem
rendering                                                                                                                                                
boxeye       0.122341  0.066642  0.439430  0.101425        0.623672  0.057207      0.457853  0.099202   0.154251  0.027837             0.155954  0.023933
highres_232  0.674726  0.034989  0.933702  0.016369        0.184532  0.017877      0.975000  0.013437   0.097505  0.015197            -0.034494  0.017630
```

## Inherited benchmark status

```text
  rendering                               criterion  pass
     boxeye       mean_slope_inside_legacy_interval False
     boxeye mean_normalized_intercept_inside_margin False
     boxeye  mean_normalized_residual_inside_margin False
     boxeye       mean_normalized_rmse_below_margin False
     boxeye            mean_cycle_rmse_below_margin False
     boxeye                   mean_r2_above_minimum False
     boxeye        mean_sign_accuracy_above_minimum False
highres_232       mean_slope_inside_legacy_interval False
highres_232 mean_normalized_intercept_inside_margin  True
highres_232  mean_normalized_residual_inside_margin False
highres_232       mean_normalized_rmse_below_margin False
highres_232            mean_cycle_rmse_below_margin  True
highres_232                   mean_r2_above_minimum  True
highres_232        mean_sign_accuracy_above_minimum  True
```

## Interpretation limits

- Diagonal zero and swap antisymmetry are construction identities and QA only.
- Legacy thresholds are inherited descriptive benchmarks, not a new preregistration.
- Vector-field additivity is excluded from the primary classification and handled in Notebook 08.

# NBA GBDT Width Sweep

Date: 2026-03-05
Method: 5-fold walk-forward cross-validation on processed NBA features
Goal: compare predictive strength for pruned `gbdt` feature widths

| Width | Log Loss | Brier | AUC | Accuracy |
| --- | ---: | ---: | ---: | ---: |
| 35 | 0.8031 | 0.2668 | 0.6364 | 0.6064 |
| 36 | 0.8077 | 0.2695 | 0.6325 | 0.5949 |
| 38 | 0.8046 | 0.2677 | 0.6378 | 0.6154 |
| 40 | 0.7995 | 0.2671 | 0.6404 | 0.6103 |
| 44 | 0.8078 | 0.2706 | 0.6338 | 0.5974 |
| 48 | 0.8053 | 0.2699 | 0.6352 | 0.6000 |
| 56 | 0.8122 | 0.2703 | 0.6419 | 0.6141 |

Winner by probabilistic CV error: 40 features

Notes:
- 35 did not improve on 40, so the sweep did not continue downward to 30.
- 56 had slightly higher AUC than 40, but materially worse log loss and Brier, which is not preferable for probability quality.
- The saved NBA `gbdt` feature map remains at 40 features.

# Market Collapse Risk Monitor — Ensemble

## What's included

- Calibrated 6-month ensemble risk engine
- Six component models: logistic regression, LDA, linear SVM, RBF SVM, random forest, and MLP neural network
- Platt/sigmoid probability calibration from purged out-of-fold predictions
- Regularized logistic meta-model combining the calibrated component probabilities
- Secondary 12-month severe-drawdown ensemble
- Component-model probabilities, ensemble weights, and holdout metrics in the dashboard
- Original transparent logistic scenario calculator retained for feature-by-feature sensitivity
- Historical stress/drawdown chart
- Source Version 2 dataset archive

## Validation

The primary target is a path drawdown of at least -10% within six months. Evaluation uses a time-ordered holdout beginning 2016-01, with target-horizon purging and exclusion of months without a complete forward outcome.

The ensemble is an additional risk engine, not a guarantee of improved forecasting. Its strict holdout metrics are displayed in the dashboard so users can see how it performed out of ts of this folder to the repository root.

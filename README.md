# Market Collapse Risk Monitor — Ensemble GitHub Pages App

A free, static GitHub Pages dashboard built from the supplied Version 2 crisis feature dataset.

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

The ensemble is an additional risk engine, not a guarantee of improved forecasting. Its strict holdout metrics are displayed in the dashboard so users can see how it performed out of sample.

## Deploy free with GitHub Pages

1. Create a GitHub repository, for example `market-collapse-risk`.
2. Upload the contents of this folder to the repository root.
3. Open **Settings → Pages**.
4. Choose **Deploy from a branch**.
5. Select your default branch and `/ (root)`.
6. Save. GitHub Pages will publish `index.html` as the site.

No server or paid API is required.

## Important

This is a statistical risk monitor, not investment advice and not a reliable collapse timer. The data and models are historical and can produce false alarms or miss events.


## Automatic refresh

The repository now includes a GitHub Actions workflow at `.github/workflows/update-risk.yml`. It refreshes the public data inputs daily, rebuilds the latest Version 2 monthly feature row, refits the six-model ensemble, and commits the refreshed JSON used by the GitHub Pages dashboard. See `AUTOMATION_SETUP.md` for details.

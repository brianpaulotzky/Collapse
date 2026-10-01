# Collapse — automatic data refresh

This package adds a GitHub Actions updater to the existing static GitHub Pages app.

## What it does

Once installed in the repository, the workflow:

1. Downloads fresh public market/economic observations.
2. Rebuilds the monthly Version 2 feature row.
3. Recalculates the rolling 60-month z-scores and stress composites.
4. Refits the six-model ensemble using only rows with known forward targets.
5. Uses a purged expanding out-of-fold meta-model for the ensemble.
6. Writes `data/risk_dashboard_data.json` and `data/ensemble_risk_engine.json`.
7. Commits the changes. Your existing GitHub Pages branch deployment then republishes the site.

## Data sources

- FRED: VIX, EPU, NFCI, Baa-10Y spread, WTI, STLFSI4, Michigan Sentiment, and 10Y-2Y term spread.
- Caldara/Iacoviello: monthly Geopolitical Risk Index.
- Yahoo Finance: SPY adjusted daily prices used for month-end price and 1-month realized volatility.

No API key is required by this updater.

## Schedule

The workflow runs daily at 18:17 UTC and can also be started manually from the repository's Actions tab.

The updater treats the most recent sufficiently populated SPY month as the live monthly feature month. Slow-moving macro series remain at their latest available observation; they are not fabricated or backfilled into future months.

## Important model note

The repository contains the original Version 2 dataset and the previously generated ensemble output, but it does not contain the private fitting code/calibration objects used to generate that earlier ensemble. Therefore this automation uses a reproducible `ensemble-v1.1-auto` refit based on the documented Version 2 feature definitions and the six model families shown in the dashboard. The original transparent scenario calculator remains unchanged.

The displayed holdout metrics are recalculated by the updater each run. They should be read as model diagnostics, not as proof that the model can reliably time market collapses.

## GitHub Pages

The repository can continue using **Settings → Pages → Deploy from a branch**. The updater only changes JSON data files, so the normal Pages deployment will publish the new values after each successful commit.

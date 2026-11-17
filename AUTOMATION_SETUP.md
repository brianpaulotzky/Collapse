# Published market observations

`refresh_market.py` uses Python's standard library to fetch eleven public FRED series, preserving each observation date, unit and source URL. Failed sources are marked unavailable, never filled with historical model means. The page independently warns about old observations and snapshots older than two days. Retrieval timestamps are not observation or publication dates.

The workflow `.github/workflows/update-risk.yml` runs weekdays at 23:30 UTC and on manual dispatch. It commits `market_snapshot.json` and explicitly requests a rebuild of the existing branch-based Pages site; a bot commit alone does not trigger a branch Pages build. Keep the existing branch-based Pages source. Enable Actions and workflow write permissions in repository settings. Scheduled runs depend on GitHub availability and repository policy.

Run locally with `python refresh_market.py`. Serve the repository over HTTP (for example `python -m http.server`); opening index.html as a file will not support JSON fetches.

The historical risk engine remains archived. It is not refreshed by the observations workflow. Its training data provenance, feature construction, target labels and probability calibration require a separate audit before presenting any output as a current risk estimate. Market observations do not automatically validate a collapse model. The standalone experimental refit script is not a production refresh path.

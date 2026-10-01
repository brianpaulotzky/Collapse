#!/usr/bin/env python3
"""Refresh Collapse dashboard data from public sources and refit the ensemble.

The updater intentionally keeps the static GitHub Pages architecture: it writes
JSON files only. GitHub Actions commits the refreshed JSON, and Pages republishes
it using the repository's existing branch deployment.
"""
from __future__ import annotations

import io
import json
import math
import os
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_JSON = DATA_DIR / "risk_dashboard_data.json"
ENGINE_JSON = DATA_DIR / "ensemble_risk_engine.json"
ARCHIVE_ZIP = DATA_DIR / "crisis_features_dataset2.zip"

FEATURES = [
    "vix", "gpr", "epu_us", "nfci", "baa10y", "wti", "stlfsi", "umcsent",
    "term_spread", "stress_composite", "stress_composite_v2", "vix_z60",
    "gpr_z60", "epu_us_z60", "nfci_z60", "baa10y_z60", "stress_l1",
    "stress_l3", "spy_vol_1m",
]

FRED_SERIES = {
    "vix": "VIXCLS",
    "epu_us": "USEPUINDXM",
    "nfci": "NFCI",
    "baa10y": "BAA10YM",
    "wti": "DCOILWTICO",
    "stlfsi": "STLFSI4",
    "umcsent": "UMCSENT",
    "term_spread": "T10Y2Y",
}
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
GPR_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls"
YAHOO_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/SPY"


def get(url: str, timeout: int = 60) -> requests.Response:
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "CollapseRiskMonitor/1.1"})
    r.raise_for_status()
    return r


def fetch_fred(series: str) -> pd.Series:
    r = get(FRED_URL, timeout=60)
    # The graph endpoint returns one series when id contains one mnemonic.
    # Retry with an explicit query parameter if the first request is unusual.
    from io import StringIO
    df = pd.read_csv(StringIO(r.text))
    if "DATE" not in df.columns:
        raise RuntimeError(f"Unexpected FRED response for {series}")
    value_col = [c for c in df.columns if c != "DATE"][0]
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    return df.set_index("DATE")[value_col].dropna().rename(series)


def fetch_fred_series(series: str) -> pd.Series:
    r = get(f"{FRED_URL}?id={series}", timeout=60)
    df = pd.read_csv(io.StringIO(r.text))
    if "DATE" not in df.columns:
        raise RuntimeError(f"Unexpected FRED response for {series}")
    value_col = [c for c in df.columns if c != "DATE"][0]
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    return df.set_index("DATE")[value_col].dropna().rename(series)


def fetch_gpr() -> pd.DataFrame:
    raw = get(GPR_URL, timeout=120).content
    df = pd.read_excel(io.BytesIO(raw))
    df.columns = [str(c).strip() for c in df.columns]
    date_col = next((c for c in df.columns if c.lower() in {"month", "date"}), None)
    if date_col is None:
        raise RuntimeError("GPR file has no month/date column")
    gpr_col = next((c for c in df.columns if c.upper() == "GPR"), None)
    if gpr_col is None:
        raise RuntimeError("GPR file has no GPR column")
    out = pd.DataFrame({
        "date": pd.to_datetime(df[date_col], errors="coerce"),
        "gpr": pd.to_numeric(df[gpr_col], errors="coerce"),
    }).dropna(subset=["date", "gpr"])
    out["date"] = out["date"].dt.to_period("M").dt.to_timestamp()
    return out.drop_duplicates("date", keep="last").set_index("date")


def fetch_spy_daily() -> pd.DataFrame:
    # Keep a generous window so the updater can be run manually after a long pause.
    now = int(datetime.now(timezone.utc).timestamp())
    start = now - 240 * 86400
    params = {"period1": start, "period2": now + 86400, "interval": "1d", "events": "div,splits", "includeAdjustedClose": "true"}
    r = requests.get(YAHOO_CHART, params=params, timeout=60, headers={"User-Agent": "CollapseRiskMonitor/1.1"})
    r.raise_for_status()
    j = r.json()["chart"]["result"][0]
    q = j["indicators"]["quote"][0]
    adj = j["indicators"].get("adjclose", [{"adjclose": []}])[0]["adjclose"]
    dates = pd.to_datetime(j["timestamp"], unit="s", utc=True).tz_convert(None)
    df = pd.DataFrame({"date": dates, "close": q["close"], "adj": adj})
    df["date"] = df["date"].dt.normalize()
    df = df.dropna(subset=["adj"]).sort_values("date")
    df["ret"] = df["adj"].pct_change()
    return df


def last_in_month(series: pd.Series, month: pd.Timestamp) -> float:
    q = series[series.index.to_period("M") == month.to_period("M")].dropna()
    return float(q.iloc[-1]) if len(q) else float("nan")


def load_history() -> pd.DataFrame:
    with zipfile.ZipFile(ARCHIVE_ZIP) as z:
        with z.open("crisis_core_monthly.csv") as f:
            df = pd.read_csv(f, parse_dates=["date"])
    df = df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    return df


def choose_target_month(spy: pd.DataFrame) -> pd.Timestamp:
    x = spy.copy()
    x["month"] = x["date"].dt.to_period("M").dt.to_timestamp()
    counts = x.groupby("month").size()
    eligible = counts[counts >= 15].index
    if len(eligible) == 0:
        raise RuntimeError("Could not find a month with enough SPY observations")
    return pd.Timestamp(eligible[-1])


def build_current_row(history: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    spy = fetch_spy_daily()
    target = choose_target_month(spy)

    fred = {}
    for feature, series_id in FRED_SERIES.items():
        fred[feature] = fetch_fred_series(series_id)

    gpr = fetch_gpr()
    row = {"date": target}
    for feature, s in fred.items():
        row[feature] = last_in_month(s, target)
    row["gpr"] = float(gpr.loc[target, "gpr"]) if target in gpr.index else float("nan")

    spym = spy[spy["date"].dt.to_period("M") == target.to_period("M")].copy()
    row["spy_adj"] = float(spym["adj"].iloc[-1])
    row["spy_vol_1m"] = float(spym["ret"].dropna().std(ddof=1) * math.sqrt(21)) if spym["ret"].notna().sum() >= 5 else float("nan")

    # Features are defined from the historical monthly panel. Recalculate the
    # rolling z-scores after appending the new raw observations.
    raw_cols = ["vix", "gpr", "epu_us", "nfci", "baa10y", "wti", "stlfsi", "umcsent"]
    combined = history.copy()
    # Replace an existing month with the fresh observation; otherwise append it.
    new_df = pd.DataFrame([row])
    combined = pd.concat([combined, new_df], ignore_index=True, sort=False)
    combined = combined.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)

    for c in raw_cols:
        roll = combined[c].rolling(60, min_periods=60)
        combined[c + "_z60"] = (combined[c] - roll.mean()) / roll.std(ddof=1)

    combined["stress_composite"] = combined[["vix_z60", "gpr_z60", "epu_us_z60", "nfci_z60"]].mean(axis=1, skipna=False)
    combined["stress_composite_v2"] = combined[["vix_z60", "gpr_z60", "epu_us_z60", "nfci_z60", "baa10y_z60", "stlfsi_z60"]].mean(axis=1, skipna=False)
    combined["stress_l1"] = combined["stress_composite"].shift(1)
    combined["stress_l3"] = combined["stress_composite"].shift(3)

    return combined, spy


def make_model(name: str, seed: int):
    if name == "Logistic":
        return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", LogisticRegression(class_weight="balanced", C=1.0, max_iter=4000, random_state=seed))])
    if name == "LDA":
        return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", LinearDiscriminantAnalysis())])
    if name == "Linear SVM":
        return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", CalibratedClassifierCV(LinearSVC(class_weight="balanced", C=0.5, random_state=seed), method="sigmoid", cv=3))])
    if name == "RBF SVM":
        return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", SVC(C=1.0, gamma="scale", class_weight="balanced", probability=True, random_state=seed))])
    if name == "Random Forest":
        return Pipeline([("impute", SimpleImputer(strategy="median")), ("model", RandomForestClassifier(n_estimators=180, max_depth=4, min_samples_leaf=5, class_weight="balanced_subsample", random_state=seed, n_jobs=-1))])
    if name == "Neural Network":
        return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", MLPClassifier(hidden_layer_sizes=(12, 6), alpha=0.02, early_stopping=True, max_iter=700, random_state=seed))])
    raise ValueError(name)


def purged_oof(X: pd.DataFrame, y: pd.Series, horizon: int, seed: int = 42):
    """Create simple expanding OOF predictions with a horizon purge.

    Each validation block is predicted only from observations strictly before the
    validation start minus the target horizon. This prevents target-window overlap.
    """
    n = len(X)
    pred = {name: np.full(n, np.nan) for name in MODEL_NAMES}
    # Start after enough history for stable models and leave a sizeable holdout-like tail.
    starts = list(range(max(60, horizon), n, 18))
    for start in starts:
        end = min(start + 12, n)
        train_end = max(0, start - horizon)
        if train_end < 50 or end <= start:
            continue
        Xt, yt = X.iloc[:train_end], y.iloc[:train_end]
        Xv = X.iloc[start:end]
        if yt.nunique() < 2:
            continue
        for name in MODEL_NAMES:
            try:
                model = make_model(name, seed)
                model.fit(Xt, yt)
                pred[name][start:end] = model.predict_proba(Xv)[:, 1]
            except Exception as exc:
                print(f"OOF warning {name}: {exc}")
    return pred


MODEL_NAMES = ["Logistic", "LDA", "Linear SVM", "RBF SVM", "Random Forest", "Neural Network"]


def fit_engine(df: pd.DataFrame, target_col: str, horizon: int, seed: int = 42):
    work = df.copy()
    labeled = work[target_col].notna()
    labeled &= work[FEATURES].notna().sum(axis=1) >= 12
    work = work.loc[labeled].copy().reset_index(drop=True)
    X = work[FEATURES]
    y = work[target_col].astype(int)
    if y.nunique() < 2 or len(y) < 80:
        raise RuntimeError(f"Not enough labeled rows for {target_col}")

    oof = purged_oof(X, y, horizon, seed)
    meta_X = pd.DataFrame(oof)
    valid = meta_X.notna().all(axis=1)
    # Fit a regularized meta-model on the OOF probabilities.
    meta = LogisticRegression(class_weight="balanced", C=0.25, max_iter=4000, random_state=seed)
    if valid.sum() >= 40 and y.loc[valid].nunique() == 2:
        meta.fit(meta_X.loc[valid], y.loc[valid])
    else:
        # Safe fallback: average component probabilities.
        meta = None

    components = {}
    for name in MODEL_NAMES:
        model = make_model(name, seed)
        model.fit(X, y)
        # Current row is the last row of the combined feature panel.
        components[name] = model

    latest = df.sort_values("date").iloc[-1]
    X_latest = pd.DataFrame([latest[FEATURES].to_dict()])
    probs = {name: float(model.predict_proba(X_latest)[:, 1][0]) for name, model in components.items()}

    if meta is not None:
        pvec = pd.DataFrame([probs])[MODEL_NAMES]
        ensemble = float(meta.predict_proba(pvec)[:, 1][0])
        weights = {name: float(v) for name, v in zip(MODEL_NAMES, meta.coef_[0])}
        # Normalize absolute coefficients for display only; signed coefficients are
        # retained separately so the dashboard does not imply they are probabilities.
        abs_sum = sum(abs(v) for v in weights.values()) or 1.0
        display_weights = {k: abs(v) / abs_sum for k, v in weights.items()}
        meta_intercept = float(meta.intercept_[0])
    else:
        ensemble = float(np.mean(list(probs.values())))
        display_weights = {name: 1 / len(MODEL_NAMES) for name in MODEL_NAMES}
        weights = {name: 0.0 for name in MODEL_NAMES}
        meta_intercept = 0.0

    # Out-of-sample metrics for the ensemble and components on the last 20% of
    # labeled observations, with a horizon-sized purge from the training set.
    cut = max(1, int(len(X) * 0.8))
    train_end = max(1, cut - horizon)
    test = np.arange(cut, len(X))
    metrics = []
    if len(test) >= 10 and y.iloc[test].nunique() == 2:
        for name in MODEL_NAMES:
            m = make_model(name, seed)
            m.fit(X.iloc[:train_end], y.iloc[:train_end])
            pp = m.predict_proba(X.iloc[test])[:, 1]
            metrics.append({"model": name, "roc_auc": float(roc_auc_score(y.iloc[test], pp)), "pr_auc": float(average_precision_score(y.iloc[test], pp)), "brier": float(brier_score_loss(y.iloc[test], pp))})
        if meta is not None:
            # Strict holdout component predictions: fit each component only on the
            # pre-test sample, then apply the meta-model fit only to pre-test OOF rows.
            ptest_dict = {}
            for name in MODEL_NAMES:
                hm = make_model(name, seed)
                hm.fit(X.iloc[:train_end], y.iloc[:train_end])
                ptest_dict[name] = hm.predict_proba(X.iloc[test])[:, 1]
            ptest = pd.DataFrame(ptest_dict)[MODEL_NAMES]
            vv = meta_X.notna().all(axis=1) & (meta_X.index < train_end)
            hold_meta = LogisticRegression(class_weight="balanced", C=0.25, max_iter=4000, random_state=seed)
            if vv.sum() >= 40 and y.loc[vv].nunique() == 2:
                hold_meta.fit(meta_X.loc[vv], y.loc[vv])
                pe = hold_meta.predict_proba(ptest)[:, 1]
                metrics.append({"model": "Ensemble", "roc_auc": float(roc_auc_score(y.iloc[test], pe)), "pr_auc": float(average_precision_score(y.iloc[test], pe)), "brier": float(brier_score_loss(y.iloc[test], pe))})

    return {
        "component_probabilities": probs,
        "ensemble_probability": ensemble,
        "meta_weights": display_weights,
        "meta_signed_coefficients": weights,
        "meta_intercept": meta_intercept,
        "holdout_metrics": metrics,
        "train_end": str(work.iloc[train_end - 1]["date"].date()) if train_end > 0 else None,
        "test_start": str(work.iloc[cut]["date"].date()) if cut < len(work) else None,
        "test_end": str(work.iloc[-1]["date"].date()),
        "test_n": int(len(test)),
        "test_events": int(y.iloc[test].sum()) if len(test) else 0,
        "test_prevalence": float(y.iloc[test].mean()) if len(test) else None,
    }


def label(prob: float) -> str:
    if prob < 0.15:
        return "Low"
    if prob < 0.30:
        return "Moderate"
    if prob < 0.50:
        return "High"
    return "Very High"


def update_files(combined: pd.DataFrame, primary: dict, secondary: dict, generated_at: str):
    combined = combined.sort_values("date").reset_index(drop=True)
    latest = combined.iloc[-1]
    latest_date = latest["date"].strftime("%Y-%m-%d")

    engine = {
        "engine_version": "ensemble-v1.1-auto",
        "generated_at": generated_at,
        "latest_date": latest_date,
        "features": FEATURES,
        "primary_target": {"id": "y_dd_6m", "description": "Probability the market reaches at least a -10% path drawdown within the next 6 months", "horizon_months": 6},
        "secondary_target": {"id": "y_dd_12m", "description": "Probability the market reaches at least a -15% path drawdown within the next 12 months", "horizon_months": 12},
        "models": MODEL_NAMES,
        "primary": primary,
        "secondary": secondary,
        "methodology": {
            "calibration": "Component probabilities are refit from the Version 2 labeled monthly panel; the ensemble meta-model is fit on purged expanding out-of-fold component predictions.",
            "ensemble": "Regularized balanced logistic meta-model combining six component probabilities.",
            "validation": "Time-ordered holdout with target-horizon purging; the displayed holdout is a fresh run of the updater.",
            "censoring": "Rows without complete forward target windows are excluded from fitting/evaluation.",
            "data_refresh": "Public FRED, Caldara-Iacoviello GPR, and Yahoo Finance SPY data; monthly feature rows use the latest available observation within the month.",
            "disclaimer": "Statistical risk engine; not investment advice and not a reliable collapse timer."
        }
    }
    ENGINE_JSON.write_text(json.dumps(engine, indent=2), encoding="utf-8")

    dash = json.loads(DATA_JSON.read_text(encoding="utf-8"))
    dash["defaults"] = {f: (None if pd.isna(latest.get(f)) else float(latest[f])) for f in FEATURES}
    dash["latest_date"] = latest_date
    # Keep the original transparent scenario model intact; its value is separate
    # from the automatically refit ensemble.
    dash["ensemble_engine"] = engine
    dash["latest_ensemble_prob"] = primary["ensemble_probability"]
    dash["latest_ensemble_label"] = label(primary["ensemble_probability"])

    hist = dash.get("history", {})
    # Rebuild the visible historical stress/drawdown series from the source panel.
    def arr(col, transform=lambda x: x):
        vals = []
        for x in combined[col]:
            vals.append(None if pd.isna(x) else transform(float(x)))
        return vals
    hist["dates"] = [d.strftime("%Y-%m") for d in combined["date"]]
    hist["stress_v1"] = arr("stress_composite")
    hist["stress_v2"] = arr("stress_composite_v2")
    hist["vix"] = arr("vix")
    hist["collapsed"] = arr("Collapsed") if "Collapsed" in combined.columns else hist.get("collapsed", [])
    hist["drawdown"] = arr("spy_dd_me") if "spy_dd_me" in combined.columns else hist.get("drawdown", [])
    dash["history"] = hist
    dash["automation"] = {
        "status": "automatic",
        "generated_at": generated_at,
        "source_summary": "FRED + Caldara/Iacoviello GPR + Yahoo Finance SPY",
        "refresh_note": "GitHub Actions refreshes this file on schedule; some macro series update less frequently than daily."
    }
    DATA_JSON.write_text(json.dumps(dash, indent=2, allow_nan=False), encoding="utf-8")


def main():
    generated_at = datetime.now(timezone.utc).isoformat()
    history = load_history()
    combined, _ = build_current_row(history)
    # The current row is the last feature row. Preserve the historical target columns.
    primary = fit_engine(combined, "y_dd_6m", 6, seed=42)
    secondary = fit_engine(combined, "y_dd_12m", 12, seed=43)
    update_files(combined, primary, secondary, generated_at)
    print(f"Updated Collapse dashboard for {combined.iloc[-1]['date']:%Y-%m-%d}")
    print(f"6m ensemble: {primary['ensemble_probability']:.4f} ({label(primary['ensemble_probability'])})")
    print(f"12m ensemble: {secondary['ensemble_probability']:.4f} ({label(secondary['ensemble_probability'])})")


if __name__ == "__main__":
    main()

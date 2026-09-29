"""
rainx_ai.py - training, verification and date-based prediction for RAIN-X.

No Streamlit code in here. Everything is computed from the uploaded CSV;
no score, probability or label is hard-coded.

CLI:  python rainx_ai.py --data data/demo_data.csv      (trains + saves models/)
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (GradientBoostingRegressor, RandomForestClassifier,
                              RandomForestRegressor)
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
DATA_LABEL = "Synthetic Demonstration Data - NOT official IMD/NCMRWF observations"
RAW_LABEL = "Raw/Input Rainfall"          # rain_raw is NOT called NWP

# Chronological split (never random). Dates are inclusive upper bounds.
TRAIN_END = "2022-12-31"                   # train      : 2021 - 2022
VAL_END = "2023-07-31"                     # validation : 2023-01 .. 2023-07
                                           # test       : 2023-08 .. 2023-12 (late 2023)

REGIMES = ["Active Monsoon", "Break Monsoon", "Monsoon Depression/Low",
           "Coastal Rainfall", "Orographic Rainfall", "Western Disturbance"]

REGIME_FEATURES = ["lat", "lon", "elevation", "coast_dist", "trough", "pwat", "temp",
                   "rh", "mslp", "wind", "wdir", "cape", "rain_raw", "prev_rain",
                   "month", "day_of_year", "monsoon_season"]
RAIN_FEATURES = ["rain_raw", "pwat", "temp", "rh", "mslp", "wind", "wdir", "cape",
                 "prev_rain", "elevation", "coast_dist", "lat", "lon", "month",
                 "day_of_year"]

# IMD-style 24 h categories used for the heavy-rain classifiers
HEAVY_THRESHOLDS = {"heavy": 64.5, "very_heavy": 115.5, "extreme": 204.5}
MIN_POSITIVES_TO_TRAIN = 10                # need this many train events, else "not estimable"
MIN_ROWS_REGIME_MODEL = 60                 # fewer train rows -> that regime falls back to global

# Thresholds for categorical verification (CSI / POD / FAR)
VERIF_THRESHOLDS = [2.5, 15.5, 64.5]
LOW_EVENT_COUNT = 5                        # flag categorical scores with fewer observed events

# Future-estimate analog window (days of year either side, expanded until enough rows)
ANALOG_WINDOWS = (7, 15, 30, 45)
ANALOG_MIN_ROWS = 12

SEED = 42


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def find_data(path: str | None = None) -> Path:
    cands = [path, "data/demo_data.csv", "demo_data.csv", "/mnt/user-data/uploads/demo_data.csv"]
    for c in cands:
        if c and Path(c).exists():
            return Path(c)
    raise FileNotFoundError("demo_data.csv not found. Put it in ./data/ or pass --data PATH.")


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    df["monsoon_season"] = df["month"].between(6, 9).astype(int)
    return df


def load_data(path: str | None = None) -> pd.DataFrame:
    df = add_derived(pd.read_csv(find_data(path)))
    return df.sort_values(["date", "district"]).reset_index(drop=True)


def split_masks(df: pd.DataFrame):
    tr = df["date"] <= TRAIN_END
    va = (df["date"] > TRAIN_END) & (df["date"] <= VAL_END)
    te = df["date"] > VAL_END
    return tr, va, te


def split_of(date) -> str:
    d = pd.Timestamp(date)
    if d <= pd.Timestamp(TRAIN_END):
        return "train"
    return "validation" if d <= pd.Timestamp(VAL_END) else "test"


# --------------------------------------------------------------------------
# Regime labels (PROTOTYPE rules - not official IMD labels)
# --------------------------------------------------------------------------
# NOTE: the uploaded CSV has no regime column. These are prototype rules written
# from the inputs available at prediction time (obs_rain is NOT used). If your
# existing RAIN-X project already has its own rules, replace `regime_rule`.
def regime_rule(r) -> str:
    if r["month"] in (11, 12, 1, 2, 3):
        return "Western Disturbance"
    if r["mslp"] <= 1003.0 and r["wind"] >= 12:
        return "Monsoon Depression/Low"
    if r["coast_dist"] <= 20 and 200 <= r["wdir"] <= 300 and r["rain_raw"] >= 5:
        return "Coastal Rainfall"
    if r["elevation"] >= 500 and 225 <= r["wdir"] <= 315 and r["wind"] >= 9 and r["rain_raw"] >= 3:
        return "Orographic Rainfall"
    if r["rain_raw"] < 2.5 and r["pwat"] < 42:
        return "Break Monsoon"
    return "Active Monsoon"


def label_regimes(df: pd.DataFrame) -> pd.Series:
    return df.apply(regime_rule, axis=1)


# --------------------------------------------------------------------------
# Verification metrics
# --------------------------------------------------------------------------
def rmse(o, p): return float(np.sqrt(np.mean((np.asarray(p) - np.asarray(o)) ** 2)))
def mae(o, p): return float(np.mean(np.abs(np.asarray(p) - np.asarray(o))))


def categorical(o, p, thr: float) -> dict:
    o, p = np.asarray(o) >= thr, np.asarray(p) >= thr
    hits, miss, fa = int((o & p).sum()), int((o & ~p).sum()), int((~o & p).sum())
    div = lambda a, b: float(a / b) if b > 0 else float("nan")
    return dict(threshold=thr, events=int(o.sum()), hits=hits, misses=miss, false_alarms=fa,
                POD=div(hits, hits + miss), FAR=div(fa, hits + fa),
                CSI=div(hits, hits + miss + fa))


def verify(obs, forecasts: dict) -> pd.DataFrame:
    rows = []
    for name, p in forecasts.items():
        rows.append(dict(forecast=name, RMSE=rmse(obs, p), MAE=mae(obs, p),
                         bias=float(np.mean(np.asarray(p) - np.asarray(obs)))))
    return pd.DataFrame(rows)


def verify_categorical(obs, forecasts: dict) -> pd.DataFrame:
    rows = []
    for name, p in forecasts.items():
        for t in VERIF_THRESHOLDS:
            c = categorical(obs, p, t)
            c["forecast"] = name
            c["low_sample"] = c["events"] < LOW_EVENT_COUNT
            rows.append(c)
    cols = ["forecast", "threshold", "events", "hits", "misses", "false_alarms",
            "POD", "FAR", "CSI", "low_sample"]
    return pd.DataFrame(rows)[cols]


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def _regressor_candidates():
    return {
        "RandomForest": RandomForestRegressor(300, min_samples_leaf=5, max_features=0.5,
                                              n_jobs=-1, random_state=SEED),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=300, max_depth=3,
                                                      learning_rate=0.05, subsample=0.8,
                                                      random_state=SEED),
    }


def _fit_best_regressor(Xtr, ytr, Xva, yva):
    """Fit both candidates on train, keep the one with the lower VALIDATION RMSE."""
    best = None
    for name, m in _regressor_candidates().items():
        m.fit(Xtr, ytr)
        score = rmse(yva, m.predict(Xva)) if len(Xva) else float("inf")
        if best is None or score < best[2]:
            best = (name, m, score)
    return best


def train_all(df: pd.DataFrame, verbose: bool = False) -> dict:
    df = df.copy()
    df["regime"] = label_regimes(df)
    tr_m, va_m, te_m = split_masks(df)
    tr, va, te = df[tr_m], df[va_m], df[te_m]
    log = print if verbose else (lambda *a, **k: None)

    # ---- 1. regime classifier ------------------------------------------------
    clf = RandomForestClassifier(300, min_samples_leaf=2, n_jobs=-1, random_state=SEED)
    clf.fit(tr[REGIME_FEATURES], tr["regime"])
    reg_report = {}
    for nm, part in (("validation", va), ("test", te)):
        pred = clf.predict(part[REGIME_FEATURES])
        reg_report[nm] = dict(accuracy=float(accuracy_score(part["regime"], pred)),
                              macro_f1=float(f1_score(part["regime"], pred, average="macro")),
                              n=int(len(part)))
    log("regime model", reg_report)

    # predicted regime for every row (used to route the regime-aware model)
    df["regime_pred"] = clf.predict(df[REGIME_FEATURES])
    proba = clf.predict_proba(df[REGIME_FEATURES])
    df["regime_conf"] = proba.max(axis=1)
    tr, va, te = df[tr_m], df[va_m], df[te_m]

    # ---- 2. global correction model ------------------------------------------
    g_name, g_model, g_val = _fit_best_regressor(tr[RAIN_FEATURES], tr["obs_rain"],
                                                 va[RAIN_FEATURES], va["obs_rain"])
    log("global model:", g_name, "val RMSE", round(g_val, 3))

    # ---- 3. one correction model per regime (trained on TRUE rule labels) ------
    regime_models, regime_info = {}, {}
    for r in REGIMES:
        tr_r, va_r = tr[tr["regime"] == r], va[va["regime"] == r]
        if len(tr_r) < MIN_ROWS_REGIME_MODEL:
            regime_info[r] = dict(model="global fallback", train_rows=int(len(tr_r)))
            continue
        name, m, s = _fit_best_regressor(tr_r[RAIN_FEATURES], tr_r["obs_rain"],
                                         va_r[RAIN_FEATURES] if len(va_r) else tr_r[RAIN_FEATURES].head(1),
                                         va_r["obs_rain"] if len(va_r) else tr_r["obs_rain"].head(1))
        regime_models[r] = m
        regime_info[r] = dict(model=name, train_rows=int(len(tr_r)), val_rows=int(len(va_r)))

    # ---- 4. heavy-rain classifiers -------------------------------------------
    heavy_models, heavy_info = {}, {}
    for key, thr in HEAVY_THRESHOLDS.items():
        n_tr, n_va, n_te = (int((p["obs_rain"] >= thr).sum()) for p in (tr, va, te))
        info = dict(threshold=thr, events_train=n_tr, events_val=n_va, events_test=n_te)
        if n_tr < MIN_POSITIVES_TO_TRAIN:
            info["status"] = (f"NOT ESTIMABLE: only {n_tr} training event(s) >= {thr} mm "
                              f"(need {MIN_POSITIVES_TO_TRAIN}). No probability is produced.")
        else:
            m = RandomForestClassifier(300, min_samples_leaf=3, n_jobs=-1, random_state=SEED)
            m.fit(tr[RAIN_FEATURES], (tr["obs_rain"] >= thr).astype(int))
            heavy_models[key] = m
            info["status"] = "trained"
            for nm, part in (("val", va), ("test", te)):
                y = (part["obs_rain"] >= thr).astype(int)
                p = m.predict_proba(part[RAIN_FEATURES])[:, 1]
                info[f"brier_{nm}"] = float(brier_score_loss(y, p)) if len(y) else None
                # skill reference: always forecasting the training base rate
                base = np.full(len(y), n_tr / len(tr))
                info[f"brier_{nm}_climatology"] = float(brier_score_loss(y, base)) if len(y) else None
        heavy_info[key] = info

    # ---- 5. predictions + verification on held-out data -----------------------
    def predict_frame(part: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=part.index)
        out["raw"] = part["rain_raw"].values
        out["global"] = np.clip(g_model.predict(part[RAIN_FEATURES]), 0, None)
        out["regime_aware"] = _regime_aware(part, part["regime_pred"], g_model, regime_models)
        return out

    bundle = dict(
        clf=clf, global_model=g_model, global_name=g_name, regime_models=regime_models,
        regime_info=regime_info, heavy_models=heavy_models, heavy_info=heavy_info,
        regime_report=reg_report, regimes_seen=list(clf.classes_),
        splits=dict(train=[str(tr["date"].min().date()), str(tr["date"].max().date()), int(len(tr))],
                    validation=[str(va["date"].min().date()), str(va["date"].max().date()), int(len(va))],
                    test=[str(te["date"].min().date()), str(te["date"].max().date()), int(len(te))]),
        regime_counts=df["regime"].value_counts().to_dict(),
    )

    verification = {}
    for nm, part in (("validation", va), ("test", te)):
        f = predict_frame(part)
        fc = {"Raw/Input": f["raw"], "Global correction": f["global"],
              "Regime-aware correction": f["regime_aware"]}
        verification[nm] = dict(continuous=verify(part["obs_rain"], fc),
                                categorical=verify_categorical(part["obs_rain"], fc))
    bundle["verification"] = verification
    return bundle


def _regime_aware(part, regimes, g_model, regime_models) -> np.ndarray:
    """Pick the correction model of the PREDICTED regime; fall back to global."""
    out = np.clip(g_model.predict(part[RAIN_FEATURES]), 0, None)
    regimes = np.asarray(regimes)
    for r, m in regime_models.items():
        idx = regimes == r
        if idx.any():
            out[idx] = np.clip(m.predict(part.loc[idx, RAIN_FEATURES]), 0, None)
    return out


# --------------------------------------------------------------------------
# Prediction helpers
# --------------------------------------------------------------------------
def predict_rows(bundle: dict, rows: pd.DataFrame) -> pd.DataFrame:
    """Regime + rainfall + heavy-rain probabilities for prepared feature rows."""
    rows = rows.copy()
    clf = bundle["clf"]
    proba = clf.predict_proba(rows[REGIME_FEATURES])
    rows["regime_pred"] = clf.classes_[proba.argmax(axis=1)]
    rows["regime_conf"] = proba.max(axis=1)
    for i, c in enumerate(clf.classes_):
        rows[f"p_{c}"] = proba[:, i]
    rows["rain_global"] = np.clip(bundle["global_model"].predict(rows[RAIN_FEATURES]), 0, None)
    rows["rain_regime_aware"] = _regime_aware(rows, rows["regime_pred"], bundle["global_model"],
                                              bundle["regime_models"])
    for key in HEAVY_THRESHOLDS:
        m = bundle["heavy_models"].get(key)
        rows[f"p_{key}"] = m.predict_proba(rows[RAIN_FEATURES])[:, 1] if m is not None else np.nan
    return rows


def historical_lookup(bundle: dict, df: pd.DataFrame, district: str, date) -> pd.Series | None:
    r = df[(df["district"] == district) & (df["date"] == pd.Timestamp(date))]
    if r.empty:
        return None
    return predict_rows(bundle, r).iloc[0]


def supported_months(df: pd.DataFrame) -> list[int]:
    return sorted(df["month"].unique().tolist())


def _circ_mean_deg(a) -> float:
    a = np.deg2rad(np.asarray(a, dtype=float))
    return float(np.rad2deg(np.arctan2(np.sin(a).mean(), np.cos(a).mean())) % 360)


def analog_rows(df: pd.DataFrame, district: str, date) -> tuple[pd.DataFrame | None, dict]:
    """
    Historical analog states for a future date: rows of the SAME district within
    +/- N days-of-year of the target date (all years 2021-2023). The window is widened
    until it holds enough rows. Returns (rows, info); rows is None if the dataset has
    no data for that time of year.
    """
    date = pd.Timestamp(date)
    doy = date.dayofyear
    d = df[df["district"] == district]
    circ = np.abs(d["day_of_year"] - doy)
    circ = np.minimum(circ, 365 - circ)
    for w in ANALOG_WINDOWS:
        sel = d[circ <= w]
        if len(sel) >= ANALOG_MIN_ROWS:
            # never borrow analogs from a different season: same monsoon_season flag only
            sel = sel[sel["monsoon_season"] == int(date.month in range(6, 10))]
            if len(sel) >= ANALOG_MIN_ROWS:
                return sel, dict(window_days=w, n=int(len(sel)),
                                 years=sorted(sel["date"].dt.year.unique().tolist()))
    return None, dict(window_days=None, n=0, years=[])


def future_estimate(bundle: dict, df: pd.DataFrame, district: str, date) -> dict:
    """
    PROTOTYPE future estimate from historical analogs. Each analog row supplies the
    atmospheric state (trough, pwat, temp, rh, mslp, wind, wdir, cape, rain_raw, prev_rain);
    static fields come from the district; month / day_of_year / monsoon_season come from
    the target date. The models are run on every analog and the results are aggregated.
    There is NO real forecast input for the target date.
    """
    date = pd.Timestamp(date)
    analogs, info = analog_rows(df, district, date)
    if analogs is None:
        return dict(ok=False, months_available=supported_months(df))
    X = analogs.copy()
    X["month"] = date.month
    X["day_of_year"] = date.dayofyear
    X["monsoon_season"] = int(date.month in range(6, 10))
    P = predict_rows(bundle, X)

    probs = P[[f"p_{c}" for c in bundle["clf"].classes_]].mean()
    probs.index = [c[2:] for c in probs.index]
    regime = probs.idxmax()
    rain = P["rain_regime_aware"]
    out = dict(
        ok=True, district=district, date=date, analog_info=info,
        regime=regime, regime_confidence=float(probs.max()),
        regime_probabilities=probs.sort_values(ascending=False).to_dict(),
        rain_mean=float(rain.mean()), rain_median=float(rain.median()),
        rain_p10=float(rain.quantile(0.10)), rain_p90=float(rain.quantile(0.90)),
        rain_global_mean=float(P["rain_global"].mean()),
        raw_input_median=float(analogs["rain_raw"].median()),
        rain_samples=rain.values,
        inputs_summary=dict(
            pwat=float(analogs["pwat"].median()), temp=float(analogs["temp"].median()),
            rh=float(analogs["rh"].median()), mslp=float(analogs["mslp"].median()),
            wind=float(analogs["wind"].median()), wdir=_circ_mean_deg(analogs["wdir"]),
            cape=float(analogs["cape"].median()), prev_rain=float(analogs["prev_rain"].median())),
    )
    for key in HEAVY_THRESHOLDS:
        out[f"p_{key}"] = (float(P[f"p_{key}"].mean()) if key in bundle["heavy_models"] else None)
    return out


# --------------------------------------------------------------------------
# Bundle persistence
# --------------------------------------------------------------------------
def train_and_save(data_path: str | None = None, out_dir: str = "models", verbose=True) -> dict:
    df = load_data(data_path)
    bundle = train_all(df, verbose=verbose)
    Path(out_dir).mkdir(exist_ok=True)
    joblib.dump(bundle, Path(out_dir) / "rainx_ai_bundle.joblib")
    summary = dict(splits=bundle["splits"], regime_counts=bundle["regime_counts"],
                   regime_report=bundle["regime_report"], heavy=bundle["heavy_info"],
                   global_model=bundle["global_name"], regime_info=bundle["regime_info"],
                   verification={k: {n: v.to_dict("records") for n, v in d.items()}
                                 for k, d in bundle["verification"].items()})
    (Path(out_dir) / "metrics.json").write_text(json.dumps(summary, indent=2, default=str))
    return bundle


def load_bundle(out_dir: str = "models"):
    p = Path(out_dir) / "rainx_ai_bundle.joblib"
    return joblib.load(p) if p.exists() else None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default="models")
    a = ap.parse_args()
    b = train_and_save(a.data, a.out)
    print("\nSplits:", b["splits"])
    for k, v in b["verification"]["test"].items():
        print(f"\nTEST {k}:\n", v.round(3).to_string(index=False))
    print("\nHeavy-rain classifiers:")
    for k, v in b["heavy_info"].items():
        print(" ", k, v["status"], {x: v[x] for x in v if x.startswith("events")})

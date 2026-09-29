import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
def _rfr(): return RandomForestRegressor(n_estimators=120, max_depth=12, min_samples_leaf=3, random_state=0, n_jobs=-1)

def train_correction(tr, feats, target="obs_rain"):
    per = {r: _rfr().fit(s[feats], s[target]) for r, s in tr.groupby("regime") if len(s) >= 80}
    return {"global": _rfr().fit(tr[feats], tr[target]), "per_regime": per}

def predict_global(b, df, feats): return np.clip(b["global"].predict(df[feats]), 0, None)

def predict_regime(b, df, feats, regimes):
    out = predict_global(b, df, feats); regimes = np.asarray(regimes)
    for r, m in b["per_regime"].items():
        i = regimes == r
        if i.any(): out[i] = np.clip(m.predict(df[feats][i]), 0, None)
    return out

def train_heavy(tr, feats, thresholds):
    ms = {}
    for t in thresholds:
        y = (tr["obs_rain"] >= t).astype(int)
        ms[t] = RandomForestClassifier(n_estimators=150, max_depth=12, min_samples_leaf=3, class_weight="balanced_subsample",
                                       random_state=0, n_jobs=-1).fit(tr[feats], y) if y.sum() >= 10 else None  # too few events -> no model
    return {"thresholds": list(thresholds), "models": ms}

def predict_heavy(b, df, feats):
    return {t: (m.predict_proba(df[feats])[:, 1] if m is not None else np.full(len(df), np.nan)) for t, m in b["models"].items()}

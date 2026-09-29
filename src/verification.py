import numpy as np, pandas as pd
def rmse(o, p): return float(np.sqrt(np.mean((o - p) ** 2)))
def mae(o, p): return float(np.mean(np.abs(o - p)))
def categorical(o, p, t):
    ho, hp = o >= t, p >= t; h = (ho & hp).sum(); m = (ho & ~hp).sum(); f = (~ho & hp).sum()
    nan = float("nan")
    return dict(events=int(ho.sum()), CSI=h/(h+m+f) if h+m+f else nan, POD=h/(h+m) if h+m else nan, FAR=f/(h+f) if h+f else nan)
def score_table(obs, preds, thresholds):
    rows = []
    for name, p in preds.items():
        r = {"method": name, "RMSE": rmse(obs, p), "MAE": mae(obs, p)}
        for t in thresholds:
            c = categorical(obs, p, t)
            r.update({f"events>={t}": c["events"], f"CSI>={t}": c["CSI"], f"POD>={t}": c["POD"], f"FAR>={t}": c["FAR"]})
        rows.append(r)
    return pd.DataFrame(rows)  # NaN = undefined (no events / no forecasts). FSS not implemented.

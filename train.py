import os, json, joblib
from src.preprocessing import FEATURES, THRESHOLDS, load_data
from src import regime as R, correction as C, verification as V
os.makedirs("models", exist_ok=True)
df = load_data(); test_year = int(df.date.dt.year.max())
tr, te = df[df.date.dt.year < test_year].reset_index(drop=True), df[df.date.dt.year == test_year].reset_index(drop=True)
clf = R.train_classifier(tr[FEATURES], tr["regime"]); cor = C.train_correction(tr, FEATURES); hv = C.train_heavy(tr, FEATURES, THRESHOLDS)
reg = clf.predict(te[FEATURES])  # regime-aware correction uses PREDICTED regimes on held-out data
preds = {"Raw / input rainfall": te.rain_raw.values, "Global correction": C.predict_global(cor, te, FEATURES),
         "Regime-aware correction": C.predict_regime(cor, te, FEATURES, reg)}
tab = V.score_table(te.obs_rain.values, preds, [15.6] + THRESHOLDS); tab.to_csv("models/metrics.csv", index=False)
joblib.dump({"features": FEATURES, "model": clf}, "models/regime_model.pkl")
joblib.dump(cor, "models/correction_models.pkl"); joblib.dump(hv, "models/heavy_rain_model.pkl")
json.dump({"test_year": test_year, "n_train": len(tr), "n_test": len(te), "regime_acc_vs_rule_labels": float((reg == te.regime.values).mean()),
           "data_source": df.data_source.iloc[0], "thresholds": THRESHOLDS}, open("models/info.json", "w"), indent=1)
print(df.data_source.iloc[0], "| train", len(tr), "test", len(te)); print(tab.round(3).T.to_string())

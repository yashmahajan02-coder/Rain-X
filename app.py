import json, joblib, numpy as np, pandas as pd, plotly.express as px, plotly.graph_objects as go, streamlit as st
from src.preprocessing import load_data
from src import correction as C
from src.regime import RULES
st.set_page_config("RAIN-X", page_icon="🌧️", layout="wide")
NAMES = {64.5: "Heavy", 115.5: "Very heavy", 204.5: "Extremely heavy"}

@st.cache_resource
def load():
    try:
        return ({k: joblib.load(f"models/{k}.pkl") for k in ["regime_model", "correction_models", "heavy_rain_model"]},
                json.load(open("models/info.json")), pd.read_csv("models/metrics.csv"))
    except FileNotFoundError: return None

@st.cache_data
def predict():
    m, info, _ = load(); df = load_data(); F = m["regime_model"]["features"]; clf = m["regime_model"]["model"]
    P = clf.predict_proba(df[F]); cls = list(clf.classes_)
    df["pred_regime"] = [cls[i] for i in P.argmax(1)]; df["confidence"] = P.max(1)
    for i, c in enumerate(cls): df["p_" + c] = P[:, i]
    df["rain_ai"] = C.predict_regime(m["correction_models"], df, F, df.pred_regime.values)
    for t, p in C.predict_heavy(m["heavy_rain_model"], df, F).items(): df[f"p_heavy_{t}"] = p
    df["split"] = np.where(df.date.dt.year == info["test_year"], "held-out test", "training")
    return df, cls

st.title("🌧️ RAIN-X"); st.caption("Regime-Aware AI Rainfall Forecast System")
res = load()
if res is None: st.error("Models not found. Run `python train.py` first."); st.stop()
m, info, metrics = res; df, cls = predict(); ths = m["heavy_rain_model"]["thresholds"]
if "Synthetic" in info["data_source"]:
    st.warning("⚠️ **Synthetic Demonstration Data** — not IMD/NCMRWF observations. Numbers below demonstrate the pipeline only, not scientific results.")
with st.sidebar:
    st.header("Selection"); dist = st.selectbox("District", sorted(df.district.unique()))
    ds = df[df.district == dist]; dates = sorted(ds.date.dt.date.unique())
    ts = ds[ds.split != "training"]; k = dates.index(ts.loc[ts.obs_rain.idxmax(), "date"].date()) if len(ts) else len(dates) - 1
    day = st.selectbox("Date", dates, index=k)
    st.caption("Default = wettest held-out day for this district.")
row = ds[ds.date.dt.date == day].iloc[0]; alld = df[df.date.dt.date == day]
c = st.columns(4)
c[0].metric("Detected Regime", row.pred_regime.upper()); c[1].metric("Regime Confidence", f"{row.confidence:.0%}")
c[2].metric("Raw Model/Input Rainfall", f"{row.rain_raw:.1f} mm")
c[3].metric("AI Corrected", f"{row.rain_ai:.1f} mm", f"{(row.rain_ai-row.rain_raw)/row.rain_raw*100:+.0f}%" if row.rain_raw > .1 else None)
c = st.columns(len(ths) + 1); c[0].metric("Observed" + (" (synthetic)" if "Synthetic" in info["data_source"] else ""), f"{row.obs_rain:.1f} mm")
for col, t in zip(c[1:], ths):
    p = row[f"p_heavy_{t}"]; col.metric(f"{NAMES.get(t, 'Event')} ≥{t} mm (configurable)", "n/a (too few events)" if np.isnan(p) else f"{p:.0%}")
l, r = st.columns([3, 2])
w = ds[(ds.date <= pd.Timestamp(day)) & (ds.date >= pd.Timestamp(day) - pd.Timedelta(days=29))]
fig = go.Figure([go.Scatter(x=w.date, y=w[k_], name=n, mode="lines+markers") for k_, n in [("rain_raw", "Raw/Input"), ("rain_ai", "AI corrected"), ("obs_rain", "Observed")]])
fig.update_layout(title=f"Raw vs AI vs Observed — {dist} (30 days to {day})", yaxis_title="mm/day", height=380, legend_orientation="h"); l.plotly_chart(fig, use_container_width=True)
pr = pd.DataFrame({"Regime": cls, "Probability": [row["p_" + x] for x in cls]})
r.plotly_chart(px.bar(pr, x="Probability", y="Regime", orientation="h", range_x=[0, 1], title="Regime probabilities", height=380), use_container_width=True)
st.subheader("Verification (held-out year, all districts)")
st.caption(f"Test year {info['test_year']} · n={info['n_test']} · regime classifier agreement with PROTOTYPE rule labels: {info['regime_acc_vs_rule_labels']:.1%} · NaN = undefined (no events). FSS not implemented.")
st.dataframe(metrics.round(3), use_container_width=True, hide_index=True)
st.subheader(f"District table — {day}")
t = alld[["district", "pred_regime", "confidence", "rain_raw", "rain_ai", "obs_rain", f"p_heavy_{ths[0]}"]].rename(columns={"rain_raw": "raw/input mm", "rain_ai": "AI mm", "obs_rain": "observed mm", f"p_heavy_{ths[0]}": f"P(≥{ths[0]} mm)"})
a, b = st.columns([3, 2]); a.dataframe(t.round(2), use_container_width=True, hide_index=True)
mp = px.scatter_geo(alld, lat="lat", lon="lon", size=alld.rain_ai.clip(lower=1), color="rain_ai", hover_name="district", color_continuous_scale="Blues", title="AI corrected rainfall (mm)")
mp.update_geos(fitbounds="locations", showcountries=True, showsubunits=True); b.plotly_chart(mp, use_container_width=True)
with st.expander("Regime label rules (prototype, NOT official IMD)"): st.code(RULES)

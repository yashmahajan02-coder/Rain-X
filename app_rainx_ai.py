"""
app_rainx_ai.py - RAIN-X dashboard modes: Historical Verification / AI Future Prediction.

Run:  streamlit run app_rainx_ai.py

render_historical() and render_future() take (bundle, df, district) so they can be
called from an existing RAIN-X app; the file only launches itself when run directly.
"""
import datetime as dt

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import rainx_ai as R

MODE_HIST, MODE_FUT = "Historical Verification", "AI Future Prediction"
COLORS = {"Raw/Input": "#9aa5b1", "Global correction": "#2f80ed",
          "Regime-aware correction": "#27ae60", "Observed": "#111827"}


# ----------------------------------------------------------------- helpers
def pct(p):
    if p is None or pd.isna(p):
        return "Not estimable"
    return "<0.1%" if p < 0.001 else f"{100 * p:.1f}%"


def mm(x):
    return f"{x:.1f} mm"


def data_banner():
    st.warning(f"**{R.DATA_LABEL}.** Demonstration/training data only. "
               f"`rain_raw` is shown as **{R.RAW_LABEL}** (a prototype forecast input), not NWP.")


@st.cache_data(show_spinner=False)
def get_data():
    return R.load_data()


@st.cache_resource(show_spinner="Training AI models from the CSV (first run only)...")
def get_bundle():
    b = R.load_bundle()
    return b if b is not None else R.train_and_save(verbose=False)


def heavy_table(bundle):
    rows = []
    for k, v in bundle["heavy_info"].items():
        rows.append({"class": k.replace("_", " "), "threshold (mm)": v["threshold"],
                     "events train/val/test": f'{v["events_train"]}/{v["events_val"]}/{v["events_test"]}',
                     "status": v["status"]})
    return pd.DataFrame(rows)


def regime_bar(probs: dict, title: str):
    s = pd.Series(probs).sort_values()
    fig = go.Figure(go.Bar(x=s.values * 100, y=s.index, orientation="h",
                           marker_color="#2f80ed", text=[f"{v * 100:.0f}%" for v in s.values],
                           textposition="outside"))
    fig.update_layout(title=title, height=280, margin=dict(l=10, r=30, t=40, b=10),
                      xaxis_title="%", xaxis_range=[0, 110])
    return fig


# ----------------------------------------------------------- historical mode
def render_historical(bundle, df, district):
    st.header("Historical Verification")
    st.caption("Historical data → train AI → verify AI → use learned patterns to estimate future rainfall.")
    d = df[df["district"] == district]
    lo, hi = d["date"].min().date(), d["date"].max().date()
    default = d[R.split_masks(d)[2]]["date"].iloc[0].date()   # first held-out test day
    date = st.date_input("Date", value=default, min_value=lo, max_value=hi)

    row = R.historical_lookup(bundle, df, district, date)
    if row is None:
        nearest = d.iloc[(d["date"] - pd.Timestamp(date)).abs().argsort()[:3]]["date"].dt.date.tolist()
        st.info(f"The dataset has no record for {district} on {date}. It covers only "
                f"months {R.supported_months(df)}. Nearest available dates: "
                + ", ".join(map(str, nearest)))
    else:
        split = R.split_of(date)
        st.subheader(f"{district.upper()} · {date:%d %B %Y}")
        if split == "test":
            st.success("Held-out TEST date: the models never saw this row during training or tuning.")
        else:
            st.warning(f"This date is in the **{split}** period, so it was used to train/tune the models "
                       "(in-sample). Pick a date from 2023-08-01 onward for an honest check.")
        c = st.columns(4)
        c[0].metric(R.RAW_LABEL, mm(row["rain_raw"]))
        c[1].metric("Global correction", mm(row["rain_global"]))
        c[2].metric("Regime-aware correction", mm(row["rain_regime_aware"]))
        c[3].metric("Observed rainfall", mm(row["obs_rain"]))
        c = st.columns(4)
        c[0].metric("Regime (prototype label)", row["regime_pred"])
        c[1].metric("Regime confidence", f'{100 * row["regime_conf"]:.0f}%')
        c[2].metric("P(≥ 64.5 mm)", pct(row["p_heavy"]))
        c[3].metric("P(≥ 115.5 mm)", pct(row["p_very_heavy"]))

        fig = go.Figure(go.Bar(
            x=["Raw/Input", "Global correction", "Regime-aware correction", "Observed"],
            y=[row["rain_raw"], row["rain_global"], row["rain_regime_aware"], row["obs_rain"]],
            marker_color=list(COLORS.values()), text=None))
        fig.update_layout(title="Rainfall comparison (mm)", height=300, margin=dict(t=40, b=10))
        cols = st.columns(2)
        cols[0].plotly_chart(fig)
        probs = {k[2:]: row[k] for k in row.index if k.startswith("p_") and k[2:] in R.REGIMES}
        cols[1].plotly_chart(regime_bar(probs, "Regime probabilities"))

    st.divider()
    st.subheader("Verification on the held-out test period")
    sp = bundle["splits"]
    st.caption(f"Train {sp['train'][0]} → {sp['train'][1]} ({sp['train'][2]} rows) · "
               f"Validation {sp['validation'][0]} → {sp['validation'][1]} ({sp['validation'][2]}) · "
               f"**Test {sp['test'][0]} → {sp['test'][1]} ({sp['test'][2]})**. Chronological split, no shuffling.")
    v = bundle["verification"]["test"]
    st.markdown("**RMSE / MAE / bias (mm)** — computed from the models' test predictions")
    st.dataframe(v["continuous"].round(3), hide_index=True)
    st.markdown("**CSI / POD / FAR** by threshold")
    cat = v["categorical"].round(3).copy()
    cat["note"] = cat["low_sample"].map(lambda x: f"< {R.LOW_EVENT_COUNT} observed events: not reliable" if x else "")
    st.dataframe(cat.drop(columns="low_sample"), hide_index=True)
    st.caption("NaN = undefined (no forecast/observed events at that threshold). FSS is not computed: it needs "
               "gridded fields, and this dataset is one point per district.")

    with st.expander("Model details and honest caveats"):
        rr = bundle["regime_report"]["test"]
        st.write(f"**Regime model** (RandomForest): test accuracy {rr['accuracy']:.1%}, macro-F1 "
                 f"{rr['macro_f1']:.3f}. These labels come from *prototype rules* applied to the inputs "
                 "(not official IMD labels), so this accuracy measures how well the forest reproduces those "
                 "rules, not real meteorological skill.")
        st.write(f"**Global correction model:** {bundle['global_name']} (chosen on validation RMSE). "
                 "Regime-aware models: " + ", ".join(f"{k}: {v['model']} ({v['train_rows']} rows)"
                                                     for k, v in bundle["regime_info"].items()))
        st.write("**Heavy-rain classifiers**")
        st.dataframe(heavy_table(bundle), hide_index=True)


# --------------------------------------------------------------- future mode
def render_future(bundle, df, district):
    st.header("AI DATE PREDICTION")
    st.caption("Operational pipeline (NOT active here — needs real forecast data): NCMRWF forecast → regime "
               "detection → AI post-processing → operational rainfall prediction.")
    st.error("**Prototype future estimate based on historical dataset — NOT an official NCMRWF/IMD forecast.** "
             "The CSV ends on 2023-12-29 and holds no forecast inputs for future dates. The estimate is built "
             "from the district's historical atmospheric states for the same time of year (2021–2023) run "
             "through the trained models, so it behaves like a learned seasonal climatology and cannot "
             "anticipate any real weather event.")
    last = df["date"].max().date()
    date = st.date_input("Forecast date", value=dt.date(2026, 9, 30),
                         min_value=last + dt.timedelta(days=1))
    res = R.future_estimate(bundle, df, district, date)
    if not res["ok"]:
        names = ", ".join(dt.date(2000, m, 1).strftime("%b") for m in res["months_available"])
        st.info(f"The dataset only has data for these months: {names}. No estimate can be made for "
                f"{date:%d %B}. Pick a date in a supported month.")
        return

    st.subheader(f"{district.upper()} · {date:%d %B %Y}")
    st.markdown("### AI PREDICTION")
    c = st.columns(3)
    c[0].metric("Regime (prototype label)", res["regime"])
    c[1].metric("Regime confidence", f'{100 * res["regime_confidence"]:.0f}%',
                help="Share of the historical analog states that the regime model assigns to this regime.")
    c[2].metric("Predicted rainfall (expected)", mm(res["rain_mean"]),
                help="Mean over analog states of the regime-aware corrected rainfall.")
    c = st.columns(3)
    c[0].metric("Heavy rain P(≥ 64.5 mm)", pct(res["p_heavy"]))
    c[1].metric("Very heavy P(≥ 115.5 mm)", pct(res["p_very_heavy"]))
    c[2].metric("Extreme P(≥ 204.5 mm)", pct(res["p_extreme"]))
    if res["p_extreme"] is None:
        st.caption("Extreme rain is not estimable: " + bundle["heavy_info"]["extreme"]["status"])
    st.write(f"Median {mm(res['rain_median'])} · 10th–90th percentile of analog states: "
             f"{mm(res['rain_p10'])} – {mm(res['rain_p90'])}")

    ai = res["analog_info"]
    st.markdown(f"**Data basis:** historical 2021–2023 dataset ({ai['n']} analog states from "
                f"±{ai['window_days']} days of {date:%d %b}, years {ai['years']})  \n"
                "**Status:** Prototype future estimate")
    cols = st.columns(2)
    cols[0].plotly_chart(regime_bar(res["regime_probabilities"], "Regime share across analog states"))
    fig = go.Figure(go.Histogram(x=res["rain_samples"], nbinsx=25, marker_color="#27ae60"))
    fig.update_layout(title="Spread of estimated rainfall across analog states", height=280,
                      xaxis_title="mm", margin=dict(l=10, r=10, t=40, b=10))
    cols[1].plotly_chart(fig)
    with st.expander("Inputs used (historical medians, not a forecast)"):
        s = res["inputs_summary"]
        st.write({**{k: round(v, 2) for k, v in s.items()},
                  R.RAW_LABEL + " (median)": round(res["raw_input_median"], 2)})
        st.caption("The same month/day gives the same estimate for any future year: there is no trend, "
                   "ENSO or current-weather signal in this dataset.")


# ---------------------------------------------------------------------- main
def main():
    st.set_page_config(page_title="RAIN-X AI", layout="wide")
    st.title("RAIN-X · Regime-aware AI rainfall post-processing")
    data_banner()
    df, bundle = get_data(), get_bundle()
    with st.sidebar:
        mode = st.radio("MODE", [MODE_HIST, MODE_FUT])
        district = st.selectbox("District", sorted(df["district"].unique()),
                                index=sorted(df["district"].unique()).index("Nashik"))
    (render_historical if mode == MODE_HIST else render_future)(bundle, df, district)


if __name__ == "__main__":
    main()

# RAIN-X — Regime-Aware AI Rainfall Forecast System (SIH 26080 MVP)
Pipeline: input data → regime classification → regime-specific rainfall correction → heavy-rain probability → district forecast → verification dashboard.

    pip install -r requirements.txt
    python train.py
    streamlit run app.py

## Data honesty
- Default data is **Synthetic Demonstration Data** (`data/demo_data.csv`, auto-generated). It is NOT IMD/NCMRWF data; all scores on it are demo numbers, not scientific results.
- `rain_raw` is "Raw Model/Input Rainfall". It becomes NWP (NCUM/NCUM-R) only when you supply operational forecasts; reanalysis (IMDAA/MERA) or observations must not be called NWP.
- To use real data: save `data/real_data.csv` with columns `date,district,lat,lon,elevation,coast_dist,rain_raw,obs_rain,temp,rh,mslp,wind,wdir,cape,pwat,trough,prev_rain` (obs_rain e.g. IMD 0.25° gridded rainfall aggregated to district), then rerun `python train.py`.
- Regime labels are prototype rules (see `src/regime.py`, also shown in the app), not official IMD labels. The classifier learns these rules, so its accuracy measures rule reproduction only.
- Heavy-rain thresholds (64.5 / 115.5 / 204.5 mm) are configurable via `THRESHOLDS` in `src/preprocessing.py`. Thresholds with <10 training events get no model (shown n/a).
- Validation: train on earlier years, test on the last year; regime-aware correction uses *predicted* regimes. Metrics: RMSE, MAE, CSI, POD, FAR. FSS not implemented.

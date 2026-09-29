import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
REGIMES = ["Active Monsoon", "Break Monsoon", "Monsoon Depression/Low", "Coastal Rainfall", "Orographic Rainfall", "Western Disturbance"]
RULES = """PROTOTYPE regime labels (transparent rules, NOT official IMD labels). First match wins:
1. Western Disturbance: Dec-Feb and upper-level trough index > 0.6
2. Monsoon Depression/Low: Jun-Sep, MSLP < 1000 hPa and wind >= 10 m/s
3. Orographic Rainfall: Jun-Sep, elevation >= 400 m, wind >= 7 m/s, wind direction 225-315 deg (westerly)
4. Coastal Rainfall: Jun-Sep, distance to coast < 60 km
5. Active Monsoon: Jun-Sep, PWAT >= 46 mm
6. Break Monsoon: Jun-Sep, PWAT < 46 mm
Other days (Oct-May without WD trough) are out of scope and dropped."""

def label_regimes(df):
    m = df["date"].dt.month; mon = m.between(6, 9); wd = m.isin([12, 1, 2])
    conds = [wd & (df.trough > 0.6), mon & (df.mslp < 1000) & (df.wind >= 10),
             mon & (df.elevation >= 400) & (df.wind >= 7) & df.wdir.between(225, 315),
             mon & (df.coast_dist < 60), mon & (df.pwat >= 46), mon]
    lab = np.select(conds, [REGIMES[i] for i in (5, 2, 4, 3, 0, 1)], default="")
    return pd.Series(lab, index=df.index).replace("", np.nan)

def train_classifier(X, y):
    return RandomForestClassifier(n_estimators=150, max_depth=12, class_weight="balanced", random_state=0, n_jobs=-1).fit(X, y)

import os, numpy as np, pandas as pd
from .regime import label_regimes
FEATURES = ["rain_raw", "temp", "rh", "mslp", "wind", "wdir", "cape", "pwat", "trough", "elevation", "coast_dist", "prev_rain"]
THRESHOLDS = [64.5, 115.5, 204.5]  # mm/day, CONFIGURABLE (heavy / very heavy / extremely heavy)
# name: (lat, lon, approx elevation m, approx coast distance km) -- approximate, for the demo
DISTRICTS = {"Mumbai": (19.08, 72.88, 10, 2), "Thane": (19.2, 72.97, 20, 15), "Raigad": (18.52, 73.18, 30, 10),
    "Ratnagiri": (16.99, 73.3, 20, 8), "Pune": (18.52, 73.86, 560, 110), "Satara": (17.68, 74.0, 700, 120),
    "Kolhapur": (16.7, 74.24, 550, 100), "Nashik": (19.99, 73.79, 600, 140), "Dhule": (20.9, 74.77, 250, 250),
    "Nagpur": (21.15, 79.09, 310, 600), "Chh. Sambhajinagar": (19.88, 75.34, 580, 300)}
FACTOR = {"Active Monsoon": 1.3, "Break Monsoon": .3, "Monsoon Depression/Low": 2.2, "Coastal Rainfall": 1.6, "Orographic Rainfall": 1.8, "Western Disturbance": .5}
BIAS = {"Active Monsoon": .85, "Break Monsoon": 1.3, "Monsoon Depression/Low": .75, "Coastal Rainfall": .7, "Orographic Rainfall": .6, "Western Disturbance": 1.2}

def make_synthetic(seed=0, years=(2021, 2022, 2023)):
    """SYNTHETIC DEMONSTRATION DATA. Not IMD/NCMRWF data. Raw-rain biases are invented per regime."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(f"{years[0]}-01-01", f"{years[-1]}-12-31"); n = len(dates)
    doy = dates.dayofyear.values; mon = ((doy >= 152) & (doy <= 273)).astype(float)
    ar = np.zeros(n)
    for i in range(1, n): ar[i] = .93 * ar[i-1] + rng.normal(0, .4)
    spell = 1 / (1 + np.exp(-2.5 * ar)); dep = (rng.random(n) < .05) * mon
    trough = np.clip(rng.normal(.2, .3, n) + (1 - mon) * (rng.random(n) < .2) * .5, 0, 1.5)
    out = []
    for name, (lat, lon, el, cd) in DISTRICTS.items():
        co = float(cd < 60)
        d = pd.DataFrame(dict(date=dates, district=name, lat=lat, lon=lon, elevation=el, coast_dist=cd, trough=trough))
        d["pwat"] = 30 + mon*(6 + 16*spell + 4*co) + rng.normal(0, 2.5, n)
        d["temp"] = 26 + mon*(4 - 4*spell) - el/1000*6 + rng.normal(0, 1.5, n)
        d["rh"] = np.clip(50 + mon*(15 + 25*spell) + rng.normal(0, 6, n), 20, 100)
        d["mslp"] = 1012 - mon*(4 + 6*spell) - dep*8 + rng.normal(0, 1, n)
        d["wind"] = 4 + mon*(3 + 4*spell + 2*co) + dep*7 + np.abs(rng.normal(0, 1.5, n))
        d["wdir"] = np.where(mon > 0, rng.normal(260, 25, n), rng.normal(300, 40, n)) % 360
        d["cape"] = np.clip(mon*(500 + 900*spell) + rng.normal(0, 250, n), 0, None)
        d["regime"] = label_regimes(d)
        fac = d.regime.map(FACTOR).fillna(.3).values
        mu = .12*np.clip(d.pwat.values - 32, 0, None)**1.6*fac + .3 + 1.5*(d.regime == "Western Disturbance").values
        d["obs_rain"] = rng.gamma(.7, mu/.7)
        d["rain_raw"] = np.clip(d.obs_rain*d.regime.map(BIAS).fillna(1).values*rng.lognormal(0, .4, n) + rng.normal(0, 1.5, n), 0, None)
        d["prev_rain"] = d.obs_rain.shift(1).fillna(0)
        out.append(d.dropna(subset=["regime"]))
    df = pd.concat(out).drop(columns="regime").sort_values(["date", "district"])
    fc = df.select_dtypes("float").columns; df[fc] = df[fc].round(2)
    df["data_source"] = "Synthetic Demonstration Data"
    return df.reset_index(drop=True)

def load_data(path=None):
    """Uses data/real_data.csv (same columns) if present, else data/demo_data.csv (auto-generated synthetic)."""
    path = path or ("data/real_data.csv" if os.path.exists("data/real_data.csv") else "data/demo_data.csv")
    if not os.path.exists(path):
        os.makedirs("data", exist_ok=True); make_synthetic().to_csv(path, index=False)
    df = pd.read_csv(path, parse_dates=["date"])
    if "data_source" not in df: df["data_source"] = "User-supplied data"
    df["regime"] = label_regimes(df)
    return df.dropna(subset=["regime"]).reset_index(drop=True)

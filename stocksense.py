#!/usr/bin/env python3
"""
StockSense - Retail Demand Estimation & Stockout Risk Analysis   (Problem Statement 10)
=========================================================================================
ONE FILE that does everything the web app will later call:

  1. Clean the UCI Online Retail data, split it into REGIONS (top countries + Rest of World)
  2. Aggregate to weekly demand per region x product and build leakage-free features
  3. Compare baselines vs ML models (chronological test) -> MAE, RMSE, R2, WAPE
  4. Forecast NEXT-WEEK demand for every product-region (best method chosen automatically)
  5. Demand class  : High / Medium / Low (terciles of forecast demand)
  6. Inventory     : buffer (safety) stock, reorder point, days of cover, stockout risk level
                     + 0-100 risk score, recommended order quantity
  7. Expiry        : Perishable / Non-perishable, "expires in N days", wastage + order cap
  8. Discount      : none for High demand, rule-based discount for Low demand (with margin cap)
  9. Reorder overview (counts by risk / demand class / region, units, cost)
 10. Add a NEW product or a NEW dataset (merge / replace) -> saved in SQLite, outputs refreshed
 11. Filter / sort the results (by demand class, risk, region ...)

QUICK START
  pip install pandas numpy scikit-learn openpyxl joblib
  python stocksense.py run                      # full pipeline  (put Online_Retail.xlsx next to this file)
  python stocksense.py run --demo-expiry        # demo: treat a random 8% of products as perishable (simulated)
  python stocksense.py show --demand Low --sort days_of_cover --top 15
  python stocksense.py show --risk High --region "United Kingdom"
  python stocksense.py reorders                 # reorder overview
  python stocksense.py discounts                # discount suggestions
  python stocksense.py product 85123A           # full analysis of one product
  python stocksense.py add-product --code NEW01 --name "TEST MUG" --price 3.5 --stock 120 \
         --region "United Kingdom" --weekly-sales 30,28,35,40 [--perishable --shelf-days 45]
  python stocksense.py set-type 22086 --perishable --shelf-days 60
  python stocksense.py set-stock 85123A --region "United Kingdom" --stock 500 --lead-time 10
  python stocksense.py add-dataset more_sales.csv --mode merge     # or --mode replace

IMPORTANT ASSUMPTIONS (state these in your report)
  * The dataset has NO stock-on-hand, NO expiry dates, NO costs  -> these are SIMULATED or
    USER-PROVIDED and are always labelled as such in the outputs.
  * Country = customer's delivery country (not a warehouse location).
  * Only ~1 year of data: yearly seasonality cannot be learned, weekly demand is lumpy
    (wholesale buyers), so forecast accuracy is limited (see the printed metrics).
  * Discount demand uplift uses an ASSUMED price elasticity (default 1.5) - not learned from data.
"""
import os, sys, json, math, argparse, sqlite3, hashlib, warnings
from datetime import date, datetime
from statistics import NormalDist

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)

# =============================================================================
# CONFIGURATION  (everything the UI will later expose as settings)
# =============================================================================
CFG = dict(
    data_path="Online_Retail.xlsx", out_dir="outputs", model_dir="models",
    db_path="stocksense.db", merged_raw_path="data_merged_raw.pkl",
    # data
    n_top_regions=5, top_n_products=300, min_active_weeks=8, test_fraction=0.20,
    qty_cap_pct=99.5, seed=42, product_code_regex=r"^\d{5}",
    # demand classes: percentile cut-offs of forecast demand
    demand_pct=(1 / 3, 2 / 3),
    # inventory
    service_level=0.95, default_lead_days=7, region_lead_days={"Rest of World": 14},
    review_weeks=1, stock_mult_range=(0.3, 6.0),      # simulated stock = 0.3 - 6 weeks of avg weekly demand
    max_safety_weeks=3,                               # cap: buffer <= 3 weeks of demand (lumpy items)
    demo_perishable_share=0.08,                       # --demo-expiry: share of products treated as perishable
    # expiry
    expiring_soon_days=30,
    # discount (all ASSUMPTIONS - editable)
    cost_ratio=0.60, min_margin=0.10, elasticity=1.5, max_discount=0.40,
    cover_tiers=[(15, 0), (30, 10), (60, 20), (10 ** 9, 30)],   # (days of cover below, discount %)
    falling_threshold=0.85, falling_extra_pct=5, expiry_extra_max_pct=20,
)

NON_PRODUCT = {"POST", "DOT", "M", "C2", "D", "S", "BANK CHARGES", "AMAZONFEE", "CRUK",
               "DCGSSGIRL", "DCGSSBOY", "B", "PADS", "ADJUST", "ADJUST2"}
FOOD_TOKENS = {"CHOCOLATE", "CHOCOLATES", "BISCUIT", "BISCUITS", "COOKIES", "FUDGE", "CANDY", "SWEETS",
               "HONEY", "COFFEE", "TEABAGS", "POPCORN", "SNACKS", "SAUCE", "MARMALADE", "CRISPS", "CEREAL"}
NONFOOD_TOKENS = {"CUTTER", "CUTTERS", "TIN", "TINS", "CASES", "MUG", "MUGS", "HOLDER", "BOTTLE", "CANDLE",
                  "CANDLES", "SIGN", "BOX", "BOXES", "CUP", "TOWEL", "TOWELS", "BAG", "BAGS", "CARD", "JAR",
                  "JARS", "PLATE", "SCENTED", "WRAP", "BADGE", "STICKERS", "LIGHT", "DECORATION"}
REQUIRED_COLS = ["InvoiceNo", "StockCode", "Quantity", "InvoiceDate", "UnitPrice", "Country"]
KEY = ["Region", "StockCode"]
BASELINES = {"Baseline: last week": "lag_0", "Baseline: 4-wk moving avg": "roll_mean_4",
             "Baseline: exp. smoothing": "ewma"}
ML_NAMES = ["Linear Regression (Ridge)", "Random Forest", "Gradient Boosting"]
ENS_ML = "Ensemble: 3 ML models avg"
ENS_ALL = "Ensemble: ML + smoothing + MA4"
FEATURES_BASE = ["x_lag0", "x_lag1", "x_lag3", "x_roll8", "x_std4", "x_level", "x_cust", "x_price",
                 "mkt_growth", "mkt_wow", "holiday_season", "product_age_weeks"]


def log(msg, verbose=True):
    if verbose:
        print(msg)


# =============================================================================
# 1. LOADING + CLEANING
# =============================================================================
def read_any(path):
    """Read csv / xlsx (xlsx cached as pickle - reading it is slow)."""
    if path.lower().endswith((".xlsx", ".xls")):
        cache = path + ".cache.pkl"
        if os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(path):
            return pd.read_pickle(cache)
        df = pd.read_excel(path)
        df.to_pickle(cache)
        return df
    if path.lower().endswith(".pkl"):
        return pd.read_pickle(path)
    return pd.read_csv(path, encoding="latin-1")


def validate_columns(df):
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}. Required: {REQUIRED_COLS}")


def load_active_raw(cfg):
    """Active dataset = merged/replaced dataset if present, else the original file."""
    if os.path.exists(cfg["merged_raw_path"]):
        return pd.read_pickle(cfg["merged_raw_path"]), cfg["merged_raw_path"]
    if not os.path.exists(cfg["data_path"]):
        raise FileNotFoundError(f"{cfg['data_path']} not found. Put Online_Retail.xlsx next to stocksense.py "
                                f"or pass --data PATH.")
    return read_any(cfg["data_path"]), cfg["data_path"]


def clean(raw, cfg, verbose=True):
    df = raw.copy()
    validate_columns(df)
    for c in ["Description", "CustomerID"]:
        if c not in df.columns:
            df[c] = np.nan
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["StockCode", "InvoiceDate"])
    df["StockCode"] = df["StockCode"].astype(str).str.strip()
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    n0 = len(df)
    df = df[~df["InvoiceNo"].str.startswith("C")]                    # cancellations
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]            # returns, free items
    df = df[~df["StockCode"].str.upper().isin(NON_PRODUCT)]          # postage, fees ...
    pat = df["StockCode"].str.match(cfg["product_code_regex"])
    if pat.mean() >= 0.5:                                            # skip rule for non-UCI datasets
        df = df[pat]
    df = df.drop_duplicates()
    cap = df.groupby("StockCode")["Quantity"].transform(lambda s: s.quantile(cfg["qty_cap_pct"] / 100))
    df["Quantity"] = np.minimum(df["Quantity"], cap)                 # cap outliers per product
    desc = df.dropna(subset=["Description"]).groupby("StockCode")["Description"].first()
    df["Description"] = df["Description"].fillna(df["StockCode"].map(desc)).fillna("UNKNOWN")
    log(f"[clean] rows {n0:,} -> {len(df):,} | products {df['StockCode'].nunique():,} | "
        f"countries {df['Country'].nunique()} | {df['InvoiceDate'].min().date()} -> {df['InvoiceDate'].max().date()}",
        verbose)
    return df


def map_regions(df, cfg):
    top = (df.groupby("Country")["Quantity"].sum().sort_values(ascending=False)
             .head(cfg["n_top_regions"]).index.tolist())
    df["Region"] = np.where(df["Country"].isin(top), df["Country"], "Rest of World")
    return df, top + ["Rest of World"]


# =============================================================================
# 2. WEEKLY AGGREGATION + FEATURES
# =============================================================================
def build_weekly(df, cfg):
    df = df.copy()
    df["Week"] = df["InvoiceDate"].dt.to_period("W").dt.start_time
    if (df["InvoiceDate"].max() - df["Week"].max()).days < 6:        # drop incomplete final week
        df = df[df["Week"] < df["Week"].max()]
    top_products = (df.groupby("StockCode")["Quantity"].sum().sort_values(ascending=False)
                      .head(cfg["top_n_products"]).index)
    d = df[df["StockCode"].isin(top_products)]
    weeks = pd.date_range(d["Week"].min(), d["Week"].max(), freq="W-MON")
    agg = (d.groupby(["Region", "StockCode", "Week"])
             .agg(demand=("Quantity", "sum"), price=("UnitPrice", "mean"), customers=("CustomerID", "nunique"))
             .reset_index())
    keys = agg[KEY].drop_duplicates()
    grid = keys.merge(pd.DataFrame({"Week": weeks}), how="cross")     # complete grid: missing weeks = 0
    g = grid.merge(agg, on=KEY + ["Week"], how="left")
    g["demand"] = g["demand"].fillna(0)
    g["customers"] = g["customers"].fillna(0)
    g["price"] = g["price"].fillna(g["StockCode"].map(d.groupby("StockCode")["UnitPrice"].mean()))
    g = g.sort_values(KEY + ["Week"]).reset_index(drop=True)
    names = d.groupby("StockCode")["Description"].first()
    return g, names


def add_features(g, alphas=(0.2, 0.3, 0.4, 0.5)):
    """All features use PAST data only (week t predicts week t+1)."""
    grp = g.groupby(KEY)["demand"]
    g["lag_0"] = g["demand"]
    g["lag_1"] = grp.shift(1)
    g["lag_3"] = grp.shift(3)
    g["roll_mean_4"] = grp.transform(lambda s: s.rolling(4, min_periods=1).mean())
    g["roll_mean_8"] = grp.transform(lambda s: s.rolling(8, min_periods=1).mean())
    g["roll_std_4"] = grp.transform(lambda s: s.rolling(4, min_periods=2).std())
    g["cust_roll_4"] = g.groupby(KEY)["customers"].transform(lambda s: s.rolling(4, min_periods=1).mean())
    for a in alphas:
        g[f"ew_{a}"] = grp.transform(lambda s, a=a: s.ewm(alpha=a, adjust=False).mean())
    # market momentum per region (is overall demand accelerating? e.g. Christmas ramp-up)
    mk = g.groupby(["Region", "Week"])["demand"].sum().rename("mkt").reset_index().sort_values(["Region", "Week"])
    mk["r4"] = mk.groupby("Region")["mkt"].transform(lambda s: s.rolling(4, min_periods=1).mean())
    mk["prev"] = mk.groupby("Region")["mkt"].transform(lambda s: s.shift(4).rolling(4, min_periods=1).mean())
    mk["mkt_growth"] = (mk["r4"] / mk["prev"].replace(0, np.nan)).fillna(1).clip(0.3, 3)
    mk["mkt_wow"] = (mk["mkt"] / mk.groupby("Region")["mkt"].shift(1).replace(0, np.nan)).fillna(1).clip(0.3, 3)
    g = g.merge(mk[["Region", "Week", "mkt_growth", "mkt_wow"]], on=["Region", "Week"], how="left")
    g["month"] = g["Week"].dt.month
    g["holiday_season"] = g["month"].isin([11, 12]).astype(int)
    first = g[g["demand"] > 0].groupby(KEY)["Week"].min().rename("first_sale")
    g = g.merge(first, on=KEY, how="left")
    g["product_age_weeks"] = ((g["Week"] - g["first_sale"]).dt.days / 7).clip(lower=0).fillna(0)
    g["target"] = g.groupby(KEY)["demand"].shift(-1)                  # NEXT week's demand
    return g


def add_relative(g, regions):
    """Scale-free features: everything relative to the recent 4-week level."""
    den = g["roll_mean_4"].clip(lower=1)
    g["x_lag0"] = g["lag_0"] / den
    g["x_lag1"] = g["lag_1"] / den
    g["x_lag3"] = g["lag_3"] / den
    g["x_roll8"] = g["roll_mean_8"] / den
    g["x_std4"] = g["roll_std_4"] / den
    g["x_level"] = np.log1p(den)
    g["x_cust"] = np.log1p(g["cust_roll_4"])
    g["x_price"] = np.log1p(g["price"])
    g["y_ratio"] = (g["target"] / den).clip(upper=6)                  # model predicts a multiple of recent level
    for r in regions:
        g["reg_" + r] = (g["Region"] == r).astype(int)
    return g


# =============================================================================
# 3. MODELS + EVALUATION
# =============================================================================
def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {"MAE": mean_absolute_error(y, p), "RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "R2": r2_score(y, p), "WAPE_%": 100 * np.abs(y - p).sum() / max(y.sum(), 1e-9)}


def make_models(seed):
    return {
        "Linear Regression (Ridge)": Ridge(alpha=5.0),
        "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=30, max_features=0.5,
                                               n_jobs=-1, random_state=seed),
        "Gradient Boosting": HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=8,
                                                           min_samples_leaf=80, l2_regularization=5.0,
                                                           random_state=seed),
    }


def predict_candidate(name, rows, fitted, features):
    """Unified prediction (units) for any candidate method on rows that have features."""
    if name in BASELINES:
        return rows[BASELINES[name]].values.astype(float)
    scale = rows["roll_mean_4"].clip(lower=1).values
    ml = {n: np.clip(fitted[n].predict(rows[features]), 0, None) * scale for n in ML_NAMES}
    if name in ml:
        return ml[name]
    ens_ml = np.mean([ml[n] for n in ML_NAMES], axis=0)
    if name == ENS_ML:
        return ens_ml
    if name == ENS_ALL:
        return (ens_ml + rows["ewma"].values + rows["roll_mean_4"].values) / 3
    raise KeyError(name)


def evaluate_models(g, regions, cfg, verbose=True):
    """Chronological evaluation. Returns everything needed for reporting and the final fit."""
    features = FEATURES_BASE + ["reg_" + r for r in regions]
    weeks = sorted(g["Week"].unique())
    cut = int(len(weeks) * (1 - cfg["test_fraction"]))
    train_end = weeks[cut - 1]
    act = g[g["Week"] <= train_end].groupby(KEY)["demand"].apply(lambda s: int((s > 0).sum())).rename("act_train")
    gg = g.merge(act, on=KEY, how="left")
    gg = gg[gg["act_train"] >= cfg["min_active_weeks"]]
    # exponential-smoothing alpha tuned on TRAIN rows only
    tr_rows = gg[(gg["Week"] <= train_end)].dropna(subset=["target"])
    alphas = sorted(float(c[3:]) for c in gg.columns if c.startswith("ew_"))
    best_a = min(alphas, key=lambda a: (tr_rows["target"] - tr_rows[f"ew_{a}"]).abs().mean())
    gg["ewma"] = gg[f"ew_{best_a}"]
    need = ["lag_1", "lag_3", "roll_std_4", "ewma"] + features
    lab = gg.dropna(subset=need + ["target"]).copy()
    train, test = lab[lab["Week"] <= train_end], lab[lab["Week"] > train_end].copy()
    log(f"[model] weeks {len(weeks)} | train up to {pd.Timestamp(train_end).date()} | test "
        f"{pd.Timestamp(test['Week'].min()).date()} -> {pd.Timestamp(test['Week'].max()).date()} | "
        f"series {lab.groupby(KEY).ngroups} | train rows {len(train):,} | test rows {len(test):,} | "
        f"smoothing alpha {best_a}", verbose)
    fitted = {}
    for n, m in make_models(cfg["seed"]).items():
        m.fit(train[features], train["y_ratio"])
        fitted[n] = m
    names = list(BASELINES) + ML_NAMES + [ENS_ML, ENS_ALL]
    preds = {n: predict_candidate(n, test, fitted, features) for n in names}
    results = pd.DataFrame({n: metrics(test["target"], p) for n, p in preds.items()}).T
    best = results["MAE"].idxmin()
    best_ml = results.loc[ML_NAMES, "MAE"].idxmin()
    test["pred"] = preds[best]
    test["abs_err"] = (test["target"] - test["pred"]).abs()
    test["err"] = test["target"] - test["pred"]
    return dict(features=features, train_end=train_end, alpha=best_a, lab=lab, train=train, test=test,
                fitted_train=fitted, results=results, best=best, best_ml=best_ml, preds=preds)


def error_tables(ev, names, cfg):
    test = ev["test"].copy()
    avg = test.groupby(KEY)["roll_mean_8"].mean()
    q1, q2 = avg.quantile(cfg["demand_pct"])
    test["Class"] = np.where(test["roll_mean_8"] >= q2, "High", np.where(test["roll_mean_8"] >= q1, "Medium", "Low"))

    def by(col):
        return (test.groupby(col).apply(lambda x: pd.Series({
            "rows": len(x), "MAE": x["abs_err"].mean(),
            "WAPE_%": 100 * x["abs_err"].sum() / max(x["target"].sum(), 1e-9)})).round(2))
    worst = test.groupby("StockCode")["abs_err"].mean().sort_values(ascending=False).head(10)
    worst = pd.DataFrame({"Description": worst.index.map(names), "MAE": worst.round(1).values}, index=worst.index)
    rf = ev["fitted_train"]["Random Forest"]
    imp = pd.Series(rf.feature_importances_, index=ev["features"]).sort_values(ascending=False).round(4)
    return {"by_class": by("Class"), "by_region": by("Region"), "by_month": by("month"),
            "worst": worst, "importance": imp.to_frame("importance")}


# =============================================================================
# 4. SQLITE STORE (user-added products, overrides, settings)
# =============================================================================
def db_connect(cfg):
    conn = sqlite3.connect(cfg["db_path"])
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS products(
        stock_code TEXT PRIMARY KEY, name TEXT, unit_price REAL, product_type TEXT DEFAULT 'Non-perishable',
        shelf_life_days INTEGER, expiry_entered_on TEXT, source TEXT DEFAULT 'user', est_weekly_demand REAL,
        date_added TEXT);
    CREATE TABLE IF NOT EXISTS inventory(
        stock_code TEXT, region TEXT, current_stock REAL, lead_time_days INTEGER,
        PRIMARY KEY(stock_code, region));
    CREATE TABLE IF NOT EXISTS weekly_sales(
        stock_code TEXT, region TEXT, week_idx INTEGER, quantity REAL,
        PRIMARY KEY(stock_code, region, week_idx));
    CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT);
    """)
    return conn


def db_add_product(cfg, code, name, price, stock, regions, lead_time, perishable, shelf_days,
                   weekly_sales=None, est_demand=None):
    conn = db_connect(cfg)
    today = date.today().isoformat()
    ptype = "Perishable" if perishable else "Non-perishable"
    conn.execute("""INSERT INTO products(stock_code,name,unit_price,product_type,shelf_life_days,expiry_entered_on,
                    source,est_weekly_demand,date_added) VALUES(?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(stock_code) DO UPDATE SET name=excluded.name, unit_price=excluded.unit_price,
                    product_type=excluded.product_type, shelf_life_days=excluded.shelf_life_days,
                    expiry_entered_on=excluded.expiry_entered_on, est_weekly_demand=excluded.est_weekly_demand,
                    source='user'""",
                 (code, name, price, ptype, shelf_days if perishable else None, today if perishable else None,
                  "user", est_demand, today))
    for r in regions:
        conn.execute("""INSERT INTO inventory VALUES(?,?,?,?) ON CONFLICT(stock_code,region)
                        DO UPDATE SET current_stock=excluded.current_stock, lead_time_days=excluded.lead_time_days""",
                     (code, r, stock, lead_time))
        if weekly_sales:
            conn.execute("DELETE FROM weekly_sales WHERE stock_code=? AND region=?", (code, r))
            for i, q in enumerate(weekly_sales):
                conn.execute("INSERT INTO weekly_sales VALUES(?,?,?,?)", (code, r, i, float(q)))
    conn.commit()
    conn.close()


def db_set_type(cfg, code, perishable, shelf_days):
    conn = db_connect(cfg)
    today = date.today().isoformat()
    ptype = "Perishable" if perishable else "Non-perishable"
    conn.execute("""INSERT INTO products(stock_code,product_type,shelf_life_days,expiry_entered_on,source,date_added)
                    VALUES(?,?,?,?,?,?) ON CONFLICT(stock_code) DO UPDATE SET product_type=excluded.product_type,
                    shelf_life_days=excluded.shelf_life_days, expiry_entered_on=excluded.expiry_entered_on""",
                 (code, ptype, shelf_days if perishable else None, today if perishable else None, "dataset", today))
    conn.commit()
    conn.close()


def db_set_stock(cfg, code, region, stock, lead_time):
    conn = db_connect(cfg)
    conn.execute("""INSERT INTO inventory VALUES(?,?,?,?) ON CONFLICT(stock_code,region)
                    DO UPDATE SET current_stock=excluded.current_stock, lead_time_days=excluded.lead_time_days""",
                 (code, region, stock, lead_time))
    conn.commit()
    conn.close()


def db_read(cfg):
    conn = db_connect(cfg)
    prod = pd.read_sql("SELECT * FROM products", conn)
    inv = pd.read_sql("SELECT * FROM inventory", conn)
    ws = pd.read_sql("SELECT * FROM weekly_sales", conn)
    conn.close()
    return prod, inv, ws


# =============================================================================
# 5. INVENTORY + EXPIRY + DISCOUNT LOGIC  (one function used for dataset AND user products)
# =============================================================================
def zscore(service_level):
    return NormalDist().inv_cdf(service_level)


def _u(code, region):
    """Deterministic pseudo-random number in [0,1) per product-region (stable across runs)."""
    return int(hashlib.md5(f"{region}|{code}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def suggest_perishable(desc):
    """Heuristic SUGGESTION only (user confirms). The UCI dataset is mostly giftware, so few/no matches."""
    toks = set(str(desc).upper().replace("-", " ").split())
    return bool(toks & FOOD_TOKENS) and not (toks & NONFOOD_TOKENS)


def evaluate_item(rec, cfg):
    """
    rec keys: forecast, err_std, stock, lead_days, price, demand_class, falling,
              product_type, shelf_life_days, days_left (None if non-perishable)
    Returns all inventory / expiry / discount outputs for one product in one region.
    """
    z = zscore(cfg["service_level"])
    f = max(float(rec["forecast"]), 0.0)
    stock = float(rec["stock"])
    lead_w = rec["lead_days"] / 7.0
    ltd = f * lead_w                                                  # lead-time demand
    safety = z * rec["err_std"] * math.sqrt(lead_w)                   # buffer stock
    safety = min(safety, cfg["max_safety_weeks"] * max(f, float(rec.get("avg8") or 0), 1.0))   # practical cap
    rop = ltd + safety                                                # reorder point
    daily = f / 7.0
    cover = stock / daily if daily > 0 else 999.0
    risk = "High" if stock < ltd else ("Medium" if stock < rop else "Low")
    ratio = stock / rop if rop > 0 else 2.0
    score = int(round(min(max(100 * (1 - ratio / 1.5), 0), 100)))
    order = max(0, math.ceil(rop + f * cfg["review_weeks"] - stock)) if stock < rop else 0   # order only below reorder point

    # ---- expiry -------------------------------------------------------------------------------
    perish = rec["product_type"] == "Perishable"
    out = dict(forecast_next_week=round(f, 1), lead_time_demand=round(ltd, 1), safety_stock=round(safety, 1),
               reorder_point=round(rop, 1), days_of_cover=round(min(cover, 999.0), 1), risk_level=risk,
               risk_score=score, shelf_life_days=rec.get("shelf_life_days"), days_to_expiry=None,
               expiry_status="N/A (no expiry)", expected_sold_before_expiry=None, wastage_units=0,
               wastage_value=0.0)
    exp_sellable = None
    if perish and rec.get("days_left") is not None:
        dl = int(rec["days_left"])
        exp_sellable = daily * max(dl, 0)
        wastage = max(0.0, stock - exp_sellable)
        if dl <= 0:
            status, order = "Expired", 0
        else:
            cap = max(0, math.floor(exp_sellable - stock))            # never order more than can be sold
            order = min(order, cap)
            status = ("Expiry Risk" if stock > exp_sellable else
                      "Expiring Soon" if dl <= cfg["expiring_soon_days"] else "Safe")
        out.update(days_to_expiry=dl, expiry_status=status, expected_sold_before_expiry=round(exp_sellable, 1),
                   wastage_units=int(round(wastage)), wastage_value=round(wastage * rec["price"], 2))
    out["recommended_order_qty"] = int(order)
    out["order_value_gbp"] = round(order * rec["price"], 2)
    out["order_cost_est_gbp"] = round(order * rec["price"] * cfg["cost_ratio"], 2)
    out["reorder_status"] = {"High": "Urgent", "Medium": "Soon", "Low": "OK"}[risk]

    # ---- discount (High demand: no change; Low demand: cover-based; expiry adds on top) ----------
    cls, price = rec["demand_class"], rec["price"]
    disc, reason = 0.0, "No change"
    if out["expiry_status"] == "Expired":
        reason = "Expired - remove from sale"
    elif cls == "High":
        reason = "High demand - no discount needed"
    else:
        if cls == "Low" and stock > 0:
            tier = next(p for lim, p in cfg["cover_tiers"] if cover < lim)
            disc += tier
            if tier:
                reason = f"Low demand, {min(cover, 999):.0f} days of cover"
            if tier and rec.get("falling"):
                disc += cfg["falling_extra_pct"]
                reason += " + falling sales"
        if out["expiry_status"] == "Expiry Risk" and stock > 0:
            extra = min(cfg["expiry_extra_max_pct"], cfg["expiry_extra_max_pct"] * out["wastage_units"] / stock)
            disc += extra
            reason = (reason + " + " if disc > extra else "") + "Expiry risk"
        if disc == 0 and cls == "Medium":
            reason = "Medium demand - no discount (monitor)"
    margin_cap = max(0.0, 1 - cfg["cost_ratio"] * (1 + cfg["min_margin"])) * 100
    disc = round(min(disc, cfg["max_discount"] * 100, margin_cap), 1)
    new_price = price * (1 - disc / 100)
    new_dem = f * (1 - disc / 100) ** (-cfg["elasticity"]) if disc > 0 else f
    cost = price * cfg["cost_ratio"]
    out.update(
        discount_pct=disc, discount_reason=reason, discounted_price=round(new_price, 2),
        new_weekly_demand_est=round(new_dem, 1),
        days_to_clear_before=round(min(cover, 999.0), 1),
        days_to_clear_after=round(min(stock / (new_dem / 7.0), 999.0), 1) if new_dem > 0 else 999.0,
        weekly_revenue_change=round(new_dem * new_price - f * price, 2),
        weekly_margin_change=round(new_dem * (new_price - cost) - f * (price - cost), 2))
    # one-line action
    if out["expiry_status"] == "Expired":
        act = "Remove expired stock"
    elif risk == "High":
        act = f"URGENT: order {order} units"
    elif risk == "Medium":
        act = f"Reorder soon ({order} units)"
    elif disc > 0:
        act = f"Apply {disc:.0f}% discount"
    else:
        act = "Healthy - no action"
    out["action"] = act
    return out


def days_left_from(row_type, shelf_days, entered_on, today):
    if row_type != "Perishable" or shelf_days is None or pd.isna(shelf_days):
        return None
    if entered_on is None or (isinstance(entered_on, float) and pd.isna(entered_on)):
        return int(shelf_days)
    return int(shelf_days) - (today - date.fromisoformat(str(entered_on))).days


# =============================================================================
# 6. MAIN PIPELINE
# =============================================================================
def run_pipeline(cfg=CFG, demo_expiry=False, verbose=True, save=True):
    os.makedirs(cfg["out_dir"], exist_ok=True)
    os.makedirs(cfg["model_dir"], exist_ok=True)
    raw, src = load_active_raw(cfg)
    log(f"[data] source: {src} | raw rows {len(raw):,}", verbose)
    df = clean(raw, cfg, verbose)
    df, regions = map_regions(df, cfg)
    log(f"[regions] {regions}", verbose)
    g, names = build_weekly(df, cfg)
    g = add_features(g)
    g = add_relative(g, regions)
    price_avg = g.groupby(KEY)["price"].mean()
    last_week = g["Week"].max()

    ev = evaluate_models(g, regions, cfg, verbose)
    et = error_tables(ev, names, cfg)
    features, best = ev["features"], ev["best"]

    # ---------- final forecast: refit models on ALL labelled rows, predict next week ----------
    act_all = g.groupby(KEY)["demand"].apply(lambda s: int((s > 0).sum()))
    universe = act_all[act_all >= cfg["min_active_weeks"]].index
    g["ewma"] = g[f"ew_{ev['alpha']}"]
    final_models = make_models(cfg["seed"])
    for n, m in final_models.items():
        m.fit(ev["lab"][features], ev["lab"]["y_ratio"])
    latest = g[g["Week"] == last_week].set_index(KEY).loc[lambda x: x.index.isin(universe)].reset_index()
    latest = latest.dropna(subset=["lag_1", "lag_3", "roll_std_4"] + features)
    latest["forecast"] = np.clip(predict_candidate(best, latest, final_models, features), 0, None)

    # ---------- forecast-error spread per series (feeds buffer stock) ----------
    res = ev["test"].groupby(KEY)["err"].agg(["std", "count"]).reset_index()
    res.loc[res["count"] < 3, "std"] = np.nan
    latest = latest.merge(res[KEY + ["std"]].rename(columns={"std": "err_series"}), on=KEY, how="left")
    cv_pool = float(np.nanmedian(latest["err_series"] / latest["forecast"].clip(lower=1)))
    latest["err_std"] = np.where(latest["err_series"].isna(), cv_pool * latest["forecast"],
                                 0.5 * latest["err_series"] + 0.5 * cv_pool * latest["forecast"])
    q1, q2 = latest["forecast"].quantile(cfg["demand_pct"])
    thresholds = (float(q1), float(q2))
    latest["demand_class"] = np.where(latest["forecast"] >= q2, "High",
                                      np.where(latest["forecast"] >= q1, "Medium", "Low"))

    # ---------- overrides from SQLite ----------
    prod_db, inv_db, ws_db = db_read(cfg)
    inv_map = {(r.stock_code, r.region): (r.current_stock, r.lead_time_days) for r in inv_db.itertuples()}
    type_map = {r.stock_code: r for r in prod_db.itertuples()}
    today = date.today()
    lo, hi = cfg["stock_mult_range"]
    rows = []
    for r in latest.itertuples(index=False):
        code, region = r.StockCode, r.Region
        desc = names.get(code, "UNKNOWN")
        price = float(price_avg.loc[(region, code)])
        stock_src, lead = "simulated", cfg["region_lead_days"].get(region, cfg["default_lead_days"])
        stock = max(1, round((lo + _u(code, region) * (hi - lo)) * max(r.roll_mean_8, r.forecast, 1)))
        if (code, region) in inv_map:
            stock, lead = inv_map[(code, region)]
            stock_src = "user-provided"
        ptype, shelf, entered, exp_src = "Non-perishable", None, None, "default"
        if code in type_map and type_map[code].product_type == "Perishable":
            t = type_map[code]
            ptype, shelf, entered, exp_src = "Perishable", t.shelf_life_days, t.expiry_entered_on, "user-provided"
        elif demo_expiry and _u(code, "perishable") < cfg["demo_perishable_share"]:
            ptype, shelf, entered, exp_src = "Perishable", int(20 + _u(code, "exp") * 100), None, "simulated (demo)"
        rec = dict(forecast=r.forecast, err_std=r.err_std, stock=stock, lead_days=lead, price=price,
                   avg8=r.roll_mean_8, demand_class=r.demand_class, falling=(r.roll_mean_8 > 0 and r.roll_mean_4 < cfg["falling_threshold"] * r.roll_mean_8),
                   product_type=ptype, shelf_life_days=shelf,
                   days_left=days_left_from(ptype, shelf, entered, today))
        out = evaluate_item(rec, cfg)
        rows.append({"region": region, "stock_code": code, "description": desc, "source": "dataset",
                     "forecast_confidence": "Model", "unit_price": round(price, 2), "demand_class": r.demand_class,
                     "avg_weekly_demand_8w": round(r.roll_mean_8, 1), "current_stock": int(stock),
                     "stock_source": stock_src, "lead_time_days": int(lead), "product_type": ptype,
                     "expiry_source": exp_src, "suggest_perishable": suggest_perishable(desc), **out})

    # ---------- user-added products (cold start / short history) ----------
    for p in prod_db[prod_db["source"] == "user"].itertuples():
        for inv in inv_db[inv_db["stock_code"] == p.stock_code].itertuples():
            hist = ws_db[(ws_db["stock_code"] == p.stock_code) & (ws_db["region"] == inv.region)] \
                .sort_values("week_idx")["quantity"].tolist()
            fc, conf, err = forecast_new_product(hist, p.est_weekly_demand, p.unit_price, inv.region, rows, cfg,
                                                 ev["alpha"], cv_pool)
            cls = "High" if fc >= q2 else ("Medium" if fc >= q1 else "Low")
            falling = len(hist) >= 8 and np.mean(hist[-4:]) < cfg["falling_threshold"] * np.mean(hist[-8:])
            rec = dict(forecast=fc, err_std=err, stock=inv.current_stock, lead_days=inv.lead_time_days,
                       avg8=float(np.mean(hist[-8:])) if hist else fc, price=p.unit_price, demand_class=cls, falling=falling, product_type=p.product_type,
                       shelf_life_days=p.shelf_life_days,
                       days_left=days_left_from(p.product_type, p.shelf_life_days, p.expiry_entered_on, today))
            out = evaluate_item(rec, cfg)
            rows.append({"region": inv.region, "stock_code": p.stock_code, "description": p.name, "source": "user-added",
                         "forecast_confidence": conf, "unit_price": p.unit_price, "demand_class": cls,
                         "avg_weekly_demand_8w": round(float(np.mean(hist[-8:])), 1) if hist else None,
                         "current_stock": int(inv.current_stock), "stock_source": "user-provided",
                         "lead_time_days": int(inv.lead_time_days), "product_type": p.product_type,
                         "expiry_source": "user-provided" if p.product_type == "Perishable" else "default",
                         "suggest_perishable": suggest_perishable(p.name), **out})

    master = pd.DataFrame(rows)
    ctx = dict(cfg=cfg, regions=regions, master=master, results=ev["results"], best=best, best_ml=ev["best_ml"],
               alpha=ev["alpha"], err_tables=et, thresholds=thresholds, cv_pool=cv_pool, last_week=last_week,
               train_end=ev["train_end"], hist=g[KEY + ["Week", "demand"]], names=names,
               n_test=len(ev["test"]), n_series=len(universe), final_models=final_models, features=features)
    if save:
        save_outputs(ctx, verbose)
    return ctx


def forecast_new_product(hist, est_demand, price, region, rows, cfg, alpha, cv_pool):
    """Cold-start logic for products without enough history."""
    if len(hist) >= 4:
        ew = pd.Series(hist).ewm(alpha=alpha, adjust=False).mean().iloc[-1]
        fc = (ew + float(np.mean(hist[-4:]))) / 2
        conf = "Medium (own history)" if len(hist) >= 8 else "Low (short history)"
        err = cv_pool * fc * (1.0 if len(hist) >= 8 else 1.5)
    elif est_demand is not None and not pd.isna(est_demand):
        fc, conf, err = float(est_demand), "Low (user estimate)", cv_pool * float(est_demand) * 2
    else:                                                              # similar-priced products in same region
        sim = [r["forecast_next_week"] for r in rows if r["region"] == region and r["source"] == "dataset"
               and 0.7 * price <= r["unit_price"] <= 1.3 * price]
        if not sim:
            sim = [r["forecast_next_week"] for r in rows if r["region"] == region and r["source"] == "dataset"]
        fc = float(np.median(sim)) if sim else 10.0
        conf, err = "Low (cold start: similar-price median)", cv_pool * fc * 2
    return fc, conf, err


# =============================================================================
# 7. OUTPUTS / REPORTS
# =============================================================================
def reorder_overview(master):
    m = master
    need = m[m["recommended_order_qty"] > 0]
    ov = {
        "kpi": pd.Series({
            "products_total": len(m),
            "urgent (High risk)": int((m["risk_level"] == "High").sum()),
            "soon (Medium risk)": int((m["risk_level"] == "Medium").sum()),
            "ok (Low risk)": int((m["risk_level"] == "Low").sum()),
            "reorders_needed (qty>0)": len(need),
            "total_units_to_order": int(need["recommended_order_qty"].sum()),
            "order_value_gbp (retail price)": round(need["order_value_gbp"].sum(), 2),
            "order_cost_est_gbp": round(need["order_cost_est_gbp"].sum(), 2)}),
        "by_risk": m.groupby("risk_level").agg(items=("stock_code", "count"), units=("recommended_order_qty", "sum")),
        "by_demand_class": pd.crosstab(m["demand_class"], m["risk_level"]),
        "by_region": m.groupby("region").agg(items=("stock_code", "count"),
                                              urgent=("risk_level", lambda s: (s == "High").sum()),
                                              reorders=("recommended_order_qty", lambda s: (s > 0).sum()),
                                              units=("recommended_order_qty", "sum"),
                                              value_gbp=("order_value_gbp", "sum")).round(2),
    }
    return ov


def rollup_all_regions(master):
    order = {"High": 2, "Medium": 1, "Low": 0}
    t = master.groupby(["stock_code", "description"]).agg(
        regions=("region", "count"), forecast_next_week=("forecast_next_week", "sum"),
        current_stock=("current_stock", "sum"), recommended_order_qty=("recommended_order_qty", "sum"),
        unit_price=("unit_price", "first"),
        worst_risk=("risk_level", lambda s: max(s, key=lambda x: order[x])),
        demand_class=("demand_class", lambda s: max(s, key=lambda x: order[x]))).reset_index()
    return t


def save_outputs(ctx, verbose=True):
    cfg, od = ctx["cfg"], ctx["cfg"]["out_dir"]
    ctx["results"].round(3).to_csv(f"{od}/model_comparison.csv")
    for k, v in ctx["err_tables"].items():
        v.to_csv(f"{od}/error_{k}.csv")
    ctx["master"].to_csv(f"{od}/master_table.csv", index=False)
    rollup_all_regions(ctx["master"]).to_csv(f"{od}/master_by_product_all_regions.csv", index=False)
    ctx["master"][ctx["master"]["recommended_order_qty"] > 0].sort_values(
        ["risk_score", "recommended_order_qty"], ascending=False).to_csv(f"{od}/reorders.csv", index=False)
    ctx["master"][ctx["master"]["discount_pct"] > 0].sort_values("discount_pct", ascending=False) \
        .to_csv(f"{od}/discounts.csv", index=False)
    for k, v in reorder_overview(ctx["master"]).items():
        v.to_csv(f"{od}/reorder_overview_{k}.csv")
    best = ctx["results"].loc[ctx["best"]]
    summary = dict(generated=datetime.now().isoformat(timespec="seconds"), best_forecaster=ctx["best"],
                   best_ml_model=ctx["best_ml"], metrics_best={k: round(float(v), 3) for k, v in best.items()},
                   n_series=ctx["n_series"], regions=ctx["regions"], demand_class_thresholds=ctx["thresholds"],
                   last_complete_week=str(pd.Timestamp(ctx["last_week"]).date()), settings=_json_safe(cfg))
    with open(f"{od}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    joblib.dump({"models": ctx["final_models"], "features": ctx["features"], "best": ctx["best"],
                 "alpha": ctx["alpha"], "thresholds": ctx["thresholds"], "regions": ctx["regions"]},
                f"{cfg['model_dir']}/stocksense_models.joblib")
    log(f"[saved] CSV/JSON outputs in ./{od}/ and models in ./{cfg['model_dir']}/", verbose)


def _json_safe(d):
    return json.loads(json.dumps(d, default=str))


SHOW_COLS = ["region", "stock_code", "description", "demand_class", "forecast_next_week", "current_stock",
             "safety_stock", "reorder_point", "days_of_cover", "risk_level", "risk_score",
             "recommended_order_qty", "expiry_status", "discount_pct", "action"]


def filter_sort(master, region=None, demand=None, risk=None, ptype=None, search=None, sort=None, desc=True, top=None):
    m = master.copy()
    if region and region != "All":
        m = m[m["region"] == region]
    if demand:
        m = m[m["demand_class"] == demand.capitalize()]
    if risk:
        m = m[m["risk_level"] == risk.capitalize()]
    if ptype:
        m = m[m["product_type"].str.lower().str.startswith(ptype.lower())]
    if search:
        m = m[m["description"].str.contains(search, case=False, na=False) | m["stock_code"].str.contains(search, case=False)]
    if sort == "demand":
        m = m.assign(_k=m["demand_class"].map({"High": 2, "Medium": 1, "Low": 0})).sort_values(
            ["_k", "forecast_next_week"], ascending=not desc).drop(columns="_k")
    elif sort == "risk":
        m = m.sort_values("risk_score", ascending=not desc)
    elif sort:
        m = m.sort_values(sort, ascending=not desc)
    return m.head(top) if top else m


def print_report(ctx):
    m, res = ctx["master"], ctx["results"]
    print("\n" + "=" * 100 + "\n MODEL COMPARISON  (chronological test set, next-week demand per product-region)\n" + "=" * 100)
    print(res.round(3).to_string())
    b = res.loc[ctx["best"]]
    print(f"\n>> Forecaster used for stock decisions: {ctx['best']}  "
          f"(MAE {b['MAE']:.1f} | RMSE {b['RMSE']:.1f} | R2 {b['R2']:.3f} | WAPE {b['WAPE_%']:.1f}%)")
    print(f">> Best pure-ML model: {ctx['best_ml']} (MAE {res.loc[ctx['best_ml'], 'MAE']:.1f}, "
          f"R2 {res.loc[ctx['best_ml'], 'R2']:.3f})")
    for k in ["by_class", "by_region", "by_month", "worst"]:
        print(f"\n--- forecast error: {k} ---\n{ctx['err_tables'][k].to_string()}")
    print("\n--- feature importance (Random Forest) ---\n" + ctx["err_tables"]["importance"].head(8).to_string())

    print("\n" + "=" * 100 + f"\n INVENTORY OUTPUT  (forecast for week after {pd.Timestamp(ctx['last_week']).date()}; "
          "stock/expiry SIMULATED unless marked user-provided)\n" + "=" * 100)
    print(f"demand-class thresholds (units/week): Low < {ctx['thresholds'][0]:.1f} <= Medium < {ctx['thresholds'][1]:.1f} <= High")
    ov = reorder_overview(m)
    print("\n--- REORDER OVERVIEW ---\n" + ov["kpi"].to_string())
    print("\nby demand class x risk level:\n" + ov["by_demand_class"].to_string())
    print("\nby region:\n" + ov["by_region"].to_string())
    print("\n--- TOP 10 URGENT REORDERS ---")
    print(filter_sort(m, risk="High", sort="risk_score", top=10)[SHOW_COLS].to_string(index=False))
    d = m[m["discount_pct"] > 0]
    print(f"\n--- DISCOUNT SUGGESTIONS: {len(d)} items (High-demand items: none) ---")
    print(d.sort_values("discount_pct", ascending=False).head(8)[
        ["region", "stock_code", "description", "demand_class", "days_of_cover", "unit_price", "discount_pct",
         "discounted_price", "days_to_clear_before", "days_to_clear_after", "weekly_margin_change",
         "discount_reason"]].to_string(index=False))
    ex = m[m["product_type"] == "Perishable"]
    if len(ex):
        print(f"\n--- PERISHABLE ITEMS: {len(ex)} | expiry status counts: {ex['expiry_status'].value_counts().to_dict()} ---")
        print(ex.sort_values("wastage_value", ascending=False).head(6)[
            ["region", "stock_code", "description", "current_stock", "days_to_expiry", "expected_sold_before_expiry",
             "expiry_status", "wastage_units", "wastage_value", "recommended_order_qty", "expiry_source"]].to_string(index=False))
    print("\nNOTE: stock levels / lead times / expiry are simulated or user-provided (dataset has none).")


def print_product(ctx, code, region=None):
    m = ctx["master"]
    p = m[(m["stock_code"] == code) & ((m["region"] == region) if region else True)]
    if p.empty:
        print(f"Product {code} not found in the forecast universe (needs >= {ctx['cfg']['min_active_weeks']} active weeks, "
              f"or add it with `add-product`).")
        return
    for _, r in p.iterrows():
        print("\n" + "-" * 90 + f"\n{r['stock_code']}  {r['description']}  |  region: {r['region']}  |  source: {r['source']}\n" + "-" * 90)
        for k in ["demand_class", "forecast_next_week", "forecast_confidence", "unit_price", "current_stock",
                  "stock_source", "lead_time_days", "lead_time_demand", "safety_stock", "reorder_point",
                  "days_of_cover", "risk_level", "risk_score", "recommended_order_qty", "order_value_gbp",
                  "product_type", "days_to_expiry", "expiry_status", "expected_sold_before_expiry",
                  "wastage_units", "wastage_value", "discount_pct", "discount_reason", "discounted_price",
                  "new_weekly_demand_est", "days_to_clear_before", "days_to_clear_after",
                  "weekly_revenue_change", "weekly_margin_change", "action"]:
            print(f"  {k:30s} {r[k]}")
        h = ctx.get("hist")
        h = h[(h["StockCode"] == code) & (h["Region"] == r["region"])].tail(12) if h is not None else []
        if len(h):
            print("  last 12 weeks demand:", [int(x) for x in h["demand"]])


# =============================================================================
# 8. ADD DATASET (merge / replace)
# =============================================================================
def add_dataset(path, mode, cfg=CFG):
    if mode not in ("merge", "replace"):
        raise ValueError("mode must be merge or replace")
    new = read_any(path)
    validate_columns(new)
    new["InvoiceDate"] = pd.to_datetime(new["InvoiceDate"], errors="coerce")
    try:
        before_raw, _ = load_active_raw(cfg)
        before_codes, before_rows = set(before_raw["StockCode"].astype(str)), len(before_raw)
        old = json.load(open(f"{cfg['out_dir']}/summary.json")) if os.path.exists(f"{cfg['out_dir']}/summary.json") else None
    except FileNotFoundError:
        before_raw, before_codes, before_rows, old = None, set(), 0, None
    skipped = 0
    if mode == "merge" and before_raw is not None:
        keys = [c for c in ["InvoiceNo", "StockCode", "InvoiceDate", "Quantity", "UnitPrice", "CustomerID"]
                if c in before_raw.columns and c in new.columns]

        def keyed(d):                                   # comparable string keys (dtype-safe)
            k = d[keys].copy()
            for c in keys:
                k[c] = pd.to_numeric(k[c], errors="coerce").astype(float).astype(str) if c in ("Quantity", "UnitPrice") \
                    else k[c].astype(str)
            return pd.MultiIndex.from_frame(k)
        dup = keyed(new).isin(set(keyed(before_raw)))
        skipped = int(dup.sum())
        merged = pd.concat([before_raw, new[~dup]], ignore_index=True)     # only NEW rows are appended
    else:
        merged = new
    merged.to_pickle(cfg["merged_raw_path"])
    print(f"[dataset] mode={mode}: {before_rows:,} rows -> {len(merged):,} rows "
          f"({len(new):,} uploaded, {skipped:,} duplicates of existing rows skipped) | "
          f"new product codes: {len(set(new['StockCode'].astype(str)) - before_codes):,}")
    ctx = run_pipeline(cfg, verbose=False)
    nm = ctx["results"].loc[ctx["best"]]
    print(f"[dataset] retrained. Forecaster: {ctx['best']} | MAE {nm['MAE']:.2f} | R2 {nm['R2']:.3f} | WAPE {nm['WAPE_%']:.1f}% "
          + (f"(before: MAE {old['metrics_best']['MAE']}, R2 {old['metrics_best']['R2']})" if old else ""))
    print(f"[dataset] {len(ctx['master'])} product-region rows refreshed -> outputs/")
    return ctx


# =============================================================================
# 9. CLI
# =============================================================================
def load_outputs_or_run(cfg, demo=False, refresh=False):
    """Read-only commands reuse the saved master table (instant); otherwise run the pipeline."""
    path = f"{cfg['out_dir']}/master_table.csv"
    if os.path.exists(path) and not demo and not refresh:
        return dict(cfg=cfg, master=pd.read_csv(path, dtype={"stock_code": str}), hist=None)
    return run_pipeline(cfg, demo_expiry=demo, verbose=False)


def main():
    ap = argparse.ArgumentParser(description="StockSense - demand estimation & stockout risk")
    ap.add_argument("--data", help="path to Online_Retail.xlsx / csv")
    ap.add_argument("--refresh", action="store_true", help="recompute instead of reading saved outputs")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("run", help="full pipeline + report")
    s.add_argument("--demo-expiry", action="store_true", help="demo: random 8%% of products treated as perishable (simulated)")
    s.add_argument("--service-level", type=float); s.add_argument("--lead-time", type=int)
    s.add_argument("--review-weeks", type=float); s.add_argument("--elasticity", type=float)

    s = sub.add_parser("show", help="filter / sort results")
    s.add_argument("--region"); s.add_argument("--demand", choices=["High", "Medium", "Low", "high", "medium", "low"])
    s.add_argument("--risk", choices=["High", "Medium", "Low", "high", "medium", "low"])
    s.add_argument("--type", choices=["perishable", "non-perishable"]); s.add_argument("--search")
    s.add_argument("--sort", default="demand"); s.add_argument("--asc", action="store_true")
    s.add_argument("--top", type=int, default=25); s.add_argument("--demo-expiry", action="store_true")

    sub.add_parser("reorders", help="reorder overview")
    sub.add_parser("discounts", help="discount suggestions")
    s = sub.add_parser("product", help="full analysis of one product"); s.add_argument("code"); s.add_argument("--region")

    s = sub.add_parser("add-product", help="add a new product")
    s.add_argument("--code", required=True); s.add_argument("--name", required=True)
    s.add_argument("--price", type=float, required=True); s.add_argument("--stock", type=float, required=True)
    s.add_argument("--region", action="append", required=True, help="repeat for several regions")
    s.add_argument("--lead-time", type=int, default=7)
    s.add_argument("--perishable", action="store_true"); s.add_argument("--shelf-days", type=int)
    s.add_argument("--weekly-sales", help="comma-separated past weekly sales, oldest first")
    s.add_argument("--est-demand", type=float, help="your weekly demand estimate if no history")

    s = sub.add_parser("set-type", help="mark a product perishable / non-perishable")
    s.add_argument("code"); g_ = s.add_mutually_exclusive_group(required=True)
    g_.add_argument("--perishable", action="store_true"); g_.add_argument("--non-perishable", action="store_true")
    s.add_argument("--shelf-days", type=int, help="expires in N days (from today)")

    s = sub.add_parser("set-stock", help="override current stock / lead time")
    s.add_argument("code"); s.add_argument("--region", required=True); s.add_argument("--stock", type=float, required=True)
    s.add_argument("--lead-time", type=int, default=7)

    s = sub.add_parser("add-dataset", help="merge or replace the dataset and retrain")
    s.add_argument("path"); s.add_argument("--mode", choices=["merge", "replace"], default="merge")
    s = sub.add_parser("reset-data", help="remove merged dataset and go back to the original file")

    a = ap.parse_args()
    cfg = dict(CFG)
    if a.data:
        cfg["data_path"] = a.data
    if a.cmd == "run":
        for k, v in [("service_level", a.service_level), ("default_lead_days", a.lead_time),
                     ("review_weeks", a.review_weeks), ("elasticity", a.elasticity)]:
            if v is not None:
                cfg[k] = v
        ctx = run_pipeline(cfg, demo_expiry=a.demo_expiry)
        print_report(ctx)
    elif a.cmd == "show":
        ctx = load_outputs_or_run(cfg, a.demo_expiry, a.refresh)
        df = filter_sort(ctx["master"], a.region, a.demand, a.risk, a.type, a.search, a.sort, not a.asc, a.top)
        print(f"{len(df)} rows shown (sorted by {a.sort}, {'ascending' if a.asc else 'descending'})")
        print(df[SHOW_COLS].to_string(index=False))
    elif a.cmd == "reorders":
        ctx = load_outputs_or_run(cfg, refresh=a.refresh)
        for k, v in reorder_overview(ctx["master"]).items():
            print(f"\n--- {k} ---\n{v.to_string()}")
        print("\n--- reorder list (most urgent first, top 20) ---")
        print(ctx["master"][ctx["master"]["recommended_order_qty"] > 0].sort_values(
            ["risk_score", "recommended_order_qty"], ascending=False).head(20)[SHOW_COLS].to_string(index=False))
    elif a.cmd == "discounts":
        ctx = load_outputs_or_run(cfg, refresh=a.refresh)
        d = ctx["master"][ctx["master"]["discount_pct"] > 0].sort_values("discount_pct", ascending=False)
        print(f"{len(d)} discount suggestions (High demand items are never discounted)")
        print(d[["region", "stock_code", "description", "demand_class", "days_of_cover", "unit_price", "discount_pct",
                 "discounted_price", "days_to_clear_before", "days_to_clear_after", "weekly_revenue_change",
                 "weekly_margin_change", "discount_reason"]].head(30).to_string(index=False))
        print(f"\nAssumptions: cost = {cfg['cost_ratio']:.0%} of price, min margin {cfg['min_margin']:.0%}, "
              f"price elasticity {cfg['elasticity']} (not learned from data).")
    elif a.cmd == "product":
        print_product(load_outputs_or_run(cfg, refresh=a.refresh), a.code, a.region)
    elif a.cmd == "add-product":
        if a.perishable and not a.shelf_days:
            sys.exit("--perishable needs --shelf-days (expires in N days)")
        ws = [float(x) for x in a.weekly_sales.split(",")] if a.weekly_sales else None
        db_add_product(cfg, a.code, a.name, a.price, a.stock, a.region, a.lead_time, a.perishable, a.shelf_days,
                       ws, a.est_demand)
        print(f"Added product {a.code} to {cfg['db_path']}. Computing its output ...")
        ctx = load_outputs_or_run(cfg, refresh=True)
        print_product(ctx, a.code)
    elif a.cmd == "set-type":
        if a.perishable and not a.shelf_days:
            sys.exit("--perishable needs --shelf-days")
        db_set_type(cfg, a.code, a.perishable, a.shelf_days)
        print_product(load_outputs_or_run(cfg, refresh=True), a.code)
    elif a.cmd == "set-stock":
        db_set_stock(cfg, a.code, a.region, a.stock, a.lead_time)
        print_product(load_outputs_or_run(cfg, refresh=True), a.code, a.region)
    elif a.cmd == "add-dataset":
        try:
            add_dataset(a.path, a.mode, cfg)
        except (ValueError, FileNotFoundError) as e:
            sys.exit(f"ERROR: {e}")
    elif a.cmd == "reset-data":
        if os.path.exists(cfg["merged_raw_path"]):
            os.remove(cfg["merged_raw_path"])
        print("Merged dataset removed; the original file will be used.")


if __name__ == "__main__":
    main()
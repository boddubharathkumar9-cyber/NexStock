from pathlib import Path
from typing import Optional
import json
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import stocksense as S

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs"
MASTER_PATH = OUT_DIR / "master_table.csv"
COMP_PATH = OUT_DIR / "model_comparison.csv"
SUMMARY_PATH = OUT_DIR / "summary.json"

if not MASTER_PATH.exists():
    raise RuntimeError(
        "Missing 'outputs/master_table.csv'. Run `python stocksense.py run` first."
    )

# Load precomputed StockSense outputs (fast startup on Render without needing Online_Retail.xlsx)
master_df = pd.read_csv(MASTER_PATH, dtype={"stock_code": str})
comp_records = (
    pd.read_csv(COMP_PATH)
    .rename(columns={"Unnamed: 0": "model"})
    .to_dict(orient="records")
    if COMP_PATH.exists()
    else []
)
summary_meta = (
    json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    if SUMMARY_PATH.exists()
    else {}
)

# Setting docs_url="/" opens Swagger UI ("Try it out") directly on the root Render URL
app = FastAPI(
    title="StockSense — Retail Product Demand & Stockout Risk API",
    description=(
        "PS #10: Product-Region Next-Week Demand Estimation, Safety Stock, "
        "Expiry Wastage & Margin-Capped Discount API (powered by stocksense.py)."
    ),
    version="1.0.0",
    docs_url="/",
    redoc_url=None,
)


class EvaluateRequest(BaseModel):
    stock_code: str = Field(
        ..., example="85123A", description="Product StockCode (e.g., 85123A, 22423)"
    )
    region: str = Field(
        ...,
        example="United Kingdom",
        description="Region (United Kingdom, Netherlands, EIRE, Germany, France, Rest of World)",
    )
    current_stock: Optional[float] = Field(
        None,
        ge=0,
        example=120,
        description="Optional override for current stock on hand",
    )
    lead_time_days: Optional[int] = Field(
        None, ge=1, le=90, example=7, description="Optional supplier lead time in days"
    )
    service_level: float = Field(
        0.95,
        ge=0.50,
        le=0.999,
        example=0.95,
        description="Target service level (0.95 = 95%)",
    )
    review_weeks: float = Field(
        1.0, ge=0.25, le=12.0, example=1.0, description="Review period in weeks"
    )
    perishable: Optional[bool] = Field(
        None, example=False, description="Optional override: mark item as Perishable"
    )
    shelf_days: Optional[int] = Field(
        None,
        ge=1,
        example=30,
        description="Days until expiry (required if perishable=True)",
    )


@app.get("/health", tags=["Monitoring"])
def health():
    return {
        "status": "ok",
        "best_forecaster": summary_meta.get(
            "best_forecaster", "Baseline: exp. smoothing"
        ),
        "metrics": summary_meta.get("metrics_best", {}),
        "regions": summary_meta.get("regions", []),
        "total_product_regions": len(master_df),
    }


@app.get("/models", tags=["Evaluation"])
def model_comparison():
    return {
        "best_forecaster": summary_meta.get("best_forecaster"),
        "best_ml_model": summary_meta.get("best_ml_model"),
        "demand_class_thresholds": summary_meta.get("demand_class_thresholds"),
        "comparison": comp_records,
    }


@app.get("/products", tags=["Inventory"])
def list_products(
    region: Optional[str] = None,
    demand: Optional[str] = None,
    risk: Optional[str] = None,
    search: Optional[str] = None,
    sort: str = "demand",
    top: int = 25,
):
    df = S.filter_sort(
        master_df,
        region=region,
        demand=demand,
        risk=risk,
        search=search,
        sort=sort,
        desc=True,
        top=min(top, 200),
    )
    cols = [
        "region",
        "stock_code",
        "description",
        "demand_class",
        "forecast_next_week",
        "current_stock",
        "stock_source",
        "lead_time_days",
        "safety_stock",
        "reorder_point",
        "days_of_cover",
        "risk_level",
        "risk_score",
        "recommended_order_qty",
        "expiry_status",
        "discount_pct",
        "action",
    ]
    return {
        "count": len(df),
        "items": df[cols].fillna("").to_dict(orient="records"),
    }


@app.get("/reorders", tags=["Inventory"])
def reorders_overview(region: Optional[str] = None, top: int = 25):
    df = master_df if not region else master_df[master_df["region"] == region]
    if df.empty:
        raise HTTPException(
            status_code=404, detail=f"No records found for region '{region}'."
        )
    ov = S.reorder_overview(df)
    urgent = (
        df[df["recommended_order_qty"] > 0]
        .sort_values(["risk_score", "recommended_order_qty"], ascending=False)
        .head(min(top, 200))
    )
    return {
        "kpi": ov["kpi"].to_dict(),
        "by_region": ov["by_region"].reset_index().to_dict(orient="records"),
        "prioritized_reorders": urgent[
            [
                "region",
                "stock_code",
                "description",
                "demand_class",
                "current_stock",
                "reorder_point",
                "risk_level",
                "recommended_order_qty",
                "order_value_gbp",
                "action",
            ]
        ].to_dict(orient="records"),
    }


@app.post("/predict", tags=["Simulation & Risk"])
def predict_and_evaluate(req: EvaluateRequest):
    code = req.stock_code.strip()
    region = req.region.strip()
    match = master_df[
        (master_df["stock_code"] == code) & (master_df["region"] == region)
    ]

    if match.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Product '{code}' in region '{region}' not found. Check GET /products for valid codes & regions.",
        )

    row = match.iloc[0]
    cfg = dict(S.CFG)
    cfg["service_level"] = req.service_level
    cfg["review_weeks"] = req.review_weeks

    stock = (
        float(req.current_stock)
        if req.current_stock is not None
        else float(row["current_stock"])
    )
    lead_days = (
        int(req.lead_time_days)
        if req.lead_time_days is not None
        else int(row["lead_time_days"])
    )
    is_perish = (
        req.perishable
        if req.perishable is not None
        else (row["product_type"] == "Perishable")
    )
    ptype = "Perishable" if is_perish else "Non-perishable"
    shelf_days = (
        req.shelf_days
        if req.shelf_days is not None
        else (
            int(row["days_to_expiry"])
            if pd.notna(row.get("days_to_expiry"))
            else (30 if is_perish else None)
        )
    )

    # Recompute exact error std from stored safety stock & default service level
    z_default = S.zscore(S.CFG["service_level"])
    lead_w_default = float(row["lead_time_days"]) / 7.0
    inferred_err_std = float(row["safety_stock"]) / max(
        z_default * (lead_w_default**0.5), 1e-6
    )

    rec = dict(
        forecast=float(row["forecast_next_week"]),
        err_std=inferred_err_std,
        stock=stock,
        lead_days=lead_days,
        price=float(row["unit_price"]),
        avg8=float(row["avg_weekly_demand_8w"])
        if pd.notna(row["avg_weekly_demand_8w"])
        else float(row["forecast_next_week"]),
        demand_class=str(row["demand_class"]),
        falling=False,
        product_type=ptype,
        shelf_life_days=shelf_days,
        days_left=shelf_days if is_perish else None,
    )
    eval_out = S.evaluate_item(rec, cfg)

    return {
        "stock_code": code,
        "description": str(row["description"]),
        "region": region,
        "demand_class": str(row["demand_class"]),
        "unit_price_gbp": float(row["unit_price"]),
        "stock_source": "user-override"
        if req.current_stock is not None
        else str(row["stock_source"]),
        "lead_time_days": lead_days,
        "product_type": ptype,
        "forecaster_used": summary_meta.get(
            "best_forecaster", "Baseline: exp. smoothing"
        ),
        "inventory_and_risk_analysis": eval_out,
        "assumption_note": "UCI Online Retail lacks on-hand stock, cost, and expiry columns; stock/expiry are simulated unless overridden.",
    }

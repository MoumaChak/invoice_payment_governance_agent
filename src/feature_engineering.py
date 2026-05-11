from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from src.config import MARKET_RISK, PROCESSED_DIR


def _date_add_days(date_series: pd.Series, days_series: pd.Series) -> pd.Series:
    return pd.to_datetime(date_series) + pd.to_timedelta(days_series.fillna(0).astype(int), unit="D")


def build_feature_table(datasets: Dict[str, pd.DataFrame], output_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    inv = datasets["invoice_master"].copy()
    vendor = datasets["vendor_master"].copy()
    po = datasets["po_master"].copy()
    contract = datasets["contract_terms"].copy()
    sap = datasets["sap_payment_data"].copy()
    cora = datasets["cora_workflow_queue"].copy()
    processor = datasets["processor_actions"].copy()
    upr = datasets["upr_requests"].copy()

    df = inv.merge(vendor, on="vendor_id", how="left")
    df = df.merge(po[["po_id", "po_payment_term", "po_amount", "buying_team"]], on="po_id", how="left")
    df = df.merge(contract[["contract_id", "contract_payment_term", "agreement_type"]], on="contract_id", how="left")
    df = df.merge(sap, on="invoice_id", how="left")
    df = df.merge(cora, on="invoice_id", how="left")
    df = df.merge(processor, on="processor_id", how="left")
    df = df.merge(upr, on="invoice_id", how="left")

    for c in ["invoice_date", "invoice_receipt_date", "sap_due_date", "sap_payment_release_date", "queue_entry_date"]:
        df[c] = pd.to_datetime(df[c], errors="coerce")

    # Source-of-truth hierarchy: contract > PO > vendor master > invoice face > SAP due date.
    df["source_of_truth_payment_term"] = (
        df["contract_payment_term"].combine_first(df["po_payment_term"])
        .combine_first(df["vendor_payment_term"])
        .combine_first(df["invoice_payment_term"])
    )
    df["contractual_due_date"] = _date_add_days(df["invoice_receipt_date"], df["source_of_truth_payment_term"])
    df["recommended_release_date"] = df["contractual_due_date"]

    df["days_to_contractual_due_date"] = (df["contractual_due_date"] - df["sap_payment_release_date"]).dt.days
    df["days_paid_early"] = df["days_to_contractual_due_date"].clip(lower=0).fillna(0).astype(int)
    df["early_payment_flag"] = (df["sap_payment_release_date"] < df["contractual_due_date"]).astype(int)
    df["payment_term_mismatch_flag"] = (df["po_payment_term"].fillna(-1).astype(int) != df["invoice_payment_term"].fillna(-2).astype(int)).astype(int)
    df["po_invoice_term_delta"] = df["po_payment_term"].fillna(df["invoice_payment_term"]) - df["invoice_payment_term"]
    df["sap_contract_due_date_delta"] = (df["sap_due_date"] - df["contractual_due_date"]).dt.days.fillna(0).astype(int)
    df["vendor_early_payment_rate"] = df["historical_early_payment_rate"].fillna(0)
    df["processor_early_release_rate"] = df["historical_early_release_rate"].fillna(0)
    df["processor_manual_override_rate"] = df["historical_manual_override_rate"].fillna(0)
    df["queue_age_days"] = df["queue_age_days"].fillna(0).astype(int)
    df["manual_override_flag"] = df["manual_override_flag"].fillna(0).astype(int)
    df["month_end_flag"] = (df["sap_payment_release_date"].dt.day >= 25).astype(int)
    df["weekend_release_flag"] = (df["sap_payment_release_date"].dt.dayofweek >= 5).astype(int)
    threshold = df["invoice_amount"].quantile(0.90)
    df["high_value_invoice_flag"] = (df["invoice_amount"] >= threshold).astype(int)
    df["upr_flag"] = df["upr_flag"].fillna(0).astype(int)
    df["upr_documented_flag"] = df["upr_documented_flag"].fillna(0).astype(int)
    df["upr_missing_but_early_flag"] = ((df["early_payment_flag"] == 1) & (df["upr_flag"] == 0)).astype(int)
    df["sap_error_present_flag"] = (df["sap_error_code"].fillna("NONE") != "NONE").astype(int)
    df["market_risk_score"] = df["market"].map(MARKET_RISK).fillna(0.25)
    notes_lower = df["workflow_notes"].fillna("").str.lower()
    df["workflow_risk_signal"] = (
        notes_lower.str.contains("urgent|payment request", regex=True).astype(int) * 0.55
        + notes_lower.str.contains("date logic|term mapping|sap|cora", regex=True).astype(int) * 0.65
        + notes_lower.str.contains("manual override|queue clearance|accelerated", regex=True).astype(int) * 0.70
        + notes_lower.str.contains("no documented|no clear audit|insufficient audit", regex=True).astype(int) * 0.60
    ).clip(0, 1)
    df["approval_risk_score"] = df["approval_status"].map({"Approved": 0.20, "Pending": 0.55, "Rejected": 0.10}).fillna(0.35)
    df["amount_log"] = np.log1p(df["invoice_amount"].clip(lower=0))
    df["amount_band"] = pd.qcut(df["invoice_amount"].rank(method="first"), q=4, labels=["Low", "Medium", "High", "Very High"])

    def root_cause(row):
        if row["early_payment_flag"] == 0:
            return "Not applicable / on-time"
        if row["upr_flag"] == 1 and str(row.get("upr_approval_status", "")).lower() in ["approved", "pending"]:
            return "UPR"
        if row["sap_error_present_flag"] == 1 or abs(row["sap_contract_due_date_delta"]) >= 3:
            return "System error"
        if row["manual_override_flag"] == 1 or row["processor_early_release_rate"] >= 0.22:
            return "Behavioural"
        return "Insufficient audit trail"

    df["root_cause_label"] = df.apply(root_cause, axis=1)
    df["invoice_month"] = df["sap_payment_release_date"].dt.to_period("M").astype(str)

    # Keep stable ordering and persist.
    df = df.sort_values("invoice_id").reset_index(drop=True)
    df.to_csv(output_dir / "invoice_model_table.csv", index=False)
    return df

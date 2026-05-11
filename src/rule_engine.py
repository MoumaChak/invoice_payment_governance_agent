from __future__ import annotations

from typing import List

import pandas as pd


def _flag_names(row: pd.Series) -> List[str]:
    flags = []
    if row.get("early_payment_flag", 0) == 1:
        flags.append("Release before contractual due date")
    if row.get("days_paid_early", 0) >= 15:
        flags.append("15+ days early")
    if row.get("upr_missing_but_early_flag", 0) == 1:
        flags.append("UPR missing but paid early")
    if abs(row.get("sap_contract_due_date_delta", 0)) >= 3:
        flags.append("SAP due date differs from contract")
    if row.get("payment_term_mismatch_flag", 0) == 1:
        flags.append("PO/invoice payment term mismatch")
    if row.get("manual_override_flag", 0) == 1:
        flags.append("Manual override")
    if row.get("high_value_invoice_flag", 0) == 1 and row.get("early_payment_flag", 0) == 1:
        flags.append("High-value early payment")
    if row.get("vendor_early_payment_rate", 0) >= 0.25:
        flags.append("Repeat vendor early payment pattern")
    return flags


def recommend_action(row: pd.Series, early_days_threshold: int = 15) -> str:
    if row.get("early_payment_flag", 0) == 0 and row.get("risk_score", 0) < 0.50:
        return "APPROVE"
    if row.get("days_paid_early", 0) >= early_days_threshold and row.get("upr_flag", 0) == 0:
        return "HOLD"
    if row.get("high_value_invoice_flag", 0) == 1 and row.get("early_payment_flag", 0) == 1:
        return "ESCALATE"
    if row.get("sap_error_present_flag", 0) == 1 or abs(row.get("sap_contract_due_date_delta", 0)) >= 3:
        return "DEFER"
    if row.get("risk_score", 0) >= 0.75:
        return "HOLD"
    return "APPROVE"


def apply_rules(df: pd.DataFrame, early_days_threshold: int = 15) -> pd.DataFrame:
    out = df.copy()
    out["rule_flags"] = out.apply(lambda r: "; ".join(_flag_names(r)) or "No hard-rule breach", axis=1)
    out["rule_flag_count"] = out["rule_flags"].apply(lambda x: 0 if x == "No hard-rule breach" else len(x.split("; ")))
    out["recommended_action"] = out.apply(lambda r: recommend_action(r, early_days_threshold), axis=1)
    out["status"] = out["recommended_action"].map({"HOLD": "Review Required", "ESCALATE": "Escalated", "DEFER": "Deferred", "APPROVE": "Ready to Release"}).fillna("Review Required")
    return out

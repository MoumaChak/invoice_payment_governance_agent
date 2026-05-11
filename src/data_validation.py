from typing import Dict, List, Tuple

import pandas as pd

REQUIRED_COLUMNS = {
    "invoice_master": ["invoice_id", "vendor_id", "po_id", "contract_id", "processor_id", "market", "currency", "invoice_date", "invoice_receipt_date", "invoice_amount", "invoice_payment_term", "invoice_priority", "approval_status"],
    "vendor_master": ["vendor_id", "vendor_name", "vendor_category", "vendor_country", "vendor_payment_term", "vendor_risk_rating", "historical_early_payment_rate"],
    "po_master": ["po_id", "vendor_id", "po_payment_term", "po_amount", "po_date", "buying_team"],
    "contract_terms": ["contract_id", "vendor_id", "contract_payment_term", "agreement_type", "contract_start_date", "contract_end_date"],
    "sap_payment_data": ["invoice_id", "sap_due_date", "sap_payment_release_date", "payment_status", "payment_method"],
    "cora_workflow_queue": ["invoice_id", "queue_entry_date", "queue_age_days", "workflow_status", "workflow_notes", "manual_override_flag", "sap_error_code", "time_of_day"],
    "processor_actions": ["processor_id", "processor_name", "team", "avg_daily_volume", "historical_manual_override_rate", "historical_early_release_rate"],
    "upr_requests": ["invoice_id", "upr_flag", "upr_reason", "upr_approval_status", "upr_requested_by", "upr_documented_flag"],
    "ocr_invoice_text": ["invoice_id", "invoice_text", "ground_truth_due_date", "ground_truth_payment_term", "ground_truth_vendor_id", "ground_truth_invoice_amount", "ground_truth_invoice_date", "ground_truth_po_number", "ground_truth_currency", "ground_truth_contract_reference"],
}

VALID_TERMS = {30, 45, 60, 90}


def _check_required_columns(name: str, df: pd.DataFrame, errors: List[str]) -> None:
    missing = [c for c in REQUIRED_COLUMNS[name] if c not in df.columns]
    if missing:
        errors.append(f"{name}: missing columns {missing}")


def validate_datasets(datasets: Dict[str, pd.DataFrame]) -> Tuple[bool, pd.DataFrame]:
    errors: List[str] = []
    warnings: List[str] = []

    for name, required in REQUIRED_COLUMNS.items():
        if name not in datasets:
            errors.append(f"Missing dataset: {name}")
            continue
        df = datasets[name]
        if df.empty:
            errors.append(f"{name}: dataset is empty")
        _check_required_columns(name, df, errors)

    if "invoice_master" in datasets:
        inv = datasets["invoice_master"]
        if inv["invoice_id"].duplicated().any():
            errors.append("invoice_master: duplicate invoice_id found")
        if (inv["invoice_amount"] < 0).any():
            errors.append("invoice_master: negative invoice_amount found")
        invalid_terms = set(inv["invoice_payment_term"].dropna().astype(int).unique()) - VALID_TERMS
        if invalid_terms:
            errors.append(f"invoice_master: invalid payment terms {sorted(invalid_terms)}")
        for col in ["invoice_date", "invoice_receipt_date"]:
            bad = pd.to_datetime(inv[col], errors="coerce").isna().sum()
            if bad:
                errors.append(f"invoice_master: {bad} invalid {col} values")

    if "vendor_master" in datasets and "invoice_master" in datasets:
        invalid_vendors = set(datasets["invoice_master"]["vendor_id"]) - set(datasets["vendor_master"]["vendor_id"])
        if invalid_vendors:
            errors.append(f"invoice_master: invalid vendor IDs detected: {len(invalid_vendors)}")

    if "po_master" in datasets:
        po = datasets["po_master"]
        invalid_po_terms = set(po["po_payment_term"].dropna().astype(int).unique()) - VALID_TERMS
        if invalid_po_terms:
            errors.append(f"po_master: invalid payment terms {sorted(invalid_po_terms)}")
        if (po["po_amount"] < 0).any():
            errors.append("po_master: negative po_amount found")

    report = pd.DataFrame(
        [{"severity": "ERROR", "message": e} for e in errors] +
        [{"severity": "WARNING", "message": w} for w in warnings]
    )
    if report.empty:
        report = pd.DataFrame([{"severity": "OK", "message": "All validation checks passed"}])
    return len(errors) == 0, report

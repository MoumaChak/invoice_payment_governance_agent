from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import OUTPUTS_DIR, SIMULATED_DIR

PATTERNS = {
    "due_date": [
        r"(?mi)^(?:DUE_DATE|DUE DATE|PAYMENT_DUE_DATE)\s*[:\-]\s*([0-9]{4}-[0-9]{2}-[0-9]{2}|[0-9]{2}/[0-9]{2}/[0-9]{4})\s*$",
    ],
    "payment_term": [
        r"(?mi)^(?:PAYMENT_TERM|PAYMENT TERM)\s*[:\-]\s*([0-9]+)\s*DAYS\s*$",
    ],
    "vendor_id": [
        r"(?mi)^(?:VENDOR_ID|VENDOR ID|SUPPLIER_ID)\s*[:\-]\s*(V[0-9]+)\s*$",
    ],
    "invoice_amount": [
        r"(?mi)^(?:INVOICE_AMOUNT|TOTAL_AMOUNT|AMOUNT)\s*[:\-]\s*([0-9,]+\.?[0-9]*)\s*$",
    ],
    "invoice_date": [
        r"(?mi)^(?:INVOICE_DATE|INVOICE DATE)\s*[:\-]\s*([0-9]{4}-[0-9]{2}-[0-9]{2}|[0-9]{2}/[0-9]{2}/[0-9]{4})\s*$",
    ],
    "po_number": [
        r"(?mi)^(?:PO_NUMBER|PO NUMBER|PO_NO)\s*[:\-]\s*(PO[0-9]+)\s*$",
    ],
    "currency": [
        r"(?mi)^(?:CURRENCY|CURR)\s*[:\-]\s*([A-Z]{3})\s*$",
    ],
    "contract_reference": [
        r"(?mi)^(?:CONTRACT_REF|CONTRACT REFERENCE|CONTRACT_NO)\s*[:\-]\s*(CT[0-9]+)\s*$",
    ],
}


def _first_match(patterns: list[str], text: str):
    text = text or ""
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def _parse_date(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    value = str(value).strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y"):
        try:
            return pd.to_datetime(value, format=fmt).date().isoformat()
        except Exception:
            continue
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date().isoformat()


def _parse_int(value):
    if value is None:
        return None
    try:
        return int(float(str(value).strip().replace(",", "")))
    except Exception:
        return None


def _parse_amount(value):
    if value is None:
        return None
    try:
        return round(float(str(value).strip().replace(",", "")), 2)
    except Exception:
        return None


def _safe_equal_series(left: pd.Series, right: pd.Series) -> pd.Series:
    return left.fillna("__missing__").astype(str) == right.fillna("__missing__").astype(str)


def run_ocr_extraction(input_path: Path = SIMULATED_DIR / "ocr_invoice_text.csv", outputs_dir: Path = OUTPUTS_DIR) -> pd.DataFrame:
    outputs_dir.mkdir(parents=True, exist_ok=True)
    if not input_path.exists():
        raise FileNotFoundError(f"OCR input file not found: {input_path}")

    df = pd.read_csv(input_path)
    extracted = pd.DataFrame({"invoice_id": df["invoice_id"]})

    extracted["extracted_due_date_raw"] = df["invoice_text"].apply(lambda x: _first_match(PATTERNS["due_date"], x))
    extracted["extracted_due_date"] = extracted["extracted_due_date_raw"].apply(_parse_date)

    extracted["extracted_payment_term_raw"] = df["invoice_text"].apply(lambda x: _first_match(PATTERNS["payment_term"], x))
    extracted["extracted_payment_term"] = extracted["extracted_payment_term_raw"].apply(_parse_int)

    extracted["extracted_vendor_id"] = df["invoice_text"].apply(lambda x: _first_match(PATTERNS["vendor_id"], x))

    extracted["extracted_invoice_amount_raw"] = df["invoice_text"].apply(lambda x: _first_match(PATTERNS["invoice_amount"], x))
    extracted["extracted_invoice_amount"] = extracted["extracted_invoice_amount_raw"].apply(_parse_amount)

    extracted["extracted_invoice_date_raw"] = df["invoice_text"].apply(lambda x: _first_match(PATTERNS["invoice_date"], x))
    extracted["extracted_invoice_date"] = extracted["extracted_invoice_date_raw"].apply(_parse_date)

    extracted["extracted_po_number"] = df["invoice_text"].apply(lambda x: _first_match(PATTERNS["po_number"], x))
    extracted["extracted_currency"] = df["invoice_text"].apply(lambda x: _first_match(PATTERNS["currency"], x))
    extracted["extracted_contract_reference"] = df["invoice_text"].apply(lambda x: _first_match(PATTERNS["contract_reference"], x))

    comparisons = {
        "due_date": _safe_equal_series(extracted["extracted_due_date"], df["ground_truth_due_date"]),
        "payment_term": extracted["extracted_payment_term"].fillna(-9999).astype(int) == df["ground_truth_payment_term"].fillna(-8888).astype(int),
        "vendor_id": _safe_equal_series(extracted["extracted_vendor_id"], df["ground_truth_vendor_id"]),
        "invoice_amount": extracted["extracted_invoice_amount"].fillna(-9999).round(2) == df["ground_truth_invoice_amount"].fillna(-8888).round(2),
        "invoice_date": _safe_equal_series(extracted["extracted_invoice_date"], df["ground_truth_invoice_date"]),
        "po_number": _safe_equal_series(extracted["extracted_po_number"], df["ground_truth_po_number"]),
        "currency": _safe_equal_series(extracted["extracted_currency"], df["ground_truth_currency"]),
        "contract_reference": _safe_equal_series(extracted["extracted_contract_reference"], df["ground_truth_contract_reference"]),
    }

    metrics = pd.DataFrame([
        {"field": field, "accuracy": float(vals.mean())} for field, vals in comparisons.items()
    ])
    metrics.loc[len(metrics)] = ["overall", float(pd.concat([v.rename(k) for k, v in comparisons.items()], axis=1).values.mean())]

    results = pd.concat([extracted, df.filter(like="ground_truth")], axis=1)
    results.to_csv(outputs_dir / "ocr_extraction_results.csv", index=False)
    metrics.to_csv(outputs_dir / "ocr_metrics.csv", index=False)
    return metrics

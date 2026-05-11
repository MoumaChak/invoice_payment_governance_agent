from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

try:
    from faker import Faker
except Exception:  # graceful fallback
    Faker = None

from src.config import (
    CURRENCIES,
    MARKET_RISK,
    MARKETS,
    N_INVOICES,
    PAYMENT_TERMS,
    RANDOM_SEED,
    SIMULATED_DIR,
)


SCENARIOS = [
    "NORMAL_ON_TIME",
    "LOW_RISK_STANDARD_PROCESS",
    "LEGITIMATE_UPR_EARLY_PAYMENT",
    "CORA_SAP_SYSTEM_ERROR",
    "BEHAVIOURAL_QUEUE_CLEARANCE",
    "PAYMENT_TERM_MISMATCH",
    "REPEAT_VENDOR_PATTERN",
    "HIGH_VALUE_MANUAL_OVERRIDE",
    "AUDIT_GAP_EARLY_RELEASE",
]

SCENARIO_PROBS = np.array([0.33, 0.12, 0.10, 0.11, 0.12, 0.08, 0.07, 0.04, 0.03])
SCENARIO_PROBS = SCENARIO_PROBS / SCENARIO_PROBS.sum()


def _fake_company(i: int) -> str:
    if Faker:
        fake = Faker()
        return fake.company()
    return f"Vendor Company {i:04d}"


def _choose_different_terms(rng: np.random.Generator, current_terms: np.ndarray) -> np.ndarray:
    out = current_terms.copy()
    for idx, current in enumerate(current_terms):
        choices = [t for t in PAYMENT_TERMS if int(t) != int(current)]
        out[idx] = rng.choice(choices)
    return out


def generate_simulated_data(n_invoices: int = N_INVOICES, output_dir: Path = SIMULATED_DIR) -> Dict[str, pd.DataFrame]:
    """Generate enterprise-style AP governance data with multiple realistic scenarios.

    Important modeling design:
    - The target is intentionally generated from a noisy business-risk process.
    - Direct leakage fields such as days_paid_early are created for dashboards/rules,
      but the M1 model training code excludes them from predictors.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_vendors = 420
    n_processors = 90
    n_pos = max(1000, n_invoices // 4)
    n_contracts = n_vendors

    vendor_ids = np.array([f"V{str(i).zfill(5)}" for i in range(1, n_vendors + 1)])
    processor_ids = np.array([f"P{str(i).zfill(4)}" for i in range(1, n_processors + 1)])
    po_ids = np.array([f"PO{str(i).zfill(7)}" for i in range(1, n_pos + 1)])
    contract_ids = np.array([f"CT{str(i).zfill(6)}" for i in range(1, n_contracts + 1)])

    vendor_market = rng.choice(MARKETS, size=n_vendors, p=np.array([0.17, 0.18, 0.12, 0.12, 0.10, 0.11, 0.09, 0.11]))
    vendor_payment_term = rng.choice(PAYMENT_TERMS, size=n_vendors, p=[0.42, 0.18, 0.30, 0.10])
    vendor_risk_rating = rng.choice(["Low", "Medium", "High"], size=n_vendors, p=[0.55, 0.32, 0.13])
    vendor_hist = np.clip(
        rng.beta(2.0, 10.0, n_vendors)
        + (vendor_risk_rating == "High") * 0.14
        + (vendor_risk_rating == "Medium") * 0.05,
        0.01,
        0.68,
    )
    # Introduce repeat-offender vendors so the repeat-vendor scenario is learnable but not deterministic.
    repeat_vendor_mask = rng.random(n_vendors) < 0.12
    vendor_hist[repeat_vendor_mask] = np.clip(vendor_hist[repeat_vendor_mask] + rng.uniform(0.10, 0.24, repeat_vendor_mask.sum()), 0.05, 0.75)

    vendor_df = pd.DataFrame({
        "vendor_id": vendor_ids,
        "vendor_name": [_fake_company(i) for i in range(1, n_vendors + 1)],
        "vendor_category": rng.choice(["Raw Materials", "Packaging", "Logistics", "Marketing", "IT Services", "Facilities", "Professional Services"], n_vendors),
        "vendor_country": vendor_market,
        "vendor_payment_term": vendor_payment_term,
        "vendor_risk_rating": vendor_risk_rating,
        "historical_early_payment_rate": np.round(vendor_hist, 4),
    })

    processor_df = pd.DataFrame({
        "processor_id": processor_ids,
        "processor_name": [f"Processor {i:03d}" for i in range(1, n_processors + 1)],
        "team": rng.choice(["AP India", "AP Americas", "AP Europe", "AP LATAM", "AP Africa", "AP SEA"], n_processors),
        "avg_daily_volume": rng.integers(35, 190, n_processors),
        "historical_manual_override_rate": np.round(np.clip(rng.beta(2, 9, n_processors), 0.01, 0.65), 4),
        "historical_early_release_rate": np.round(np.clip(rng.beta(2, 8, n_processors), 0.01, 0.70), 4),
    })
    high_override_processors = rng.choice(processor_ids, size=max(6, n_processors // 10), replace=False)
    processor_df.loc[processor_df["processor_id"].isin(high_override_processors), "historical_manual_override_rate"] = np.round(rng.uniform(0.28, 0.55, len(high_override_processors)), 4)
    processor_df.loc[processor_df["processor_id"].isin(high_override_processors), "historical_early_release_rate"] = np.round(rng.uniform(0.25, 0.55, len(high_override_processors)), 4)

    contract_df = pd.DataFrame({
        "contract_id": contract_ids,
        "vendor_id": vendor_ids,
        "contract_payment_term": np.where(rng.random(n_contracts) < 0.82, vendor_payment_term, rng.choice(PAYMENT_TERMS, n_contracts)),
        "agreement_type": rng.choice(["Standard", "Strategic", "Exception"], n_contracts, p=[0.72, 0.22, 0.06]),
        "contract_start_date": pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 180, n_contracts), unit="D"),
        "contract_end_date": pd.to_datetime("2027-01-01") + pd.to_timedelta(rng.integers(0, 365, n_contracts), unit="D"),
    })

    po_vendor = rng.choice(vendor_ids, n_pos)
    vendor_term_map = dict(zip(vendor_ids, vendor_payment_term))
    po_terms = np.array([vendor_term_map[v] for v in po_vendor])
    base_mismatch = rng.random(n_pos) < 0.18
    po_terms[base_mismatch] = _choose_different_terms(rng, po_terms[base_mismatch])
    po_df = pd.DataFrame({
        "po_id": po_ids,
        "vendor_id": po_vendor,
        "po_payment_term": po_terms,
        "po_amount": np.round(rng.lognormal(mean=10.6, sigma=0.85, size=n_pos), 2),
        "po_date": pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 420, n_pos), unit="D"),
        "buying_team": rng.choice(["Foods", "Beauty", "Home Care", "Ice Cream", "Nutrition", "Corporate"], n_pos),
    })

    inv_ids = np.array([f"INV{str(i).zfill(8)}" for i in range(1, n_invoices + 1)])
    scenarios = rng.choice(SCENARIOS, n_invoices, p=SCENARIO_PROBS)
    inv_vendor = rng.choice(vendor_ids, n_invoices)

    # Make repeat-vendor scenarios more likely to use high-history vendors.
    high_hist_vendors = vendor_df.loc[vendor_df["historical_early_payment_rate"] >= vendor_df["historical_early_payment_rate"].quantile(0.75), "vendor_id"].to_numpy()
    repeat_idx = np.where(scenarios == "REPEAT_VENDOR_PATTERN")[0]
    if len(repeat_idx):
        inv_vendor[repeat_idx] = rng.choice(high_hist_vendors, len(repeat_idx))

    vendor_idx = {v: i for i, v in enumerate(vendor_ids)}
    inv_contract = np.array([contract_ids[vendor_idx[v]] for v in inv_vendor])
    vendor_to_pos = po_df.groupby("vendor_id")["po_id"].apply(list).to_dict()
    inv_po = np.array([rng.choice(vendor_to_pos.get(v, list(po_ids))) for v in inv_vendor])

    inv_processor = rng.choice(processor_ids, n_invoices)
    behavioural_idx = np.where(np.isin(scenarios, ["BEHAVIOURAL_QUEUE_CLEARANCE", "HIGH_VALUE_MANUAL_OVERRIDE"]))[0]
    if len(behavioural_idx):
        inv_processor[behavioural_idx] = rng.choice(high_override_processors, len(behavioural_idx))

    inv_market = np.array([vendor_df.loc[vendor_idx[v], "vendor_country"] for v in inv_vendor])
    inv_currency = rng.choice(CURRENCIES, n_invoices)
    inv_date = pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 455, n_invoices), unit="D")
    receipt_lag = rng.integers(0, 8, n_invoices)
    inv_receipt = inv_date + pd.to_timedelta(receipt_lag, unit="D")
    inv_amount = np.round(rng.lognormal(mean=9.7, sigma=1.05, size=n_invoices), 2)
    high_value_idx = np.where(scenarios == "HIGH_VALUE_MANUAL_OVERRIDE")[0]
    if len(high_value_idx):
        inv_amount[high_value_idx] *= rng.uniform(4, 10, len(high_value_idx))
    random_high_spike = rng.random(n_invoices) < 0.025
    inv_amount[random_high_spike] *= rng.uniform(5, 12, random_high_spike.sum())

    invoice_terms = np.array([vendor_term_map[v] for v in inv_vendor])
    term_noise = rng.random(n_invoices) < 0.18
    invoice_terms[term_noise] = _choose_different_terms(rng, invoice_terms[term_noise])
    mismatch_invoice_idx = np.where(scenarios == "PAYMENT_TERM_MISMATCH")[0]
    if len(mismatch_invoice_idx):
        invoice_terms[mismatch_invoice_idx] = _choose_different_terms(rng, invoice_terms[mismatch_invoice_idx])

    invoice_priority = rng.choice(["Low", "Medium", "High", "Critical"], n_invoices, p=[0.20, 0.56, 0.19, 0.05])
    invoice_priority[np.isin(scenarios, ["LEGITIMATE_UPR_EARLY_PAYMENT", "HIGH_VALUE_MANUAL_OVERRIDE"])] = rng.choice(["High", "Critical"], np.isin(scenarios, ["LEGITIMATE_UPR_EARLY_PAYMENT", "HIGH_VALUE_MANUAL_OVERRIDE"]).sum(), p=[0.62, 0.38])
    approval_status = rng.choice(["Pending", "Approved", "Rejected"], n_invoices, p=[0.22, 0.75, 0.03])
    approval_status[np.isin(scenarios, ["LEGITIMATE_UPR_EARLY_PAYMENT", "BEHAVIOURAL_QUEUE_CLEARANCE", "HIGH_VALUE_MANUAL_OVERRIDE"])] = rng.choice(["Approved", "Pending"], np.isin(scenarios, ["LEGITIMATE_UPR_EARLY_PAYMENT", "BEHAVIOURAL_QUEUE_CLEARANCE", "HIGH_VALUE_MANUAL_OVERRIDE"]).sum(), p=[0.72, 0.28])

    invoice_df = pd.DataFrame({
        "invoice_id": inv_ids,
        "vendor_id": inv_vendor,
        "po_id": inv_po,
        "contract_id": inv_contract,
        "processor_id": inv_processor,
        "market": inv_market,
        "currency": inv_currency,
        "invoice_date": inv_date,
        "invoice_receipt_date": inv_receipt,
        "invoice_amount": np.round(inv_amount, 2),
        "invoice_payment_term": invoice_terms,
        "invoice_priority": invoice_priority,
        "approval_status": approval_status,
        "scenario_label": scenarios,
    })

    contract_term_map = dict(zip(contract_df.contract_id, contract_df.contract_payment_term))
    contract_due = inv_receipt + pd.to_timedelta([contract_term_map[c] for c in inv_contract], unit="D")

    vendor_hist_map = dict(zip(vendor_df.vendor_id, vendor_df.historical_early_payment_rate))
    proc_early_map = dict(zip(processor_df.processor_id, processor_df.historical_early_release_rate))
    proc_override_map = dict(zip(processor_df.processor_id, processor_df.historical_manual_override_rate))
    market_risk_arr = np.array([MARKET_RISK[m] for m in inv_market])
    vendor_hist_arr = np.array([vendor_hist_map[v] for v in inv_vendor])
    proc_early_arr = np.array([proc_early_map[p] for p in inv_processor])
    proc_override_arr = np.array([proc_override_map[p] for p in inv_processor])

    is_upr_scenario = scenarios == "LEGITIMATE_UPR_EARLY_PAYMENT"
    is_system_scenario = scenarios == "CORA_SAP_SYSTEM_ERROR"
    is_behaviour_scenario = scenarios == "BEHAVIOURAL_QUEUE_CLEARANCE"
    is_mismatch_scenario = scenarios == "PAYMENT_TERM_MISMATCH"
    is_repeat_scenario = scenarios == "REPEAT_VENDOR_PATTERN"
    is_high_value_scenario = scenarios == "HIGH_VALUE_MANUAL_OVERRIDE"
    is_audit_gap_scenario = scenarios == "AUDIT_GAP_EARLY_RELEASE"

    upr_flag_prob = np.where(is_upr_scenario, 0.94, np.where(is_high_value_scenario, 0.12, 0.035))
    upr_flag = rng.random(n_invoices) < upr_flag_prob
    manual_override_prob = np.clip(
        proc_override_arr * 0.65 + 0.03
        + is_behaviour_scenario * 0.70
        + is_high_value_scenario * 0.62
        + is_audit_gap_scenario * 0.25,
        0.02,
        0.85,
    )
    manual_override = rng.random(n_invoices) < manual_override_prob
    sap_error_prob = np.clip(0.015 + market_risk_arr * 0.04 + is_system_scenario * 0.84 + is_mismatch_scenario * 0.48, 0.01, 0.94)
    sap_error_present = rng.random(n_invoices) < sap_error_prob
    queue_age = rng.integers(0, 18, n_invoices)
    queue_age[is_behaviour_scenario | is_audit_gap_scenario] += rng.integers(5, 16, (is_behaviour_scenario | is_audit_gap_scenario).sum())
    queue_age = np.clip(queue_age, 0, 35)

    scenario_base = np.select(
        [
            scenarios == "NORMAL_ON_TIME",
            scenarios == "LOW_RISK_STANDARD_PROCESS",
            is_upr_scenario,
            is_system_scenario,
            is_behaviour_scenario,
            is_mismatch_scenario,
            is_repeat_scenario,
            is_high_value_scenario,
            is_audit_gap_scenario,
        ],
        [0.01, 0.025, 0.80, 0.82, 0.84, 0.68, 0.64, 0.76, 0.66],
        default=0.10,
    )
    # Noisy probability: high enough to be learnable, not high enough to become perfect.
    early_probability = np.clip(
        scenario_base
        + vendor_hist_arr * 0.11
        + proc_early_arr * 0.09
        + manual_override.astype(float) * 0.06
        + sap_error_present.astype(float) * 0.06
        + market_risk_arr * 0.025
        + (queue_age > 14).astype(float) * 0.035
        + (invoice_priority == "Critical").astype(float) * 0.035
        + rng.normal(0, 0.030, n_invoices),
        0.02,
        0.88,
    )
    early_flag = rng.random(n_invoices) < early_probability

    early_days = np.zeros(n_invoices, dtype=int)
    # Scenario-specific early-day severity.
    for mask, low, high in [
        (is_upr_scenario, 3, 18),
        (is_system_scenario, 7, 38),
        (is_behaviour_scenario, 8, 45),
        (is_mismatch_scenario, 5, 35),
        (is_repeat_scenario, 5, 32),
        (is_high_value_scenario, 10, 42),
        (is_audit_gap_scenario, 12, 46),
    ]:
        idx = np.where(mask & early_flag)[0]
        if len(idx):
            early_days[idx] = rng.integers(low, high, len(idx))
    default_early_idx = np.where((early_flag) & (early_days == 0))[0]
    if len(default_early_idx):
        early_days[default_early_idx] = rng.integers(2, 18, len(default_early_idx))
    normal_delay = rng.integers(0, 8, n_invoices)
    release_dates = pd.Series(contract_due).reset_index(drop=True) - pd.to_timedelta(early_days, unit="D") + pd.to_timedelta(np.where(early_flag, 0, normal_delay), unit="D")

    sap_due_noise = rng.choice([-3, 0, 0, 0, 2, 5], n_invoices, p=[0.08, 0.38, 0.25, 0.14, 0.09, 0.06])
    sys_idx = np.where(is_system_scenario | is_mismatch_scenario)[0]
    if len(sys_idx):
        sap_due_noise[sys_idx] = rng.choice([-14, -10, -7, -3, 0, 5, 10], len(sys_idx), p=[0.16, 0.18, 0.22, 0.16, 0.12, 0.08, 0.08])
    sap_due = pd.Series(contract_due).reset_index(drop=True) + pd.to_timedelta(sap_due_noise, unit="D")

    sap_df = pd.DataFrame({
        "invoice_id": inv_ids,
        "sap_due_date": sap_due,
        "sap_payment_release_date": release_dates,
        "payment_status": np.where(rng.random(n_invoices) < 0.10, "Held", "Queued"),
        "payment_method": rng.choice(["ACH", "Wire", "Cheque", "SEPA", "NEFT"], n_invoices, p=[0.42, 0.22, 0.06, 0.16, 0.14]),
    })

    sap_error_codes = np.where(sap_error_present, rng.choice(["DATE_MAP_ERR", "TERM_CFG_ERR", "CORA_SYNC_ERR", "SAP_BLOCK_MISS"], n_invoices), "NONE")
    time_of_day = rng.choice(["Morning", "Afternoon", "Evening", "Late Night"], n_invoices, p=[0.38, 0.34, 0.20, 0.08])
    time_of_day[is_behaviour_scenario | is_audit_gap_scenario] = rng.choice(["Evening", "Late Night", "Afternoon"], (is_behaviour_scenario | is_audit_gap_scenario).sum(), p=[0.42, 0.28, 0.30])

    root_cause = []
    notes = []
    for i in range(n_invoices):
        if not early_flag[i]:
            rc = "Not applicable / on-time"
            note = f"Scenario {scenarios[i]}: standard queue movement; payment not before contractual due date."
        elif upr_flag[i] and is_upr_scenario[i]:
            rc = "UPR"
            note = "Urgent payment request submitted by business; approval/documentation requires validation before release."
        elif sap_error_present[i] or is_system_scenario[i]:
            rc = "System error"
            note = f"Cora/SAP date logic issue detected with code {sap_error_codes[i]}; payment term mapping requires review."
        elif manual_override[i] or is_behaviour_scenario[i] or is_high_value_scenario[i]:
            rc = "Behavioural"
            note = "Manual override or queue clearance behaviour observed; payment appears accelerated before contractual due date."
        else:
            rc = "Insufficient audit trail"
            note = "Early release detected with no documented UPR, system defect, or clear audit owner. Categorized as insufficient audit trail requiring investigation."
        root_cause.append(rc)
        notes.append(note)

    cora_df = pd.DataFrame({
        "invoice_id": inv_ids,
        "queue_entry_date": inv_receipt + pd.to_timedelta(rng.integers(0, 5, n_invoices), unit="D"),
        "queue_age_days": queue_age,
        "workflow_status": rng.choice(["Pending", "Approved", "Released", "Held"], n_invoices, p=[0.24, 0.42, 0.22, 0.12]),
        "workflow_notes": notes,
        "manual_override_flag": manual_override.astype(int),
        "sap_error_code": sap_error_codes,
        "time_of_day": time_of_day,
    })

    upr_df = pd.DataFrame({
        "invoice_id": inv_ids,
        "upr_flag": upr_flag.astype(int),
        "upr_reason": np.where(upr_flag, rng.choice(["Critical supply continuity", "Legal settlement", "Port clearance", "Urgent production need", "Executive exception"], n_invoices), "None"),
        "upr_approval_status": np.where(upr_flag, rng.choice(["Approved", "Pending", "Rejected"], n_invoices, p=[0.68, 0.26, 0.06]), "Not Applicable"),
        "upr_requested_by": np.where(upr_flag, rng.choice(["Business Unit", "Procurement", "Finance Lead", "Plant Manager"], n_invoices), "None"),
        "upr_documented_flag": np.where(upr_flag, (rng.random(n_invoices) < 0.78).astype(int), 0),
    })

    ocr_texts = []
    for i in range(n_invoices):
        due_dt = pd.Timestamp(contract_due[i]).date()
        inv_dt = pd.Timestamp(inv_date[i]).date()
        term = int(contract_term_map[inv_contract[i]])
        amount = float(inv_amount[i])

        vendor_label = rng.choice(["VENDOR_ID", "VENDOR ID", "SUPPLIER_ID"], p=[0.84, 0.11, 0.05])
        po_label = rng.choice(["PO_NUMBER", "PO NUMBER", "PO_NO"], p=[0.82, 0.12, 0.06])
        contract_label = rng.choice(["CONTRACT_REF", "CONTRACT REFERENCE", "CONTRACT_NO"], p=[0.83, 0.12, 0.05])
        inv_date_label = rng.choice(["INVOICE_DATE", "INVOICE DATE"], p=[0.86, 0.14])
        due_label = rng.choice(["DUE_DATE", "DUE DATE", "PAYMENT_DUE_DATE"], p=[0.82, 0.13, 0.05])
        term_label = rng.choice(["PAYMENT_TERM", "PAYMENT TERM"], p=[0.86, 0.14])
        currency_label = rng.choice(["CURRENCY", "CURR"], p=[0.9, 0.1])
        amount_label = rng.choice(["INVOICE_AMOUNT", "TOTAL_AMOUNT", "AMOUNT"], p=[0.8, 0.12, 0.08])

        due_str = due_dt.strftime("%Y-%m-%d") if rng.random() < 0.9 else due_dt.strftime("%d/%m/%Y")
        inv_date_str = inv_dt.strftime("%Y-%m-%d") if rng.random() < 0.9 else inv_dt.strftime("%d/%m/%Y")
        amount_str = f"{amount:.2f}" if rng.random() < 0.82 else f"{amount:,.2f}"

        # Mild OCR-style imperfections so M3 is realistic but still strong.
        if rng.random() < 0.03:
            due_label = "DUE_DAIE"  # not captured by parser
        if rng.random() < 0.025:
            vendor_label = "VENDOR_1D"  # OCR confusion I vs 1
        if rng.random() < 0.025:
            po_label = "P0_NUMBER"
        if rng.random() < 0.025:
            contract_label = "CONTR4CT_REF"
        if rng.random() < 0.025:
            inv_date_label = "INVOICE_DAIE"
        if rng.random() < 0.025:
            term_label = "PAYMENT_T3RM"
        if rng.random() < 0.02:
            currency_label = "CURRENXY"
        if rng.random() < 0.03:
            amount_label = "INV0ICE_AMOUNT"  # OCR confusion O vs 0
        if rng.random() < 0.02:
            due_str = due_dt.strftime("%d-%m-%Y")
        if rng.random() < 0.02:
            inv_date_str = inv_dt.strftime("%d-%m-%Y")

        sep = rng.choice([":", " - ", ": ", " : "], p=[0.55, 0.1, 0.25, 0.10])
        lines = [
            f"INVOICE_ID{sep}{inv_ids[i]}",
            f"{vendor_label}{sep}{inv_vendor[i]}",
            f"{po_label}{sep}{inv_po[i]}",
            f"{contract_label}{sep}{inv_contract[i]}",
            f"{inv_date_label}{sep}{inv_date_str}",
            f"{due_label}{sep}{due_str}",
            f"{term_label}{sep}{term} DAYS",
            f"{currency_label}{sep}{inv_currency[i]}",
            f"{amount_label}{sep}{amount_str}",
        ]
        ocr_texts.append("\n".join(lines) + "\n")

    ocr_df = pd.DataFrame({
        "invoice_id": inv_ids,
        "invoice_text": ocr_texts,
        "ground_truth_due_date": pd.Series(contract_due).dt.date.astype(str),
        "ground_truth_payment_term": [contract_term_map[c] for c in inv_contract],
        "ground_truth_vendor_id": inv_vendor,
        "ground_truth_invoice_amount": np.round(inv_amount, 2),
        "ground_truth_invoice_date": pd.Series(inv_date).dt.date.astype(str),
        "ground_truth_po_number": inv_po,
        "ground_truth_currency": inv_currency,
        "ground_truth_contract_reference": inv_contract,
    })

    frames = {
        "invoice_master": invoice_df,
        "vendor_master": vendor_df,
        "po_master": po_df,
        "contract_terms": contract_df,
        "sap_payment_data": sap_df,
        "cora_workflow_queue": cora_df,
        "processor_actions": processor_df,
        "upr_requests": upr_df,
        "ocr_invoice_text": ocr_df,
    }
    for name, df in frames.items():
        df.to_csv(output_dir / f"{name}.csv", index=False)
    return frames


if __name__ == "__main__":
    generate_simulated_data()
    print(f"Simulated data generated in {SIMULATED_DIR}")

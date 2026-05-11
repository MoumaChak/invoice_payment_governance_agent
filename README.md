# AI-Powered Invoice Payment Governance & Early Payment Leakage Control Agent

## 1. Project Overview

This is a complete local enterprise AI/analytics POC for detecting and preventing early invoice payments before cash leaves the system. It simulates invoice, vendor, PO, contract, SAP, Cora workflow, UPR, OCR, audit, and follow-up data, then trains ML models, scores invoices, generates audit logs, and launches a Streamlit dashboard.

## 2. Business Problem

Large AP operations may pay vendor invoices before their contractual due date. Early payment releases cash too soon, reduces working capital benefit, and can hide operational control issues such as manual queue clearing, SAP/Cora date mapping defects, or undocumented urgent payment requests.

The objective is simple: **pay on the due date — not before, not after.**

## 3. Solution Modules

- Data simulation layer for 5,000 invoices
- Data ingestion and validation layer
- Feature engineering layer
- Deterministic AP governance rule engine
- M1 early payment risk scorer
- M2 root cause classifier
- M3 OCR/document extraction simulator
- M4 due date recommender
- M5 leakage forecasting module
- Template-based GenAI explanation and email layer
- Agentic workflow layer
- Streamlit dashboard
- Audit trail and follow-up tracker

## 4. DS/ML Models Implemented

### M1: Early Payment Risk Scorer
Binary classification model predicting whether an invoice is likely to be paid early.

Models trained and compared:
- Logistic Regression
- Random Forest
- XGBoost, if installed
- LightGBM, if installed
- PyTorch feed-forward neural network, if installed

Metrics:
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix
- Classification report

### M2: Root Cause Classifier
Multi-class classifier for:
- UPR
- System error
- Behavioural
- Insufficient audit trail

Models:
- TF-IDF + Logistic Regression using workflow notes
- Structured Random Forest

### M3: OCR Extraction Simulator
Regex-based extraction from simulated invoice text for:
- Due date
- Payment term
- Vendor ID
- Invoice amount
- Invoice date
- PO number
- Currency
- Contract reference

### M4: Due Date Recommender
Uses source-of-truth hierarchy:
1. Contract payment term
2. PO payment term
3. Vendor master payment term
4. Invoice face payment term
5. SAP due date

### M5: Forecasting
Forecasts monthly early payment count, early payment value, leakage, and intervention-retained value. Prophet is used if available; otherwise a safe fallback method is used.

## 5. Folder Structure

```text
invoice_payment_governance_agent/
│
├── app.py
├── run_pipeline.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── simulated/
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_generator.py
│   ├── data_ingestion.py
│   ├── data_validation.py
│   ├── feature_engineering.py
│   ├── rule_engine.py
│   ├── model_training.py
│   ├── model_scoring.py
│   ├── due_date_recommender.py
│   ├── ocr_extraction_simulator.py
│   ├── root_cause_classifier.py
│   ├── forecasting.py
│   ├── genai_templates.py
│   ├── agents.py
│   ├── audit_logger.py
│   └── utils.py
│
├── models/
│   ├── .gitkeep
│
└── outputs/
    ├── .gitkeep
```

## 6. Setup Instructions

```bash
cd invoice_payment_governance_agent
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Mac/Linux

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Optional libraries such as XGBoost, LightGBM, PyTorch, and Prophet are handled gracefully. If one of them is unavailable, the pipeline skips it and continues.

## 7. How to Run Locally

Run the full pipeline:

```bash
python run_pipeline.py
```

Launch the dashboard:

```bash
streamlit run app.py
```

## 8. Demo Walkthrough

Use this leadership demo storyline:

1. Open the Executive Overview page.
2. Show total invoices, early payment value, working capital at risk, retained working capital, value-share revenue, and ROI multiple.
3. Move to Risk Queue and filter high-risk invoices.
4. Pick an invoice with HOLD or ESCALATE action.
5. Open Invoice Detail View.
6. Show scheduled release date versus contractual due date.
7. Show rule flags such as early release, missing UPR, SAP due date mismatch, and manual override.
8. Show the GenAI-style explanation.
9. Show the email draft to AP processor.
10. Move to Root Cause Analytics and show whether issues are UPR, System error, Behavioural, or Insufficient audit trail.
11. Move to Follow-up Tracker and show governance closure loop.
12. Move to Audit Trail and show evidence of control and traceability.
13. End with Forecasting page to show future leakage and retained value with intervention.

## 9. Success Metrics

### M1 Early Payment Risk Scorer
Target:
- Precision > 80%
- Recall > 75%
- F1 > 0.77
- ROC-AUC reported

### M2 Root Cause Classifier
Target:
- Macro F1 > 0.70
- Per-class recall > 65%

### M3 OCR Extraction
Target:
- Field extraction accuracy > 92%

### M4 Due Date Recommender
Target:
- MAE < 2 days
- High percentage of recommendations within +/- 2 days

### Business KPIs
- Early payment exposure
- Retained working capital
- Genpact value-share revenue
- Net value after hosting cost
- ROI multiple
- Number of invoices held/deferred/escalated
- Audit-gap root cause reduction opportunity

## 10. Troubleshooting

### Missing output files
Run:

```bash
python run_pipeline.py
```

### Streamlit duplicate widget error
All widgets use unique keys. If you modify the app, ensure every widget has a unique `key`.

### Optional library missing
The code skips optional models if the package is not available. For faster setup, you can temporarily remove heavy libraries like `prophet`, `torch`, `lightgbm`, or `xgboost` from `requirements.txt`. The app still runs with scikit-learn.

### Pipeline seems slow
Training XGBoost, LightGBM, PyTorch, and Prophet can take longer on low-memory machines. You can reduce `N_INVOICES` in `src/config.py` for quick demos.

## 11. Resume Bullet Points

- Built an end-to-end AI-powered Invoice Payment Governance Agent to detect and prevent premature vendor payments before cash release, reducing working-capital leakage risk across AP workflows.
- Developed a full ML pipeline using simulated SAP/Cora invoice data, rule engines, Logistic Regression, Random Forest, optional XGBoost/LightGBM, and PyTorch to score early payment risk and recommend hold/defer/escalation actions.
- Designed a root cause classification module to categorize early payments into UPR, system error, behavioural, and insufficient-audit-trail buckets, enabling targeted AP process correction and governance reporting.
- Implemented due-date governance using contract/PO/vendor/invoice/SAP source-of-truth hierarchy, OCR-style invoice field extraction, audit trail generation, follow-up tracking, and GenAI-style explanation/email templates.
- Built an interactive Streamlit leadership dashboard showing risk queue, invoice-level explanations, root cause analytics, model performance, leakage forecasting, and enterprise value-share ROI simulation.

## 12. Interview Explanation Points

- The project is not price prediction; it is AP governance and working-capital control.
- The business goal is to intercept risky early payments before payment release.
- Rules provide deterministic governance; ML adds prioritization and scalability.
- The early payment model is optimized for F1 and recall because missing risky invoices can cause cash leakage.
- Root cause classification converts detection into actionable process improvement.
- GenAI-style templates make outputs explainable for AP processors, finance leads, and vendor managers without requiring paid APIs.
- The audit trail and follow-up tracker make the POC enterprise-ready because every action is traceable.

## Optional Heavyweight Models

The project implements XGBoost, LightGBM, PyTorch, and Prophet hooks. For a fast default POC run, heavyweight optional model training is skipped unless explicitly enabled.

Enable optional ML models:

```bash
ENABLE_OPTIONAL_MODELS=1 python run_pipeline.py
```

Enable Prophet forecasting:

```bash
ENABLE_PROPHET=1 python run_pipeline.py
```

On Windows PowerShell:

```powershell
$env:ENABLE_OPTIONAL_MODELS="1"
python run_pipeline.py
```

## Model realism notes

This version uses a scenario-driven simulated data generator so the dashboard covers multiple AP governance situations:

- Normal on-time payment
- Low-risk standard processing
- Legitimate UPR early payment
- Cora/SAP system error
- Behavioural queue clearance
- Payment term mismatch
- Repeat vendor pattern
- High-value manual override
- Audit-gap early release

The M1 early payment model intentionally excludes direct target-leakage fields such as `days_paid_early`, `days_to_contractual_due_date`, and `upr_missing_but_early_flag` from model training. These fields remain available for the rule engine and dashboard, but they are not used as ML predictors. This prevents unrealistic 100% model performance and makes the model comparison more credible.

XGBoost and LightGBM train automatically when installed. PyTorch is implemented but opt-in to keep the default pipeline fast on laptops:

```bash
ENABLE_TORCH=1 python run_pipeline.py
```

For a faster run without optional models:

```bash
DISABLE_OPTIONAL_MODELS=1 python run_pipeline.py
```

## v7 Enhancement Notes

This version includes additional model-governance improvements:

- M2 root-cause classifier now trains only on early-payment cases and avoids direct leakage fields, making root-cause performance more realistic.
- M1 now generates scenario-wise performance in `outputs/scenario_m1_metrics.csv`.
- M1 now generates a benchmark comparison between a legacy rule guardrail, the best ML model, and the target threshold in `outputs/benchmark_metrics.csv`.
- M4 due-date recommender now outputs top adjustment drivers in `outputs/due_date_feature_effects.csv`.
- Streamlit now includes an Explainability page showing M1 risk drivers, M2 evidence, and M4 due-date adjustment logic.

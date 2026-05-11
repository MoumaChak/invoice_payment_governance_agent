from pathlib import Path
import traceback

from src.agents import FollowUpEmailAgent, GovernanceReportingAgent, InvoiceRiskMonitoringAgent, RootCauseDiagnosisAgent
from src.config import N_INVOICES, OUTPUTS_DIR, PROCESSED_DIR, SIMULATED_DIR
from src.data_generator import generate_simulated_data
from src.data_ingestion import load_simulated_data
from src.data_validation import validate_datasets
from src.due_date_recommender import train_due_date_recommender
from src.feature_engineering import build_feature_table
from src.forecasting import create_leakage_forecast
from src.model_training import train_early_payment_models
from src.ocr_extraction_simulator import run_ocr_extraction
from src.root_cause_classifier import train_root_cause_models


def main() -> None:
    print("[1/9] Generating simulated enterprise data...")
    generate_simulated_data(n_invoices=N_INVOICES, output_dir=SIMULATED_DIR)

    print("[2/9] Loading simulated data...")
    datasets = load_simulated_data(SIMULATED_DIR)

    print("[3/9] Validating data quality...")
    valid, report = validate_datasets(datasets)
    report.to_csv(OUTPUTS_DIR / "data_validation_report.csv", index=False)
    if not valid:
        print(report)
        raise ValueError("Data validation failed. See outputs/data_validation_report.csv")

    print("[4/9] Building feature table...")
    features = build_feature_table(datasets, PROCESSED_DIR)

    print("[5/9] Training M1 early payment risk models...")
    train_early_payment_models(features)

    print("[6/9] Training M2 root cause classifier...")
    train_root_cause_models(features)

    print("[7/9] Running M3 OCR extraction and M4 due date recommender...")
    run_ocr_extraction()
    features = train_due_date_recommender(features)

    print("[8/9] Scoring invoices, generating audit trail and follow-ups...")
    scored = InvoiceRiskMonitoringAgent(risk_threshold=0.60).run(features)
    scored = RootCauseDiagnosisAgent().run(scored)
    FollowUpEmailAgent().run(scored)
    GovernanceReportingAgent().run(scored)

    print("[9/9] Creating monthly leakage forecast...")
    create_leakage_forecast(scored)

    print("\nPipeline completed successfully.")
    print(f"Processed feature table: {PROCESSED_DIR / 'invoice_model_table.csv'}")
    print(f"Scored invoices: {OUTPUTS_DIR / 'scored_invoices.csv'}")
    print(f"Flagged invoices: {OUTPUTS_DIR / 'flagged_invoices.csv'}")
    print("Run the Streamlit app with: streamlit run app.py")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("Pipeline failed. Error details below:")
        traceback.print_exc()
        raise

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SIMULATED_DIR = DATA_DIR / "simulated"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

RANDOM_SEED = 42
N_INVOICES = 5000

SPEND_BASE = 38_000_000_000
EARLY_PAYMENT_RATE = 0.02
CAPTURE_RATE = 0.40
VALUE_SHARE = 0.50
HOSTING_COST = 122_000
RISK_THRESHOLD = 0.60
EARLY_DAYS_THRESHOLD = 15

MARKET_RISK = {
    "India": 0.35,
    "USA": 0.25,
    "UK": 0.22,
    "Brazil": 0.42,
    "Germany": 0.18,
    "Indonesia": 0.38,
    "South Africa": 0.40,
    "Mexico": 0.36,
}

PAYMENT_TERMS = [30, 45, 60, 90]
CURRENCIES = ["USD", "EUR", "GBP", "INR", "BRL", "IDR", "MXN", "ZAR"]
MARKETS = list(MARKET_RISK.keys())
ROOT_CAUSES = ["UPR", "System error", "Behavioural", "Insufficient audit trail"]

REQUIRED_FILES = {
    "invoice_master": "invoice_master.csv",
    "vendor_master": "vendor_master.csv",
    "po_master": "po_master.csv",
    "contract_terms": "contract_terms.csv",
    "sap_payment_data": "sap_payment_data.csv",
    "cora_workflow_queue": "cora_workflow_queue.csv",
    "processor_actions": "processor_actions.csv",
    "upr_requests": "upr_requests.csv",
    "ocr_invoice_text": "ocr_invoice_text.csv",
}

for _path in [RAW_DIR, SIMULATED_DIR, PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR]:
    _path.mkdir(parents=True, exist_ok=True)

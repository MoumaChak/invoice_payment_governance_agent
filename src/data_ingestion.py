from pathlib import Path
from typing import Dict

import pandas as pd

from src.config import REQUIRED_FILES, SIMULATED_DIR
from src.utils import safe_read_csv

DATE_COLUMNS = {
    "invoice_master": ["invoice_date", "invoice_receipt_date"],
    "po_master": ["po_date"],
    "contract_terms": ["contract_start_date", "contract_end_date"],
    "sap_payment_data": ["sap_due_date", "sap_payment_release_date"],
    "cora_workflow_queue": ["queue_entry_date"],
}


def load_simulated_data(input_dir: Path = SIMULATED_DIR) -> Dict[str, pd.DataFrame]:
    datasets = {}
    missing = []
    for key, filename in REQUIRED_FILES.items():
        path = input_dir / filename
        if not path.exists():
            missing.append(str(path))
            continue
        datasets[key] = safe_read_csv(path, parse_dates=DATE_COLUMNS.get(key))
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))
    return datasets

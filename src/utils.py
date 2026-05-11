import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_read_csv(path: Path, parse_dates=None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path, parse_dates=parse_dates)
    if df.empty:
        raise ValueError(f"Dataset is empty: {path}")
    return df


def save_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path: Path, default=None):
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def money(value: float) -> str:
    try:
        return f"${value:,.0f}"
    except Exception:
        return "$0"


def pct(value: float) -> str:
    try:
        return f"{value * 100:.1f}%"
    except Exception:
        return "0.0%"


def value_calculation(spend_base: float, early_payment_rate: float, capture_rate: float,
                      value_share: float, hosting_cost: float) -> dict:
    exposure = spend_base * early_payment_rate
    retained = exposure * capture_rate
    revenue = retained * value_share
    net_value = revenue - hosting_cost
    roi_multiple = revenue / hosting_cost if hosting_cost else 0
    return {
        "early_payment_exposure": exposure,
        "retained_working_capital": retained,
        "genpact_value_share_revenue": revenue,
        "net_value": net_value,
        "roi_multiple": roi_multiple,
    }

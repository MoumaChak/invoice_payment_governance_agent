from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from src.config import MODELS_DIR, OUTPUTS_DIR, RANDOM_SEED

NUMERIC = [
    "invoice_payment_term",
    "po_payment_term",
    "vendor_payment_term",
    "contract_payment_term",
    "sap_contract_due_date_delta",
    "queue_age_days",
    "amount_log",
    "market_risk_score",
    "processor_manual_override_rate",
    "vendor_early_payment_rate",
    "payment_term_mismatch_flag",
    "upr_flag",
    "high_value_invoice_flag",
    "sap_error_present_flag",
]
CATEGORICAL = ["agreement_type", "invoice_priority", "market", "root_cause_label"]


def apply_due_date_hierarchy(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["rule_based_term"] = (
        out["contract_payment_term"]
        .combine_first(out["po_payment_term"])
        .combine_first(out["vendor_payment_term"])
        .combine_first(out["invoice_payment_term"])
    )
    out["rule_based_recommended_release_date"] = pd.to_datetime(out["invoice_receipt_date"]) + pd.to_timedelta(
        out["rule_based_term"].fillna(30).astype(int), unit="D"
    )
    return out


def _simulate_adjustment_target(data: pd.DataFrame, seed: int = RANDOM_SEED) -> np.ndarray:
    """Create a realistic residual adjustment target for M4.

    Business logic:
    - Base recommendation comes from the hierarchy.
    - Some invoices still require small operational corrections due to exceptions,
      UPRs, term mismatches, or system-date issues.
    - This creates a non-trivial but learnable regression target.
    """
    rng = np.random.default_rng(seed)
    n = len(data)
    adj = np.zeros(n, dtype=float)

    upr_mask = data["root_cause_label"].eq("UPR") & data["upr_flag"].fillna(0).eq(1)
    system_mask = data["root_cause_label"].eq("System error")
    behavioural_mask = data["root_cause_label"].eq("Behavioural")
    mismatch_mask = data["payment_term_mismatch_flag"].fillna(0).eq(1)
    high_value_mask = data["high_value_invoice_flag"].fillna(0).eq(1)
    strategic_mask = data.get("agreement_type", pd.Series(index=data.index, dtype=object)).astype(str).eq("Strategic")
    priority_mask = data.get("invoice_priority", pd.Series(index=data.index, dtype=object)).astype(str).isin(["High", "Critical"])
    sap_error_mask = data["sap_error_present_flag"].fillna(0).eq(1)

    # Negative means recommend slightly earlier than pure contractual date (e.g., justified UPR/strategic exception).
    adj += np.where(upr_mask, rng.normal(-2.5, 1.1, n), 0)
    adj += np.where(strategic_mask & priority_mask, rng.normal(-1.0, 0.8, n), 0)

    # Positive means release should be pushed later / corrected because current schedule is unreliable.
    adj += np.where(system_mask, rng.normal(2.8, 1.2, n), 0)
    adj += np.where(mismatch_mask, rng.normal(1.8, 1.0, n), 0)
    adj += np.where(sap_error_mask, rng.normal(1.4, 0.8, n), 0)
    adj += np.where(behavioural_mask & high_value_mask, rng.normal(1.2, 0.7, n), 0)

    # Small market/noise effects to avoid perfect predictability.
    adj += data["market_risk_score"].fillna(0).to_numpy() * rng.normal(1.0, 0.25, n)
    adj += rng.normal(0, 0.9, n)

    return np.clip(np.round(adj), -6, 8).astype(int)


def _prepare_matrix(data: pd.DataFrame) -> pd.DataFrame:
    X_num = data[NUMERIC].apply(pd.to_numeric, errors="coerce").fillna(0)
    X_cat = pd.get_dummies(data[CATEGORICAL].fillna("Missing").astype(str), drop_first=False)
    return pd.concat([X_num, X_cat], axis=1)


def train_due_date_recommender(df: pd.DataFrame, models_dir: Path = MODELS_DIR, outputs_dir: Path = OUTPUTS_DIR) -> pd.DataFrame:
    models_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    data = apply_due_date_hierarchy(df)
    data["contractual_due_date"] = pd.to_datetime(data["contractual_due_date"])
    data["rule_based_recommended_release_date"] = pd.to_datetime(data["rule_based_recommended_release_date"])

    data["target_adjustment_days"] = _simulate_adjustment_target(data)
    data["true_recommended_release_date"] = data["rule_based_recommended_release_date"] + pd.to_timedelta(
        data["target_adjustment_days"], unit="D"
    )

    X = _prepare_matrix(data)
    y = data["target_adjustment_days"].astype(float)

    split = int(len(data) * 0.75)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = LinearRegression()
    model.fit(X_train, y_train)

    pred_test = np.round(model.predict(X_test)) if len(X_test) else np.array([])
    mae = float(mean_absolute_error(y_test, pred_test)) if len(X_test) else 0.0
    within_2 = float((np.abs(y_test.to_numpy() - pred_test) <= 2).mean()) if len(X_test) else 1.0

    pred_all = np.round(model.predict(X)).astype(int)
    # Confidence proxy: smaller residual-prone changes when magnitude is modest.
    confidence = 1 - np.clip(np.abs(pred_all - data["target_adjustment_days"].to_numpy()) / 6, 0, 1)
    use_override = confidence >= 0.55

    data["predicted_adjustment_days"] = pred_all
    data["override_confidence"] = np.round(confidence, 4)
    data["recommended_release_date"] = np.where(
        use_override,
        (data["rule_based_recommended_release_date"] + pd.to_timedelta(data["predicted_adjustment_days"], unit="D")).astype(str),
        data["rule_based_recommended_release_date"].astype(str),
    )
    data["recommended_release_date"] = pd.to_datetime(data["recommended_release_date"], errors="coerce")

    metrics = pd.DataFrame([
        {
            "model": "Rule Engine + Linear Regression Override",
            "mae_days": round(mae, 3),
            "within_plus_minus_2_days": round(within_2, 4),
        }
    ])
    metrics.to_csv(outputs_dir / "due_date_metrics.csv", index=False)
    data.to_csv(outputs_dir / "due_date_recommendations.csv", index=False)

    effects = pd.DataFrame({
        "feature": list(X.columns),
        "coefficient": model.coef_.tolist(),
    })
    effects["abs_coefficient"] = effects["coefficient"].abs()
    effects.sort_values("abs_coefficient", ascending=False).head(30).to_csv(outputs_dir / "due_date_feature_effects.csv", index=False)

    joblib.dump({"model": model, "columns": list(X.columns)}, models_dir / "due_date_model.pkl")

    # Return enriched feature table so downstream scoring/dashboard sees the improved recommendation.
    return data

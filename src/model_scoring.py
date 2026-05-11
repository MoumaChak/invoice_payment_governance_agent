from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import MODELS_DIR, OUTPUTS_DIR, RISK_THRESHOLD
from src.genai_templates import invoice_risk_explanation, root_cause_narrative
from src.model_training import M1_CATEGORICAL, M1_NUMERIC
from src.root_cause_classifier import predict_root_cause
from src.rule_engine import apply_rules


def _risk_scores(df: pd.DataFrame, models_dir: Path) -> np.ndarray:
    model_path = models_dir / "early_payment_model.pkl"
    if not model_path.exists():
        # Fallback rule-like score if model is not trained yet.
        score = (
            0.18 * df["vendor_early_payment_rate"].fillna(0).clip(0, 1) +
            0.16 * df["processor_early_release_rate"].fillna(0).clip(0, 1) +
            0.14 * df["processor_manual_override_rate"].fillna(0).clip(0, 1) +
            0.14 * df["manual_override_flag"].fillna(0) +
            0.13 * df["sap_error_present_flag"].fillna(0) +
            0.10 * df["payment_term_mismatch_flag"].fillna(0) +
            0.08 * df["high_value_invoice_flag"].fillna(0) +
            0.07 * df["market_risk_score"].fillna(0).clip(0, 1)
        ).clip(0, 1)
        return score.values
    model = joblib.load(model_path)
    X = df[M1_NUMERIC + M1_CATEGORICAL]
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.predict(X)


def score_invoices(df: pd.DataFrame, models_dir: Path = MODELS_DIR, outputs_dir: Path = OUTPUTS_DIR, risk_threshold: float = RISK_THRESHOLD) -> pd.DataFrame:
    outputs_dir.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out["risk_score"] = _risk_scores(out, models_dir)
    out["predicted_root_cause"] = predict_root_cause(out, models_dir)
    out = apply_rules(out)
    out["risk_band"] = pd.cut(out["risk_score"], bins=[-0.01, 0.35, 0.60, 0.80, 1.01], labels=["Low", "Medium", "High", "Critical"])
    out["risk_explanation"] = out.apply(invoice_risk_explanation, axis=1)
    out["root_cause_narrative"] = out.apply(root_cause_narrative, axis=1)
    out["is_flagged"] = ((out["risk_score"] >= risk_threshold) | (out["recommended_action"].isin(["HOLD", "ESCALATE", "DEFER"]))).astype(int)
    out.to_csv(outputs_dir / "scored_invoices.csv", index=False)
    out[out["is_flagged"] == 1].to_csv(outputs_dir / "flagged_invoices.csv", index=False)
    return out

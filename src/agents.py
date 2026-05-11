from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.audit_logger import generate_audit_trail, generate_follow_up_tracker
from src.config import OUTPUTS_DIR
from src.genai_templates import email_to_ap_processor, escalation_note, root_cause_narrative
from src.model_scoring import score_invoices
from src.utils import value_calculation


@dataclass
class InvoiceRiskMonitoringAgent:
    risk_threshold: float = 0.60

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        return score_invoices(df, risk_threshold=self.risk_threshold)


@dataclass
class RootCauseDiagnosisAgent:
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["root_cause_narrative"] = out.apply(root_cause_narrative, axis=1)
        return out


@dataclass
class FollowUpEmailAgent:
    outputs_dir: Path = OUTPUTS_DIR

    def run(self, scored_df: pd.DataFrame) -> pd.DataFrame:
        out = scored_df.copy()
        out["processor_email_draft"] = out.apply(email_to_ap_processor, axis=1)
        out["finance_escalation_note"] = out.apply(escalation_note, axis=1)
        generate_audit_trail(out, self.outputs_dir)
        return generate_follow_up_tracker(out, self.outputs_dir)


@dataclass
class GovernanceReportingAgent:
    spend_base: float = 38_000_000_000
    early_payment_rate: float = 0.02
    capture_rate: float = 0.40
    value_share: float = 0.50
    hosting_cost: float = 122_000

    def run(self, scored_df: pd.DataFrame) -> dict:
        early = scored_df[scored_df["early_payment_flag"] == 1]
        value = value_calculation(self.spend_base, self.early_payment_rate, self.capture_rate, self.value_share, self.hosting_cost)
        insights = {
            "total_invoices": int(len(scored_df)),
            "total_invoice_value": float(scored_df["invoice_amount"].sum()),
            "early_payment_count": int(len(early)),
            "early_payment_value": float(early["invoice_amount"].sum()),
            "flagged_count": int(scored_df["is_flagged"].sum()) if "is_flagged" in scored_df else 0,
            **value,
        }
        pd.DataFrame([insights]).to_csv(OUTPUTS_DIR / "governance_summary.csv", index=False)
        return insights

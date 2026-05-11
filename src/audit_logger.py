from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from src.config import OUTPUTS_DIR
from src.genai_templates import email_to_ap_processor, follow_up_reminder


def generate_audit_trail(scored_df: pd.DataFrame, outputs_dir: Path = OUTPUTS_DIR) -> pd.DataFrame:
    outputs_dir.mkdir(parents=True, exist_ok=True)
    candidates = scored_df[(scored_df["recommended_action"].isin(["HOLD", "ESCALATE", "DEFER"])) | (scored_df["risk_score"] >= 0.60)].copy()
    rows = []
    now = datetime.now().replace(microsecond=0)
    for i, (_, row) in enumerate(candidates.iterrows(), start=1):
        rows.append({
            "audit_id": f"AUD{i:08d}",
            "invoice_id": row["invoice_id"],
            "action_taken": row["recommended_action"],
            "triggered_by": "ML Risk Scorer + Rule Engine",
            "timestamp": now + timedelta(seconds=i),
            "root_cause": row.get("predicted_root_cause", row.get("root_cause_label", "Insufficient audit trail")),
            "recommendation": f"{row['recommended_action']} until {row.get('recommended_release_date')}",
            "risk_score": round(float(row.get("risk_score", 0)), 4),
            "final_status": "Open" if row["recommended_action"] in ["HOLD", "ESCALATE", "DEFER"] else "Closed",
            "comments": row.get("risk_explanation", "Auto-generated audit entry"),
        })
    audit = pd.DataFrame(rows)
    audit.to_csv(outputs_dir / "audit_trail.csv", index=False)
    return audit


def generate_follow_up_tracker(scored_df: pd.DataFrame, outputs_dir: Path = OUTPUTS_DIR) -> pd.DataFrame:
    outputs_dir.mkdir(parents=True, exist_ok=True)
    candidates = scored_df[scored_df["recommended_action"].isin(["HOLD", "ESCALATE", "DEFER"])].copy()
    today = datetime.now().date()
    rows = []
    for i, (_, row) in enumerate(candidates.iterrows(), start=1):
        status = "Escalated" if row["recommended_action"] == "ESCALATE" else "Open"
        rows.append({
            "follow_up_id": f"FUP{i:08d}",
            "invoice_id": row["invoice_id"],
            "owner": "Finance Lead" if row["recommended_action"] == "ESCALATE" else "AP Processor",
            "follow_up_type": "Escalation" if row["recommended_action"] == "ESCALATE" else "Email",
            "follow_up_status": status,
            "created_date": today,
            "due_date": today + timedelta(days=2 if row["recommended_action"] == "ESCALATE" else 3),
            "aging_days": 0,
            "latest_response": "Pending response",
            "next_action": "Confirm approval evidence or hold payment",
            "generated_email_text": email_to_ap_processor(row),
            "reminder_text": follow_up_reminder(row),
        })
    follow = pd.DataFrame(rows)
    follow.to_csv(outputs_dir / "follow_up_tracker.csv", index=False)
    return follow

from __future__ import annotations

import pandas as pd


def invoice_risk_explanation(row: pd.Series) -> str:
    return (
        f"Invoice {row.get('invoice_id')} has been flagged because it is scheduled to release "
        f"{int(row.get('days_paid_early', 0))} days before the contractual due date, "
        f"UPR flag is {'missing' if int(row.get('upr_flag', 0)) == 0 else 'present'}, "
        f"manual override is {'present' if int(row.get('manual_override_flag', 0)) == 1 else 'not present'}, "
        f"and the vendor historical early payment rate is {float(row.get('vendor_early_payment_rate', 0)):.1%}. "
        f"Recommended action: {row.get('recommended_action', 'Review')}."
    )


def root_cause_narrative(row: pd.Series) -> str:
    rc = row.get("predicted_root_cause", row.get("root_cause_label", "Insufficient audit trail"))
    if rc == "Not applicable / on-time":
        return "No early-payment root cause is required because the invoice is not scheduled before the contractual due date."
    if rc == "UPR":
        return "The invoice appears linked to an urgent payment request. Governance should verify that the request is approved and documented before release."
    if rc == "System error":
        return "The invoice shows indicators of SAP/Cora date or payment term logic issues. The due date mapping should be corrected before payment release."
    if rc == "Behavioural":
        return "The invoice appears to be driven by manual release behaviour, likely related to queue clearance or backlog pressure. AP leadership should review processor behaviour."
    return "The invoice lacks enough audit evidence to explain the early release. Treat this as a priority governance case until ownership is confirmed."


def email_to_ap_processor(row: pd.Series) -> str:
    return f"""Subject: Action Required - Early Payment Review for {row.get('invoice_id')}

Hi AP Team,

Invoice {row.get('invoice_id')} for vendor {row.get('vendor_name', row.get('vendor_id'))} is scheduled to release {int(row.get('days_paid_early', 0))} days before the contractual due date.

Risk score: {float(row.get('risk_score', 0)):.2f}
Root cause: {row.get('predicted_root_cause', row.get('root_cause_label', 'Insufficient audit trail'))}
Recommended action: {row.get('recommended_action', 'Review')}
Recommended release date: {row.get('recommended_release_date')}

Please confirm whether there is a documented business-approved UPR or whether this payment should be held until the correct release date.

Regards,
Invoice Payment Governance Agent
"""


def email_to_vendor_manager(row: pd.Series) -> str:
    return f"""Subject: Vendor Payment Governance Review - {row.get('vendor_name', row.get('vendor_id'))}

Hi Vendor Management Team,

Vendor {row.get('vendor_name', row.get('vendor_id'))} has an invoice flagged for potential early payment leakage.

Invoice: {row.get('invoice_id')}
Amount: {row.get('currency', '')} {float(row.get('invoice_amount', 0)):,.2f}
Days early: {int(row.get('days_paid_early', 0))}
Vendor historical early payment rate: {float(row.get('vendor_early_payment_rate', 0)):.1%}

Please validate if any commercial exception exists.
"""


def escalation_note(row: pd.Series) -> str:
    return (
        f"Escalation: Invoice {row.get('invoice_id')} is high-risk with action {row.get('recommended_action')}. "
        f"Amount is {row.get('currency', '')} {float(row.get('invoice_amount', 0)):,.2f}, "
        f"scheduled {int(row.get('days_paid_early', 0))} days early, root cause is "
        f"{row.get('predicted_root_cause', row.get('root_cause_label', 'Insufficient audit trail'))}. Leadership review is recommended."
    )


def follow_up_reminder(row: pd.Series) -> str:
    return f"Reminder: Follow-up pending for invoice {row.get('invoice_id')}. Please confirm hold/release decision and attach supporting approval evidence."


def chat_assistant_response(question: str, row: pd.Series | None = None) -> str:
    q = (question or "").lower()
    if row is None:
        return "I can answer invoice-level questions once you select an invoice from the dashboard."
    if "why" in q or "flag" in q:
        return invoice_risk_explanation(row)
    if "root" in q or "cause" in q:
        return root_cause_narrative(row)
    if "action" in q or "recommend" in q:
        return f"Recommended action is {row.get('recommended_action')} with release date {row.get('recommended_release_date')}."
    return invoice_risk_explanation(row)

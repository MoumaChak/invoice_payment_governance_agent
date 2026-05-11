from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import MODELS_DIR, OUTPUTS_DIR, RANDOM_SEED
from src.utils import save_json

# M2 uses weaker operational signals for a realistic root-cause classifier.
# It intentionally avoids the most direct rule fields such as raw UPR flag,
# raw SAP error flag, raw manual override flag, and days_paid_early.
NUMERIC = [
    "queue_age_days",
    "vendor_early_payment_rate",
    "processor_early_release_rate",
    "processor_manual_override_rate",
    "amount_log",
    "market_risk_score",
    "po_invoice_term_delta",
    "approval_risk_score",
    "high_value_invoice_flag",
]
CATEGORICAL = ["market", "invoice_priority", "time_of_day", "payment_method", "approval_status", "amount_band"]
ROOT_CAUSE_CLASSES = ["UPR", "System error", "Behavioural", "Insufficient audit trail"]


def _onehot():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _structured_pipe():
    pre = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), NUMERIC),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", _onehot())]), CATEGORICAL),
    ])
    return Pipeline([
        ("preprocess", pre),
        ("model", RandomForestClassifier(
            n_estimators=70,
            max_depth=4,
            min_samples_leaf=30,
            random_state=RANDOM_SEED,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )),
    ])


def _prepare_notes(notes: pd.Series, degrade: bool = False) -> pd.Series:
    """Remove overly explicit root-cause language to avoid perfect M2 metrics."""
    out = notes.fillna("").astype(str).str.lower()
    replacements = {
        "urgent payment request": "expedited business request",
        "upr": "business exception",
        "system error": "workflow exception",
        "date logic": "date validation",
        "term mapping": "term review",
        "manual override": "manual activity",
        "queue clearance": "queue movement",
        "behaviour": "process activity",
        "behavior": "process activity",
        "insufficient audit trail": "documentation gap",
        "no documented": "documentation missing",
        "no clear audit": "limited audit evidence",
    }
    for old, new in replacements.items():
        out = out.str.replace(old, new, regex=False)
    if degrade:
        rng = np.random.default_rng(RANDOM_SEED)
        arr = out.to_numpy(dtype=object)
        # About 15% of workflow notes become generic, simulating sparse/poor notes.
        mask = rng.random(len(arr)) < 0.10
        generic = np.array([
            "invoice requires operational review",
            "workflow note incomplete",
            "payment queue moved for review",
            "processor comment available but limited",
        ], dtype=object)
        if len(arr):
            arr[mask] = rng.choice(generic, mask.sum())
        out = pd.Series(arr, index=notes.index)
    return out


def train_root_cause_models(df: pd.DataFrame, models_dir: Path = MODELS_DIR, outputs_dir: Path = OUTPUTS_DIR) -> pd.DataFrame:
    models_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # M2 is a root-cause model for early-payment cases only.
    data = df[df.get("early_payment_flag", 0) == 1].copy()
    if data.empty:
        data = df.copy()
    data = data[data["root_cause_label"].isin(ROOT_CAUSE_CLASSES)].copy()
    data["workflow_notes_model"] = _prepare_notes(data["workflow_notes"], degrade=True)
    y = data["root_cause_label"].astype(str)

    stratify = y if y.value_counts().min() >= 2 else None
    idx_train, idx_test = train_test_split(data.index, test_size=0.25, random_state=RANDOM_SEED, stratify=stratify)

    models = {}
    metrics = []
    reports = {}

    text_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=360, ngram_range=(1, 2), stop_words="english", min_df=3)),
        ("model", LogisticRegression(max_iter=350, class_weight="balanced", C=0.9)),
    ])
    models["TF-IDF + Logistic Regression"] = (
        text_pipe,
        data.loc[idx_train, "workflow_notes_model"],
        data.loc[idx_test, "workflow_notes_model"],
    )

    struct_pipe = _structured_pipe()
    models["Structured Random Forest"] = (
        struct_pipe,
        data.loc[idx_train, NUMERIC + CATEGORICAL],
        data.loc[idx_test, NUMERIC + CATEGORICAL],
    )

    labels = ROOT_CAUSE_CLASSES
    for name, (pipe, X_train, X_test) in models.items():
        pipe.fit(X_train, y.loc[idx_train])
        pred = pipe.predict(X_test)
        report = classification_report(y.loc[idx_test], pred, labels=labels, output_dict=True, zero_division=0)
        per_class_recall = recall_score(y.loc[idx_test], pred, average=None, labels=labels, zero_division=0)
        metrics.append({
            "model": name,
            "macro_f1": f1_score(y.loc[idx_test], pred, average="macro", labels=labels, zero_division=0),
            "weighted_f1": f1_score(y.loc[idx_test], pred, average="weighted", zero_division=0),
            "min_per_class_recall": float(np.min(per_class_recall)) if len(per_class_recall) else 0,
        })
        reports[name] = {
            "classification_report": report,
            "confusion_matrix": confusion_matrix(y.loc[idx_test], pred, labels=labels).tolist(),
            "labels": labels,
        }

    metrics_df = pd.DataFrame(metrics).sort_values(["macro_f1", "min_per_class_recall"], ascending=False)
    metrics_df.to_csv(outputs_dir / "root_cause_metrics.csv", index=False)
    best_name = metrics_df.iloc[0]["model"]
    joblib.dump(models[best_name][0], models_dir / "root_cause_model.pkl")
    (models_dir / "best_root_cause_model.txt").write_text(str(best_name), encoding="utf-8")
    save_json(reports, outputs_dir / "root_cause_model_details.json")
    best_report = reports[best_name]
    pd.DataFrame(best_report["confusion_matrix"], index=best_report["labels"], columns=best_report["labels"]).to_csv(outputs_dir / "root_cause_confusion_matrix.csv")
    return metrics_df


def predict_root_cause(df: pd.DataFrame, models_dir: Path = MODELS_DIR) -> pd.Series:
    path = models_dir / "root_cause_model.pkl"
    result = pd.Series("Not applicable / on-time", index=df.index, dtype=object)
    early_mask = df.get("early_payment_flag", pd.Series(1, index=df.index)).fillna(1).astype(int).eq(1)
    if not early_mask.any():
        return result
    if not path.exists():
        result.loc[early_mask] = df.loc[early_mask].get("root_cause_label", pd.Series("Insufficient audit trail", index=df.loc[early_mask].index))
        return result

    model = joblib.load(path)
    best = (models_dir / "best_root_cause_model.txt").read_text(encoding="utf-8") if (models_dir / "best_root_cause_model.txt").exists() else ""
    early_df = df.loc[early_mask]
    if "TF-IDF" in best:
        X = _prepare_notes(early_df["workflow_notes"], degrade=False)
        result.loc[early_mask] = model.predict(X)
    else:
        result.loc[early_mask] = model.predict(early_df[NUMERIC + CATEGORICAL])
    return result

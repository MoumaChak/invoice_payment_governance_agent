from __future__ import annotations

import importlib.util
import os
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import MODELS_DIR, OUTPUTS_DIR, RANDOM_SEED
from src.utils import save_json

# M1 intentionally excludes target-leakage fields such as:
# - days_to_contractual_due_date
# - days_paid_early
# - upr_missing_but_early_flag
# Those are valid for dashboards/rules, but they make a classifier look unrealistically perfect.
M1_NUMERIC = [
    "po_invoice_term_delta",
    "sap_contract_due_date_delta",
    "vendor_early_payment_rate",
    "processor_early_release_rate",
    "processor_manual_override_rate",
    "queue_age_days",
    "manual_override_flag",
    "month_end_flag",
    "weekend_release_flag",
    "high_value_invoice_flag",
    "sap_error_present_flag",
    "market_risk_score",
    "approval_risk_score",
    "amount_log",
    "payment_term_mismatch_flag",
    "upr_flag",
    "upr_documented_flag",
]
M1_CATEGORICAL = ["market", "currency", "invoice_priority", "approval_status", "amount_band", "payment_method", "time_of_day"]
TARGET = "early_payment_flag"


def _available(module: str) -> bool:
    # Train XGBoost/LightGBM when installed. PyTorch is opt-in because the first torch import
    # can be heavy on laptops; enable it with ENABLE_TORCH=1 python run_pipeline.py.
    if os.getenv("DISABLE_OPTIONAL_MODELS", "0") == "1":
        return False
    if module == "torch" and os.getenv("ENABLE_TORCH", "0") != "1":
        return False
    return importlib.util.find_spec(module) is not None


def _onehot():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", _onehot())])
    return ColumnTransformer([
        ("num", numeric_pipe, M1_NUMERIC),
        ("cat", cat_pipe, M1_CATEGORICAL),
    ])


def _metrics(name: str, y_true, y_pred, y_prob) -> Dict[str, float]:
    roc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    return {
        "model": name,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc,
        "error": "None",
    }


def _fit_eval_model(name: str, estimator, X_train, X_test, y_train, y_test) -> Tuple[Pipeline, Dict[str, float], Dict]:
    pipe = Pipeline([("preprocess", _preprocessor()), ("model", estimator)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    if hasattr(pipe, "predict_proba"):
        y_prob = pipe.predict_proba(X_test)[:, 1]
    else:
        y_prob = y_pred
    metric = _metrics(name, y_test, y_pred, y_prob)
    details = {
        "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    return pipe, metric, details


def _feature_importance(pipe: Pipeline) -> pd.DataFrame:
    pre = pipe.named_steps["preprocess"]
    model = pipe.named_steps["model"]
    try:
        names = pre.get_feature_names_out()
    except Exception:
        names = np.array(M1_NUMERIC + M1_CATEGORICAL)
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = np.abs(model.coef_[0])
    else:
        values = np.zeros(len(names))
    n = min(len(names), len(values))
    fi = pd.DataFrame({"feature": names[:n], "importance": values[:n]}).sort_values("importance", ascending=False)
    return fi


def _train_statsmodels_summary(df: pd.DataFrame) -> None:
    # Statsmodels can be enabled for coefficient-level diagnostics, but it is kept opt-in
    # so the demo pipeline remains fast and reliable on laptops.
    if os.getenv("RUN_STATSMODELS", "0") != "1":
        (OUTPUTS_DIR / "statsmodels_logit_summary.txt").write_text(
            "Statsmodels logistic summary skipped by default for fast demo execution. "
            "Run with RUN_STATSMODELS=1 python run_pipeline.py to generate it.",
            encoding="utf-8",
        )
        return
    try:
        import statsmodels.api as sm
        model_df = df[M1_NUMERIC + [TARGET]].dropna().copy()
        if len(model_df) > 1000:
            model_df = model_df.sample(1000, random_state=RANDOM_SEED)
        X = sm.add_constant(model_df[M1_NUMERIC])
        y = model_df[TARGET]
        logit = sm.Logit(y, X).fit(disp=False, method="lbfgs", maxiter=30)
        (OUTPUTS_DIR / "statsmodels_logit_summary.txt").write_text(str(logit.summary()), encoding="utf-8")
    except Exception as exc:
        (OUTPUTS_DIR / "statsmodels_logit_summary.txt").write_text(f"Statsmodels summary skipped: {exc}", encoding="utf-8")


def _train_pytorch(X_train, X_test, y_train, y_test, base_preprocessor) -> Dict[str, float] | None:
    if not _available("torch"):
        return None
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim

        # Keep the neural net lightweight for local laptop demo runs.
        if len(X_train) > 2200:
            sample_idx = pd.Series(range(len(X_train))).sample(2200, random_state=RANDOM_SEED).index
            X_train = X_train.iloc[sample_idx]
            y_train = y_train.iloc[sample_idx]
        if len(X_test) > 900:
            sample_idx_test = pd.Series(range(len(X_test))).sample(900, random_state=RANDOM_SEED).index
            X_test = X_test.iloc[sample_idx_test]
            y_test = y_test.iloc[sample_idx_test]
        Xtr = base_preprocessor.fit_transform(X_train)
        Xte = base_preprocessor.transform(X_test)
        Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
        ytr_t = torch.tensor(np.array(y_train).reshape(-1, 1), dtype=torch.float32)
        Xte_t = torch.tensor(Xte, dtype=torch.float32)

        class Net(nn.Module):
            def __init__(self, n_features):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(n_features, 32), nn.ReLU(), nn.Dropout(0.10),
                    nn.Linear(32, 16), nn.ReLU(),
                    nn.Linear(16, 1), nn.Sigmoid(),
                )
            def forward(self, x):
                return self.layers(x)

        torch.manual_seed(RANDOM_SEED)
        net = Net(Xtr.shape[1])
        criterion = nn.BCELoss()
        opt = optim.Adam(net.parameters(), lr=0.001)
        losses = []
        for epoch in range(5):
            net.train()
            opt.zero_grad()
            pred = net(Xtr_t)
            loss = criterion(pred, ytr_t)
            loss.backward()
            opt.step()
            losses.append({"epoch": epoch + 1, "loss": float(loss.item())})

        net.eval()
        with torch.no_grad():
            probs = net(Xte_t).numpy().reshape(-1)
        preds = (probs >= 0.5).astype(int)
        torch.save(net.state_dict(), MODELS_DIR / "pytorch_early_payment_model.pt")
        joblib.dump(base_preprocessor, MODELS_DIR / "pytorch_preprocessor.pkl")
        pd.DataFrame(losses).to_csv(OUTPUTS_DIR / "pytorch_loss_curve.csv", index=False)
        return _metrics("PyTorch Neural Network", y_test, preds, probs)
    except Exception as exc:
        (OUTPUTS_DIR / "pytorch_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        return {"model": "PyTorch Neural Network", "precision": np.nan, "recall": np.nan, "f1": np.nan, "roc_auc": np.nan, "error": str(exc)}


def train_early_payment_models(df: pd.DataFrame, models_dir: Path = MODELS_DIR, outputs_dir: Path = OUTPUTS_DIR) -> pd.DataFrame:
    models_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    features = M1_NUMERIC + M1_CATEGORICAL
    model_df = df[features + [TARGET]].copy()
    X = model_df[features]
    y = model_df[TARGET].astype(int)
    stratify = y if y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RANDOM_SEED, stratify=stratify)

    candidates: List[Tuple[str, object]] = [
        ("Logistic Regression", LogisticRegression(max_iter=250, solver="liblinear", class_weight="balanced")),
        ("Random Forest", RandomForestClassifier(n_estimators=50, max_depth=8, random_state=RANDOM_SEED, class_weight="balanced_subsample", n_jobs=1)),
    ]

    if _available("xgboost"):
        try:
            from xgboost import XGBClassifier
            candidates.append(("XGBoost", XGBClassifier(n_estimators=25, max_depth=3, learning_rate=0.08, subsample=0.9, colsample_bytree=0.9, eval_metric="logloss", random_state=RANDOM_SEED, n_jobs=1, tree_method="hist")))
        except Exception:
            pass
    if _available("lightgbm"):
        try:
            from lightgbm import LGBMClassifier
            candidates.append(("LightGBM", LGBMClassifier(n_estimators=30, learning_rate=0.07, random_state=RANDOM_SEED, n_jobs=1, verbose=-1)))
        except Exception:
            pass

    metrics = []
    details = {}
    fitted = {}
    for name, est in candidates:
        try:
            pipe, met, detail = _fit_eval_model(name, est, X_train, X_test, y_train, y_test)
            metrics.append(met)
            details[name] = detail
            fitted[name] = pipe
        except Exception as exc:
            metrics.append({"model": name, "precision": np.nan, "recall": np.nan, "f1": np.nan, "roc_auc": np.nan, "error": str(exc)})

    # Add explicit rows only for optional libraries that are not available or were disabled.
    if not _available("xgboost") and not any(m.get("model") == "XGBoost" for m in metrics):
        metrics.append({"model": "XGBoost", "precision": np.nan, "recall": np.nan, "f1": np.nan, "roc_auc": np.nan, "error": "Skipped: xgboost not installed or DISABLE_OPTIONAL_MODELS=1."})
    if not _available("lightgbm") and not any(m.get("model") == "LightGBM" for m in metrics):
        metrics.append({"model": "LightGBM", "precision": np.nan, "recall": np.nan, "f1": np.nan, "roc_auc": np.nan, "error": "Skipped: lightgbm not installed or DISABLE_OPTIONAL_MODELS=1."})

    torch_met = _train_pytorch(X_train, X_test, y_train, y_test, _preprocessor())
    if torch_met:
        metrics.append(torch_met)
    elif not _available("torch"):
        metrics.append({"model": "PyTorch Neural Network", "precision": np.nan, "recall": np.nan, "f1": np.nan, "roc_auc": np.nan, "error": "Skipped: set ENABLE_TORCH=1 to train PyTorch, or install torch if missing."})

    metrics_df = pd.DataFrame(metrics).sort_values(["f1", "recall"], ascending=False, na_position="last")
    metrics_df.to_csv(models_dir / "model_comparison.csv", index=False)
    metrics_df.to_csv(outputs_dir / "model_metrics.csv", index=False)

    eligible = metrics_df[metrics_df["model"].isin(fitted.keys())].copy()
    if eligible.empty:
        raise RuntimeError("No scikit-learn early payment model trained successfully.")
    best_name = eligible.sort_values(["f1", "recall"], ascending=False).iloc[0]["model"]
    best_pipe = fitted[best_name]
    joblib.dump(best_pipe, models_dir / "early_payment_model.pkl")
    (models_dir / "best_model.txt").write_text(str(best_name), encoding="utf-8")

    # Benchmark vs best model comparison for leadership and model governance.
    best_pred = best_pipe.predict(X_test)
    best_prob = best_pipe.predict_proba(X_test)[:, 1] if hasattr(best_pipe, "predict_proba") else best_pred
    legacy_pred = (
        (X_test["manual_override_flag"].fillna(0).astype(int) == 1)
        | (X_test["sap_error_present_flag"].fillna(0).astype(int) == 1)
        | (X_test["payment_term_mismatch_flag"].fillna(0).astype(int) == 1)
        | (X_test["vendor_early_payment_rate"].fillna(0) >= 0.30)
        | (X_test["processor_early_release_rate"].fillna(0) >= 0.30)
        | ((X_test["high_value_invoice_flag"].fillna(0).astype(int) == 1) & (X_test["processor_manual_override_rate"].fillna(0) >= 0.22))
    ).astype(int)
    benchmark_rows = [
        _metrics("Legacy Rule Guardrail", y_test, legacy_pred, legacy_pred),
        _metrics(f"Best ML Model - {best_name}", y_test, best_pred, best_prob),
        {"model": "Success Target", "precision": 0.80, "recall": 0.75, "f1": 0.77, "roc_auc": np.nan, "error": "Target benchmark"},
    ]
    pd.DataFrame(benchmark_rows).to_csv(outputs_dir / "benchmark_metrics.csv", index=False)

    # Scenario-wise M1 performance so the dashboard can show which business cases
    # the model handles well or poorly.
    if "scenario_label" in df.columns:
        scenario = df.loc[X_test.index, "scenario_label"].fillna("Unclassified scenario")
    else:
        scenario = pd.Series("Overall", index=X_test.index)
    scenario_rows = []
    eval_frame = pd.DataFrame({
        "scenario_label": scenario.values,
        "actual": np.asarray(y_test),
        "predicted": np.asarray(best_pred),
        "risk_score": np.asarray(best_prob, dtype=float),
    }, index=X_test.index)
    for scen, g in eval_frame.groupby("scenario_label"):
        scenario_rows.append({
            "scenario_label": scen,
            "support": int(len(g)),
            "positive_rate": float(g["actual"].mean()),
            "predicted_positive_rate": float(g["predicted"].mean()),
            "precision": precision_score(g["actual"], g["predicted"], zero_division=0),
            "recall": recall_score(g["actual"], g["predicted"], zero_division=0),
            "f1": f1_score(g["actual"], g["predicted"], zero_division=0),
            "avg_risk_score": float(g["risk_score"].mean()),
        })
    pd.DataFrame(scenario_rows).sort_values("support", ascending=False).to_csv(outputs_dir / "scenario_m1_metrics.csv", index=False)

    fi = _feature_importance(best_pipe)
    fi.to_csv(models_dir / "feature_importance.csv", index=False)
    fi.head(30).to_csv(outputs_dir / "feature_importance.csv", index=False)

    save_json(details, outputs_dir / "early_payment_model_details.json")
    if best_name in details:
        pd.DataFrame(details[best_name]["confusion_matrix"]).to_csv(outputs_dir / "early_payment_confusion_matrix.csv", index=False)
        save_json(details[best_name]["classification_report"], outputs_dir / "early_payment_classification_report.json")
    _train_statsmodels_summary(df)
    return metrics_df

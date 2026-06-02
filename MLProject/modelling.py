"""
MLflow Project Entry Point — Tuned RandomForest Training
Student : Timotius Kristafael Harjanto
Course  : SMSML — Dicoding

This script is invoked by `mlflow run .` using the parameters defined in
the MLproject file.  All logging is done MANUALLY (no autolog).
"""

import argparse
import json
import os
import sys
import logging
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)

# dagshub not needed — tracking URI set directly via mlflow.set_tracking_uri()
DAGSHUB_AVAILABLE = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

TARGET_COLUMN = "income"
ARTIFACTS_DIR = "./artifacts/"


# ---------------------------------------------------------------------------
# CLI arguments (injected by MLproject entry_point)
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RandomForest on Adult Income dataset")
    parser.add_argument("--n_estimators",      type=int,   default=200)
    parser.add_argument("--max_depth",         type=int,   default=20)
    parser.add_argument("--min_samples_split", type=int,   default=2)
    parser.add_argument("--min_samples_leaf",  type=int,   default=1)
    parser.add_argument("--data_path",         type=str,   default="./dataset_preprocessed/")
    parser.add_argument("--dagshub_username",  type=str,   default="")
    parser.add_argument("--dagshub_repo",      type=str,   default="Eksperimen_SML_TimotiusKristafaelHarjanto")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# MLflow setup
# ---------------------------------------------------------------------------

def setup_mlflow(dagshub_username: str, dagshub_repo: str) -> None:
    # MLFLOW_RUN_ID is set by `mlflow run .` — tells us we're inside a project
    inside_project = bool(os.getenv("MLFLOW_RUN_ID"))

    # Only set tracking URI if not already provided by the environment
    # (CI sets MLFLOW_TRACKING_URI before calling `mlflow run .`)
    if not os.getenv("MLFLOW_TRACKING_URI"):
        if dagshub_username:
            uri = f"https://dagshub.com/{dagshub_username}/{dagshub_repo}.mlflow"
        else:
            abs_path = os.path.abspath("./mlruns").replace("\\", "/")
            uri = f"file:///{abs_path}"
        mlflow.set_tracking_uri(uri)

    logger.info("MLflow tracking URI: %s", mlflow.get_tracking_uri())

    if not inside_project:
        mlflow.set_experiment("mlproject_training")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(data_path: str):
    logger.info("Loading data from %s", data_path)
    train_df = pd.read_csv(os.path.join(data_path, "train.csv"))
    test_df  = pd.read_csv(os.path.join(data_path, "test.csv"))

    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]
    X_test  = test_df.drop(columns=[TARGET_COLUMN])
    y_test  = test_df[TARGET_COLUMN]

    logger.info("Train %s | Test %s", X_train.shape, X_test.shape)
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Artifact helpers
# ---------------------------------------------------------------------------

def save_confusion_matrix(y_test, y_pred, path: str) -> None:
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["<=50K", ">50K"], yticklabels=["<=50K", ">50K"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — MLflow Project Run")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def save_feature_importance(model, feature_names: list, path: str) -> None:
    importances = pd.Series(model.feature_importances_, index=feature_names).nlargest(20).sort_values()
    fig, ax = plt.subplots(figsize=(10, 8))
    importances.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("Top 20 Feature Importances")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def save_artifacts(model, y_test, y_pred, X_train, run_id: str) -> None:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    cm_path = os.path.join(ARTIFACTS_DIR, "confusion_matrix.png")
    fi_path = os.path.join(ARTIFACTS_DIR, "feature_importance.png")
    cr_path = os.path.join(ARTIFACTS_DIR, "classification_report.txt")
    ms_path = os.path.join(ARTIFACTS_DIR, "model_summary.json")

    save_confusion_matrix(y_test, y_pred, cm_path)
    save_feature_importance(model, X_train.columns.tolist(), fi_path)

    report = classification_report(y_test, y_pred, target_names=["<=50K", ">50K"], digits=4)
    with open(cr_path, "w") as f:
        f.write(f"Classification Report\nRun ID: {run_id}\n{'='*60}\n{report}")

    summary: dict[str, Any] = {
        "run_id": run_id,
        "model_type": "RandomForestClassifier",
        "dataset": "adult_income_uci",
        "student": "Timotius Kristafael Harjanto",
        "n_estimators": model.n_estimators,
        "n_features": X_train.shape[1],
    }
    with open(ms_path, "w") as f:
        json.dump(summary, f, indent=2)

    for path in [cm_path, fi_path, cr_path, ms_path]:
        mlflow.log_artifact(path)
        logger.info("Logged artifact: %s", path)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    X_train, X_test, y_train, y_test = load_dataset(args.data_path)

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth if args.max_depth > 0 else None,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        random_state=42,
        n_jobs=-1,
    )

    # If `mlflow run .` already created a run, reuse it; otherwise create one
    existing_run_id = os.getenv("MLFLOW_RUN_ID")
    run_ctx = (
        mlflow.start_run(run_id=existing_run_id)
        if existing_run_id
        else mlflow.start_run(run_name="mlproject_rf_run")
    )
    with run_ctx as run:
        run_id = run.info.run_id

        # Manual parameter logging
        mlflow.log_params({
            "n_estimators":      args.n_estimators,
            "max_depth":         args.max_depth,
            "min_samples_split": args.min_samples_split,
            "min_samples_leaf":  args.min_samples_leaf,
            "model_type":        "RandomForestClassifier",
            "dataset":           "adult_income_uci",
            "student":           "TimotiusKristafaelHarjanto",
        })

        logger.info("Fitting RandomForestClassifier …")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy":  round(accuracy_score(y_test, y_pred), 6),
            "precision": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 6),
            "recall":    round(recall_score(y_test, y_pred,    average="weighted", zero_division=0), 6),
            "f1_score":  round(f1_score(y_test, y_pred,        average="weighted", zero_division=0), 6),
            "roc_auc":   round(roc_auc_score(y_test, y_prob),  6),
        }
        mlflow.log_metrics(metrics)

        logger.info("=== Metrics ===")
        for k, v in metrics.items():
            logger.info("  %-12s: %.4f", k, v)

        save_artifacts(model, y_test, y_pred, X_train, run_id)

        # Save model locally so the Dockerfile can COPY it
        model_output_path = "./model"
        mlflow.sklearn.save_model(model, model_output_path)
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name="adult_income_rf_ci",
        )

        logger.info("Model saved to %s", model_output_path)
        logger.info("Run ID: %s", run_id)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    logger.info("=" * 60)
    logger.info("MLflow Project Training — START")
    logger.info("Args: %s", vars(args))
    logger.info("=" * 60)

    setup_mlflow(args.dagshub_username, args.dagshub_repo)
    train(args)

    logger.info("=" * 60)
    logger.info("MLflow Project Training — COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()


"""
============================================================
AUTOMATED MODEL RETRAINING
RECRUITMENT PREDICTION MLOPS PLATFORM
============================================================

Features:
1. Load labelled recruitment candidate data
2. Split data into training and validation sets
3. Preprocess missing values and scale features
4. Train Logistic Regression and Random Forest
5. Evaluate the existing production model
6. Compare candidate models against production
7. Track retraining experiments using MLflow
8. Deploy only when all approval conditions pass
9. Back up and protect production model artifacts
10. Generate a retraining comparison report
"""

import shutil
import joblib
import mlflow
import mlflow.sklearn

import pandas as pd
import numpy as np

from pathlib import Path
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR / "data" / "raw" /
    "recruitment_candidates.csv"
)

MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "best_model.joblib"

PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.joblib"

BACKUP_DIR = MODEL_DIR / "backups"

REPORT_DIR = BASE_DIR / "reports"

MLFLOW_DB = BASE_DIR / "mlflow.db"

EXPERIMENT_NAME = "Recruitment_Automated_Retraining"

FEATURES = [
    "experience_years",
    "skills_match_score",
    "assessment_score",
    "industry_experience_years",
    "interview_score"
]

TARGET = "assessment_outcome"

RANDOM_STATE = 42

TEST_SIZE = 0.20

# Production model approval conditions.
MINIMUM_F1_IMPROVEMENT = 0.01

MINIMUM_ACCURACY_IMPROVEMENT = 0.0

MINIMUM_ROC_AUC_IMPROVEMENT = 0.0

MODEL_DIR.mkdir(parents=True, exist_ok=True)
BACKUP_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD LABELLED DATASET
# ============================================================

def load_data():

    print("\nLoading labelled recruitment data...")

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = FEATURES + [TARGET]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    X = df[FEATURES].copy()

    X = X.apply(
        pd.to_numeric,
        errors="coerce"
    )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    y = df[TARGET]

    if not pd.api.types.is_numeric_dtype(y):

        y = (
            y.astype(str)
            .str.strip()
            .str.lower()
            .map({
                "pass": 1,
                "fail": 0
            })
        )

    else:

        y = pd.to_numeric(
            y,
            errors="coerce"
        )

    if y.isnull().any():

        raise ValueError(
            "Target column contains missing or invalid labels."
        )

    y = y.astype(int)

    if set(y.unique()) != {0, 1}:

        raise ValueError(
            "Target must contain both classes: 0 and 1."
        )

    print("Dataset loaded:", df.shape)

    print("\nTarget distribution:")
    print(y.value_counts())

    print("\nFeature missing values:")
    print(X.isnull().sum())

    return X, y


# ============================================================
# 3. SPLIT TRAINING AND VALIDATION DATA
# ============================================================

def split_data(X, y):

    print("\nSplitting labelled dataset...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("Training records:", len(X_train))
    print("Validation records:", len(X_test))

    return X_train, X_test, y_train, y_test


# ============================================================
# 4. CREATE PREPROCESSOR
# ============================================================

def create_preprocessor():

    return Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ])


# ============================================================
# 5. EVALUATE MODEL
# ============================================================

def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {

        "accuracy": accuracy_score(
            y_test,
            predictions
        ),

        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "f1_score": f1_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "roc_auc": roc_auc_score(
            y_test,
            probabilities
        )
    }

    return metrics


# ============================================================
# 6. TRAIN CANDIDATE MODELS
# ============================================================

def train_candidate_models(
    X_train,
    X_test,
    y_train,
    y_test
):

    print("\nTraining candidate models...")

    preprocessor = create_preprocessor()

    # Fit preprocessing only on training records.
    X_train_processed = preprocessor.fit_transform(
        X_train
    )

    X_test_processed = preprocessor.transform(
        X_test
    )

    models = {

        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            class_weight="balanced"
        )
    }

    results = {}

    for model_name, model in models.items():

        print("\nTraining:", model_name)

        model.fit(
            X_train_processed,
            y_train
        )

        metrics = evaluate_model(
            model,
            X_test_processed,
            y_test
        )

        results[model_name] = {
            "model": model,
            "metrics": metrics
        }

        print("\nEvaluation metrics:")

        for metric, value in metrics.items():

            print(f"{metric}: {value:.4f}")

    best_name = max(
        results,
        key=lambda name: results[name]["metrics"]["f1_score"]
    )

    best_model = results[best_name]["model"]

    best_metrics = results[best_name]["metrics"]

    print("\nBest candidate model:", best_name)

    print(
        "Candidate F1-score:",
        f"{best_metrics['f1_score']:.4f}"
    )

    return (
        results,
        best_model,
        preprocessor,
        best_name,
        best_metrics
    )


# ============================================================
# 7. EVALUATE CURRENT PRODUCTION MODEL
# ============================================================

def evaluate_production_model(X_test, y_test):

    print("\nEvaluating existing production model...")

    if not MODEL_PATH.exists():

        print("No existing production model found.")

        return None

    if not PREPROCESSOR_PATH.exists():

        raise FileNotFoundError(
            "Production preprocessor is missing. "
            "Cannot safely evaluate the production model."
        )

    existing_model = joblib.load(MODEL_PATH)

    existing_preprocessor = joblib.load(
        PREPROCESSOR_PATH
    )

    X_test_processed = existing_preprocessor.transform(
        X_test
    )

    metrics = evaluate_model(
        existing_model,
        X_test_processed,
        y_test
    )

    print("\nProduction model metrics:")

    for metric, value in metrics.items():

        print(f"{metric}: {value:.4f}")

    return metrics


# ============================================================
# 8. TRACK EXPERIMENTS USING MLFLOW
# ============================================================

def track_experiments(results):

    print("\nStarting MLflow experiment tracking...")

    tracking_uri = (
        "sqlite:///" + MLFLOW_DB.as_posix()
    )

    mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(EXPERIMENT_NAME)

    for model_name, result in results.items():

        model = result["model"]

        metrics = result["metrics"]

        with mlflow.start_run(
            run_name=model_name
        ):

            mlflow.log_param(
                "model_name",
                model_name
            )

            mlflow.log_param(
                "random_state",
                RANDOM_STATE
            )

            mlflow.log_param(
                "validation_split",
                TEST_SIZE
            )

            mlflow.log_param(
                "features",
                ", ".join(FEATURES)
            )

            mlflow.log_param(
                "preprocessing",
                "Median Imputation + Standard Scaling"
            )

            mlflow.log_metrics(metrics)

            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                serialization_format="pickle"
            )

            print(
                f"MLflow logged: {model_name}"
            )

    print("MLflow experiment tracking completed.")


# ============================================================
# 9. COMPARE PRODUCTION AND CANDIDATE MODELS
# ============================================================

def compare_models(old_metrics, new_metrics):

    print("\n" + "=" * 60)
    print("PRODUCTION MODEL APPROVAL EVALUATION")
    print("=" * 60)

    if old_metrics is None:

        print(
            "No production model exists. "
            "The candidate is eligible for initial deployment."
        )

        return True, {
            "f1_improved": True,
            "accuracy_not_decreased": True,
            "roc_auc_not_decreased": True
        }

    f1_improvement = (
        new_metrics["f1_score"]
        - old_metrics["f1_score"]
    )

    accuracy_improvement = (
        new_metrics["accuracy"]
        - old_metrics["accuracy"]
    )

    roc_auc_improvement = (
        new_metrics["roc_auc"]
        - old_metrics["roc_auc"]
    )

    conditions = {

        "f1_improved": (
            f1_improvement >= MINIMUM_F1_IMPROVEMENT
        ),

        "accuracy_not_decreased": (
            accuracy_improvement
            >= MINIMUM_ACCURACY_IMPROVEMENT
        ),

        "roc_auc_not_decreased": (
            roc_auc_improvement
            >= MINIMUM_ROC_AUC_IMPROVEMENT
        )
    }

    print("\nProduction model:")
    print(old_metrics)

    print("\nCandidate model:")
    print(new_metrics)

    print("\nMetric improvements:")

    print(
        f"F1 improvement: {f1_improvement:+.4f}"
    )

    print(
        f"Accuracy improvement: {accuracy_improvement:+.4f}"
    )

    print(
        f"ROC-AUC improvement: {roc_auc_improvement:+.4f}"
    )

    print("\nApproval conditions:")

    for condition, passed in conditions.items():

        print(
            f"{condition}: "
            f"{'PASSED' if passed else 'FAILED'}"
        )

    approved = all(conditions.values())

    print("\nFinal decision:")

    if approved:

        print("CANDIDATE MODEL APPROVED!")

    else:

        print("CANDIDATE MODEL REJECTED.")

        print("Keeping the existing production model.")

    return approved, conditions


# ============================================================
# 10. SAVE COMPARISON REPORT
# ============================================================

def save_comparison_report(
    old_metrics,
    new_metrics,
    approved,
    conditions,
    best_name
):

    rows = []

    for metric in [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc"
    ]:

        old_value = (
            old_metrics[metric]
            if old_metrics is not None
            else np.nan
        )

        new_value = new_metrics[metric]

        rows.append({

            "metric": metric,

            "production_score": old_value,

            "candidate_score": new_value,

            "improvement": (
                new_value - old_value
                if old_metrics is not None
                else np.nan
            )
        })

    report_path = (
        REPORT_DIR / "retraining_comparison.csv"
    )

    pd.DataFrame(rows).to_csv(
        report_path,
        index=False
    )

    decision_path = (
        REPORT_DIR / "retraining_decision.txt"
    )

    with open(
        decision_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "AUTOMATED RETRAINING DECISION\n"
        )

        file.write(
            f"Timestamp: {datetime.now().isoformat()}\n"
        )

        file.write(
            f"Candidate model: {best_name}\n"
        )

        file.write(
            f"Approved: {approved}\n\n"
        )

        for condition, passed in conditions.items():

            file.write(
                f"{condition}: {passed}\n"
            )

    print("\nComparison report saved:", report_path)

    print("Decision report saved:", decision_path)


# ============================================================
# 11. BACK UP PRODUCTION ARTIFACTS
# ============================================================

def backup_production_artifacts():

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    backup_paths = {}

    if MODEL_PATH.exists():

        model_backup = (
            BACKUP_DIR /
            f"best_model_backup_{timestamp}.joblib"
        )

        shutil.copy2(
            MODEL_PATH,
            model_backup
        )

        backup_paths["model"] = model_backup

    if PREPROCESSOR_PATH.exists():

        preprocessor_backup = (
            BACKUP_DIR /
            f"preprocessor_backup_{timestamp}.joblib"
        )

        shutil.copy2(
            PREPROCESSOR_PATH,
            preprocessor_backup
        )

        backup_paths["preprocessor"] = preprocessor_backup

    return backup_paths


# ============================================================
# 12. SAVE APPROVED PRODUCTION MODEL SAFELY
# ============================================================

def save_production_model(model, preprocessor):

    print("\nPreparing production model deployment...")

    temp_model = MODEL_DIR / "best_model_temp.joblib"

    temp_preprocessor = (
        MODEL_DIR / "preprocessor_temp.joblib"
    )

    backup_paths = {}

    try:

        # Save new artifacts to temporary files first.
        joblib.dump(
            model,
            temp_model
        )

        joblib.dump(
            preprocessor,
            temp_preprocessor
        )

        # Back up current production artifacts.
        backup_paths = backup_production_artifacts()

        # Replace the preprocessor first, then the model.
        temp_preprocessor.replace(
            PREPROCESSOR_PATH
        )

        temp_model.replace(
            MODEL_PATH
        )

        print("\nProduction model updated successfully!")

        print("Model:", MODEL_PATH)

        print("Preprocessor:", PREPROCESSOR_PATH)

        for name, path in backup_paths.items():

            print(f"Previous {name} backup:", path)

    except Exception:

        print("\nDeployment failed. Attempting rollback...")

        # Restore previous production artifacts.
        if "model" in backup_paths:

            shutil.copy2(
                backup_paths["model"],
                MODEL_PATH
            )

        if "preprocessor" in backup_paths:

            shutil.copy2(
                backup_paths["preprocessor"],
                PREPROCESSOR_PATH
            )

        raise

    finally:

        if temp_model.exists():

            temp_model.unlink()

        if temp_preprocessor.exists():

            temp_preprocessor.unlink()


# ============================================================
# 13. MAIN AUTOMATED RETRAINING WORKFLOW
# ============================================================

def main():

    print("\n" + "=" * 60)

    print("AUTOMATED MODEL RETRAINING")

    print("=" * 60)

    try:

        # Step 1: Load labelled data.
        X, y = load_data()

        # Step 2: Split training and validation records.
        (
            X_train,
            X_test,
            y_train,
            y_test
        ) = split_data(X, y)

        # Step 3: Train candidate models.
        (
            results,
            best_model,
            new_preprocessor,
            best_name,
            best_metrics
        ) = train_candidate_models(
            X_train,
            X_test,
            y_train,
            y_test
        )

        # Step 4: Evaluate current production model.
        old_metrics = evaluate_production_model(
            X_test,
            y_test
        )

        # Step 5: Track experiments.
        track_experiments(results)

        # Step 6: Compare candidate and production models.
        approved, conditions = compare_models(
            old_metrics,
            best_metrics
        )

        # Step 7: Save comparison report.
        save_comparison_report(
            old_metrics,
            best_metrics,
            approved,
            conditions,
            best_name
        )

        # Step 8: Deploy only if all conditions pass.
        if approved:

            save_production_model(
                best_model,
                new_preprocessor
            )

            print("\nPRODUCTION MODEL DEPLOYMENT APPROVED!")

        else:

            print("\nPRODUCTION MODEL NOT UPDATED.")

            print(
                "The existing production model and "
                "preprocessor remain in use."
            )

        print("\n" + "=" * 60)

        print("AUTOMATED RETRAINING COMPLETED!")

        print("=" * 60)

    except Exception as error:

        print("\nAUTOMATED RETRAINING FAILED!")

        print("Error:", error)

        raise


# ============================================================
# 14. EXECUTION
# ============================================================

if __name__ == "__main__":

    main()

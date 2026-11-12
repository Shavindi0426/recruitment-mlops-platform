
# ============================================
# RECRUITMENT MLOPS PLATFORM
# MODEL TRAINING + MLFLOW EXPERIMENT TRACKING
# SQLITE BACKEND
# ============================================

import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


# ============================================
# 1. DEFINE PROJECT PATHS
# ============================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================
# 2. CONFIGURE MLFLOW WITH SQLITE
# ============================================

DATABASE_PATH = PROJECT_ROOT / "mlflow.db"

tracking_uri = f"sqlite:///{DATABASE_PATH.as_posix()}"

mlflow.set_tracking_uri(tracking_uri)

mlflow.set_experiment(
    "Recruitment_Prediction_Experiments"
)

print("MLflow SQLite tracking configured successfully!")


# ============================================
# 3. LOAD PROCESSED DATASETS
# ============================================

X_train = pd.read_csv(DATA_DIR / "X_train.csv")
X_test = pd.read_csv(DATA_DIR / "X_test.csv")

y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze()
y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze()

print("\nDatasets loaded successfully!")

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)


# ============================================
# 4. DEFINE MACHINE LEARNING MODELS
# ============================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )

}


# ============================================
# 5. TRAIN AND EVALUATE MODELS
# ============================================

results = []

trained_models = {}

for model_name, model in models.items():

    print("\nTraining:", model_name)

    # Train model
    model.fit(X_train, y_train)

    # Generate predictions
    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test, predictions, zero_division=0
    )

    recall = recall_score(
        y_test, predictions, zero_division=0
    )

    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    auc = roc_auc_score(y_test, probabilities)


    # ========================================
    # 6. MLFLOW EXPERIMENT TRACKING
    # ========================================

    with mlflow.start_run(run_name=model_name):

        # Log model parameters
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("random_state", 42)

        if model_name == "Logistic Regression":

            mlflow.log_param("max_iter", 1000)

        else:

            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("max_depth", 10)

        # Log evaluation metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)

        # Log trained model
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            serialization_format="pickle"
        )

        print(
            "MLflow tracking completed for:",
            model_name
        )


    # Store results
    results.append({

        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC AUC": auc

    })

    trained_models[model_name] = model


    # Display evaluation results
    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    print("\nConfusion Matrix:")

    print(confusion_matrix(y_test, predictions))


# ============================================
# 7. COMPARE MODEL PERFORMANCE
# ============================================

results_df = pd.DataFrame(results)

print("\nMODEL COMPARISON")
print("-" * 60)

print(results_df.round(4).to_string(index=False))


# ============================================
# 8. SELECT THE BEST MODEL
# ============================================

best_model_name = results_df.sort_values(
    by="F1 Score",
    ascending=False
).iloc[0]["Model"]

best_model = trained_models[best_model_name]


# ============================================
# 9. SAVE THE BEST MODEL
# ============================================

model_path = MODEL_DIR / "best_model.joblib"

joblib.dump(best_model, model_path)

results_df.to_csv(
    REPORT_DIR / "model_comparison.csv",
    index=False
)


# ============================================
# 10. DISPLAY FINAL RESULTS
# ============================================

print("\n" + "=" * 60)

print("BEST MODEL:", best_model_name)

print("Best F1 Score:",
      results_df.loc[
          results_df["Model"] == best_model_name,
          "F1 Score"
      ].iloc[0])

print("\nBest model saved to:", model_path)

print("Comparison report saved successfully!")

print("MLflow database:", DATABASE_PATH)

print("\nMODEL TRAINING COMPLETED SUCCESSFULLY!")

print("=" * 60)


import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# 1. Define project paths

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT / "data" / "raw"
    / "recruitment_candidates.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

MODEL_DIR = PROJECT_ROOT / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# 2. Load the raw dataset

df = pd.read_csv(INPUT_PATH)

print("Raw dataset loaded successfully!")
print("Original shape:", df.shape)


# 3. Remove duplicate records

df = df.drop_duplicates(subset="candidate_id")


# 4. Separate features and target

X = df.drop(
    columns=["candidate_id", "assessment_outcome"]
)

y = df["assessment_outcome"].map({
    "Fail": 0,
    "Pass": 1
})

# Check target values
if y.isnull().any():
    raise ValueError("Unexpected target values found!")

print("\nFeatures:", list(X.columns))


# 5. Split the dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# 6. Create preprocessing pipeline

preprocessor = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


# 7. Fit only on training data

X_train_processed = preprocessor.fit_transform(X_train)

# Transform test data using training statistics
X_test_processed = preprocessor.transform(X_test)


# 8. Convert processed data into DataFrames

X_train_processed = pd.DataFrame(
    X_train_processed,
    columns=X.columns,
    index=X_train.index
)

X_test_processed = pd.DataFrame(
    X_test_processed,
    columns=X.columns,
    index=X_test.index
)


# 9. Save processed datasets

X_train_processed.to_csv(
    OUTPUT_DIR / "X_train.csv",
    index=False
)

X_test_processed.to_csv(
    OUTPUT_DIR / "X_test.csv",
    index=False
)

y_train.to_csv(
    OUTPUT_DIR / "y_train.csv",
    index=False
)

y_test.to_csv(
    OUTPUT_DIR / "y_test.csv",
    index=False
)


# 10. Save preprocessing pipeline

joblib.dump(
    preprocessor,
    MODEL_DIR / "preprocessor.joblib"
)


# 11. Display final results

print("\nPREPROCESSING COMPLETED!")
print("-" * 40)

print("Training shape:", X_train_processed.shape)
print("Testing shape:", X_test_processed.shape)

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nMissing values after preprocessing:")
print(X_train_processed.isnull().sum())

print("\nSaved processed datasets to:", OUTPUT_DIR)
print("Saved preprocessor to:", MODEL_DIR)


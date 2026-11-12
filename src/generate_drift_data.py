
"""
Generate simulated incoming recruitment data
for testing data drift detection.

Original dataset remains unchanged.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

REFERENCE_PATH = BASE_DIR / "data" / "raw" / "recruitment_candidates.csv"

INCOMING_DIR = BASE_DIR / "data" / "incoming"

INCOMING_PATH = INCOMING_DIR / "candidate_data.csv"


# --------------------------------------------------
# 2. FEATURES
# --------------------------------------------------

FEATURES = [
    "experience_years",
    "skills_match_score",
    "assessment_score",
    "industry_experience_years",
    "interview_score"
]


# --------------------------------------------------
# 3. GENERATE SIMULATED DATA
# --------------------------------------------------

def generate_drift_data():

    print("\n" + "=" * 55)
    print("RECRUITMENT DATA DRIFT SIMULATION")
    print("=" * 55)

    # Load original dataset
    print("\nLoading original recruitment dataset...")

    if not REFERENCE_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {REFERENCE_PATH}"
        )

    reference = pd.read_csv(REFERENCE_PATH)

    print("Original dataset shape:", reference.shape)

    # Check required columns
    missing_columns = [
        col for col in FEATURES
        if col not in reference.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required features: {missing_columns}"
        )

    # Keep only the required features
    reference = reference[FEATURES].copy()

    # Convert feature columns to numeric
    for col in FEATURES:
        reference[col] = pd.to_numeric(
            reference[col], errors="coerce"
        )

    # Remove rows with missing feature values
    reference = reference.dropna().reset_index(drop=True)

    if len(reference) < 100:
        raise ValueError(
            "Not enough valid reference records."
        )

    # Number of incoming records
    sample_size = min(600, len(reference))

    rng = np.random.default_rng(42)

    print("\nGenerating", sample_size, "incoming records...")

    incoming = pd.DataFrame()

    # --------------------------------------------------
    # 4. INTRODUCE ARTIFICIAL DISTRIBUTION CHANGES
    # --------------------------------------------------

    print("\nIntroducing artificial feature changes...\n")

    for col in FEATURES:

        minimum = reference[col].min()
        maximum = reference[col].max()

        data_range = maximum - minimum

        if data_range == 0:
            incoming[col] = minimum
            print(f"{col}: Constant feature - unchanged")
            continue

        # Generate values within the lower 25%
        # of the original feature range.
        upper_limit = minimum + (0.25 * data_range)

        simulated_values = rng.uniform(
            low=minimum,
            high=upper_limit,
            size=sample_size
        )

        # Preserve integer-valued features
        if pd.api.types.is_integer_dtype(reference[col]):
            simulated_values = np.rint(simulated_values).astype(int)

        incoming[col] = simulated_values

        print(f"Feature: {col}")
        print(f"Original range: {minimum} - {maximum}")
        print(f"Simulated range: {incoming[col].min()} - {incoming[col].max()}")
        print("-" * 40)

    # --------------------------------------------------
    # 5. SAVE INCOMING DATASET
    # --------------------------------------------------

    INCOMING_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    incoming.to_csv(
        INCOMING_PATH,
        index=False
    )

    print("\nIncoming dataset generated successfully!")

    print("Incoming dataset shape:", incoming.shape)

    print("Saved location:", INCOMING_PATH)

    print("\nIncoming dataset preview:")
    print(incoming.head())

    print("\n" + "=" * 55)
    print("DATA DRIFT SIMULATION COMPLETED!")
    print("=" * 55)


if __name__ == "__main__":
    generate_drift_data()


import pandas as pd
import numpy as np
from pathlib import Path


# 1. Set random seed for reproducibility
np.random.seed(42)


# 2. Generate synthetic candidate data
def generate_candidate_data(n_samples=3000):

    data = pd.DataFrame({
        "candidate_id": [
            f"CAND_{i:04d}" for i in range(1, n_samples + 1)
        ],

        "experience_years": np.random.randint(
            0, 16, n_samples
        ),

        "skills_match_score": np.random.randint(
            30, 101, n_samples
        ),

        "assessment_score": np.random.randint(
            25, 101, n_samples
        ),

        "industry_experience_years": np.random.randint(
            0, 11, n_samples
        ),

        "interview_score": np.random.randint(
            30, 101, n_samples
        )
    })


    # 3. Generate a synthetic assessment outcome
    candidate_score = (
        data["skills_match_score"] * 0.30
        + data["assessment_score"] * 0.35
        + data["interview_score"] * 0.25
        + data["experience_years"] * 0.5
        + data["industry_experience_years"] * 0.5
    )

    noise = np.random.normal(0, 8, n_samples)

    data["assessment_outcome"] = np.where(
        candidate_score + noise >= 55,
        "Pass",
        "Fail"
    )


    # 4. Introduce a small amount of missing data
    for column in [
        "industry_experience_years",
        "interview_score"
    ]:
        missing_indices = data.sample(
            frac=0.02, random_state=42
        ).index

        data.loc[missing_indices, column] = np.nan


    return data


# 5. Generate the dataset
df = generate_candidate_data(3000)


# 6. Save the dataset
project_root = Path(__file__).resolve().parents[1]

output_path = (
    project_root / "data" / "raw"
    / "recruitment_candidates.csv"
)

output_path.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(output_path, index=False)


# 7. Display dataset information
print("Dataset generated successfully!")

print("\nDataset shape:", df.shape)

print("\nFirst five records:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nAssessment outcome distribution:")
print(df["assessment_outcome"].value_counts())

print("\nSaved to:", output_path)


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# 1. Load the dataset
project_root = Path(__file__).resolve().parents[1]

file_path = (
    project_root / "data" / "raw"
    / "recruitment_candidates.csv"
)

df = pd.read_csv(file_path)


# 2. Basic dataset information
print("DATASET INFORMATION")
print("-" * 40)

print("Shape:", df.shape)

print("\nData types:")
print(df.dtypes)

print("\nStatistical summary:")
print(df.describe())


# 3. Check missing values
print("\nMISSING VALUES")
print("-" * 40)

print(df.isnull().sum())


# 4. Check duplicate records
print("\nDUPLICATE RECORDS")
print("-" * 40)

print("Duplicates:", df.duplicated().sum())

print("Duplicate candidate IDs:", 
      df["candidate_id"].duplicated().sum())


# 5. Target distribution
print("\nTARGET DISTRIBUTION")
print("-" * 40)

print(df["assessment_outcome"].value_counts(normalize=True) * 100)


# 6. Create output folder
output_dir = project_root / "reports" / "figures"
output_dir.mkdir(parents=True, exist_ok=True)


# 7. Plot target distribution
plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="assessment_outcome",
    hue="assessment_outcome",
    palette="Set2",
    legend=False
)

plt.title("Assessment Outcome Distribution")
plt.xlabel("Assessment Outcome")
plt.ylabel("Number of Candidates")

plt.tight_layout()

plt.savefig(output_dir / "outcome_distribution.png")

plt.show()


# 8. Feature correlation
numeric_df = df.select_dtypes(include="number")

plt.figure(figsize=(9, 6))

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Feature Correlation Heatmap")

plt.tight_layout()

plt.savefig(output_dir / "correlation_heatmap.png")

plt.show()


print("\nEDA completed successfully!")


# Compare average scores by assessment outcome

print("\nAVERAGE SCORES BY OUTCOME")
print("-" * 40)

score_comparison = df.groupby(
    "assessment_outcome"
)[
    [
        "skills_match_score",
        "assessment_score",
        "interview_score"
    ]
].mean()

print(score_comparison)


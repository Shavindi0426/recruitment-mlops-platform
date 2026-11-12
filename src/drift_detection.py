
"""
DATA DRIFT DETECTION
Recruitment Prediction MLOps Platform

Features:
1. Load original training data
2. Load incoming candidate data
3. Compare feature distributions
4. Calculate Population Stability Index (PSI)
5. Perform Kolmogorov-Smirnov (KS) tests
6. Detect significant data drift
7. Generate drift reports
8. Maintain drift monitoring history
"""

import json
import warnings

import numpy as np
import pandas as pd

from pathlib import Path
from datetime import datetime

from scipy.stats import ks_2samp

from sklearn.model_selection import train_test_split


# =====================================================
# 1. PROJECT CONFIGURATION
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR / "data" / "raw" /
    "recruitment_candidates.csv"
)

INCOMING_DIR = BASE_DIR / "data" / "incoming"

INCOMING_PATH = (
    INCOMING_DIR / "candidate_data.csv"
)

MONITORING_DIR = BASE_DIR / "monitoring"

REPORT_DIR = BASE_DIR / "reports"

DRIFT_REPORT_PATH = (
    MONITORING_DIR / "drift_report.json"
)

HISTORY_PATH = (
    MONITORING_DIR / "drift_history.csv"
)

FEATURES = [
    "experience_years",
    "skills_match_score",
    "assessment_score",
    "industry_experience_years",
    "interview_score"
]

TARGET = "assessment_outcome"

RANDOM_STATE = 42

PSI_THRESHOLD = 0.20

KS_P_VALUE_THRESHOLD = 0.05

MINIMUM_SAMPLES = 20

# Create required directories.
INCOMING_DIR.mkdir(parents=True, exist_ok=True)

MONITORING_DIR.mkdir(parents=True, exist_ok=True)

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# 2. LOAD REFERENCE DATA
# =====================================================

def load_reference_data():

    print("\nLoading original reference data...")

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"Reference dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    missing_columns = [
        col for col in FEATURES
        if col not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing feature columns: {missing_columns}"
        )

    X = df[FEATURES].copy()

    for col in FEATURES:

        X[col] = pd.to_numeric(
            X[col],
            errors="coerce"
        )

        X[col] = X[col].replace(
            [np.inf, -np.inf],
            np.nan
        )

    # Use a consistent training reference sample.
    reference_data, _ = train_test_split(
        X,
        test_size=0.20,
        random_state=RANDOM_STATE
    )

    print(
        "Reference data shape:",
        reference_data.shape
    )

    return reference_data


# =====================================================
# 3. LOAD INCOMING DATA
# =====================================================

def load_incoming_data():

    print("\nLoading incoming candidate data...")

    if INCOMING_PATH.exists():

        incoming_data = pd.read_csv(
            INCOMING_PATH
        )

        print(
            "Source: Incoming candidate dataset"
        )

    else:

        warnings.warn(
            "No incoming dataset found. "
            "Using held-out reference data for a "
            "demonstration run."
        )

        df = pd.read_csv(DATA_PATH)

        X = df[FEATURES].copy()

        # Use the same split as the reference dataset.
        _, incoming_data = train_test_split(
            X,
            test_size=0.20,
            random_state=RANDOM_STATE
        )

        print(
            "Source: Demonstration test dataset"
        )

    missing_columns = [
        col for col in FEATURES
        if col not in incoming_data.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Incoming data missing columns: {missing_columns}"
        )

    incoming_data = incoming_data[
        FEATURES
    ].copy()

    for col in FEATURES:

        incoming_data[col] = pd.to_numeric(
            incoming_data[col],
            errors="coerce"
        )

        incoming_data[col] = incoming_data[col].replace(
            [np.inf, -np.inf],
            np.nan
        )

    print(
        "Incoming data shape:",
        incoming_data.shape
    )

    return incoming_data


# =====================================================
# 4. CALCULATE POPULATION STABILITY INDEX
# =====================================================

def calculate_psi(reference, current):

    reference = np.asarray(reference, dtype=float)

    current = np.asarray(current, dtype=float)

    reference = reference[np.isfinite(reference)]

    current = current[np.isfinite(current)]

    if len(reference) < MINIMUM_SAMPLES:

        raise ValueError(
            "Insufficient reference samples for PSI."
        )

    if len(current) < MINIMUM_SAMPLES:

        raise ValueError(
            "Insufficient incoming samples for PSI."
        )

    # Create bins using reference-data quantiles.
    quantiles = np.linspace(0, 1, 11)

    breakpoints = np.unique(
        np.quantile(reference, quantiles)
    )

    # Constant or nearly constant reference feature.
    if len(breakpoints) < 2:

        if np.allclose(
            current,
            reference[0]
        ):
            return 0.0

        return float("inf")

    # Extend bins to cover incoming observations.
    breakpoints[0] = -np.inf

    breakpoints[-1] = np.inf

    reference_counts, _ = np.histogram(
        reference,
        bins=breakpoints
    )

    current_counts, _ = np.histogram(
        current,
        bins=breakpoints
    )

    # Calculate proportions.
    reference_pct = (
        reference_counts / len(reference)
    )

    current_pct = (
        current_counts / len(current)
    )

    # Avoid division by zero and log(0).
    epsilon = 0.000001

    reference_pct = np.maximum(
        reference_pct,
        epsilon
    )

    current_pct = np.maximum(
        current_pct,
        epsilon
    )

    # Normalize after applying smoothing.
    reference_pct /= reference_pct.sum()

    current_pct /= current_pct.sum()

    psi = np.sum(
        (current_pct - reference_pct)
        * np.log(current_pct / reference_pct)
    )

    return float(psi)


# =====================================================
# 5. PERFORM KS TEST
# =====================================================

def perform_ks_test(reference, current):

    reference = np.asarray(reference, dtype=float)

    current = np.asarray(current, dtype=float)

    reference = reference[np.isfinite(reference)]

    current = current[np.isfinite(current)]

    if (
        len(reference) < MINIMUM_SAMPLES
        or len(current) < MINIMUM_SAMPLES
    ):

        raise ValueError(
            "Insufficient samples for KS test."
        )

    result = ks_2samp(
        reference,
        current
    )

    return (
        float(result.statistic),
        float(result.pvalue)
    )


# =====================================================
# 6. ANALYZE INDIVIDUAL FEATURES
# =====================================================

def analyze_feature(reference, current):

    psi = calculate_psi(
        reference,
        current
    )

    ks_statistic, ks_p_value = perform_ks_test(
        reference,
        current
    )

    if psi > PSI_THRESHOLD:

        psi_status = "Drift"

    elif psi >= 0.10:

        psi_status = "Warning"

    else:

        psi_status = "Stable"

    ks_drift = (
        ks_p_value < KS_P_VALUE_THRESHOLD
    )

    # Overall feature-level drift decision.
    drift_detected = (
        psi_status == "Drift"
        or ks_drift
    )

    return {

        "psi": psi,

        "psi_status": psi_status,

        "ks_statistic": ks_statistic,

        "ks_p_value": ks_p_value,

        "ks_drift_detected": bool(ks_drift),

        "drift_detected": bool(drift_detected),

        "reference_mean": float(
            np.mean(reference)
        ),

        "incoming_mean": float(
            np.mean(current)
        ),

        "reference_samples": int(
            np.isfinite(reference).sum()
        ),

        "incoming_samples": int(
            np.isfinite(current).sum()
        )
    }


# =====================================================
# 7. GENERATE DRIFT REPORT
# =====================================================

def generate_drift_report(
    reference_data,
    incoming_data
):

    print("\nAnalyzing feature distributions...")

    feature_results = {}

    for feature in FEATURES:

        print(
            f"\nChecking feature: {feature}"
        )

        reference = reference_data[
            feature
        ].dropna().values

        current = incoming_data[
            feature
        ].dropna().values

        result = analyze_feature(
            reference,
            current
        )

        feature_results[feature] = result

        print(
            f"PSI: {result['psi']:.4f}"
        )

        print(
            f"KS p-value: {result['ks_p_value']:.6f}"
        )

        print(
            f"Status: {result['psi_status']}"
        )

    drifted_features = [

        feature

        for feature, result
        in feature_results.items()

        if result["drift_detected"]

    ]

    warning_features = [

        feature

        for feature, result
        in feature_results.items()

        if result["psi_status"] == "Warning"

        and not result["drift_detected"]

    ]

    overall_status = (
        "DRIFT DETECTED"
        if drifted_features
        else "WARNING"
        if warning_features
        else "STABLE"
    )

    report = {

        "timestamp": datetime.now().isoformat(),

        "reference_data_source": str(DATA_PATH),

        "incoming_data_source": (
            str(INCOMING_PATH)
            if INCOMING_PATH.exists()
            else "Demonstration test dataset"
        ),

        "reference_samples": int(
            len(reference_data)
        ),

        "incoming_samples": int(
            len(incoming_data)
        ),

        "total_features": len(FEATURES),

        "drifted_features": drifted_features,

        "warning_features": warning_features,

        "drift_detected": bool(
            len(drifted_features) > 0
        ),

        "overall_status": overall_status,

        "feature_results": feature_results

    }

    return report


# =====================================================
# 8. SAVE JSON REPORT
# =====================================================

def save_drift_report(report):

    with open(
        DRIFT_REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
            allow_nan=False
        )

    print(
        "\nDrift report saved:",
        DRIFT_REPORT_PATH
    )


# =====================================================
# 9. SAVE MONITORING HISTORY
# =====================================================

def save_monitoring_history(report):

    history_record = {

        "timestamp": report["timestamp"],

        "overall_status": report["overall_status"],

        "drift_detected": report["drift_detected"],

        "total_features": report["total_features"],

        "drifted_features": ", ".join(
            report["drifted_features"]
        ),

        "warning_features": ", ".join(
            report["warning_features"]
        ),

        "reference_samples": report["reference_samples"],

        "incoming_samples": report["incoming_samples"]

    }

    history_df = pd.DataFrame([
        history_record
    ])

    if HISTORY_PATH.exists():

        history_df.to_csv(
            HISTORY_PATH,
            mode="a",
            header=False,
            index=False
        )

    else:

        history_df.to_csv(
            HISTORY_PATH,
            index=False
        )

    print(
        "Monitoring history saved:",
        HISTORY_PATH
    )


# =====================================================
# 10. DISPLAY FINAL RESULTS
# =====================================================

def display_results(report):

    print("\n" + "=" * 55)

    print("DATA DRIFT DETECTION SUMMARY")

    print("=" * 55)

    print(
        "Overall status:",
        report["overall_status"]
    )

    print(
        "Total features:",
        report["total_features"]
    )

    print(
        "Drifted features:",
        len(report["drifted_features"])
    )

    print(
        "Warning features:",
        len(report["warning_features"])
    )

    if report["drifted_features"]:

        print("\nDRIFT ALERT!")

        for feature in report["drifted_features"]:

            print(" -", feature)

    else:

        print("\nNo significant feature drift detected.")

    if report["warning_features"]:

        print("\nFeatures requiring attention:")

        for feature in report["warning_features"]:

            print(" -", feature)

    print("\nDATA DRIFT DETECTION COMPLETED!")

    print("=" * 55)


# =====================================================
# 11. MAIN WORKFLOW
# =====================================================

def main():

    print("\n" + "=" * 55)

    print("RECRUITMENT DATA DRIFT MONITORING")

    print("=" * 55)

    # Step 1: Load reference dataset.
    reference_data = load_reference_data()

    # Step 2: Load incoming dataset.
    incoming_data = load_incoming_data()

    # Step 3: Generate drift analysis.
    report = generate_drift_report(
        reference_data,
        incoming_data
    )

    # Step 4: Save JSON report.S
    save_drift_report(report)

    # Step 5: Save monitoring history.
    save_monitoring_history(report)

    # Step 6: Display results.
    display_results(report)


# =====================================================
# 12. EXECUTION
# =====================================================

if __name__ == "__main__":

    main()

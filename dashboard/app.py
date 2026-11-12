"""
RECRUITMENT PREDICTION MLOPS DASHBOARD

Features:
1. Dashboard Overview
2. Data Drift Monitoring
3. Feature Distribution Analysis
4. Model Performance Monitoring
5. Historical Monitoring

Technologies:
Streamlit, Plotly, Pandas, Scikit-learn, MLflow
"""

from pathlib import Path
import json
import ast

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MONITORING_DIR = BASE_DIR / "monitoring"

REFERENCE_PATH = (
    DATA_DIR / "raw" / "recruitment_candidates.csv"
)

INCOMING_PATHS = [
    DATA_DIR / "incoming" / "candidate_data.csv",
    DATA_DIR / "incoming" / "candidate_data_drift_test.csv"
]

DRIFT_REPORT_PATH = (
    MONITORING_DIR / "drift_report.json"
)

DRIFT_HISTORY_PATH = (
    MONITORING_DIR / "drift_history.csv"
)

MODEL_PATH = MODELS_DIR / "best_model.joblib"

PREPROCESSOR_PATH = (
    MODELS_DIR / "preprocessor.joblib"
)

MLFLOW_DB_PATH = BASE_DIR / "mlflow.db"

FEATURES = [
    "experience_years",
    "skills_match_score",
    "assessment_score",
    "industry_experience_years",
    "interview_score"
]

RANDOM_STATE = 42


# ============================================================
# 2. STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Recruitment Prediction MLOps",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 3. CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

.dashboard-title {
    font-size: 32px;
    font-weight: 700;
    color: #FFFFFF;
}

.dashboard-subtitle {
    color: #AAB4C3;
    font-size: 15px;
}

div[data-testid="stMetric"] {
    background-color: #1B2430;
    border: 1px solid #303B4B;
    padding: 17px;
    border-radius: 10px;
}

div[data-testid="stMetricLabel"] {
    color: #AAB4C3;
}

div[data-testid="stMetricValue"] {
    color: #FFFFFF;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 4. DATA LOADING FUNCTIONS
# ============================================================

@st.cache_data(ttl=30)
def load_drift_report():

    if not DRIFT_REPORT_PATH.exists():
        return {}

    try:

        with open(
            DRIFT_REPORT_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except Exception as error:

        st.error(f"Drift report error: {error}")

        return {}


@st.cache_data(ttl=30)
def load_drift_history():

    if not DRIFT_HISTORY_PATH.exists():
        return pd.DataFrame()

    try:

        return pd.read_csv(DRIFT_HISTORY_PATH)

    except Exception as error:

        st.warning(f"History loading error: {error}")

        return pd.DataFrame()


@st.cache_data(ttl=30)
def load_reference_data():

    if not REFERENCE_PATH.exists():
        return pd.DataFrame()

    try:

        df = pd.read_csv(REFERENCE_PATH)

        missing = [
            col for col in FEATURES
            if col not in df.columns
        ]

        if missing:
            return pd.DataFrame()

        X = df[FEATURES].copy()

        X = X.apply(
            pd.to_numeric,
            errors="coerce"
        )

        X = X.replace(
            [np.inf, -np.inf],
            np.nan
        )

        reference, _ = train_test_split(
            X,
            test_size=0.20,
            random_state=RANDOM_STATE
        )

        return reference.dropna(subset=FEATURES)

    except Exception as error:

        st.error(f"Reference data error: {error}")

        return pd.DataFrame()


@st.cache_data(ttl=30)
def load_incoming_data():

    # First, try to follow the source recorded in the report.
    report = load_drift_report()

    report_source = report.get(
        "incoming_data_source",
        ""
    )

    selected_path = None

    if report_source:

        try:

            candidate = Path(report_source)

            if candidate.is_file():
                selected_path = candidate

        except (TypeError, OSError, ValueError):
            pass

    # If the report source cannot be resolved, use
    # the same candidate file priority as drift detection.
    if selected_path is None:

        selected_path = next(
            (
                path for path in INCOMING_PATHS
                if path.exists()
            ),
            None
        )

    try:

        if selected_path is not None:

            df = pd.read_csv(selected_path)

            source = selected_path.name

        else:

            # Demonstration fallback:
            # use the held-out 20% from original dataset.
            if not REFERENCE_PATH.exists():

                return pd.DataFrame(), "Not available"

            df = pd.read_csv(REFERENCE_PATH)

            X = df[FEATURES].copy()

            _, incoming = train_test_split(
                X,
                test_size=0.20,
                random_state=RANDOM_STATE
            )

            source = "Held-out demonstration dataset"

            return clean_feature_data(incoming), source

        return clean_feature_data(df), source

    except Exception as error:

        st.error(f"Incoming data error: {error}")

        return pd.DataFrame(), "Error"


def clean_feature_data(df):

    missing = [
        col for col in FEATURES
        if col not in df.columns
    ]

    if missing:
        return pd.DataFrame()

    result = df[FEATURES].copy()

    result = result.apply(
        pd.to_numeric,
        errors="coerce"
    )

    result = result.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return result.dropna(subset=FEATURES)


# ============================================================
# 5. MLFLOW MODEL METRICS
# ============================================================

@st.cache_data(ttl=30)
def load_model_metrics():

    if not MLFLOW_DB_PATH.exists():

        return {}, "MLflow database not found"

    try:

        import mlflow

        from mlflow.tracking import MlflowClient

        tracking_uri = (
            "sqlite:///" +
            MLFLOW_DB_PATH.as_posix()
        )

        client = MlflowClient(
            tracking_uri=tracking_uri
        )

        experiments = client.search_experiments()

        best_run = None
        best_f1 = -1

        for experiment in experiments:

            runs = client.search_runs(
                experiment_ids=[
                    experiment.experiment_id
                ],
                order_by=[
                    "attributes.start_time DESC"
                ],
                max_results=100
            )

            for run in runs:

                if run.info.status != "FINISHED":
                    continue

                metrics = run.data.metrics

                if "f1_score" not in metrics:
                    continue

                f1 = metrics["f1_score"]

                if f1 > best_f1:

                    best_f1 = f1
                    best_run = run

        if best_run is None:

            return {}, "No completed MLflow model found"

        metrics = best_run.data.metrics

        model_name = (
            best_run.data.params.get(
                "model_name",
                best_run.data.tags.get(
                    "model_name",
                    "MLflow Model"
                )
            )
        )

        return metrics, model_name

    except Exception as error:

        return {}, f"MLflow error: {error}"


# ============================================================
# 6. MODEL ARTIFACT INFORMATION
# ============================================================

def check_model_artifacts():

    return {
        "best_model": MODEL_PATH.exists(),
        "preprocessor": PREPROCESSOR_PATH.exists()
    }


# ============================================================
# 7. HELPER FUNCTIONS
# ============================================================

def safe_float(value, default=0.0):

    try:

        result = float(value)

        if not np.isfinite(result):
            return default

        return result

    except (ValueError, TypeError):

        return default


def format_number(value, decimals=4):

    try:

        if value is None or pd.isna(value):
            return "N/A"

        return f"{float(value):.{decimals}f}"

    except (ValueError, TypeError):

        return "N/A"


def format_percentage(value):

    return f"{safe_float(value) * 100:.2f}%"


def get_feature_status(result):

    if result.get("drift_detected", False):
        return "Drift"

    return result.get("psi_status", "Stable")


def count_drifted_features(value):

    # Handle missing values
    if value is None:
        return 0

    if isinstance(value, (list, tuple, set)):
        return len(value)

    # Handle numeric values
    if isinstance(value, (int, float, np.integer)):
        return int(value) if pd.notna(value) else 0

    if isinstance(value, str):

        value = value.strip()

        if value.lower() in ["", "none", "nan", "null"]:
            return 0

        # First, try to parse a Python list
        try:

            parsed = ast.literal_eval(value)

            if isinstance(parsed, (list, tuple, set)):
                return len(parsed)

            if isinstance(parsed, (int, float)):
                return int(parsed)

        except (ValueError, SyntaxError, TypeError):
            pass

        # Handle comma-separated feature names
        features = [
            feature.strip()
            for feature in value.split(",")
            if feature.strip()
            and feature.strip().lower() not in [
                "none", "nan", "null"
            ]
        ]

        return len(features)

    return 0


def get_status_color(status):

    status = str(status).upper()

    if status in ["DRIFT", "DRIFT DETECTED"]:
        return "#EF4444"

    if status == "WARNING":
        return "#F59E0B"

    return "#22C55E"


# ============================================================
# 8. LOAD DASHBOARD DATA
# ============================================================

report = load_drift_report()

history = load_drift_history()

reference_data = load_reference_data()

incoming_data, incoming_filename = load_incoming_data()

model_metrics, model_name = load_model_metrics()

artifacts = check_model_artifacts()

feature_results = report.get(
    "feature_results",
    {}
)

overall_status = report.get(
    "overall_status",
    "UNKNOWN"
).upper()

drifted_features = report.get(
    "drifted_features",
    []
)

warning_features = report.get(
    "warning_features",
    []
)

total_features = report.get(
    "total_features",
    len(FEATURES)
)


# ============================================================
# 9. SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("📊 Recruitment MLOps")

st.sidebar.markdown("---")

st.sidebar.subheader("Navigation")

page = st.sidebar.radio(
    "Select Dashboard Page",
    [
        "Dashboard Overview",
        "Data Drift Monitoring",
        "Feature Analysis",
        "Model Performance",
        "Monitoring History"
    ],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")

st.sidebar.subheader("System Information")

st.sidebar.write(f"Model: {model_name}")

st.sidebar.write(
    "Model file: "
    + (
        "Available"
        if artifacts["best_model"]
        else "Not Available"
    )
)

st.sidebar.write(
    "Preprocessor: "
    + (
        "Available"
        if artifacts["preprocessor"]
        else "Not Available"
    )
)

st.sidebar.write(
    "Drift report: "
    + (
        "Available"
        if report
        else "Not Available"
    )
)

st.sidebar.write(
    f"Incoming dataset: {incoming_filename}"
)

st.sidebar.markdown("---")

if st.sidebar.button(
    "🔄 Refresh Dashboard",
    use_container_width=True
):

    st.cache_data.clear()

    st.rerun()

st.sidebar.markdown("---")

st.sidebar.caption(
    "Recruitment Prediction MLOps Platform"
)


# ============================================================
# 10. DASHBOARD HEADER
# ============================================================

st.markdown(
    '<div class="dashboard-title">'
    'Recruitment Prediction MLOps Dashboard'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="dashboard-subtitle">'
    'Real-time model monitoring, data drift detection '
    'and model performance analysis'
    '</div>',
    unsafe_allow_html=True
)

st.markdown("---")


# ============================================================
# 11. DASHBOARD OVERVIEW
# ============================================================

if page == "Dashboard Overview":

    st.header("System Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Overall Drift Status",
            overall_status
        )

    with col2:

        st.metric(
            "Total Features",
            total_features
        )

    with col3:

        st.metric(
            "Drifted Features",
            len(drifted_features)
        )

    with col4:

        st.metric(
            "Warning Features",
            len(warning_features)
        )

    st.markdown("---")

    # Model performance

    st.header("Model Performance")

    metric_columns = st.columns(5)

    metric_names = [
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1_score", "F1 Score"),
        ("roc_auc", "ROC-AUC")
    ]

    for column, (key, label) in zip(
        metric_columns,
        metric_names
    ):

        with column:

            value = model_metrics.get(key)

            if value is not None:

                st.metric(
                    label,
                    format_percentage(value)
                )

            else:

                st.metric(label, "N/A")

    st.caption(
        f"Source: MLflow | Model: {model_name}"
    )

    st.markdown("---")

    # Feature drift summary

    st.header("Feature Drift Summary")

    if feature_results:

        rows = []

        for feature, result in feature_results.items():

            rows.append({

                "Feature": feature,

                "PSI": round(
                    safe_float(result.get("psi")),
                    4
                ),

                "KS p-value": round(
                    safe_float(
                        result.get("ks_p_value"),
                        1
                    ),
                    4
                ),

                "Status": get_feature_status(result)

            })

        summary_df = pd.DataFrame(rows)

        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No drift results available. "
            "Run drift detection first."
        )

    st.markdown("---")

    # Drift alerts

    st.header("Drift Detection Status")

    if overall_status == "DRIFT DETECTED":

        st.error(
            "⚠️ Significant data drift detected!"
        )

        for feature in drifted_features:
            st.write(f"- {feature}")

    elif overall_status == "WARNING":

        st.warning(
            "Some features show warning-level drift."
        )

        for feature in warning_features:
            st.write(f"- {feature}")

    elif overall_status == "STABLE":

        st.success(
            "No significant feature drift detected."
        )

    else:

        st.info(
            "Drift detection information is unavailable."
        )

    st.markdown("---")

    # Dataset information

    st.header("Dataset Information")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Reference Dataset")

        st.metric(
            "Reference Records",
            len(reference_data)
        )

        st.write(
            f"Features: {reference_data.shape[1]}"
        )

    with col2:

        st.subheader("Incoming Dataset")

        st.metric(
            "Incoming Records",
            len(incoming_data)
        )

        st.write(
            f"Features: {incoming_data.shape[1]}"
        )

# ============================================================
# 12. DATA DRIFT MONITORING
# ============================================================

elif page == "Data Drift Monitoring":

    st.header("Data Drift Monitoring")

    st.write(
        "Population Stability Index (PSI) "
        "and Kolmogorov-Smirnov (KS) test results."
    )

    if not feature_results:

        st.warning(
            "No drift report available. "
            "Run src/drift_detection.py first."
        )

    else:

        drift_rows = []

        for feature, result in feature_results.items():

            drift_rows.append({

                "Feature": feature,

                "PSI": safe_float(
                    result.get("psi")
                ),

                "KS Statistic": safe_float(
                    result.get("ks_statistic")
                ),

                "KS p-value": safe_float(
                    result.get("ks_p_value"),
                    1
                ),

                "Status": get_feature_status(result)

            })

        drift_df = pd.DataFrame(drift_rows)

        col1, col2 = st.columns(2)

        # PSI chart

        with col1:

            fig = px.bar(
                drift_df,
                x="Feature",
                y="PSI",
                color="Status",
                title="PSI by Feature",
                color_discrete_map={
                    "Stable": "#22C55E",
                    "Warning": "#F59E0B",
                    "Drift": "#EF4444"
                }
            )

            fig.add_hline(
                y=0.10,
                line_dash="dash",
                annotation_text="PSI 0.10"
            )

            fig.add_hline(
                y=0.20,
                line_dash="dot",
                annotation_text="PSI 0.20"
            )

            fig.update_layout(
                template="plotly_dark",
                xaxis_title="Feature",
                yaxis_title="PSI",
                xaxis_tickangle=-30
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        # KS chart
        with col2:

            ks_plot_df = drift_df.copy()

            # Convert very small p-values into a readable scale
            ks_plot_df["KS -log10(p-value)"] = -np.log10(
                pd.to_numeric(
                    ks_plot_df["KS p-value"],
                    errors="coerce"
                ).clip(
                    lower=np.finfo(float).tiny,
                    upper=1.0
                )
            )

            fig = px.bar(
                ks_plot_df,
                x="Feature",
                y="KS -log10(p-value)",
                color="Status",
                title="KS Test Significance (-log10 p-value)",
                color_discrete_map={
                    "Stable": "#22C55E",
                    "Warning": "#F59E0B",
                    "Drift": "#EF4444"
                }
            )

            # Significance threshold: p-value = 0.05
            significance_threshold = -np.log10(0.05)

            fig.add_hline(
                y=significance_threshold,
                line_dash="dash",
                annotation_text="Significance level (p = 0.05)",
                annotation_position="top right"
            )

            fig.update_layout(
                template="plotly_dark",
                xaxis_title="Feature",
                yaxis_title="-log10(KS p-value)",
                xaxis_tickangle=-30
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.markdown("---")

        # Detailed drift results

        st.subheader("Detailed Drift Results")

        display_df = drift_df.copy()

        for col in [
            "PSI",
            "KS Statistic",
            "KS p-value"
        ]:

            display_df[col] = display_df[col].round(4)

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("---")

        # Alerts

        st.subheader("Drift Alerts")

        if drifted_features:

            for feature in drifted_features:

                st.error(
                    f"Drift detected in: {feature}"
                )

        elif warning_features:

            for feature in warning_features:

                st.warning(
                    f"Distribution warning: {feature}"
                )

        else:

            st.success(
                "No drift alerts. "
                "All monitored features are stable."
            )

        st.markdown("---")

        st.subheader("Detection Information")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Reference Samples",
                report.get("reference_samples", 0)
            )

        with col2:

            st.metric(
                "Incoming Samples",
                report.get("incoming_samples", 0)
            )

        with col3:

            st.metric(
                "Total Features",
                total_features
            )

        st.caption(
            f"Last detection: "
            f"{report.get('timestamp', 'Not available')}"
        )


# ============================================================
# 13. FEATURE DISTRIBUTION ANALYSIS
# ============================================================

elif page == "Feature Analysis":

    st.header("Feature Distribution Analysis")

    if reference_data.empty:

        st.error(
            "Reference dataset is not available."
        )

    elif incoming_data.empty:

        st.warning(
            "Incoming dataset is not available."
        )

    else:

        available_features = [
            feature for feature in FEATURES
            if feature in reference_data.columns
            and feature in incoming_data.columns
        ]

        selected_feature = st.selectbox(
            "Select Recruitment Feature",
            available_features
        )

        reference_values = reference_data[
            selected_feature
        ].dropna()

        incoming_values = incoming_data[
            selected_feature
        ].dropna()

        result = feature_results.get(
            selected_feature,
            {}
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Reference Mean",
                f"{reference_values.mean():.2f}"
            )

        with col2:

            st.metric(
                "Incoming Mean",
                f"{incoming_values.mean():.2f}"
            )

        with col3:

            st.metric(
                "PSI",
                format_number(result.get("psi"))
            )

        st.markdown("---")

        # Box plot

        st.subheader(
            f"Reference vs Incoming: {selected_feature}"
        )

        comparison_df = pd.DataFrame({

            "Value": np.concatenate([
                reference_values.to_numpy(),
                incoming_values.to_numpy()
            ]),

            "Dataset": (
                ["Reference"] * len(reference_values)
                +
                ["Incoming"] * len(incoming_values)
            )

        })

        fig = px.box(
            comparison_df,
            x="Value",
            y="Dataset",
            color="Dataset",
            color_discrete_map={
                "Reference": "#3B82F6",
                "Incoming": "#F59E0B"
            },
            orientation="h",
            title="Reference vs Incoming Distribution"
        )

        fig.update_layout(
            template="plotly_dark"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # Histogram

        st.subheader("Feature Distribution Comparison")

        histogram_df = pd.DataFrame({

            "Value": np.concatenate([
                reference_values.to_numpy(),
                incoming_values.to_numpy()
            ]),

            "Dataset": (
                ["Reference"] * len(reference_values)
                +
                ["Incoming"] * len(incoming_values)
            )

        })

        fig = px.histogram(
            histogram_df,
            x="Value",
            color="Dataset",
            barmode="overlay",
            histnorm="probability density",
            opacity=0.65,
            nbins=25,
            color_discrete_map={
                "Reference": "#3B82F6",
                "Incoming": "#F59E0B"
            },
            title="Probability Density Comparison"
        )

        fig.update_layout(
            template="plotly_dark",
            xaxis_title=selected_feature,
            yaxis_title="Probability Density"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.markdown("---")

        # Feature statistics

        st.subheader("Feature Statistics")

        statistics = pd.DataFrame({

            "Statistic": [
                "Count",
                "Mean",
                "Standard Deviation",
                "Minimum",
                "Median",
                "Maximum"
            ],

            "Reference": [
                reference_values.count(),
                reference_values.mean(),
                reference_values.std(),
                reference_values.min(),
                reference_values.median(),
                reference_values.max()
            ],

            "Incoming": [
                incoming_values.count(),
                incoming_values.mean(),
                incoming_values.std(),
                incoming_values.min(),
                incoming_values.median(),
                incoming_values.max()
            ]

        })

        statistics["Reference"] = (
            statistics["Reference"].round(4)
        )

        statistics["Incoming"] = (
            statistics["Incoming"].round(4)
        )

        st.dataframe(
            statistics,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# 14. MODEL PERFORMANCE MONITORING
# ============================================================

elif page == "Model Performance":

    st.header("Model Performance Monitoring")

    st.subheader("Model Evaluation Metrics")

    metric_names = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc"
    ]

    metric_labels = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ]

    if model_metrics:

        metric_values = [
            safe_float(model_metrics.get(name))
            for name in metric_names
        ]

        metric_df = pd.DataFrame({

            "Metric": metric_labels,

            "Score": metric_values

        })

        fig = px.bar(
            metric_df,
            x="Metric",
            y="Score",
            color="Score",
            color_continuous_scale="Blues",
            text="Score",
            title="Model Evaluation Metrics"
        )

        fig.update_traces(
            texttemplate="%{text:.3f}",
            textposition="outside"
        )

        fig.update_layout(
            template="plotly_dark",
            yaxis_title="Score",
            xaxis_title="Evaluation Metric",
            yaxis_range=[0, 1.1]
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.markdown("---")

        st.subheader("Detailed Model Metrics")

        display_metrics = metric_df.copy()

        display_metrics["Score"] = (
            display_metrics["Score"].round(4)
        )

        st.dataframe(
            display_metrics,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            f"MLflow experiment model: {model_name}"
        )

    else:

        st.warning(
            "Model performance metrics are not available. "
            "Check your MLflow tracking database."
        )

    st.markdown("---")

    # Model artifacts

    st.subheader("Model Artifacts")

    col1, col2 = st.columns(2)

    with col1:

        if artifacts["best_model"]:

            st.success(
                "Best model artifact is available."
            )

        else:

            st.error(
                "Best model artifact is missing."
            )

    with col2:

        if artifacts["preprocessor"]:

            st.success(
                "Preprocessor artifact is available."
            )

        else:

            st.error(
                "Preprocessor artifact is missing."
            )

    st.markdown("---")

    st.subheader("Model Information")

    st.write(f"**Model:** {model_name}")

    st.write(
        f"**Model path:** {MODEL_PATH}"
    )

    st.write(
        f"**MLflow database:** {MLFLOW_DB_PATH}"
    )

# ============================================================
# 15. HISTORICAL MONITORING
# ============================================================

elif page == "Monitoring History":

    st.header("Historical Data Drift Monitoring")

    if history.empty:

        st.warning(
            "No monitoring history records found. "
            "Run drift detection to generate history."
        )

    else:

        st.success(
            f"Loaded {len(history)} monitoring history records."
        )

        st.subheader("Monitoring History Records")

        st.dataframe(
            history,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("---")

        # Find timestamp column

        date_candidates = [
            "timestamp",
            "date",
            "monitoring_date",
            "created_at"
        ]

        date_column = next(
            (
                col for col in date_candidates
                if col in history.columns
            ),
            None
        )

        # Find drift count column

        count_candidates = [
            "drifted_features_count",
            "drift_count",
            "num_drifted_features",
            "drifted_features"
        ]

        count_column = next(
            (
                col for col in count_candidates
                if col in history.columns
            ),
            None
        )

        # Historical drift chart

        st.subheader(
            "Historical Drifted Feature Count"
        )

        if date_column:

            chart_df = history.copy()

            chart_df[date_column] = pd.to_datetime(
                chart_df[date_column],
                errors="coerce"
            )

            chart_df = chart_df.dropna(
                subset=[date_column]
            )

            if count_column:

                # Convert the stored drifted_features field into a
                # numeric count. This supports Python-list strings,
                # comma-separated feature names, and numeric counts.

                if count_column == "drifted_features":

                    chart_df["drifted_count"] = chart_df[
                        count_column
                    ].apply(count_drifted_features)

                    # If a row is marked as drifted but its feature
                    # list is empty/unreadable, use the explicit
                    # drift status as a safe fallback for the chart.

                    if "overall_status" in chart_df.columns:

                        drift_status = (
                            chart_df["overall_status"]
                            .astype(str)
                            .str.upper()
                            .eq("DRIFT DETECTED")
                        )

                        missing_count = (
                            chart_df["drifted_count"].fillna(0).eq(0)
                            & drift_status
                        )

                        if "total_features" in chart_df.columns:

                            fallback_count = pd.to_numeric(
                                chart_df["total_features"],
                                errors="coerce"
                            )

                            chart_df.loc[
                                missing_count, "drifted_count"
                            ] = fallback_count[missing_count].fillna(1)

                        else:

                            chart_df.loc[
                                missing_count, "drifted_count"
                            ] = 1

                    y_column = "drifted_count"

                else:

                    chart_df[count_column] = pd.to_numeric(
                        chart_df[count_column],
                        errors="coerce"
                    )

                    y_column = count_column

                chart_df[y_column] = pd.to_numeric(
                    chart_df[y_column],
                    errors="coerce"
                ).fillna(0)

                chart_df = chart_df.sort_values(
                    date_column
                )

                # Display calculated values for verification

                chart_display = chart_df[
                    [date_column, y_column]
                ].copy()

                chart_display = chart_display.rename(
                    columns={
                        date_column: "Monitoring Date",
                        y_column: "Drifted Feature Count"
                    }
                )

                st.caption(
                    "Calculated historical drift counts"
                )

                st.dataframe(
                    chart_display,
                    use_container_width=True,
                    hide_index=True
                )

                # Historical drift line chart

                fig = px.line(
                    chart_df,
                    x=date_column,
                    y=y_column,
                    markers=True,
                    title="Historical Drifted Feature Count"
                )

                max_count = max(
                    1,
                    int(chart_df[y_column].max())
                )

                fig.update_layout(
                    template="plotly_dark",
                    xaxis_title="Monitoring Date",
                    yaxis_title="Number of Drifted Features",
                    yaxis=dict(
                        range=[0, max_count + 1],
                        dtick=1
                    )
                )

                fig.update_traces(
                    marker=dict(size=9),
                    line=dict(width=3)
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            else:

                st.info(
                    "No recognized drift-count column "
                    "was found for plotting."
                )

        else:

            st.info(
                "No recognized timestamp column was found."
            )

        st.markdown("---")

        # Overall monitoring status chart

        if "overall_status" in history.columns:

            st.subheader(
                "Monitoring Status Distribution"
            )

            status_counts = (
                history["overall_status"]
                .value_counts()
                .reset_index()
            )

            status_counts.columns = [
                "Status",
                "Count"
            ]

            fig = px.pie(
                status_counts,
                names="Status",
                values="Count",
                hole=0.4,
                title="Monitoring Status Distribution",
                color="Status",
                color_discrete_map={
                    "STABLE": "#22C55E",
                    "WARNING": "#F59E0B",
                    "DRIFT DETECTED": "#EF4444"
                }
            )

            fig.update_layout(
                template="plotly_dark"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


# ============================================================
# 16. FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; color:#8492A6;">

        Recruitment Prediction MLOps Platform

        <br><br>

        Data Drift Monitoring |
        MLflow |
        Streamlit |
        Plotly

    </div>
    """,
    unsafe_allow_html=True
)            
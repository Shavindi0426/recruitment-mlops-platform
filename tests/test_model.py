
import joblib
import pandas as pd

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
PREPROCESSOR_PATH = PROJECT_ROOT / "models" / "preprocessor.joblib"

FEATURES = [
    "experience_years",
    "skills_match_score",
    "assessment_score",
    "industry_experience_years",
    "interview_score"
]


def load_model_and_preprocessor():
    assert MODEL_PATH.exists(), "Model file not found"
    assert PREPROCESSOR_PATH.exists(), "Preprocessor file not found"

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    return model, preprocessor


def test_model_file_exists():
    assert MODEL_PATH.exists()


def test_preprocessor_file_exists():
    assert PREPROCESSOR_PATH.exists()


def test_model_loads_successfully():
    model, _ = load_model_and_preprocessor()

    assert model is not None
    assert hasattr(model, "predict")


def test_preprocessor_loads_successfully():
    _, preprocessor = load_model_and_preprocessor()

    assert preprocessor is not None
    assert hasattr(preprocessor, "transform")


def test_model_prediction():
    model, preprocessor = load_model_and_preprocessor()

    candidate = pd.DataFrame([{
        "experience_years": 5,
        "skills_match_score": 85,
        "assessment_score": 90,
        "industry_experience_years": 4,
        "interview_score": 88
    }], columns=FEATURES)

    processed_candidate = preprocessor.transform(candidate)

    prediction = model.predict(processed_candidate)

    assert len(prediction) == 1
    assert prediction[0] in [0, 1]


def test_model_probability_prediction():
    model, preprocessor = load_model_and_preprocessor()

    assert hasattr(model, "predict_proba")

    candidate = pd.DataFrame([{
        "experience_years": 5,
        "skills_match_score": 85,
        "assessment_score": 90,
        "industry_experience_years": 4,
        "interview_score": 88
    }], columns=FEATURES)

    processed_candidate = preprocessor.transform(candidate)

    probabilities = model.predict_proba(processed_candidate)

    assert probabilities.shape == (1, 2)
    assert abs(probabilities.sum() - 1.0) < 0.0001

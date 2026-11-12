
# ============================================================
# RECRUITMENT MLOPS PLATFORM
# REAL-TIME PREDICTION API
# ============================================================

from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"

PREPROCESSOR_PATH = (
    PROJECT_ROOT / "models" / "preprocessor.joblib"
)

FEATURES = [
    "experience_years",
    "skills_match_score",
    "assessment_score",
    "industry_experience_years",
    "interview_score"
]


# ============================================================
# 2. LOAD MODEL AND PREPROCESSOR
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}. "
        "Run python src/train.py first."
    )

if not PREPROCESSOR_PATH.exists():
    raise FileNotFoundError(
        f"Preprocessor not found: {PREPROCESSOR_PATH}. "
        "Run the training pipeline first."
    )

model = joblib.load(MODEL_PATH)

preprocessor = joblib.load(PREPROCESSOR_PATH)

print("Recruitment model loaded successfully!")

print("Preprocessor loaded successfully!")


# ============================================================
# 3. CREATE FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Recruitment Prediction API",

    description=(
        "An AI-powered recruitment prediction platform "
        "with automated model retraining and MLOps monitoring."
    ),

    version="2.0.0"
)


# ============================================================
# 4. INPUT DATA VALIDATION
# ============================================================

class CandidateInput(BaseModel):

    experience_years: float = Field(
        ge=0,
        le=50,
        description="Total years of work experience"
    )

    skills_match_score: float = Field(
        ge=0,
        le=100,
        description="Candidate skills match score"
    )

    assessment_score: float = Field(
        ge=0,
        le=100,
        description="Candidate assessment score"
    )

    industry_experience_years: float = Field(
        ge=0,
        le=50,
        description="Years of industry experience"
    )

    interview_score: float = Field(
        ge=0,
        le=100,
        description="Candidate interview score"
    )


# ============================================================
# 5. ROOT ENDPOINT
# ============================================================

@app.get("/")
def home():

    return {
        "message": "Welcome to Recruitment Prediction API",

        "status": "Running",

        "version": "2.0.0",

        "documentation": "/docs"
    }


# ============================================================
# 6. HEALTH CHECK ENDPOINT
# ============================================================

@app.get("/health")
def health_check():

    return {

        "status": "Healthy",

        "model_loaded": model is not None,

        "preprocessor_loaded": preprocessor is not None

    }


# ============================================================
# 7. REAL-TIME PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict_candidate(candidate: CandidateInput):

    try:

        # ----------------------------------------------------
        # STEP 1: RECEIVE CANDIDATE INFORMATION
        # ----------------------------------------------------

        input_data = pd.DataFrame(
            [candidate.model_dump()],
            columns=FEATURES
        )

        # ----------------------------------------------------
        # STEP 2: APPLY TRAINED PREPROCESSING PIPELINE
        # ----------------------------------------------------

        processed_data = preprocessor.transform(
            input_data
        )

        # ----------------------------------------------------
        # STEP 3: GENERATE MODEL PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(
            processed_data
        )[0]

        # ----------------------------------------------------
        # STEP 4: GENERATE PREDICTION PROBABILITY
        # ----------------------------------------------------

        probabilities = model.predict_proba(
            processed_data
        )[0]

        class_index = list(model.classes_).index(
            prediction
        )

        confidence = float(
            probabilities[class_index]
        )

        # ----------------------------------------------------
        # STEP 5: CONVERT PREDICTION INTO OUTCOME
        # ----------------------------------------------------

        outcome_mapping = {
            1: "Pass",
            0: "Fail"
        }

        outcome = outcome_mapping.get(
            int(prediction),
            str(prediction)
        )

        # ----------------------------------------------------
        # STEP 6: RETURN PREDICTION RESULT
        # ----------------------------------------------------

        return {

            "candidate_information": candidate.model_dump(),

            "prediction": outcome,

            "prediction_class": int(prediction),

            "confidence": round(
                confidence * 100,
                2
            ),

            "status": "Success"

        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )


# ============================================================
# 8. API INFORMATION
# ============================================================

@app.get("/api-info")
def api_info():

    return {

        "project": "Recruitment MLOps Platform",

        "model": type(model).__name__,

        "preprocessor": type(preprocessor).__name__,

        "purpose": "Predict candidate assessment outcomes",

        "features": FEATURES,

        "available_endpoints": [

            "/",

            "/health",

            "/predict",

            "/api-info"

        ]

    }


# ============================================================
# 9. EXECUTION INFORMATION
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )

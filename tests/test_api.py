
import sys
from pathlib import Path

# Add the project root to Python's import path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from api.main import app


client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "Healthy"
    assert data["model_loaded"] is True
    assert data["preprocessor_loaded"] is True


def test_api_info_endpoint():
    response = client.get("/api-info")

    assert response.status_code == 200


def test_prediction_endpoint():
    candidate = {
        "experience_years": 5,
        "skills_match_score": 85,
        "assessment_score": 90,
        "industry_experience_years": 4,
        "interview_score": 88
    }

    response = client.post("/predict", json=candidate)

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] in ["Pass", "Fail"]
    assert isinstance(data["confidence"], (int, float))
    assert 0 <= data["confidence"] <= 100
    assert data["status"] == "Success"


def test_prediction_with_valid_low_scores():
    candidate = {
        "experience_years": 1,
        "skills_match_score": 20,
        "assessment_score": 25,
        "industry_experience_years": 0,
        "interview_score": 30
    }

    response = client.post("/predict", json=candidate)

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] in ["Pass", "Fail"]

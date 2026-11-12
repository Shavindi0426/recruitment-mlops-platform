
"""
============================================================
RECRUITMENT MLOPS PLATFORM
MAIN PIPELINE ORCHESTRATOR
============================================================

Coordinates:
1. Synthetic recruitment dataset generation
2. Data preparation and preprocessing
3. Initial model training and experiment tracking
4. Incoming drift-test data generation
5. Data drift detection and monitoring
6. Optional automated model retraining

Usage:
    python main.py

    python main.py --skip-generation

    python main.py --run-retraining
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

SCRIPTS = {
    "generate_data": PROJECT_ROOT / "src" / "preprocessing.py",
    "prepare_data": PROJECT_ROOT / "src" / "prepare_data.py",
    "train_model": PROJECT_ROOT / "src" / "train.py",
    "generate_drift_data": (
        PROJECT_ROOT / "src" / "generate_drift_data.py"
    ),
    "detect_drift": PROJECT_ROOT / "src" / "drift_detection.py",
    "retrain_model": PROJECT_ROOT / "src" / "retrain.py",
}


# ============================================================
# 2. DISPLAY PIPELINE INFORMATION
# ============================================================

def display_header():

    print("\n" + "=" * 65)
    print("RECRUITMENT MLOPS PLATFORM")
    print("MAIN PIPELINE ORCHESTRATOR")
    print("=" * 65)

    print("Project directory:", PROJECT_ROOT)

    print(
        "Execution started:",
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )


# ============================================================
# 3. EXECUTE PIPELINE SCRIPTS
# ============================================================

def run_script(script_name, description):

    script_path = SCRIPTS[script_name]

    print("\n" + "-" * 65)
    print(description)
    print("-" * 65)

    if not script_path.exists():

        raise FileNotFoundError(
            f"Required script not found: {script_path}"
        )

    command = [
        sys.executable,
        str(script_path)
    ]

    print("Executing:", script_path.name)

    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True
    )

    print("\nCompleted:", description)


# ============================================================
# 4. RUN COMPLETE MLOPS PIPELINE
# ============================================================

def run_pipeline(skip_generation=False, run_retraining=False):

    display_header()

    try:

        # ----------------------------------------------------
        # STEP 1: GENERATE RECRUITMENT DATASET
        # ----------------------------------------------------

        if not skip_generation:

            run_script(
                "generate_data",
                "STEP 1: GENERATING RECRUITMENT DATASET"
            )

        else:

            print("\nSkipping dataset generation.")

        # ----------------------------------------------------
        # STEP 2: DATA PREPARATION AND PREPROCESSING
        # ----------------------------------------------------

        run_script(
            "prepare_data",
            "STEP 2: DATA PREPARATION AND PREPROCESSING"
        )

        # ----------------------------------------------------
        # STEP 3: TRAIN INITIAL MACHINE LEARNING MODELS
        # ----------------------------------------------------

        run_script(
            "train_model",
            "STEP 3: MODEL TRAINING AND EXPERIMENT TRACKING"
        )

        # ----------------------------------------------------
        # STEP 4: GENERATE INCOMING DRIFT-TEST DATA
        # ----------------------------------------------------

        run_script(
            "generate_drift_data",
            "STEP 4: GENERATING INCOMING DRIFT-TEST DATA"
        )

        # ----------------------------------------------------
        # STEP 5: DETECT DATA DRIFT
        # ----------------------------------------------------

        run_script(
            "detect_drift",
            "STEP 5: DATA DRIFT DETECTION AND MONITORING"
        )

        # ----------------------------------------------------
        # STEP 6: OPTIONAL AUTOMATED MODEL RETRAINING
        # ----------------------------------------------------

        if run_retraining:

            print("\nNOTE:")
            print(
                "Retraining uses the labelled raw recruitment "
                "dataset, not the unlabelled drift-test dataset."
            )

            run_script(
                "retrain_model",
                "STEP 6: AUTOMATED MODEL RETRAINING"
            )

        else:

            print("\nAutomated retraining was not requested.")

        # ----------------------------------------------------
        # FINAL PIPELINE STATUS
        # ----------------------------------------------------

        print("\n" + "=" * 65)
        print("MLOPS PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 65)

        print("Dataset generation: Completed or skipped")
        print("Data preprocessing: Completed")
        print("Model training: Completed")
        print("MLflow tracking: Completed")
        print("Drift data generation: Completed")
        print("Drift monitoring: Completed")

        print(
            "Automated retraining:",
            "Executed" if run_retraining else "Not executed"
        )

        print(
            "\nExecution finished:",
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

    except subprocess.CalledProcessError as error:

        print("\n" + "=" * 65)
        print("PIPELINE FAILED!")
        print("=" * 65)

        print("A pipeline script returned an error.")
        print("Exit code:", error.returncode)
        print("Failed command:", error.cmd)

        raise

    except Exception as error:

        print("\nPIPELINE ERROR:", str(error))

        raise


# ============================================================
# 5. COMMAND-LINE ARGUMENTS
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Recruitment MLOps Pipeline Orchestrator"
    )

    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Use the existing raw recruitment dataset."
    )

    parser.add_argument(
        "--run-retraining",
        action="store_true",
        help="Run the automated retraining script."
    )

    args = parser.parse_args()

    run_pipeline(
        skip_generation=args.skip_generation,
        run_retraining=args.run_retraining
    )


# ============================================================
# 6. APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()

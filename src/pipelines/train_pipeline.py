import subprocess

def run_step(step_name, command):
    print(f"\n===== Running {step_name} =====")
    result = subprocess.run(command, shell=True)

    if result.returncode != 0:
        raise RuntimeError(f"{step_name} failed with return code {result.returncode}")
    
    print(f"\n===== {step_name} completed successfully =====")

def main():
    run_step(
        "Data Preprocessing",
        "Python -m src.data.run_preprocessing"
    )

    run_step(
        "Feature Engineering",
        "python -m src.features.run_features"
    )

    run_step(
        "Model Training with MLflow",
        "python -m src.models.train_mlflow"
    )

    print("\n Training Pipeline completed successfully")

if __name__ == "__main__":
    main()
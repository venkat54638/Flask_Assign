import mlflow
from mlflow.tracking import MlflowClient

# Set the tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")

def register_model_from_run(run_id="104e824fe1974574be82cc8acba449b7", 
                           model_registry_name="Loan_Default_Best_Model", 
                           model_artifact_path="artifacts/model.pkl"):
    client = MlflowClient()

    # Get run details
    try:
        run = client.get_run(run_id)
        print(f"\n✅ Found run ID: {run_id}")
        print(f"   Status: {run.info.status}")
        print(f"   Metrics: {run.data.metrics}")
        print(f"   Tags: {run.data.tags}")
    except mlflow.exceptions.MlflowException as e:
        print(f"❌ Run ID '{run_id}' not found: {str(e)}")
        return

    # Check if model artifact exists
    best_model_uri = f"runs:/{run_id}/{model_artifact_path}"
    try:
        artifacts = client.list_artifacts(run_id)
        artifact_paths = [artifact.path for artifact in artifacts]
        print(f"Artifacts for run {run_id}: {artifact_paths}")
        if model_artifact_path not in artifact_paths:
            print(f"❌ Model artifact '{model_artifact_path}' not found in run {run_id}.")
            return
    except Exception as e:
        print(f"❌ Error checking artifacts for run {run_id}: {str(e)}")
        return

    # Extract model details
    accuracy = run.data.metrics.get("accuracy", "N/A")
    model_name = run.data.tags.get("model_name", "Unknown")

    print(f"\n✅ Model details for run ID: {run_id}")
    print(f"   Model: {model_name}")
    print(f"   Accuracy: {accuracy}")
    print(f"   URI: {best_model_uri}")

    # Register model
    try:
        result = mlflow.register_model(model_uri=best_model_uri, name=model_registry_name)
        print(f"\n✅ Model registered as '{model_registry_name}' with version {result.version}")
    except mlflow.exceptions.MlflowException as e:
        print(f"❌ Failed to register model: {str(e)}")
        return

    # Promote to Production
    try:
        client.transition_model_version_stage(
            name=model_registry_name,
            version=result.version,
            stage="Production",
            archive_existing_versions=True
        )
        print(f"\n🚀 Model '{model_registry_name}' (version {result.version}) promoted to 'Production'.")
    except mlflow.exceptions.MlflowException as e:
        print(f"❌ Failed to promote model to Production: {str(e)}")
        return

if __name__ == "__main__":
    register_model_from_run(run_id="104e824fe1974574be82cc8acba449b7", 
                            model_registry_name="Loan_Default_Best_Model", 
                            model_artifact_path="artifacts/model.pkl")

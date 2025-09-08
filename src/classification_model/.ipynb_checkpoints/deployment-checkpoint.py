# deployment.py

import os
import shutil
import mlflow
from mlflow.tracking import MlflowClient

def deploy_best_model(tracking_uri="http://localhost:5000", 
                     registered_model_name="TitanicClassifier", 
                     output_dir="deployment/model",
                     metric_name="accuracy"):
    """Deploy the best model based on specified metric."""
    
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    
    # Find best version by metric
    all_versions = client.get_latest_versions(name=registered_model_name, stages=["None", "Staging", "Production"])
    best_version = None
    best_score = -1
    
    for v in all_versions:
        run = mlflow.get_run(v.run_id)
        score = run.data.metrics.get(metric_name)
        if score is not None and score > best_score:
            best_score = score
            best_version = v.version
    
    if best_version is None:
        print(f"No model found with {metric_name} metric.")
        return False
    
    print(f"Best model: version {best_version} with {metric_name} {best_score:.4f}")
    
    # Promote to production
    client.transition_model_version_stage(
        name=registered_model_name,
        version=best_version,
        stage="Production",
        archive_existing_versions=True
    )
    print(f"Model {registered_model_name} version {best_version} promoted to Production.")
    
    # Package model
    model_uri = f"models:/{registered_model_name}/Production"
    spark_model = mlflow.spark.load_model(model_uri)
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    mlflow.spark.save_model(spark_model=spark_model, path=output_dir)
    print(f"Packaged Production model v{best_version} saved locally at: {output_dir}")
    return True


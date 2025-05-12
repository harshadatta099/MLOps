import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()
experiment = client.get_experiment_by_name("iris_rf")
run = client.search_runs(experiment.experiment_id, order_by=["metrics.accuracy DESC"])[0]

model_uri = f"runs:/{run.info.run_id}/model"
model_name = "IrisRandomForest"

mlflow.register_model(model_uri=model_uri, name=model_name)
client.transition_model_version_stage(name=model_name, version=1, stage="Production")

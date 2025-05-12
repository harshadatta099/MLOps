import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Set the MLflow tracking server URI
mlflow.set_tracking_uri("http://10.110.0.143:5000/")
mlflow.set_experiment("iris_rf")

with mlflow.start_run():
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target)

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)

    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")

from fastapi import FastAPI
import mlflow.sklearn
import numpy as np

app = FastAPI()
model = mlflow.sklearn.load_model("models:/IrisRandomForest/Production")

@app.get("/predict/")
def predict(sepal_length: float, sepal_width: float, petal_length: float, petal_width: float):
    inputs = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(inputs)
    return {"prediction": prediction.tolist()}

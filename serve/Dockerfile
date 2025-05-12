FROM python:3.9-slim
WORKDIR /app
COPY app.py .
RUN pip install fastapi uvicorn mlflow scikit-learn numpy
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7001"]

from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
import pandas as pd
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from deployment.drift_detection import detect_drift
from deployment.scheduler import schedule_retraining
import subprocess
import sys
import os

spark = SparkSession.builder.appName("TitanicClassifierAPI").getOrCreate()

schema = StructType([
    StructField("PassengerId", IntegerType(), True),
    StructField("Pclass", IntegerType(), True),
    StructField("Name", StringType(), True),
    StructField("Sex", StringType(), True),
    StructField("Age", DoubleType(), True),
    StructField("SibSp", IntegerType(), True),
    StructField("Parch", IntegerType(), True),
    StructField("Ticket", StringType(), True),
    StructField("Fare", DoubleType(), True),
    StructField("Cabin", StringType(), True),
    StructField("Embarked", StringType(), True),
])

MODEL_PATH = "deployment/model/sparkml"
model = PipelineModel.load(MODEL_PATH)

app = FastAPI(title="TitanicClassifier Inference API")

class PredictionRequest(BaseModel):
    PassengerId: Optional[int] = None
    Pclass: int
    Name: Optional[str] = None
    Sex: str
    Age: Optional[float] = None
    SibSp: Optional[int] = None
    Parch: int
    Ticket: Optional[str] = None
    Fare: Optional[float] = None
    Cabin: Optional[str] = None
    Embarked: Optional[str] = None


# 🔹 Initialize buffer at startup
BUFFER_FILE = "buffer.txt"
if os.path.exists(BUFFER_FILE):
    buffer_df = pd.read_json(BUFFER_FILE, lines=True)
else:
    buffer_df = pd.DataFrame(columns=[
        "PassengerId", "Pclass", "Name", "Sex", "Age", 
        "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"
    ])
    buffer_df.to_json(BUFFER_FILE, orient="records", lines=True)

def create_dataframe(data, df: pd.DataFrame):
    df = df.copy()
    if len(df) > 500:
        df = df.iloc[-499:]
    df = pd.concat([df, pd.DataFrame(data, index=[0])], ignore_index=True)
    return df

schedule_retraining(interval_days=7)

def check_drift_task(df: pd.DataFrame, drift_threshold: float = 0.5):
    drift_detected, drift_score, drift_flags = detect_drift(df, threshold=0.05)
    print("Background task is initiated")
    print("Drift score:", drift_score)

    if drift_detected and drift_score >= drift_threshold:
        print(f"Drift detected (score={drift_score:.3f}). Triggering async retraining...")

        subprocess.Popen([sys.executable, "train/train.py"],
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    else:
        print("No significant drift detected. No retraining triggered.")


@app.post("/predict")
def predict(request: PredictionRequest, background_tasks: BackgroundTasks):
    global buffer_df

    # 🔹 Update buffer with new request
    buffer_df = create_dataframe(request.dict(), buffer_df)
    buffer_df.to_json(BUFFER_FILE, orient="records", lines=True)

    input_data = [request.dict()]
    input_df = spark.createDataFrame(input_data, schema=schema)
    preds = model.transform(input_df)
    result_df = preds.select("prediction", "probability").toPandas()
    result_df['probability'] = result_df['probability'].apply(lambda vec: vec.tolist())
    predictions = result_df.to_dict('records')

    # 🔹 Run drift detection in background
    background_tasks.add_task(check_drift_task, buffer_df)

    return {"predictions": predictions}

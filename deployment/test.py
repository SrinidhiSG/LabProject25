import requests
import pandas as pd
from classification_model.spark_Session import spark_session
url = "http://localhost:8000/predict"

test_df=pd.read_csv(r"/home/sridharsg/Documents/AILabProject_Updated/test.csv")
test_df = test_df.where(pd.notnull(test_df), None)

predictions = []

for _, row in test_df.iterrows():
    payload = row.to_dict()
    response = requests.post(url, json=payload)
    predictions.append(response.json()["predictions"][0])

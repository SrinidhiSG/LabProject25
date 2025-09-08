# Importing the required packages
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, Imputer, OneHotEncoder
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier, DecisionTreeClassifier, NaiveBayes, LinearSVC
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql.types import *
from .deployment import deploy_best_model

import mlflow
import os
import json
import shutil
from mlflow.exceptions import MlflowException

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import seaborn as sns
from datetime import datetime
import time

from .spark_Session import spark_session
from .preprocessing import preprocess_data
from .ref_profile import build_reference_profile

import warnings
warnings.filterwarnings('ignore')
# import mlflow.spark # Removed due to compatibility issue

import os

client = mlflow.tracking.MlflowClient()
registered_model_name="TitanicClassifier"


# Create artifacts folder and two sub folders for confusion matrix and evaluation report
artifact_folder = "artifacts_temp"
conf_mat_folder = os.path.join(artifact_folder, "confusion_matrix")
eval_report_folder =  os.path.join(artifact_folder, "evaluation_report")
if not os.path.exists(conf_mat_folder):
  os.makedirs(conf_mat_folder)
if not os.path.exists(eval_report_folder):
  os.makedirs(eval_report_folder)

# Loading the dataset
df = pd.read_csv(r"/home/sridharsg/Documents/AILabProject_Updated/train.csv")
reference_profile = build_reference_profile(df)

os.makedirs("deployment", exist_ok=True)
with open("deployment/reference_profile.json", "w") as f:
    json.dump(reference_profile, f, indent=4)

print("Reference profile saved to reference_profile.json")

# Initiate spark session and mlflow
spark = spark_session()
mlflow.set_tracking_uri("http://localhost:5000")

schema = StructType([
    StructField("PassengerId", IntegerType(), True),
    StructField("Survived", IntegerType(), True),
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
df = spark.read.csv(r"/home/sridharsg/Documents/AILabProject_Updated/train.csv", header=True, inferSchema=True)
df = df.withColumnRenamed('Survived', 'label')
print('spark is initiated')
# Splitting the dataset
train_df = df.randomSplit([0.8, 0.2], seed=42)[0]
test_df = df.randomSplit([0.8, 0.2], seed=42)[1]

# Function call of Data Preprocessing Pipeline
preprocess_stages = preprocess_data()
print('preprocess_stages pipeline is called')
# Defining the Models
lr = LogisticRegression(featuresCol='features', labelCol='label')
rf = RandomForestClassifier(featuresCol='features', labelCol='label')
gbt = GBTClassifier(featuresCol='features', labelCol='label')
dt = DecisionTreeClassifier(featuresCol='features', labelCol='label')
nb = NaiveBayes(featuresCol='features', labelCol='label')
lsvc = LinearSVC(featuresCol='features', labelCol='label')

# Hyperparameter tuning
lr_paramGrid = (ParamGridBuilder()
              .addGrid(lr.regParam, [0.1, 0.01])
              .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])
              .build())

rf_paramGrid = (ParamGridBuilder()
              .addGrid(rf.numTrees, [10, 20])
              .addGrid(rf.maxDepth, [5, 10])
              .addGrid(rf.impurity, ['gini', 'entropy'])
              .build())

gbt_paramGrid = (ParamGridBuilder()
                .addGrid(gbt.maxIter, [10, 20])
                .addGrid(gbt.stepSize, [0.1, 0.01])
                .addGrid(gbt.maxDepth, [5, 10])
                .build())

dt_paramGrid = (ParamGridBuilder()
              .addGrid(dt.maxDepth, [5, 10])
              .addGrid(dt.impurity, ['gini', 'entropy'])
              .build())

nb_paramGrid = (ParamGridBuilder()
              .addGrid(nb.smoothing, [1.0, 2.0])
              .build())

lsvc_paramGrid = (ParamGridBuilder()
                .addGrid(lsvc.maxIter, [10, 20])
                .addGrid(lsvc.regParam, [0.1, 0.01])
                .build())

pipelines = [(Pipeline(stages=preprocess_stages + [lr]), lr_paramGrid),
            (Pipeline(stages=preprocess_stages + [rf]), rf_paramGrid),
            (Pipeline(stages=preprocess_stages + [gbt]), gbt_paramGrid),
            (Pipeline(stages=preprocess_stages + [dt]), dt_paramGrid),
            (Pipeline(stages=preprocess_stages + [nb]), nb_paramGrid),
            (Pipeline(stages=preprocess_stages + [lsvc]), lsvc_paramGrid)]

#  Performance Metrics
accuracy = MulticlassClassificationEvaluator(labelCol='label', metricName='accuracy')
f1 = MulticlassClassificationEvaluator(labelCol='label', metricName='f1')
auc = BinaryClassificationEvaluator(labelCol='label', metricName='areaUnderROC')
precision = MulticlassClassificationEvaluator(labelCol='label', metricName='precisionByLabel')

best_model = None
best_acc = 0
best_auc = 0
best_precision = 0
best_f1 = 0

train_df = train_df.coalesce(4).cache()
test_df = test_df.coalesce(2).cache()
print('CV is initiated')
# Training the models
for pipeline, paramGrid in pipelines:
  model_name = pipeline.getStages()[-1].__class__.__name__
  print(model_name)        
  crossval = CrossValidator(estimator=pipeline,
                              estimatorParamMaps=paramGrid,
                              evaluator=accuracy,
                              numFolds=5,
                              seed=42,
                              parallelism=4)

  #time=datetime.now().strftime("%d%m%Y%H%M%S")
  with mlflow.start_run(run_name=model_name):
      # mlflow.spark.autolog() # Removed due to compatibility issue

      cvModel = crossval.fit(train_df)
      predictions = cvModel.transform(test_df)

      accuracy_score = accuracy.evaluate(predictions)
      f1_score = f1.evaluate(predictions)
      auc_score = auc.evaluate(predictions)
      precision_score = precision.evaluate(predictions)

      # Log parameters and metrics manually
      best_model_stage = cvModel.bestModel.stages[-1]
      best_params = {param.name: best_model_stage.getOrDefault(param)
                    for param in best_model_stage.extractParamMap()}
      mlflow.log_params(best_params)
      '''
      # Save params to JSON for artifacts
      with open("artifacts/best_params.json", "w") as f:
          json.dump(best_params, f, indent=2)
      
      unique_params = {f"{model_name}.{k}": v for k, v in best_params.items()}
      mlflow.log_params(unique_params)
      '''
      
      mlflow.log_metric('accuracy', accuracy_score)
      mlflow.log_metric('f1', f1_score)
      mlflow.log_metric('auc', auc_score)
      mlflow.log_metric('precision', precision_score)

      # Confusion Matrix plot
      conf_mat = confusion_matrix(predictions.select('label').toPandas(), predictions.select('prediction').toPandas())

      plt.figure(figsize=(10, 5))
      sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=['Survived', 'Not Survived'], yticklabels=['Survived', 'Not Survived'])
      plt.xlabel('Predicted')
      plt.ylabel('Actual')
      plt.title('Confusion Matrix')
      cm_file = os.path.join(conf_mat_folder, f"{model_name}_confusion_matrix.png")
      plt.savefig(cm_file)
      plt.close()

      # Log confusion matrix
      mlflow.log_artifact(cm_file)

      report = {
          'accuracy': accuracy_score,
          'f1': f1_score,
          'auc': auc_score,
          'precision': precision_score
      }
      report_file = os.path.join(eval_report_folder, f"{model_name}_evaluation_report.json")
      with open(report_file, 'w') as f:
          json.dump(report, f, indent=2)

      #Log evaluation report
      mlflow.log_artifact(report_file)
    
      if accuracy_score > best_acc:
        best_acc = accuracy_score
        best_f1 = f1_score
        best_auc = auc_score
        best_precision = precision_score
        best_model = cvModel

best_pipeline_model = best_model.bestModel
print('\n Best Pipeline Model:', best_pipeline_model.stages[-1])

best_pred = best_pipeline_model.transform(test_df)

# Inspect stages + log parameters uniquely
for stage in best_pipeline_model.stages:
      stage_name = stage.__class__.__name__
      stage_params = {
          f"{stage_name}.{p.name}": stage.getOrDefault(p) if stage.isSet(p) else None
          for p in stage.params
      }
      prefixed_params = {f"{stage_name}.{k}": v for k, v in stage_params.items()}
      #mlflow.log_params(prefixed_params)
      #mlflow.log_params(stage_params)

    
    # ---------------- Log Metrics ----------------
with mlflow.start_run(run_name=registered_model_name):

    mlflow.log_metric("accuracy", best_acc)
    mlflow.log_metric("f1", best_f1)
    mlflow.log_metric("auc", best_auc)
    mlflow.log_metric("precision", best_precision)
    print('Logs are added')
    # Log model
    model_info = mlflow.spark.log_model(
            spark_model=best_pipeline_model,
            artifact_path="spark-model",
            registered_model_name=registered_model_name,
            await_registration_for=300
        )
    print(model_info)
    print(model_info.signature)

time.sleep(200)


new_version = None
timeout = 1000   # seconds
poll_interval = 5
start = time.time()

while time.time() - start < timeout:
    latest_versions = client.get_latest_versions(registered_model_name)
    print(latest_versions)
    if latest_versions:
        # Take the most recent version
        new_version = max(int(v.version) for v in latest_versions)
        print(new_version)
        break
    time.sleep(poll_interval)

print(latest_versions)
print(new_version)

# Move to Staging
client.transition_model_version_stage(
    name=registered_model_name,
    version=new_version,
    stage="Staging"
)
print(f"Model {registered_model_name} version {new_version} moved to Staging")
print("Accuracy Score:", best_acc)

import sys
import os

from deployment import deploy_best_model

deploy_best_model()
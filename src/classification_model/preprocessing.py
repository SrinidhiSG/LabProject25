# preprocessing.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pyspark.ml import Pipeline, Transformer
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.ml.feature import (
    Imputer,
    StringIndexer,
    OneHotEncoder,
    VectorAssembler
)
from pyspark.sql.functions import col, when, mean
from pyspark.sql import DataFrame

# Custom Transformer for creating the Master column
class MasterColumnCreator(Transformer,DefaultParamsReadable, DefaultParamsWritable):
    def __init__(self, column_to_add="Master"):
        super().__init__()
        self.column_to_add = column_to_add

    def _transform(self, df: DataFrame) -> DataFrame:
        return df.withColumn(self.column_to_add, when(col("Name").contains("Master."), 1).otherwise(0))

# Custom Transformer for filling Age
class AgeImputer(Transformer, DefaultParamsReadable, DefaultParamsWritable):
    def __init__(self, mean_master_age=None, mean_female_age=None,mean_male_age=None, inputCol="Age", outputCol="Age"):
        super().__init__()
        self.mean_master_age = None
        self.mean_female_age = None
        self.mean_male_age = None
        self.mean_overall_age = None  #UPDATED
        self.inputCol = inputCol
        self.outputCol = outputCol

    def _transform(self, df: DataFrame) -> DataFrame:
        if self.mean_master_age is None:
             self.mean_master_age = df.filter(col("Master") == 1).select(mean("Age")).collect()[0][0]
             self.mean_female_age = df.filter(col("Sex") == "female").select(mean("Age")).collect()[0][0]
             self.mean_male_age   = df.filter(col("Sex") == "male").select(mean("Age")).collect()[0][0]
             self.mean_overall_age = df.select(mean("Age")).collect()[0][0]  # 🔹 UPDATED

        return df.withColumn(
           self.outputCol,
            when(col(self.inputCol).isNull() & (col("Master") == 1), self.mean_master_age)
             .when(col(self.inputCol).isNull() & (col("Sex") == "female"), self.mean_female_age)
             .when(col(self.inputCol).isNull() & (col("Sex") == "male"), self.mean_male_age)
             .when(col(self.inputCol).isNull(), self.mean_overall_age)  # 🔹 UPDATED fallback
             .otherwise(col(self.inputCol))
        )

# Custom Transformer for filling Gender
class GenderFiller(Transformer, DefaultParamsReadable, DefaultParamsWritable):
    def __init__(self,inputCol="Name", outputCol="Sex"):
        super().__init__()
        self.inputCol = inputCol
        self.outputCol = outputCol

    def _transform(self, df: DataFrame) -> DataFrame:
        return df.withColumn(
            self.outputCol,
            when(
                col(self.outputCol).isNull() &
                (col(self.inputCol).contains("Mr.") | col(self.inputCol).contains("Master.")),
                "male"
            ).when(
                col(self.outputCol).isNull(), "female"
            ).otherwise(col(self.outputCol))
        )

# Custom Transformer for dropping columns
class ColumnDropper(Transformer, DefaultParamsReadable, DefaultParamsWritable):
    def __init__(self, columns_to_drop=None):
        super().__init__()
        self.columns_to_drop = columns_to_drop or []

    def _transform(self, df: DataFrame) -> DataFrame:
        return df.drop(*self.columns_to_drop)


def preprocess_data():

    # --- Step 1: Create master columns
    master_cols = MasterColumnCreator()

    # --- Step 2: Impute numerical features ---
    # Median imputation for discrete numeric variables
    cat_num_imputer = Imputer(
        inputCols=["Pclass", "SibSp", "Parch"],
        outputCols=["Pclass", "SibSp", "Parch"],
        strategy="median"
    )

    # Mean imputation for continuous numeric variable
    fare_imputer = Imputer(
        inputCols=["Fare"],
        outputCols=["Fare"],
        strategy="mean"
    )

    # Custom imputers for Age and Gender
    age_filler = AgeImputer()
    gender_filler = GenderFiller()

    # --- Step 3: Drop unnecessary columns
    drop_unused = ColumnDropper(
        columns_to_drop=["PassengerId", "Name", "Ticket", "Cabin", "Embarked"]
    )

    # --- Step 4: Encode categorical variable (Sex)
    sex_indexer = StringIndexer(
        inputCol="Sex",
        outputCol="idx_Sex",
        handleInvalid="keep"   # UPDATED
    )
    sex_encoder = OneHotEncoder(
        inputCol="idx_Sex",
        outputCol="ohe_Sex",
        handleInvalid="keep"   # UPDATED
    )

    # --- Step 5: Assemble features
    feature_cols = ["Pclass", "ohe_Sex", "Age", "SibSp", "Parch", "Fare"]
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features"
    )

    # --- Return pipeline stages ---
    return [
        master_cols,
        gender_filler, # Fill gender before imputing age which depends on gender
        age_filler,
        cat_num_imputer,
        fare_imputer,
        drop_unused,
        sex_indexer,
        sex_encoder,
        assembler
    ]
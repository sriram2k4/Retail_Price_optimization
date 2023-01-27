import pandas as pd
import numpy as np
from make_dataset import (Ingestor, LabelEncoder, ProcessData)
from build_feature import BuildFeatures
from utils import split_dataset
from build_model import ModelBuilder
from evaluate import Evaluate
from sklearn.metrics import (explained_variance_score, max_error,mean_absolute_error,mean_squared_error,r2_score)

# [X] Config
# [X] Ingester
# [X] Encoding
# [X] Processing
# [X] Build Feature
# [X] Build Model

# Ingester
ingestor = Ingestor("data.csv")
df = ingestor.load_dataset()

#Encode
cats_col = df.select_dtypes(include=["object"]).columns.tolist()
cats_col.remove("date")
label_encoder = LabelEncoder(df,cats_col)
df = label_encoder.fit_transform()

#Process
procesor = ProcessData(df)
df = procesor.remove_null_values()
df = df.drop(["id", "sku_id","date"], axis=1)
# print(df)

#Build_Feature
builder = BuildFeatures(df)
df = builder.build_features()
df.fillna(0, inplace=True)
x_train, x_test, y_train, y_test = split_dataset(df)

#Build model
builder = ModelBuilder(x_train, y_train)
model = builder.build_model()

# Evaluating
evaluate = Evaluate(x=x_test, y=y_test, model=model)
evaluate.evaluate()

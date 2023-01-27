import pandas as pd
from make_dataset import (Ingestor, LabelEncoder, ProcessData)

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
print(df)

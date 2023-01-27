from make_dataset import LabelEncoder
from make_dataset import Ingestor

df = Ingestor('data.csv').load_dataset()
# print(df)
cats_col = df.select_dtypes("object")

print(cats_col)
cats_col = df.select_dtypes("object").columns.tolist()
cats_col.remove("date")

le = LabelEncoder(df,cats_col)
le.fit_transform()
